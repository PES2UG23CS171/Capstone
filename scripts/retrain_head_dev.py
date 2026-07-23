"""
Retrain the 12-class head on FSD50K.dev_audio with the CORRECTED protocol.

Protocol (audit fix — see ARCHITECTURE.md §10 / README):
  * train/val  ← FSD50K.dev_audio, UPLOADER-GROUPED (no Freesound uploader spans
    train and val), so val is an honest generalisation estimate.
  * test       ← FSD50K.eval_audio only (the official held-out set, uploader-
    disjoint from dev by FSD50K construction).
  * backbone   ← FROZEN.  We never touch it: the 960-d embedding is read straight
    out of the already-exported frozen backbone inside efficientat_head12.onnx
    (via an "expose the head's Gemm input" extractor ONNX), so NO torch backbone /
    torchvision / weight download is needed and training is deployment-consistent.
  * head       ← a single Linear(960→12) (matches the deployed Gemm).  We train on
    standardised embeddings and FOLD the standardisation into the exported Gemm
    weights, so the ONNX structure is byte-for-byte the same shape — we only swap
    hd.weight / hd.bias.  No new nodes, no new models.

Window policy (v3 — rolling-window-consistent training):
  The deployed classifier sees a ROLLING 4 s window of a live stream: events
  usually occupy a fraction of the window at an arbitrary position, embedded
  in room tone.  v2 trained on tile-first-4s crops and measurably collapsed on
  such windows (true-class posterior 0.88 → 0.26 for an event at the window
  edge).  v3 therefore extracts, per clip:
    * clips ≥ 4 s: TWO random 4 s crops (deterministic per fname), and
    * clips < 4 s: one cyclic tile from a random phase (matches the runtime
      partial-buffer tile-pad) PLUS one "sparse" variant — the clip placed at
      a random offset inside 4 s of low-level noise floor (event-in-context).
  Same uploader-grouped split as v2 (loaded from fsd50k_split_dev.json), so
  the window policy is the only variable.

Outputs:
  * checkpoints/efficientat_head12_v3.onnx (+ .data)  — window-robust head.
  * checkpoints/dev_emb_cache_v3.npz                  — cached embeddings.
  * checkpoints/fsd50k_split_dev.json                 — the grouped split (reused).

Then evaluates the baselines on the eval test set and prints macro-F1 /
precision / recall / per-class / top-1 / confusion matrix.

Run:  PYTHONPATH=<repo> .venv_poc/bin/python -m scripts.retrain_head_dev
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from math import gcd
from pathlib import Path

import numpy as np

_TARGETS = ("clean", "fan", "hvac", "traffic", "keyboard", "dog", "wind",
            "music", "tv", "speech", "competing_speech", "other")
_TRAINABLE = ("fan", "traffic", "keyboard", "dog", "wind", "music",
              "speech", "competing_speech", "other")
_LAB_IDX = {t: i for i, t in enumerate(_TARGETS)}

ROOT = "dataset/FSD50K"
SR = 32000
WIN_S = 4.0
WIN_N = int(WIN_S * SR)
# v3 window policy: variants per clip and the sparse-context noise floor
# (dB relative to the clip's RMS — quiet room tone under a dominant event).
N_VARIANTS = 2
SPARSE_FLOOR_DB = -35.0


# ── data ────────────────────────────────────────────────────────────────────
def _read_mapped(csv_path: Path):
    from voiceiso.data.fsd50k_labelmap import fsd50k_labels_to_target
    rows = []
    for r in csv.DictReader(open(csv_path, newline="")):
        t = {x for x in fsd50k_labels_to_target(r["labels"]) if x in _TRAINABLE}
        if t:
            rows.append((r["fname"], sorted(t)))
    return rows


def _grouped_split(rows, uploader_of, val_frac, rng):
    by_up = defaultdict(list)
    for fn, t in rows:
        by_up[uploader_of.get(fn, f"__none__{fn}")].append((fn, t))
    ups = sorted(by_up)
    rng.shuffle(ups)
    n_val_target = int(len(rows) * val_frac)
    val, train = [], []
    for up in ups:
        (val if len(val) < n_val_target else train).extend(by_up[up])
    return train, val


def _cap_per_class(rows, cap, rng):
    if not cap:
        return rows
    idx = np.arange(len(rows))
    rng.shuffle(idx)
    per = defaultdict(int)
    keep = []
    for i in idx:
        fn, t = rows[i]
        if any(per[c] < cap for c in t):
            keep.append(rows[i])
            for c in t:
                per[c] += 1
    return keep


# ── embedding extractor (frozen backbone, via ONNX) ──────────────────────────
class _Embedder:
    def __init__(self, head_onnx="checkpoints/efficientat_head12.onnx",
                 out="checkpoints/_embed_extractor.onnx"):
        import onnx
        import onnxruntime as ort
        from voiceiso.stages._efficientat_mel import EfficientATMel
        if not Path(out).exists():
            m = onnx.load(head_onnx)
            gemm = [n for n in m.graph.node if n.op_type == "Gemm" and "hd.weight" in n.input][0]
            self.emb_name = gemm.input[0]
            vi = onnx.helper.ValueInfoProto(); vi.name = self.emb_name
            m.graph.output.append(vi)
            onnx.save(m, out, save_as_external_data=True, location=Path(out).name + ".data")
        opts = ort.SessionOptions(); opts.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(out, sess_options=opts, providers=["CPUExecutionProvider"])
        self.emb_name = [o.name for o in self.sess.get_outputs() if o.name != "posteriors"][0]
        self.in_name = self.sess.get_inputs()[0].name
        self.mel = EfficientATMel(sr=SR)

    def _load_full(self, path: Path):
        """Load the WHOLE clip at the model SR (no cropping here)."""
        import soundfile as sf
        try:
            data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        except Exception:
            return None
        x = data.mean(axis=1)
        if sr != SR:
            from scipy.signal import resample_poly
            g = gcd(sr, SR)
            x = resample_poly(x, SR // g, sr // g).astype("float32")
        return x.astype("float32")

    def _embed_window(self, win: np.ndarray) -> np.ndarray:
        feats = self.mel(win).unsqueeze(1).detach().cpu().numpy().astype(np.float32)
        out = self.sess.run([self.emb_name], {self.in_name: feats})[0]
        return np.asarray(out).reshape(-1).astype(np.float32)

    def embed_variants(self, path: Path, rng: np.random.Generator):
        """Rolling-window-consistent 4 s variants of one clip (v3 policy)."""
        x = self._load_full(path)
        if x is None or len(x) == 0:
            return None
        wins = []
        if len(x) >= WIN_N:
            # Long clip: random crops — the model must recognise the class
            # from an arbitrary 4 s slice, exactly like the live rolling window.
            for _ in range(N_VARIANTS):
                s = int(rng.integers(0, len(x) - WIN_N + 1))
                wins.append(x[s:s + WIN_N])
        else:
            # Short clip, variant A: cyclic tile from a random phase — matches
            # the runtime partial-buffer tile-pad (noise_classifier H2 path).
            reps = int(np.ceil(WIN_N / len(x))) + 1
            tiled = np.tile(x, reps)
            ph = int(rng.integers(0, len(x)))
            wins.append(tiled[ph:ph + WIN_N])
            # Variant B: sparse event-in-context — the clip at a random offset
            # inside 4 s of low-level noise floor (the live-mic regime: a
            # transient occupying a fraction of an otherwise-quiet window).
            rms = float(np.sqrt(np.mean(x * x)) + 1e-9)
            floor = rms * 10.0 ** (SPARSE_FLOOR_DB / 20.0)
            buf = (floor * rng.standard_normal(WIN_N)).astype("float32")
            off = int(rng.integers(0, WIN_N - len(x) + 1))
            buf[off:off + len(x)] += x
            wins.append(buf)
        return [w.astype("float32") for w in wins]


def _embed_rows(emb, rows, audio_dir: Path, name: str, seed: int):
    """One embedding per WINDOW VARIANT (v3: ≥1 per clip, usually 2).

    Variant randomness is seeded per (seed, fname) via crc32 so the cache is
    reproducible regardless of row order.
    """
    import zlib
    X, Y = [], []
    t0 = time.time()
    for i, (fn, tgts) in enumerate(rows):
        rng = np.random.default_rng((seed, zlib.crc32(str(fn).encode())))
        wins = emb.embed_variants(audio_dir / f"{fn}.wav", rng)
        if not wins:
            continue
        y = np.zeros(len(_TARGETS), dtype="float32")
        for t in tgts:
            y[_LAB_IDX[t]] = 1.0
        for w in wins:
            X.append(emb._embed_window(w))
            Y.append(y)
        if (i + 1) % 400 == 0:
            print(f"    {name}: {i+1}/{len(rows)} clips → {len(X)} windows  ({time.time()-t0:.0f}s)", flush=True)
    return np.stack(X), np.stack(Y)


# ── train ────────────────────────────────────────────────────────────────────
def _val_macro_f1(P, Yva):
    """Top-1 protocol over trainable classes (matches runtime + eval harness)."""
    ti = [_LAB_IDX[t] for t in _TRAINABLE]
    pred = P.argmax(1)
    f1s = []
    for c in ti:
        pr = (pred == c); tr = Yva[:, c] > 0.5
        tp = (pr & tr).sum(); fp = (pr & ~tr).sum(); fn = (~pr & tr).sum()
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * p * r / (p + r) if (p + r) else 0.0)
    return float(np.mean(f1s))


def train_head(Xtr, Ytr, Xva, Yva, epochs, lr, hidden, seed):
    import torch
    torch.manual_seed(seed)
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
    Ztr = (Xtr - mu) / sd; Zva = (Xva - mu) / sd
    n_cls = len(_TARGETS); emb = Xtr.shape[1]
    if hidden > 0:
        head = torch.nn.Sequential(torch.nn.Linear(emb, hidden), torch.nn.ReLU(),
                                   torch.nn.Dropout(0.3), torch.nn.Linear(hidden, n_cls))
    else:
        head = torch.nn.Linear(emb, n_cls)
    pos = Ytr.sum(0) + 1e-6; neg = len(Ytr) - pos
    pw = torch.tensor(np.clip(neg / pos, 1.0, 50.0), dtype=torch.float32)
    crit = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    Zt = torch.tensor(Ztr); Yt = torch.tensor(Ytr); Zv = torch.tensor(Zva)
    bs = 256; n = len(Zt)
    best_f1, best_state, bad, patience = -1.0, None, 0, 30
    for ep in range(epochs):
        head.train(); perm = torch.randperm(n)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            opt.zero_grad()
            loss = crit(head(Zt[idx]), Yt[idx]); loss.backward(); opt.step()
        if (ep + 1) % 5 == 0:
            head.eval()
            with torch.no_grad():
                P = torch.sigmoid(head(Zv)).numpy()
            f1 = _val_macro_f1(P, Yva)
            if f1 > best_f1:
                best_f1, bad = f1, 0
                best_state = {k: v.clone() for k, v in head.state_dict().items()}
            else:
                bad += 1
            if (ep + 1) % 25 == 0:
                print(f"    epoch {ep+1}: loss {loss.item():.3f}  val_top1_macroF1 {f1:.3f} (best {best_f1:.3f})", flush=True)
            if bad >= patience:
                print(f"    early stop @ {ep+1} (best {best_f1:.3f})", flush=True)
                break
    if best_state:
        head.load_state_dict(best_state)
    # Fold standardisation into a single Linear(960→12) for export (linear only).
    if hidden == 0:
        W = head.weight.detach().numpy().astype(np.float64)      # [12,960]
        b = head.bias.detach().numpy().astype(np.float64)        # [12]
        mu_, sd_ = mu.astype(np.float64), sd.astype(np.float64)
        W_fold = (W / sd_[None, :]).astype(np.float32)
        b_fold = (b - (W * (mu_ / sd_)[None, :]).sum(1)).astype(np.float32)
        return W_fold, b_fold, best_f1, head, (mu, sd)
    return None, None, best_f1, head, (mu, sd)


def export_v2(W, b, src="checkpoints/efficientat_head12.onnx",
              dst="checkpoints/efficientat_head12_v2.onnx"):
    import onnx
    from onnx import numpy_helper as nh
    m = onnx.load(src)
    repl = 0
    for i, init in enumerate(m.graph.initializer):
        if init.name == "hd.weight":
            m.graph.initializer[i].CopyFrom(nh.from_array(W.astype(np.float32), "hd.weight")); repl += 1
        elif init.name == "hd.bias":
            m.graph.initializer[i].CopyFrom(nh.from_array(b.astype(np.float32), "hd.bias")); repl += 1
    assert repl == 2, f"expected to replace 2 initializers, did {repl}"
    onnx.save(m, dst, save_as_external_data=True, location=Path(dst).name + ".data")
    # carry labels metadata
    m2 = onnx.load(dst)
    if not any(p.key == "labels" for p in m2.metadata_props):
        e = m2.metadata_props.add(); e.key = "labels"; e.value = "|".join(_TARGETS)
        onnx.save(m2, dst, save_as_external_data=True, location=Path(dst).name + ".data")
    return dst


# ── evaluation ────────────────────────────────────────────────────────────────
def _render_confusion(rep, labels):
    present = [l for l in labels if rep.per_class.get(l, {}).get("support", 0) > 0
               or any(rep.confusion.get(t, {}).get(l, 0) for t in labels)]
    w = max(len(l) for l in present) + 1
    head = " " * (w + 1) + "".join(f"{l[:6]:>7}" for l in present)
    lines = ["  confusion (rows=truth, cols=pred):", head]
    for t in present:
        row = rep.confusion.get(t, {})
        lines.append(f"  {t:<{w}}" + "".join(f"{row.get(p,0):>7}" for p in present))
    return "\n".join(lines)


def evaluate_all(head_path, args):
    v2_path = head_path  # newest head under evaluation (v3 by default)
    from voiceiso.config import PipelineConfig
    from voiceiso.stages.noise_classifier import EfficientATNoiseClassifier, HeuristicNoiseClassifier
    from voiceiso.bench.classifier_eval import evaluate_classifier
    from voiceiso.data.fsd50k_eval import iter_fsd50k_eval

    cfg = PipelineConfig(); cfg.noise_classifier_window_s = 4.0
    print("\nloading eval test clips…", flush=True)
    clips = list(iter_fsd50k_eval(root=ROOT, sr=48000, clip_seconds=5.0,
                                  max_per_class=args.test_cap, seed=0))
    print(f"  {len(clips)} eval test clips\n", flush=True)

    def mk_eat(path):
        c = PipelineConfig(); c.noise_classifier_window_s = 4.0
        c.noise_classifier_model_path = path
        return EfficientATNoiseClassifier(c)

    all_backends = {
        "heuristic": lambda: HeuristicNoiseClassifier(cfg),
        "pretrained-direct (527→12)": lambda: mk_eat("checkpoints/efficientat_s_4s.onnx"),
        "head v1 (eval-pool)": lambda: mk_eat("checkpoints/efficientat_head12.onnx"),
        "head v2 (dev, first-4s)": lambda: mk_eat("checkpoints/efficientat_head12_v2.onnx"),
        "head v3 (dev, window-robust)": lambda: mk_eat("checkpoints/efficientat_head12_v3.onnx"),
        "head UNDER TEST": lambda: mk_eat(v2_path),
    }
    sel = [s.strip() for s in (args.backends or "").split(",") if s.strip()]
    backends = {}
    for name, mk in all_backends.items():
        if sel and not any(s in name for s in sel):
            continue
        try:
            backends[name] = mk()
        except Exception as exc:
            print(f"  (skipping {name}: {type(exc).__name__}: {exc})", flush=True)
    reports = {}
    for name, clf in backends.items():
        t0 = time.time()
        reports[name] = evaluate_classifier(clf, clips)
        print(f"  {name:<28} macro-F1={reports[name].macro_f1:.3f}  "
              f"top-1={reports[name].micro_acc:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    print("\n" + "=" * 72)
    print("  FINAL TEST (FSD50K eval — held out; dev-trained head never saw it)")
    print("=" * 72)
    print(f"  {'backend':<28}{'macro-F1':>10}{'top-1':>8}")
    for name, r in reports.items():
        print(f"  {name:<28}{r.macro_f1:>10.3f}{r.micro_acc:>8.3f}")

    rep = reports.get("head UNDER TEST") or reports.get("head v3 (dev, window-robust)")
    if rep is None:
        return reports
    print("\n  head v3 — per-class (precision / recall / F1 / support):")
    print(f"  {'label':<18}{'P':>7}{'R':>7}{'F1':>7}{'supp':>7}")
    for lab in _TARGETS:
        m = rep.per_class.get(lab, {})
        if m.get("support", 0) == 0:
            continue
        print(f"  {lab:<18}{m['precision']:>7.3f}{m['recall']:>7.3f}{m['f1']:>7.3f}{int(m['support']):>7}")
    print()
    print(_render_confusion(rep, list(_TARGETS)))
    return reports


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cap", type=int, default=600)
    ap.add_argument("--val-cap", type=int, default=150)
    ap.add_argument("--test-cap", type=int, default=150)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default="checkpoints/dev_emb_cache_v3.npz")
    ap.add_argument("--out", default="checkpoints/efficientat_head12_v3.onnx")
    ap.add_argument("--backends", default="",
                    help="comma-separated name filter for the eval table (e.g. 'v2,v3')")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    from voiceiso.data.fsd50k_eval import load_uploader_map
    dev_dir = Path(ROOT) / "FSD50K.dev_audio"

    if not args.eval_only:
        if Path(args.cache).exists():
            print(f"loading cached embeddings {args.cache}")
            d = np.load(args.cache)
            Xtr, Ytr, Xva, Yva = d["Xtr"], d["Ytr"], d["Xva"], d["Yva"]
        else:
            dev_rows = _read_mapped(Path(ROOT) / "FSD50K.ground_truth" / "dev.csv")
            split_path = Path("checkpoints/fsd50k_split_dev.json")
            if split_path.exists():
                # Reuse the EXACT v2 split (same clips, same uploader grouping)
                # so the window policy is the only variable between v2 and v3.
                sp = json.load(open(split_path))
                lab = {fn: t for fn, t in dev_rows}
                train = [(f, lab[f]) for f in sp["train_fnames"] if f in lab]
                val = [(f, lab[f]) for f in sp["val_fnames"] if f in lab]
                print(f"reusing split {split_path}: train={len(train)} val={len(val)} clips")
            else:
                up_dev = load_uploader_map(ROOT, "dev")
                train, val = _grouped_split(dev_rows, up_dev, args.val_frac, rng)
                tu = {up_dev.get(f, f) for f, _ in train}; vu = {up_dev.get(f, f) for f, _ in val}
                print(f"dev split: train={len(train)} val={len(val)}  "
                      f"uploader overlap train∩val={len(tu & vu)} (must be 0)")
                train = _cap_per_class(train, args.train_cap, rng)
                val = _cap_per_class(val, args.val_cap, rng)
                json.dump({"mode": "standard grouped (dev→train/val by uploader, eval→test)",
                           "grouped_by": "uploader", "val_frac": args.val_frac,
                           "train_fnames": [f for f, _ in train], "val_fnames": [f for f, _ in val]},
                          open(split_path, "w"), indent=0)
            pc = defaultdict(int)
            for _f, t in train:
                for c in t:
                    pc[c] += 1
            print(f"  train clips={len(train)} per-class={dict(pc)}")
            print("extracting embeddings (frozen backbone via ONNX, v3 window policy)…", flush=True)
            emb = _Embedder()
            Xtr, Ytr = _embed_rows(emb, train, dev_dir, "train", args.seed)
            Xva, Yva = _embed_rows(emb, val, dev_dir, "val", args.seed)
            np.savez(args.cache, Xtr=Xtr, Ytr=Ytr, Xva=Xva, Yva=Yva)
            print(f"  cached embeddings → {args.cache}  train{Xtr.shape} val{Xva.shape}")

        print("training head (frozen backbone, standardisation folded into export)…", flush=True)
        W, b, best_f1, _head, _stats = train_head(Xtr, Ytr, Xva, Yva,
                                                  args.epochs, args.lr, args.hidden, args.seed)
        print(f"  best val top-1 macro-F1 (uploader-disjoint, window-robust) = {best_f1:.3f}")
        if W is None:
            print("  (MLP head — weight-swap export not supported; needs graph rebuild)")
            return 1
        dst = export_v2(W, b, dst=args.out)
        print(f"  exported → {dst}")

    evaluate_all(args.out, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
