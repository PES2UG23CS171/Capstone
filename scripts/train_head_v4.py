"""v4 head: activate the TV/loudspeaker class + robustify fan — frozen backbone.

Changes vs v3 (same protocol: uploader-grouped dev train/val, FSD50K eval as
untouched test, embeddings from the frozen backbone ONNX, linear head with
standardisation folded into the export):

  * ``tv`` becomes trainable: positives are SYNTHESIZED by playing dev speech
    (70 %) and music (30 %) clips through a loudspeaker simulation (band-limit
    + compression + small-room reverb).  The clean originals stay in training
    as their own class — paired contrast teaches the head the CHANNEL, not the
    content.  The backbone already encodes it (AudioSet has Television/Radio).
  * ``fan`` (64 clips) gets 6 window variants per clip (crops/tiles + gain
    jitter ±6 dB + mild spectral tilt) instead of 2.
  * Stratified uploader-grouped val: greedy assignment with per-class floors
    so early stopping / threshold calibration stop selecting on noise
    (v3's val had fan=0, keyboard=4).
  * Per-class decision thresholds calibrated on val under the RUNTIME rule
    (argmax over smoothed-posterior/threshold) and shipped in ONNX metadata.

Deploy rule: v4 replaces v3 only if FSD50K-eval macro-F1/top-1 do not regress
(>0.01) AND the held-out loudspeaker-discrimination test improves decisively.

Run:  PYTHONPATH=<repo> .venv_poc/bin/python -m scripts.train_head_v4
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import zlib
from collections import defaultdict
from math import gcd
from pathlib import Path

import numpy as np

from scripts.retrain_head_dev import (_Embedder, _read_mapped, export_v2,
                                      evaluate_all, ROOT, SR, WIN_S, WIN_N)

_TARGETS = ("clean", "fan", "hvac", "traffic", "keyboard", "dog", "wind",
            "music", "tv", "speech", "competing_speech", "other")
_TRAINABLE = ("fan", "traffic", "keyboard", "dog", "wind", "music",
              "tv", "speech", "competing_speech", "other")
_LAB_IDX = {t: i for i, t in enumerate(_TARGETS)}

N_VAR_DEFAULT = 2
N_VAR_FAN = 6
TV_TRAIN_CAP = 700
TV_VAL_FLOOR = 60
VAL_FLOORS = {"fan": 12, "keyboard": 30, "dog": 30, "wind": 30, "tv": TV_VAL_FLOOR,
              "traffic": 30, "music": 30, "speech": 30, "competing_speech": 30, "other": 30}


# ── loudspeaker / TV simulation (applied at the model SR, 32 kHz) ────────────
def loudspeaker_sim(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    from scipy.signal import butter, sosfilt, fftconvolve
    sr = SR
    # 1. band-limit — small-speaker response
    hp = float(rng.uniform(180.0, 400.0))
    lp = float(rng.uniform(3200.0, 5200.0))
    sos = butter(2, [hp / (sr / 2), lp / (sr / 2)], btype="band", output="sos")
    y = sosfilt(sos, x)
    # 2. drive / compression (broadcast-ish loudness)
    g = float(rng.uniform(2.0, 5.0))
    y = np.tanh(g * y) / g
    # 3. small-room reverb — exponential-decay noise IR
    t60 = float(rng.uniform(0.06, 0.18))
    n_ir = int(t60 * sr)
    ir = rng.standard_normal(n_ir) * np.exp(-6.9 * np.arange(n_ir) / n_ir)
    ir[0] = 1.0
    wet = float(rng.uniform(0.15, 0.40))
    rev = fftconvolve(y, ir)[: len(y)]
    rev /= (np.max(np.abs(rev)) + 1e-9) / (np.max(np.abs(y)) + 1e-9)
    y = (1.0 - wet) * y + wet * rev
    # normalise RMS back to the source's
    y *= (np.sqrt(np.mean(x ** 2)) + 1e-9) / (np.sqrt(np.mean(y ** 2)) + 1e-9)
    return y.astype("float32")


def fan_jitter(win: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    from scipy.signal import sosfilt, butter
    y = win * float(10.0 ** (rng.uniform(-6, 6) / 20.0))
    # mild spectral tilt: gentle 1st-order shelf up or down
    f0 = float(rng.uniform(400.0, 2000.0))
    sos = butter(1, f0 / (SR / 2), btype=("low" if rng.random() < 0.5 else "high"),
                 output="sos")
    tilt = sosfilt(sos, y)
    mix = float(rng.uniform(0.0, 0.5))
    y = (1 - mix) * y + mix * tilt
    return np.clip(y, -1.0, 1.0).astype("float32")


# ── stratified uploader-grouped split ────────────────────────────────────────
def stratified_split(rows, uploader_of, rng, floors, val_frac=0.15):
    by_up = defaultdict(list)
    for fn, t in rows:
        by_up[uploader_of.get(fn, f"__none__{fn}")].append((fn, t))
    ups = sorted(by_up)
    rng.shuffle(ups)
    need = dict(floors)
    val_ups, val_count = [], 0
    n_val_target = int(len(rows) * val_frac)
    # pass 1: satisfy per-class floors
    for up in ups:
        classes = {c for _f, t in by_up[up] for c in t}
        if any(need.get(c, 0) > 0 for c in classes):
            val_ups.append(up)
            val_count += len(by_up[up])
            for _f, t in by_up[up]:
                for c in t:
                    if c in need:
                        need[c] -= 1
    # pass 2: fill to the target fraction
    for up in ups:
        if up in val_ups:
            continue
        if val_count >= n_val_target:
            break
        val_ups.append(up)
        val_count += len(by_up[up])
    val_set = set(val_ups)
    train = [r for up in ups if up not in val_set for r in by_up[up]]
    val = [r for up in val_set for r in by_up[up]]
    return train, val


def _cap_per_class(rows, cap, rng):
    idx = np.arange(len(rows)); rng.shuffle(idx)
    per = defaultdict(int); keep = []
    for i in idx:
        fn, t = rows[i]
        if any(per[c] < cap for c in t):
            keep.append(rows[i])
            for c in t:
                per[c] += 1
    return keep


# ── embedding assembly ───────────────────────────────────────────────────────
def embed_side(emb, rows, audio_dir, name, seed, tv_sources):
    """Real rows (v3 window policy; fan gets N_VAR_FAN variants) + tv synths."""
    X, Y = [], []
    t0 = time.time()
    for i, (fn, tgts) in enumerate(rows):
        rng = np.random.default_rng((seed, zlib.crc32(str(fn).encode())))
        wins = emb.embed_variants(audio_dir / f"{fn}.wav", rng)
        if not wins:
            continue
        if "fan" in tgts:
            x = emb._load_full(audio_dir / f"{fn}.wav")
            while x is not None and len(wins) < N_VAR_FAN:
                if len(x) >= WIN_N:
                    s = int(rng.integers(0, len(x) - WIN_N + 1))
                    w = x[s:s + WIN_N]
                else:
                    reps = int(np.ceil(WIN_N / len(x))) + 1
                    ph = int(rng.integers(0, len(x)))
                    w = np.tile(x, reps)[ph:ph + WIN_N]
                wins.append(fan_jitter(w, rng))
        y = np.zeros(len(_TARGETS), dtype="float32")
        for t in tgts:
            y[_LAB_IDX[t]] = 1.0
        for w in wins:
            X.append(emb._embed_window(w)); Y.append(y)
        if (i + 1) % 400 == 0:
            print(f"    {name}: {i+1}/{len(rows)} clips → {len(X)} windows ({time.time()-t0:.0f}s)", flush=True)
    # tv synthesis
    ytv = np.zeros(len(_TARGETS), dtype="float32"); ytv[_LAB_IDX["tv"]] = 1.0
    for j, fn in enumerate(tv_sources):
        rng = np.random.default_rng((seed, 7777, zlib.crc32(str(fn).encode())))
        wins = emb.embed_variants(audio_dir / f"{fn}.wav", rng)
        if not wins:
            continue
        for w in wins[:1]:                      # one synth per source clip
            X.append(emb._embed_window(loudspeaker_sim(w, rng)))
            Y.append(ytv)
        if (j + 1) % 200 == 0:
            print(f"    {name}-tv: {j+1}/{len(tv_sources)} ({time.time()-t0:.0f}s)", flush=True)
    return np.stack(X), np.stack(Y)


# ── training (extended-trainable copy of the v3 trainer) ─────────────────────
def _val_macro_f1(P, Yva, thresholds=None):
    ti = [_LAB_IDX[t] for t in _TRAINABLE]
    S = P / thresholds[None, :] if thresholds is not None else P
    pred = S.argmax(1)
    f1s = []
    for c in ti:
        pr = (pred == c); tr = Yva[:, c] > 0.5
        tp = (pr & tr).sum(); fp = (pr & ~tr).sum(); fn = (~pr & tr).sum()
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * p * r / (p + r) if (p + r) else 0.0)
    return float(np.mean(f1s))


def train_head(Xtr, Ytr, Xva, Yva, epochs, lr, seed):
    import torch
    torch.manual_seed(seed)
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
    Ztr = (Xtr - mu) / sd; Zva = (Xva - mu) / sd
    head = torch.nn.Linear(Xtr.shape[1], len(_TARGETS))
    pos = Ytr.sum(0) + 1e-6; neg = len(Ytr) - pos
    pw = torch.tensor(np.clip(neg / pos, 1.0, 50.0), dtype=torch.float32)
    crit = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    Zt = torch.tensor(Ztr); Yt = torch.tensor(Ytr); Zv = torch.tensor(Zva)
    bs, n = 256, len(Zt)
    best, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        head.train(); perm = torch.randperm(n)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            opt.zero_grad(); crit(head(Zt[idx]), Yt[idx]).backward(); opt.step()
        if (ep + 1) % 5 == 0:
            head.eval()
            with torch.no_grad():
                P = torch.sigmoid(head(Zv)).numpy()
            f1 = _val_macro_f1(P, Yva)
            if f1 > best:
                best, bad = f1, 0
                best_state = {k: v.clone() for k, v in head.state_dict().items()}
            else:
                bad += 1
            if (ep + 1) % 25 == 0:
                print(f"    epoch {ep+1}: val macro-F1 {f1:.3f} (best {best:.3f})", flush=True)
            if bad >= 30:
                print(f"    early stop @ {ep+1} (best {best:.3f})", flush=True)
                break
    if best_state:
        head.load_state_dict(best_state)
    W = head.weight.detach().numpy().astype(np.float64)
    b = head.bias.detach().numpy().astype(np.float64)
    W_fold = (W / sd[None, :]).astype(np.float32)
    b_fold = (b - (W * (mu / sd)[None, :]).sum(1)).astype(np.float32)
    import torch as _t
    with _t.no_grad():
        Pva = _t.sigmoid(head(_t.tensor(Zva))).numpy()
    return W_fold, b_fold, best, Pva


def calibrate_thresholds(Pva, Yva):
    """Per-class thresholds maximizing val macro-F1 under the runtime rule."""
    t = np.ones(len(_TARGETS), dtype=np.float64)
    grid = np.concatenate([np.arange(0.05, 1.01, 0.05)])
    for _sweep in range(2):                      # two coordinate passes
        for c in [_LAB_IDX[x] for x in _TRAINABLE]:
            best_f1, best_t = -1.0, t[c]
            for g in grid:
                t[c] = g
                f1 = _val_macro_f1(Pva, Yva, thresholds=t)
                if f1 > best_f1:
                    best_f1, best_t = f1, g
            t[c] = best_t
    return t, _val_macro_f1(Pva, Yva, thresholds=t)


# ── held-out loudspeaker discrimination test ─────────────────────────────────
def loudspeaker_test(head_paths: dict, n=80, seed=123):
    """EVAL-set speech clips, live vs loudspeaker-simulated: does the head
    separate speech from tv?  Sources never seen by any head."""
    import onnxruntime as ort
    from voiceiso.stages._efficientat_mel import EfficientATMel
    import soundfile as sf
    from scipy.signal import resample_poly
    mel = EfficientATMel(sr=SR)
    rows = []
    from voiceiso.data.fsd50k_labelmap import fsd50k_labels_to_target
    for r in csv.DictReader(open(Path(ROOT) / "FSD50K.ground_truth" / "eval.csv")):
        t = fsd50k_labels_to_target(r["labels"])
        if "speech" in t and "music" not in t:
            rows.append(r["fname"])
    rng = np.random.default_rng(seed)
    rng.shuffle(rows)
    rows = rows[:n]
    audio_dir = Path(ROOT) / "FSD50K.eval_audio"

    def load_win(fn, rng2):
        x, sr = sf.read(str(audio_dir / f"{fn}.wav"), dtype="float32", always_2d=True)
        x = x.mean(1)
        if sr != SR:
            g = gcd(sr, SR)
            x = resample_poly(x, SR // g, sr // g).astype("float32")
        if len(x) < WIN_N:
            x = np.tile(x, int(np.ceil(WIN_N / max(len(x), 1))))
        s = int(rng2.integers(0, max(len(x) - WIN_N, 0) + 1))
        return x[s:s + WIN_N].astype("float32")

    results = {}
    for name, path in head_paths.items():
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        labels = sess.get_modelmeta().custom_metadata_map["labels"].split("|")
        thr_meta = sess.get_modelmeta().custom_metadata_map.get("thresholds")
        thr = (np.array([float(v) for v in thr_meta.split("|")])
               if thr_meta else np.ones(len(labels)))

        def top(win):
            feats = mel(win).unsqueeze(1).numpy().astype(np.float32)
            p = sess.run(None, {sess.get_inputs()[0].name: feats})[0].reshape(-1)
            s = p / thr
            order = [i for i, l in enumerate(labels) if l != "clean"]
            return labels[max(order, key=lambda i: s[i])]

        live_ok = spk_tv = spk_notspeech = 0
        for fn in rows:
            rng2 = np.random.default_rng((seed, zlib.crc32(fn.encode())))
            w = load_win(fn, rng2)
            if top(w) == "speech":
                live_ok += 1
            lab = top(loudspeaker_sim(w, rng2))
            if lab == "tv":
                spk_tv += 1
            if lab != "speech":
                spk_notspeech += 1
        results[name] = (live_ok / len(rows), spk_tv / len(rows), spk_notspeech / len(rows))
        print(f"  {name:24s} live→speech {live_ok/len(rows):.2f}  "
              f"loudspeaker→tv {spk_tv/len(rows):.2f}  loudspeaker→NOT-speech {spk_notspeech/len(rows):.2f}",
              flush=True)
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cap", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default="checkpoints/dev_emb_cache_v4.npz")
    ap.add_argument("--out", default="checkpoints/efficientat_head12_v4.onnx")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    if not args.eval_only:
        if Path(args.cache).exists():
            d = np.load(args.cache)
            Xtr, Ytr, Xva, Yva = d["Xtr"], d["Ytr"], d["Xva"], d["Yva"]
            print(f"loaded cache {args.cache}: train{Xtr.shape} val{Xva.shape}")
        else:
            from voiceiso.data.fsd50k_eval import load_uploader_map
            dev_rows = _read_mapped(Path(ROOT) / "FSD50K.ground_truth" / "dev.csv")
            up = load_uploader_map(ROOT, "dev")
            train, val = stratified_split(dev_rows, up, rng, VAL_FLOORS)
            tu = {up.get(f, f) for f, _ in train}; vu = {up.get(f, f) for f, _ in val}
            print(f"stratified split: train={len(train)} val={len(val)} "
                  f"uploader overlap={len(tu & vu)} (must be 0)")
            pcv = defaultdict(int)
            for _f, t in val:
                for c in t:
                    pcv[c] += 1
            print(f"  val per-class: {dict(pcv)}")
            train = _cap_per_class(train, args.train_cap, rng)
            val = _cap_per_class(val, 150, rng)
            # tv sources: speech (70%) + music (30%), from each side separately
            def tv_srcs(rows, cap):
                sp = [f for f, t in rows if "speech" in t][: int(cap * 0.7)]
                mu = [f for f, t in rows if "music" in t and "speech" not in t][: int(cap * 0.3)]
                return sp + mu
            tv_tr = tv_srcs(train, TV_TRAIN_CAP)
            tv_va = tv_srcs(val, TV_VAL_FLOOR * 2)
            print(f"  tv synth sources: train={len(tv_tr)} val={len(tv_va)}")
            json.dump({"mode": "v4 stratified grouped + tv-synth + fan-boost",
                       "train_fnames": [f for f, _ in train], "val_fnames": [f for f, _ in val],
                       "tv_train_sources": tv_tr, "tv_val_sources": tv_va},
                      open("checkpoints/fsd50k_split_dev_v4.json", "w"), indent=0)
            emb = _Embedder()
            dev_dir = Path(ROOT) / "FSD50K.dev_audio"
            print("extracting embeddings (v4 policy)…", flush=True)
            Xtr, Ytr = embed_side(emb, train, dev_dir, "train", args.seed, tv_tr)
            Xva, Yva = embed_side(emb, val, dev_dir, "val", args.seed, tv_va)
            np.savez(args.cache, Xtr=Xtr, Ytr=Ytr, Xva=Xva, Yva=Yva)
            print(f"  cached → {args.cache} train{Xtr.shape} val{Xva.shape}")

        print("training v4 head…", flush=True)
        W, b, best_f1, Pva = train_head(Xtr, Ytr, Xva, Yva, args.epochs, args.lr, args.seed)
        print(f"  best val macro-F1 (stratified, uploader-disjoint) = {best_f1:.3f}")
        thr, f1_cal = calibrate_thresholds(Pva, Yva)
        print(f"  after per-class threshold calibration: {f1_cal:.3f}")
        print(f"  thresholds: {dict(zip(_TARGETS, np.round(thr, 2)))}")
        dst = export_v2(W, b, dst=args.out)
        # attach thresholds metadata
        import onnx
        m = onnx.load(dst)
        for key, val_s in (("thresholds", "|".join(f"{v:.4f}" for v in thr)),):
            found = False
            for p in m.metadata_props:
                if p.key == key:
                    p.value = val_s; found = True
            if not found:
                e = m.metadata_props.add(); e.key = key; e.value = val_s
        onnx.save(m, dst, save_as_external_data=True, location=Path(dst).name + ".data")
        print(f"  exported → {dst} (with thresholds metadata)")

    class A:  # minimal args shim for evaluate_all
        test_cap = 150
        backends = "v3,v4"
    print("\nFSD50K eval (held-out):", flush=True)
    import scripts.retrain_head_dev as rhd
    rhd.evaluate_all(args.out, A)

    print("\nloudspeaker discrimination (EVAL speech clips, held out):", flush=True)
    loudspeaker_test({"v3": "checkpoints/efficientat_head12_v3.onnx", "v4": args.out})
    return 0


if __name__ == "__main__":
    sys.exit(main())
