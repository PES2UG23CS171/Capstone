# Real-world condition matrix

Full deployed pipeline, 100 ms blocks, real FSD50K **eval** clips (n=4/condition, held out from all training). Silence regime: noise attenuation (dB, higher = better). Speech regime: SI-SDRi at the given input SNR (dB, higher = better). All intrusive numbers are scored on the valid reference/output overlap only (pipeline output delay 975 samples; no zero padding is scored).

| condition | silence: atten (dB) | speech @0 dB: SI-SDRi | speech @5 dB: SI-SDRi | speech @10 dB: SI-SDRi |
|---|---|---|---|---|
| fan | +41.9 | +10.4 | +8.5 | +5.5 |
| AC (→fan) | +41.9 | +10.4 | +8.5 | +5.5 |
| keyboard | +27.6 | +10.8 | +8.8 | +5.3 |
| mouse (→clicks) | +27.6 | +10.8 | +8.8 | +5.3 |
| clap | +42.7 | +7.3 | +5.6 | +3.5 |
| TV (loudspeaker) | +13.7 | +5.1 | +3.8 | +2.0 |
| competing speech | +31.4 | +7.5 | +5.9 | +3.5 |
| café (babble) | +31.4 | +7.5 | +5.9 | +3.5 |
| traffic | +41.5 | +6.9 | +4.9 | +2.5 |
