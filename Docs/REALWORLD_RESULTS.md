# Real-world condition matrix

Full deployed pipeline, 100 ms blocks, real FSD50K **eval** clips (n=4/condition, held out from all training). Silence regime: noise attenuation (dB, higher = better). Speech regime: SI-SDRi at the given input SNR (dB, higher = better). All intrusive numbers are scored on the valid reference/output overlap only (pipeline output delay 975 samples; no zero padding is scored).

| condition | silence: atten (dB) | speech @0 dB: SI-SDRi | speech @5 dB: SI-SDRi | speech @10 dB: SI-SDRi |
|---|---|---|---|---|
| fan | +41.9 | +10.4 | +8.5 | +5.5 |
| AC (→fan) | +41.9 | +10.4 | +8.5 | +5.5 |
| keyboard | +27.2 | +10.9 | +8.8 | +5.3 |
| mouse (→clicks) | +27.2 | +10.9 | +8.8 | +5.3 |
| clap | +42.7 | +7.2 | +5.6 | +3.5 |
| TV (loudspeaker) | +13.6 | +5.2 | +3.8 | +2.0 |
| competing speech | +31.3 | +7.3 | +5.9 | +3.5 |
| café (babble) | +31.3 | +7.3 | +5.9 | +3.5 |
| traffic | +41.5 | +6.0 | +4.6 | +2.3 |
