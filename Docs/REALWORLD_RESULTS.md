# Real-world condition matrix

Full deployed pipeline, 100 ms blocks, real FSD50K **eval** clips (n=4/condition, held out from all training). Silence regime: noise attenuation (dB, higher = better). Speech regime: SI-SDRi at the given input SNR (dB, higher = better).

| condition | silence: atten (dB) | speech @0 dB: SI-SDRi | speech @5 dB: SI-SDRi | speech @10 dB: SI-SDRi |
|---|---|---|---|---|
| fan | +38.0 | +8.3 | +6.2 | +3.7 |
| AC (→fan) | +38.0 | +8.3 | +6.2 | +3.7 |
| keyboard | +26.6 | +10.2 | +8.2 | +4.9 |
| mouse (→clicks) | +26.6 | +10.2 | +8.2 | +4.9 |
| clap | +44.8 | +7.2 | +5.0 | +2.5 |
| TV (loudspeaker) | +19.0 | +4.1 | +3.0 | +1.3 |
| competing speech | +23.2 | +6.3 | +4.7 | +2.2 |
| café (babble) | +23.2 | +6.3 | +4.7 | +2.2 |
| traffic | +38.3 | +5.8 | +4.3 | +2.0 |
