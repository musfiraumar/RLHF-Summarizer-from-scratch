# Table 3 – Final comparison (300 evaluation posts)

| Policy | Mean rφ | Win rate vs. SFT | KL to SFT | Length | ROUGE-L |
|---|---:|---:|---:|---:|---:|
| SFT | -0.556 | – | 0 | 30.4 | 0.253 |
| PPO | -0.513 | 52.3% (157/300) | 0.41 | 21.5 | 0.241 |
| DPO | -0.713 | 48.3% (145/300) | 4.03 | 44.2 | 0.223 |

Greedy decoding; rφ is the reward model score; KL is the mean per-summary KL to the SFT model (nats); length is in tokens.
