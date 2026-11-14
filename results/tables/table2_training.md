# Table 2 – Training

| Stage | Final train loss | Validation metric | Time / GPU |
|---|---:|---:|---|
| SFT | 2.3905 | 2.4447 loss | 89.52 min / Tesla T4 |
| Reward Model | 0.6911 | 0.5340 accuracy | 7781.3s / Tesla T4 |
| PPO | -0.0281 (policy loss, iter 200) | -0.527 mean reward-model score (last 20 iters) | 60.8 min / Tesla T4 |
| DPO | 0.6096 | 0.6090 accuracy | 12342.6s (205.7 min) / Tesla T4 |