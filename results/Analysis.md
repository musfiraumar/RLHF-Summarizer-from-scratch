
## 7.1 Reward model accuracy and mistakes

The reward model gets 53.4% of the 1,000 validation pairs right. Before training it was at 49.7%. The training loss never moved off about 0.691 over the 2,500 steps, and chance level is ln 2 = 0.693, so it basically did not learn. With 1,000 pairs the 95% interval is about plus or minus 3 points, which puts 53.4% only just above a coin flip.

The 20 mistakes saved in `reward_mistakes.json` are the first 20 errors in validation order, not a random sample, so I am careful about what I generalise from them. The clearest thing is that none of them is a confident error. The average gap between the score of the chosen and rejected summary is -0.02 and the worst is -0.10. The model just cannot tell the two apart.

Three examples:

1. The "TIFU by rescuing a feral kitten" post. The chosen summary is only the title repeated ("TIFU by rescuing a feral kitten."), and the rejected one is a longer sentence about keeping the kitten for a year. The reward model scored the longer one higher (-0.982 vs -1.009). The label itself is a bit questionable here, but it shows the model leaning toward longer, more detailed text.
2. The long-distance relationship post asking when it is acceptable to "go dark". The rejected summary is almost a copy of the post title, while the chosen one actually rewrites the situation ("Is it ever okay to not officially end things? emotionally abusive LDR..."). The model scored the copy higher (-0.295 vs -0.326), so it does not separate copying from summarising.
3. The post by a 21-year-old whose girlfriend seems to have lost interest. This has the biggest gap of the 20 (-0.806 vs -0.702). Both summaries are fluent and pick out different parts of the post. Deciding which is better takes real judgment about what matters in the post, and a 135M model trained on 20k pairs does not get there.

Most of the other saved mistakes look like these: two fluent, on-topic summaries separated by a small difference in detail or faithfulness.

## 7.2 PPO training

The plot is in `results/figures/figure5_ppo_training.png` (reward, KL, length and clip fraction against iteration, raw plus a 10-iteration moving average).

Comparing the first 20 iterations with the last 20:

| | first 20 | last 20 |
|---|---:|---:|
| mean reward model score | -0.552 | -0.527 |
| mean KL to SFT (nats) | 1.23 | 5.15 |
| mean summary length (tokens) | 34.4 | 24.8 |

The score did go up, but only by about 0.025. Single iterations bounce around by roughly 0.06 (16 prompts per rollout), so the rise is about the size of the noise. The best single iteration was -0.381 at iteration 50 and it did not hold.

Length changed a lot. Summaries started at about 34 tokens, which matches the SFT data (33.8), and ended near 25, a drop of about 28%. Across iterations, reward and length are weakly negatively correlated (r = -0.28), so the reward model seems to like shorter outputs a little, and the policy picked that up. Token entropy also rose from 2.96 to 3.55, which is not what you would expect from a policy that is getting sharper.

For that small reward gain the policy moved a long way from SFT: KL reached 5 to 6 nats per summary, with a peak of 8.2.

I also ran `evaluate.py` on 300 held-out posts with greedy decoding, scoring each model's summaries with the reward model:

| model | mean reward | mean KL to SFT | mean length | ROUGE-L | win rate vs SFT |
|---|---:|---:|---:|---:|---:|
| SFT | -0.556 | 0.00 | 30.4 | 0.253 | n/a |
| PPO | -0.513 | 0.41 | 21.5 | 0.241 | 52.3% (157/300) |
| DPO | -0.713 | 4.03 | 44.2 | 0.223 | 48.3% (145/300) |

The KL values here are not comparable with the 5 to 6 nats in the training log, because training sampled at temperature 1 and this evaluation is greedy. They are only useful for comparing PPO with DPO.

PPO's mean reward is 0.043 higher than SFT's and its summaries are about 9 tokens shorter, but it wins on only 52.3% of posts. With 300 posts the 95% interval is about plus or minus 5.7 points, so 52.3% (and DPO's 48.3%) cannot be told apart from 50%. DPO went the other way: its summaries got longer (44 tokens, close to the 48-token limit), its reward model score is the lowest of the three, and its ROUGE-L is the lowest. The reward model seems to dislike long outputs, which fits the negative reward and length correlation seen during PPO.

So is a higher reward model score the same as a better summary? In this run, no, or at least there is no evidence that it is. The reward model is close to chance, so a gap of 0.04 in its score says almost nothing about quality. PPO's gain comes together with shorter outputs and a lower ROUGE-L against the reference summaries, and DPO's lower score comes together with longer outputs. Both look like the score tracking length more than quality. The human evaluation is what should settle this:

| comparison (300 posts) | human win rate | reward-model win rate |
|---|---:|---:|
| PPO vs SFT | [TODO] | 52.3% |
| DPO vs SFT | [TODO] | 48.3% |
| PPO vs DPO | [TODO] | [TODO] |

If humans do not prefer PPO over SFT even though its reward model score is higher, that is direct evidence the score and quality came apart. If they do prefer it, the likely reason is that shorter summaries are tighter, and it is worth checking how often the shorter ones dropped an important fact. The reverse question applies to DPO: humans may like its longer summaries even though the reward model does not.

## 7.3 The role of beta_KL

In `train_ppo.py` beta_KL is 0.05. Each generated token gets a reward of -beta_KL times (log pi_theta - log pi_ref), and the reward model score is added on the last token (Eq. 41). Summed over the sequence this is a sample estimate of the KL between the policy and the SFT model, so the objective is the expected reward model score minus beta_KL times that KL. It keeps the policy near the SFT model, where the reward model was trained and where the text is still fluent. Because it is estimated from one sample per token it can come out negative, which happened once (iteration 2, -0.035).

With beta_KL = 0 the penalty disappears and the policy just maximises the reward model score. The only limits left would be the PPO clip and the learning rate, and those slow each step down but do not bound the total drift. I would expect KL to keep growing and the policy to exploit whatever the reward model likes, such as length, copying the title, or odd token patterns, while fluency falls apart and the score keeps going up. With a reward model this weak I think that would happen quickly, since most of its signal is noise.

Even at 0.05 the penalty is not small next to the reward. A KL of about 5 nats costs about 0.25, roughly ten times the 0.025 reward gain in the logs. So the term is doing real work, and removing it removes the only thing holding the policy back. (It would also mean the reference model is not needed during PPO.)

## 7.4 Clip fraction

The mean clip fraction over the 200 iterations was 0.027, with a median of 0.026, and 98.5% of iterations were under 0.05. Only iteration 1 stands out at 0.169, probably because the optimizer and value head start untrained.

`epoch0_clip` is exactly 0 in every iteration and `ratio_dev` is around 1e-4 there. That is what it should be: before the first gradient step the policy equals the rollout policy, so every ratio is 1 apart from float noise. The logged clip fraction averages over four epochs including that zero one, so in epochs 1 to 3 about 3.5% of tokens were clipped.

So about 96 to 97% of tokens stay inside [0.8, 1.2] after an update. Each update is small, and the clip rarely binds. Step size is set mostly by the learning rate (1e-5) and gradient clipping, not by the PPO clip. Small steps still add up, and after 200 iterations the policy is 5 to 6 nats from SFT. A low, steady clip fraction alongside a rising KL means the drift was slow and steady, not a few large jumps.

## 7.5 DPO: where the reward went and why Z(x) is not needed

The KL-regularised objective has the optimal policy pi*(y|x) = pi_ref(y|x) exp(r(x,y)/beta) / Z(x) (Eq. 55). Solving for the reward gives r(x,y) = beta log(pi*(y|x) / pi_ref(y|x)) + beta log Z(x) (Eqs. 56 and 57). So the reward is written in terms of the policy itself: the log ratio of the trained policy to the SFT reference is an implicit reward, and there is no separate reward model to train.

Z(x) is a sum over every possible summary, so it cannot be computed. But it depends only on the prompt x and not on y. The Bradley-Terry preference probability only uses the difference r(x, y_w) - r(x, y_l), and the beta log Z(x) term is the same for both and cancels (Eq. 58):

P(y_w preferred over y_l) = sigmoid( beta * [ log(pi_theta(y_w)/pi_ref(y_w)) - log(pi_theta(y_l)/pi_ref(y_l)) ] )

`train_dpo.py` computes exactly this margin, summing log-probs over the summary tokens only, with beta = 0.1, learning rate 2e-6, batch size 8 and 2,500 steps. The loss is minus log sigmoid of the margin, and implicit reward accuracy is the share of validation pairs where the margin is positive.

On the same 1,000 validation pairs:

| | accuracy |
|---|---:|
| reward model, untrained | 49.7% |
| reward model, final | 53.4% |
| DPO implicit reward, final | [TODO] (see `dpo_log.json`) |

The DPO accuracy is still missing, but the evaluation already shows how far DPO moved: its greedy summaries are about 14 tokens longer than SFT's and it is 4.0 nats from the reference. Once the DPO number is in: DPO trains the whole language model directly on the preference loss, so its implicit reward is usually at least as accurate as a separate reward model of the same size. If it is clearly above 53.4%, the same data gave a better signal when used through the policy's log ratios. If it is also around 50 to 55%, the limit is the noisy labels and the small model, not the method.

## 7.6 Memory, time, and which method I would use

PPO had 3 models on the GPU: the trainable policy (with a linear value head on the same backbone), the frozen reference, and the frozen reward model. A textbook setup with a separate critic would need 4. DPO had 2: the trainable policy and the frozen reference, with no reward model, value head or sampling. All models are fp32 and about 135M parameters each.

Time on the Tesla T4:

| stage | time |
|---|---:|
| SFT | 89.5 min |
| reward model (2,500 steps) | 129.7 min |
| PPO (200 iterations, 16 prompts each) | 60.8 min |
| DPO (2,500 steps) | [TODO] |

The PPO route is SFT + reward model + PPO, about 280 minutes. The DPO route is SFT + DPO. PPO also has to generate inside the loop, which is part of why 3,200 rollouts took an hour even at this size.

For a new project I would still lean toward DPO, but the evidence from this repo for it is weaker than I first thought, so I will say what it does and does not show. On the reward model's own metric PPO looks slightly better than DPO (mean reward -0.513 against -0.713, win rate 52.3% against 48.3%), and neither win rate is distinguishable from 50%. That metric is not trustworthy, because the reward model is at 53.4% accuracy and seems to penalise length, which DPO increased. So these numbers cannot say which method produced better summaries. What I can say is about cost and risk. The reward model was the weak stage: it gave PPO a signal close to noise, and PPO paid a large KL cost during training for a +0.025 reward change and shorter outputs. DPO skips that stage, needs one fewer model in memory, has no sampling loop or value function, and has fewer hyperparameters to tune (beta and learning rate, against beta_KL, clip, value and entropy coefficients and rollout size). It is also easier to run on a single T4. The cost to watch is that DPO drifted toward longer outputs here, so a length check or a length-normalised loss would be needed in a real project. I would go back to PPO only if I had a reward that is accurate and cheap to query, such as a verifiable checker or a strong reward model, or if on-policy exploration mattered, since DPO only learns from the fixed pairs. The human-evaluation result should go here: if humans prefer DPO's summaries over PPO's, that supports the choice; if not, the choice rests on cost and simplicity alone.