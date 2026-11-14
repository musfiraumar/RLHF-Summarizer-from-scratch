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

So is a higher reward model score the same as a better summary? In this run, no, or at least there is no evidence that it is. The reward model is close to chance, so a gap of 0.04 in its score says almost nothing about quality. PPO's gain comes together with shorter outputs and a lower ROUGE-L against the reference summaries, and DPO's lower score comes together with longer outputs. Both look like the score tracking length more than quality. The human evaluation is what should settle this.

Two members each looked at 30 posts (fixed seed 42, drawn from the posts where the three summaries differ). For each post the SFT, PPO and DPO summaries were shown in random order without saying which model wrote which, and each member independently picked the best one (`results/human_eval.csv`, key in `results/human_eval_key.csv`). Each pick counts for one method, so this is a best-of-three choice and not a pairwise win rate:

| picked as best (30 posts) | Member 1 | Member 2 | both members (60 picks) |
|---|---:|---:|---:|
| SFT | 11 (36.7%) | 13 (43.3%) | 24 (40.0%) |
| PPO | 14 (46.7%) | 9 (30.0%) | 23 (38.3%) |
| DPO | 5 (16.7%) | 8 (26.7%) | 13 (21.7%) |

The two members chose the same summary on 15 of the 30 posts (50.0%). Picking at random among three summaries would agree about 33% of the time, and the kappa computed from these counts is about 0.24, so agreement is only fair. With 30 posts the 95% interval on the 50% is roughly plus or minus 18 points.

PPO and SFT are not separated. The members disagree about which is better (Member 1 picked PPO 14 times to SFT's 11, Member 2 picked SFT 13 times to PPO's 9) and together they picked SFT 24 times and PPO 23 times. So the reward model's 0.043 higher mean score for PPO did not turn into a human preference. That fits the earlier point that a gap this small in a near-chance reward model says almost nothing about quality, but 30 posts could only show a large difference, so this does not prove PPO and SFT are equal.

DPO was picked least by both members: 5 and 8 times, 13 of 60 picks (21.7%) against the 33% expected if all three were equal. That is suggestive and not conclusive (about 1.9 standard errors below expectation, and the two members rated the same posts). Here the humans and the reward model agree, since DPO also had the lowest reward model score. One thing that could explain it is length: DPO summaries hit the 48-token limit on 190 of the 300 evaluation posts (63%), against 30 for SFT and 4 for PPO, so many of them are cut off or run on. The picks do not say why the members preferred the others, and I did not ask them, so this is a possible reason and not a result.

The comparison table from the plan is therefore replaced by the counts above. The reward model's own win rates on all 300 posts were:

| comparison (300 posts) | reward-model win rate |
|---|---:|
| PPO vs SFT | 52.3% |
| DPO vs SFT | 48.3% |
| PPO vs DPO | 58.3% (175/300) |

The reward model and the members agree on the ranking of DPO and disagree on PPO against SFT, where the reward model has a small lead and the humans see none.

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
| DPO implicit reward, final | 60.9% (see `dpo_log.json`) |

DPO's implicit reward gets 60.9% of the 1,000 validation pairs right, against 53.4% for the separate reward model. With 1,000 pairs the 95% interval is about plus or minus 3 points, so the 7.5 point gap is clear. Accuracy was already about 59% at step 250, rose slowly to a peak of 61.9% at step 2250 and ended at 60.9%, while the train loss only fell from 0.693 to 0.610. So the same preference data gave a better signal when used through the policy's log ratios than through a separate reward model of the same size. The absolute level is still modest. About 4 in 10 validation pairs are ranked the wrong way, which fits noisy labels and a small model, and the method alone does not remove that limit.

Accuracy on preference pairs does not mean better summaries, though. The evaluation shows how far DPO moved: its greedy summaries are about 14 tokens longer than SFT's and it is 4.0 nats from the reference. A policy can rank the pairs well and still drift toward longer outputs, which is what happened here.

One note on the log: the baseline validation accuracy before training is printed as 0.0. At that point the policy equals the reference, so every margin is exactly zero, and a strict margin > 0 test counts none of them as correct. It is an artefact of the comparison and not a real accuracy; chance is about 50%.

## 7.6 Memory, time, and which method I would use

PPO had 3 models on the GPU: the trainable policy (with a linear value head on the same backbone), the frozen reference, and the frozen reward model. A textbook setup with a separate critic would need 4. DPO had 2: the trainable policy and the frozen reference, with no reward model, value head or sampling. All models are fp32 and about 135M parameters each.

Time on the Tesla T4:

| stage | time |
|---|---:|
| SFT | 89.5 min |
| reward model (2,500 steps) | 129.7 min |
| PPO (200 iterations, 16 prompts each) | 60.8 min |
| DPO (2,500 steps) | 205.7 min |

The PPO route is SFT + reward model + PPO, about 280 minutes. The DPO route is SFT + DPO, about 295 minutes, so DPO was not faster in wall-clock time here. Its 2,500 steps each need forward passes through both the policy and the reference on two sequences, and they cost more than the whole PPO loop. The saving is in the number of stages and models, not in training time. PPO also has to generate inside the loop, which is part of why 3,200 rollouts took an hour even at this size.

For a new project I would start with DPO for its cost and simplicity, but the human evaluation weakens the case for DPO as it was trained here, so I will say what the evidence does and does not show. On the reward model's own metric PPO looks slightly better than DPO (mean reward -0.513 against -0.713, win rate 52.3% against 48.3%), and neither win rate is distinguishable from 50%. That metric is not trustworthy, because the reward model is at 53.4% accuracy and seems to penalise length, which DPO increased. So these numbers cannot say which method produced better summaries. What I can say is about cost and risk. The reward model was the weak stage: it gave PPO a signal close to noise, and PPO paid a large KL cost during training for a +0.025 reward change and shorter outputs. DPO skips that stage, needs one fewer model in memory, has no sampling loop or value function, reached a higher accuracy on the preference pairs than the reward model did (60.9% against 53.4%), and has fewer hyperparameters to tune (beta and learning rate, against beta_KL, clip, value and entropy coefficients and rollout size). It is also easier to run on a single T4, although it was not faster in total time (about 295 minutes against 280). The cost to watch is that DPO drifted toward longer outputs here, so a length check or a length-normalised loss would be needed in a real project. I would go back to PPO only if I had a reward that is accurate and cheap to query, such as a verifiable checker or a strong reward model, or if on-policy exploration mattered, since DPO only learns from the fixed pairs. The human evaluation does not support DPO's summaries: the two members picked DPO as best on 13 of 60 picks, against 23 for PPO and 24 for SFT, so the choice rests on cost and simplicity alone and not on better output. Together with the 63% of DPO summaries that hit the length limit, it means a real project would need a length check or a length-normalised loss, or a lower beta or fewer steps, before DPO could be trusted. With 30 posts and two members this is a small study, so I would repeat it with more posts and members before deciding.