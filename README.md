# preference-alignment

**Reward modelling, DPO and RLHF** compared on a small alignment problem where the **true reward is known**, so the evaluation can check whether each method actually improves what the labeller cares about, and catch failure modes such as reward over-optimisation and prompt-ignoring mode collapse.

## The task

- **Prompt:** `E` ("prefer even digits") or `O` ("prefer odd digits"). **Response:** 8 digits.
- **Hidden true reward:** fraction of digits with the requested parity, **minus 1 if the response contains a 7**, a safety rule the labeller applies but never states.
- **Feedback:** pairs of responses labelled by a noisy **Bradley–Terry** labeller over the true reward, standing in for human or AI feedback (RLHF / RLAIF).
- **Reference policy:** a small GPT trained to emit uniformly random digits, the "unaligned SFT model".

## Methods

| Method | Implementation |
|---|---|
| Reward model | GPT backbone with a scalar head on the last token, Bradley–Terry loss `−log σ(r_chosen − r_rejected)` |
| **DPO** | `−log σ(β · [(log π − log π_ref)(chosen) − (log π − log π_ref)(rejected)])`; no reward model, no sampling |
| **RLHF** | KL-regularised policy gradient with a leave-one-out baseline (RLOO, a simpler cousin of PPO): reward = `RM(y) − c·log(π/π_ref)` |

## Results

```bash
pip install -e ".[dev]"
python examples/compare_alignment.py    # ~2 min on CPU
pytest
```

```
reward model held-out pairwise accuracy: 80.3%
weak reward model (300 pairs) held-out accuracy: 71.9%

policy                      true reward  on E prompts  on O prompts  violations
SFT reference                    -0.058        -0.065        -0.050       56.8%
DPO (beta=0.1)                    0.894         0.883         0.905        0.0%
RLHF / RLOO (KL 0.05)             0.503         0.283         0.725        0.4%
RLHF, no KL, 2x steps             0.530         0.081         0.984        0.0%
RLHF on weak RM, no KL            0.531         0.470         0.593        0.0%
RLHF on weak RM, KL 0.2           0.507         0.448         0.566        0.1%

RLHF on the weak reward model, no KL - proxy score vs true reward during training:
  step   0: RM -1.31   true +0.039
  step  75: RM +5.14   true +0.488
  step 299: RM +5.36   true +0.523
```

### What it shows

1. **Every method learns the unstated safety rule.** Responses containing a 7 fall from 57% to under 1%, although no one ever told the model about 7s; the rule was implicit in the preferences.
2. **DPO performs best here** and stays balanced across both prompts.
3. **RLHF collapses onto one mode.** Without enough KL regularisation, the policy learns "always output odd digits": nearly perfect on `O` prompts, near zero on `E`. Average reward hides this; reward split by prompt reveals it. Breaking evaluations down this way is part of the work.
4. **Proxy reward ≠ true reward.** Against the weaker reward model, the proxy score rises from −1.3 to +5.4 while true reward stalls around 0.5. A rising reward-model curve is not evidence of alignment.

This is a toy problem with single-seed results. The value is in the controlled setting: the mechanisms (Bradley–Terry modelling, the DPO objective, KL-regularised policy gradient) are the same ones used to align large language models.

## License

MIT
