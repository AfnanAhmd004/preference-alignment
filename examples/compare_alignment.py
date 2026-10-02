"""SFT reference vs DPO vs RLHF, plus what happens to RLHF without a KL penalty."""
import copy

import torch

from align import dpo, evaluate, new_policy, pairwise_accuracy, preference_pairs, rlhf, sft_uniform, train_reward_model

torch.set_num_threads(4)
ref = sft_uniform(new_policy())
train_pairs, test_pairs = preference_pairs(6000, seed=0), preference_pairs(1000, seed=1)
rm = train_reward_model(list(train_pairs))
print(f"reward model held-out pairwise accuracy: {pairwise_accuracy(rm, test_pairs):.1%}\n")

results = {"SFT reference": evaluate(ref)}
results["DPO (beta=0.1)"] = evaluate(dpo(copy.deepcopy(ref), list(train_pairs)))
pol, log_kl = rlhf(copy.deepcopy(ref), rm, kl_coef=0.05)
results["RLHF / RLOO (KL 0.05)"] = evaluate(pol)
pol, log_nokl = rlhf(copy.deepcopy(ref), rm, kl_coef=0.0, steps=300, lr=1e-3)
results["RLHF, no KL, 2x steps"] = evaluate(pol)

weak_rm = train_reward_model(list(preference_pairs(300, seed=5)), epochs=20)
print(f"weak reward model (300 pairs) held-out accuracy: {pairwise_accuracy(weak_rm, test_pairs):.1%}")
pol, log_weak = rlhf(copy.deepcopy(ref), weak_rm, kl_coef=0.0, steps=300, lr=1e-3)
results["RLHF on weak RM, no KL"] = evaluate(pol)
pol, _ = rlhf(copy.deepcopy(ref), weak_rm, kl_coef=0.2, steps=300, lr=1e-3)
results["RLHF on weak RM, KL 0.2"] = evaluate(pol)

print(f"\n{'policy':<26}{'true reward':>13}{'on E prompts':>14}{'on O prompts':>14}{'violations':>12}")
for k, v in results.items():
    print(f"{k:<26}{v['true_reward']:>13.3f}{v['reward_E']:>14.3f}{v['reward_O']:>14.3f}{v['violations']:>12.1%}")

print("\nRLHF on the weak reward model, no KL - proxy score vs true reward during training:")
for row in log_weak[::3]:
    print(f"  step {row['step']:>3}: RM {row['rm_reward']:+.2f}   true {row['true_reward']:+.3f}   KL {row['kl']:.1f}")
