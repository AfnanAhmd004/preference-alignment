import copy

import torch

from align import dpo, evaluate, new_policy, pairwise_accuracy, preference_pairs, sequence_logprob, sft_uniform, train_reward_model, true_reward
from align.task import E, O, STOI


def test_true_reward():
    assert true_reward(E, [0, 2, 4, 6, 8, 0, 2, 4]) == 1.0
    assert true_reward(O, [1, 3, 5, 9, 1, 3, 5, 9]) == 1.0
    assert true_reward(O, [7, 3, 5, 9, 1, 3, 5, 9]) == 0.0  # perfect parity, but contains a 7


def test_preferences_track_true_reward():
    pairs = preference_pairs(2000, seed=0)
    agree = sum(true_reward(p, c) >= true_reward(p, r) for p, c, r in pairs) / len(pairs)
    assert agree > 0.8


def test_sequence_logprob_of_uniform_policy():
    pol = sft_uniform(new_policy(), steps=300)
    seqs = torch.tensor([[STOI["<bos>"], E, *range(8)]])
    assert abs(float(sequence_logprob(pol, seqs)) - 8 * torch.log(torch.tensor(0.1))) < 1.0


def test_reward_model_and_dpo_improve_true_reward():
    ref = sft_uniform(new_policy(), steps=300)
    pairs = preference_pairs(3000, seed=0)
    rm = train_reward_model(list(pairs), epochs=3)
    assert pairwise_accuracy(rm, preference_pairs(500, seed=1)) > 0.75
    aligned = dpo(copy.deepcopy(ref), list(pairs), epochs=2)
    assert evaluate(aligned, n=400)["true_reward"] > evaluate(ref, n=400)["true_reward"] + 0.2
