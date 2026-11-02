"""Batching sorts texts by length; activations must still land on the right rows."""

from types import SimpleNamespace

import numpy as np
import torch

from src.extract import final_token_activations


class _Enc(dict):
    __getattr__ = dict.__getitem__

    def to(self, device):
        return self


class StubTokenizer:
    """Each character becomes one token whose id is the text's length."""

    def __call__(self, texts, **kw):
        n = max(len(t) for t in texts)
        ids = torch.zeros(len(texts), n, dtype=torch.long)
        mask = torch.zeros(len(texts), n, dtype=torch.long)
        for i, t in enumerate(texts):  # left padding
            ids[i, n - len(t):] = len(t)
            mask[i, n - len(t):] = 1
        return _Enc(input_ids=ids, attention_mask=mask)


class StubModel:
    device = "cpu"

    def __call__(self, input_ids, attention_mask, position_ids, **kw):
        h = input_ids.float()[..., None].repeat(1, 1, 3)  # [B, T, d=3]
        hs = tuple(h * k for k in range(4))  # layer k = k * token id
        return SimpleNamespace(hidden_states=hs)


def test_rows_are_written_back_in_input_order():
    texts = ["aaaaa", "a", "aaa", "aaaaaaa", "aa", "aaaa"]
    acts = final_token_activations(StubModel(), StubTokenizer(), texts,
                                   batch_size=2)
    assert acts.shape == (4, len(texts), 3)
    expected = np.array([len(t) for t in texts], dtype=np.float32)
    for k in range(4):
        np.testing.assert_array_equal(acts[k, :, 0], k * expected)
