"""Model loading, knowledge filtering and final-token activation extraction."""

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_id: str, device: str = "cuda", dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, attn_implementation="eager").to(device)
    model.eval()
    return model, tok


def _batches(texts, batch_size):
    """Index batches with similar lengths together to minimise padding."""
    order = np.argsort([len(t) for t in texts], kind="stable")
    for start in range(0, len(order), batch_size):
        yield order[start:start + batch_size]


def _encode(tok, texts, device):
    enc = tok(texts, return_tensors="pt", padding=True,
              add_special_tokens=False).to(device)
    # Left padding: positions must count real tokens only.
    pos = (enc.attention_mask.cumsum(-1) - 1).clamp(min=0)
    return enc, pos


@torch.no_grad()
def generate_answers(model, tok, prompts, max_new_tokens=8, batch_size=32):
    out = [None] * len(prompts)
    for idx in tqdm(list(_batches(prompts, batch_size)), desc="generate"):
        enc, _ = _encode(tok, [prompts[i] for i in idx], model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                             do_sample=False)
        new = gen[:, enc.input_ids.shape[1]:]
        for i, text in zip(idx, tok.batch_decode(new, skip_special_tokens=True)):
            out[i] = text.strip()
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def final_token_activations(model, tok, texts, batch_size=16):
    """Residual stream at the last token of each text, for every layer.

    Returns float32 array [n_layers + 1, N, d_model], where index 0 is the
    embedding output and index i is the output of decoder block i - 1
    (Hugging Face `output_hidden_states` convention).
    """
    acts = None
    for idx in tqdm(list(_batches(texts, batch_size)), desc="activations"):
        enc, pos = _encode(tok, [texts[i] for i in idx], model.device)
        # logits_to_keep=1 avoids materialising full-vocabulary logits.
        hs = model(**enc, position_ids=pos, output_hidden_states=True,
                   logits_to_keep=1).hidden_states
        last = torch.stack([h[:, -1] for h in hs]).float().cpu().numpy()
        if acts is None:
            acts = np.empty((last.shape[0], len(texts), last.shape[2]),
                            dtype=np.float32)
        acts[:, idx] = last
    return acts


def final_token_strings(tok, texts):
    """Decoded last token of each text, as the model sees it."""
    ids = tok(texts, add_special_tokens=False).input_ids
    return [tok.decode(x[-1]) for x in ids]
