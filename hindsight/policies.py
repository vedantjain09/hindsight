"""Policies. A policy maps (live env, history so far, rng) -> one JSON action.

NoisyExpert is the free, scripted policy used for the attribution benchmark: it follows the oracle plan but injects
mistakes (harmless lookups/errors and consequential wrong actions) at configurable rates.
LocalLM is the real policy for Colab: it sees ONLY the rendered prompt (goal, tool docs, history), never the plan.
"""
from __future__ import annotations

import json
import random
import re
from typing import Protocol

from .core import FINISH, Env, Step, akey, render_prompt


class Policy(Protocol):
    name: str

    def act(self, env: Env, history: list[Step], rng: random.Random) -> dict: ...


class NoisyExpert:
    def __init__(self, eps_dec: float = 0.04, eps_harm: float = 0.08, name: str | None = None):
        self.eps_dec, self.eps_harm = eps_dec, eps_harm
        self.name = name or f"noisy-expert(dec={eps_dec},harm={eps_harm})"

    def act(self, env: Env, history: list[Step], rng: random.Random) -> dict:
        done = {akey(s.action) for s in history if not s.error}
        r = rng.random()
        if r < self.eps_dec + self.eps_harm and env.mistakes:
            return rng.choice(env.mistakes)
        for a in env.plan:
            if akey(a) not in done:
                return a
        return FINISH

    def act_batch(self, items, rngs):
        return [self.act(e, h, r) for (e, h), r in zip(items, rngs)]


_JSON = re.compile(r"\{.*?\}(?=\s*$|\s*\n|\s*```)|\{.*\}", re.S)


def parse_action(text: str) -> dict:
    """Pull the first well-formed JSON action out of a model completion; invalid output becomes a no-op error."""
    text = text.strip().strip("`")
    dec, i = json.JSONDecoder(), 0
    while i < len(text):
        j = text.find("{", i)
        if j < 0:
            break
        try:
            obj, _ = dec.raw_decode(text[j:])
            if isinstance(obj, dict) and "tool" in obj:
                obj.setdefault("args", {})
                return obj
        except json.JSONDecodeError:
            pass
        i = j + 1
    return {"tool": "invalid", "args": {}}


class LocalLM:
    """Hugging Face causal LM policy (run on Colab). Batches generation across concurrent rollouts."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", adapter: str | None = None,
                 temperature: float = 0.7, max_new_tokens: int = 80, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if dev == "cuda" else None).to(dev)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.dev, self.temperature, self.max_new = dev, temperature, max_new_tokens
        self.name = f"lm:{model_name}" + (f"+{adapter}" if adapter else "")

    def _chat(self, prompt: str) -> str:
        return self.tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)

    def act_batch(self, items, rngs):
        torch = self.torch
        texts = [self._chat(render_prompt(e, h)) for e, h in items]
        out_all = []
        for i in range(0, len(texts), 16):
            enc = self.tok(texts[i:i + 16], return_tensors="pt", padding=True).to(self.dev)
            with torch.no_grad():
                out = self.model.generate(**enc, do_sample=self.temperature > 0, temperature=max(self.temperature, 1e-5),
                                          max_new_tokens=self.max_new, pad_token_id=self.tok.pad_token_id)
            out_all += self.tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return [parse_action(t) for t in out_all]

    def act(self, env, history, rng):
        return self.act_batch([(env, history)], [rng])[0]


def act_batch(policy, items, rngs):
    if hasattr(policy, "act_batch"):
        return policy.act_batch(items, rngs)
    return [policy.act(e, h, r) for (e, h), r in zip(items, rngs)]
