"""LoRA DPO on {prompt, chosen, rejected} jsonl (Colab T4). Reference model = the same weights with the adapter disabled.
  python scripts/train_dpo.py --data runs/dpo.jsonl --out runs/lora_dpo"""
import argparse, json, random

p = argparse.ArgumentParser()
p.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
p.add_argument("--data", required=True); p.add_argument("--out", required=True)
p.add_argument("--epochs", type=int, default=2); p.add_argument("--lr", type=float, default=5e-5)
p.add_argument("--beta", type=float, default=0.1); p.add_argument("--max-examples", type=int, default=None)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from hindsight.train_utils import dpo_loss, seq_logp

random.seed(a.seed); torch.manual_seed(a.seed)
tok = AutoTokenizer.from_pretrained(a.base)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.float16 if DEV == "cuda" else torch.float32).to(DEV)
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
rows = [json.loads(l) for l in open(a.data)]
random.shuffle(rows)
rows = rows[: a.max_examples] if a.max_examples else rows
chat = lambda pr: tok.apply_chat_template([{"role": "user", "content": pr}], tokenize=False, add_generation_prompt=True)


def logp(m, prompt, completion):
    pi = tok(chat(prompt), return_tensors="pt").input_ids
    ci = tok(completion + tok.eos_token, return_tensors="pt", add_special_tokens=False).input_ids
    ids = torch.cat([pi, ci], 1).to(DEV)
    logits = m(input_ids=ids).logits[:, :-1]
    labels = ids[:, 1:]
    mask = torch.zeros_like(labels, dtype=torch.float32); mask[:, pi.shape[1] - 1:] = 1
    return seq_logp(logits, labels, mask)


opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=a.lr)
for ep in range(a.epochs):
    random.shuffle(rows)
    for i in range(0, len(rows), 4):
        tot, acc = 0.0, 0.0
        for r in rows[i:i + 4]:
            with torch.no_grad(), model.disable_adapter():
                rc, rr = logp(model, r["prompt"], r["chosen"]), logp(model, r["prompt"], r["rejected"])
            pc, pr = logp(model, r["prompt"], r["chosen"]), logp(model, r["prompt"], r["rejected"])
            loss, a_ = dpo_loss(pc, pr, rc, rr, a.beta)
            (loss / 4).backward(); tot += loss.item() / 4; acc += a_.item() / 4
        opt.step(); opt.zero_grad()
        if (i // 4) % 10 == 0:
            print(f"ep{ep} step{i // 4} loss {tot:.4f} pref-acc {acc:.2f}", flush=True)
model.save_pretrained(a.out)
