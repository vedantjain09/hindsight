"""LoRA SFT on {prompt, completion} jsonl (Colab T4). Loss on completion tokens only.
  python scripts/train_sft.py --data runs/salvage_sft.jsonl --out runs/lora_salvage"""
import argparse, json, random

p = argparse.ArgumentParser()
p.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
p.add_argument("--data", required=True); p.add_argument("--out", required=True)
p.add_argument("--epochs", type=int, default=2); p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--max-examples", type=int, default=None, help="cap so conditions are compared at equal data budget")
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

random.seed(a.seed); torch.manual_seed(a.seed)
tok = AutoTokenizer.from_pretrained(a.base)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.float16 if DEV == "cuda" else torch.float32).to(DEV)
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
model.print_trainable_parameters()
rows = [json.loads(l) for l in open(a.data)]
random.shuffle(rows)
rows = rows[: a.max_examples] if a.max_examples else rows
chat = lambda pr: tok.apply_chat_template([{"role": "user", "content": pr}], tokenize=False, add_generation_prompt=True)
opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=a.lr)
model.train()
for ep in range(a.epochs):
    random.shuffle(rows)
    for i in range(0, len(rows), 8):
        loss_sum = 0.0
        for r in rows[i:i + 8]:
            pi = tok(chat(r["prompt"]), return_tensors="pt").input_ids
            ci = tok(r["completion"] + tok.eos_token, return_tensors="pt", add_special_tokens=False).input_ids
            ids = torch.cat([pi, ci], 1).to(DEV)
            labels = ids.clone(); labels[:, : pi.shape[1]] = -100
            loss = model(input_ids=ids, labels=labels).loss / 8
            loss.backward(); loss_sum += loss.item()
        opt.step(); opt.zero_grad()
        if (i // 8) % 20 == 0:
            print(f"ep{ep} step{i // 8} loss {loss_sum:.4f}", flush=True)
model.save_pretrained(a.out)
