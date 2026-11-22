"""
Fine-tuning LoRA du petit modèle sur GPU (Google Colab, Kaggle ou une machine avec carte NVIDIA).
Même données, même consigne et mêmes réglages que entrainer_mac.sh, pour que les résultats soient comparables.

Sur Kaggle (2 GPU T4) :
    !git clone https://github.com/adam-hassen/assuria-profil-slm.git
    %cd assuria-profil-slm
    !pip install -q "transformers==4.57.6" "peft==0.17.1" accelerate datasets
    !torchrun --nproc_per_node 2 entrainer_gpu.py --epochs 1 --accumulation 2

Sur un seul GPU (Colab) : !python entrainer_gpu.py --epochs 1
Lot effectif = lot x accumulation x nombre de GPU (4 dans les deux cas ci-dessus).

Sortie : modeles/<modele>/fusionne (modèle complet, format Hugging Face)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

DOSSIER = Path(__file__).parent


def lire(nom):
    with open(DOSSIER / "donnees" / f"{nom}.jsonl", encoding="utf-8") as f:
        return [json.loads(l)["messages"] for l in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modele", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lot", type=int, default=1)
    ap.add_argument("--accumulation", type=int, default=4, help="lot effectif = lot x accumulation")
    ap.add_argument("--max-len", type=int, default=3072)
    a = ap.parse_args()

    nom = a.modele.split("/")[-1].lower()
    sortie = DOSSIER / "modeles" / nom
    tok = AutoTokenizer.from_pretrained(a.modele)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    coupes = [0]

    def encoder(messages):
        # la perte ne porte que sur la réponse (le JSON), comme --mask-prompt avec MLX
        invite = tok.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True)
        reponse = messages[2]["content"] + tok.eos_token
        ids_invite = tok(invite, add_special_tokens=False)["input_ids"]
        ids_reponse = tok(reponse, add_special_tokens=False)["input_ids"]
        if len(ids_invite) + len(ids_reponse) > a.max_len:
            # trop long : on retire le début de la conversation, jamais la réponse attendue
            coupes[0] += 1
            garde = a.max_len - len(ids_reponse)
            ids_invite = ids_invite[:200] + ids_invite[-(garde - 200):]
        ids = ids_invite + ids_reponse
        labels = [-100] * len(ids_invite) + ids_reponse
        return {"input_ids": ids, "labels": labels}

    train = Dataset.from_list([encoder(m) for m in lire("train")])
    valid = Dataset.from_list([encoder(m) for m in lire("valid")])
    print(f"{len(train)} exemples d'entraînement, {len(valid)} de validation, {coupes[0]} raccourcis (plus de {a.max_len} tokens)")

    def assembler(lot):
        n = max(len(x["input_ids"]) for x in lot)
        pad = tok.pad_token_id
        return {
            "input_ids": torch.tensor([x["input_ids"] + [pad] * (n - len(x["input_ids"])) for x in lot]),
            "attention_mask": torch.tensor([[1] * len(x["input_ids"]) + [0] * (n - len(x["input_ids"])) for x in lot]),
            "labels": torch.tensor([x["labels"] + [-100] * (n - len(x["labels"])) for x in lot]),
        }

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    modele = AutoModelForCausalLM.from_pretrained(a.modele, torch_dtype=torch.bfloat16 if bf16 else torch.float16)
    modele.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    modele.enable_input_require_grads()
    modele = get_peft_model(modele, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    modele.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=str(sortie / "checkpoints"), num_train_epochs=a.epochs,
        per_device_train_batch_size=a.lot, per_device_eval_batch_size=a.lot, gradient_accumulation_steps=a.accumulation,
        learning_rate=1e-4, lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=20, eval_strategy="steps", eval_steps=250, save_steps=250, save_total_limit=2,
        bf16=bf16, fp16=not bf16, report_to="none", remove_unused_columns=False, ddp_find_unused_parameters=False)
    entraineur = Trainer(model=modele, args=args, train_dataset=train, eval_dataset=valid, data_collator=assembler)
    if entraineur.is_world_process_zero():
        n = max(1, torch.cuda.device_count())
        print(f"GPU : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'aucun (CPU)'} x {n}")
    entraineur.train()
    if not entraineur.is_world_process_zero():  # avec 2 GPU (torchrun), un seul processus enregistre le modèle
        return

    modele.save_pretrained(sortie / "adapters")
    fusionne = modele.merge_and_unload()
    fusionne.save_pretrained(sortie / "fusionne", safe_serialization=True)
    tok.save_pretrained(sortie / "fusionne")
    print(f"Modèle fine-tuné : {sortie / 'fusionne'}")


if __name__ == "__main__":
    main()
