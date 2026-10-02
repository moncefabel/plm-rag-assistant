"""LoRA fine-tuning of a sentence-transformers embedding model on PLM pairs.

Training pairs (question, change order) come from change orders that are NOT
used by the evaluation set, so the recall gain measured by ``rag.evaluate`` is
not leakage.

    python -m rag.finetune --out models/lora-plm
    python -m rag.evaluate --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
        --adapter models/lora-plm --tag lora

Only the LoRA matrices on the attention query/value projections are trained;
the base encoder stays frozen. Loss: MultipleNegativesRankingLoss (in-batch
negatives), the standard contrastive objective for retrieval embeddings.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .data_gen import load_jsonl, write
from .embeddings import DEFAULT_MODEL


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="models/lora-plm")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=8)
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments, losses)
    from sentence_transformers.training_args import BatchSamplers

    torch.manual_seed(0)
    if not (Path(args.data) / "train_pairs.jsonl").exists():
        write(args.data)
    pairs = load_jsonl(Path(args.data) / "train_pairs.jsonl")
    dataset = Dataset.from_list([{"anchor": p["query"], "positive": p["positive"]}
                                 for p in pairs])

    model = SentenceTransformer(args.model)
    encoder = model[0].auto_model
    encoder.add_adapter(LoraConfig(r=args.rank, lora_alpha=2 * args.rank,
                                   lora_dropout=0.1, target_modules=["query", "value"]))
    for name, param in encoder.named_parameters():
        param.requires_grad = "lora_" in name
    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    total = sum(p.numel() for p in encoder.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f} %)")

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(Path(args.out) / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # no false negatives in a batch
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        seed=0,
    )
    trainer = SentenceTransformerTrainer(
        model=model, args=training_args, train_dataset=dataset,
        loss=losses.MultipleNegativesRankingLoss(model))
    trainer.train()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    encoder.save_pretrained(args.out)  # writes adapter_config.json + adapter weights only
    print(f"LoRA adapter saved to {args.out}")


if __name__ == "__main__":
    main()
