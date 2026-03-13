import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import torch
from chat_preprocess import ChatDistillOptions, distill_chat_text
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a BART model on chat->summary JSONL pairs."
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default=Path("bart_chat_summary_pairs.jsonl"),
        help="Path to JSONL file with chat/summary pairs.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="facebook/bart-base",
        help="Hugging Face model id or local path.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("bart_chat_model"),
        help="Directory for checkpoints and final model.",
    )
    parser.add_argument(
        "--input_key",
        type=str,
        default="chat",
        help="JSON key containing source text.",
    )
    parser.add_argument(
        "--target_key",
        type=str,
        default="summary",
        help="JSON key containing summary text.",
    )
    parser.add_argument(
        "--source_prefix",
        type=str,
        default="",
        help="Optional prefix prepended to every source input (useful for T5-style prompts).",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation split ratio in [0, 1).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_source_length", type=int, default=1024)
    parser.add_argument("--max_target_length", type=int, default=64)
    parser.add_argument(
        "--max_examples_per_target",
        type=int,
        default=0,
        help=(
            "Optional cap for identical target summaries in the training split. "
            "Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--max_examples_per_target_prefix",
        type=int,
        default=0,
        help=(
            "Optional cap for summaries sharing the same opening words. "
            "Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--target_prefix_words",
        type=int,
        default=4,
        help="Number of summary words used for target-prefix capping.",
    )
    parser.add_argument(
        "--drop_low_info_targets",
        action="store_true",
        help="Drop target summaries with obviously low-information phrasing.",
    )
    parser.add_argument(
        "--clean_chat_input",
        dest="clean_chat_input",
        action="store_true",
        help=(
            "Distill chat input before training (strip metadata, remove system lines, "
            "compress duplicate/flood messages)."
        ),
    )
    parser.add_argument(
        "--no_clean_chat_input",
        dest="clean_chat_input",
        action="store_false",
        help="Disable chat-input distillation.",
    )
    parser.set_defaults(clean_chat_input=True)
    parser.add_argument(
        "--strip_metadata",
        dest="strip_metadata",
        action="store_true",
        help="Drop timestamps/usernames from each chat line during distillation.",
    )
    parser.add_argument(
        "--keep_metadata",
        dest="strip_metadata",
        action="store_false",
        help="Keep timestamps/usernames in distilled input.",
    )
    parser.set_defaults(strip_metadata=True)
    parser.add_argument(
        "--drop_system_lines",
        dest="drop_system_lines",
        action="store_true",
        help="Remove bot/system lines during chat-input distillation.",
    )
    parser.add_argument(
        "--keep_system_lines",
        dest="drop_system_lines",
        action="store_false",
        help="Keep bot/system lines in distilled input.",
    )
    parser.set_defaults(drop_system_lines=True)
    parser.add_argument(
        "--collapse_duplicate_messages",
        dest="collapse_duplicate_messages",
        action="store_true",
        help="Collapse repeated chat messages into one line with a count suffix.",
    )
    parser.add_argument(
        "--no_collapse_duplicate_messages",
        dest="collapse_duplicate_messages",
        action="store_false",
        help="Do not collapse repeated chat messages.",
    )
    parser.set_defaults(collapse_duplicate_messages=True)
    parser.add_argument(
        "--duplicate_count_threshold",
        type=int,
        default=3,
        help="Minimum repeat count to collapse duplicated messages.",
    )
    parser.add_argument(
        "--collapse_token_floods",
        dest="collapse_token_floods",
        action="store_true",
        help="Collapse emote/token floods inside a single chat line.",
    )
    parser.add_argument(
        "--no_collapse_token_floods",
        dest="collapse_token_floods",
        action="store_false",
        help="Do not collapse emote/token floods inside a line.",
    )
    parser.set_defaults(collapse_token_floods=True)
    parser.add_argument(
        "--token_flood_threshold",
        type=int,
        default=4,
        help="Minimum repeated-token run length to collapse within one line.",
    )
    parser.add_argument(
        "--min_message_chars",
        type=int,
        default=2,
        help="Minimum characters required for a chat line after cleanup.",
    )
    parser.add_argument(
        "--max_messages_per_example",
        type=int,
        default=0,
        help="Optional cap on distilled chat lines per example (0 disables cap).",
    )
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    return parser.parse_args()


LOW_INFO_TARGET_PATTERNS = [
    re.compile(r"\bother things\b", re.I),
    re.compile(r"\bwatching some goofball\b", re.I),
    re.compile(r"\btopic(?:\s+is)?\s+unclear\b", re.I),
]


def text_word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text))


def target_prefix_key(target: str, prefix_words: int) -> str:
    words = re.findall(r"[A-Za-z0-9']+", target.lower())
    if not words:
        return target.lower().strip()
    return " ".join(words[: max(1, prefix_words)])


def is_low_info_target(target: str) -> bool:
    trimmed = target.strip()
    lowered = trimmed.lower()
    if any(pattern.search(lowered) for pattern in LOW_INFO_TARGET_PATTERNS):
        return True
    tokens = re.findall(r"[A-Za-z0-9']+", lowered)
    if len(tokens) < 6:
        return True
    if len(set(tokens)) <= 2 and len(tokens) >= 6:
        return True
    return False


def load_pairs(
    data_path: Path,
    input_key: str,
    target_key: str,
    clean_chat_input: bool,
    chat_options: ChatDistillOptions,
    drop_low_info_targets: bool,
) -> tuple[list[tuple[str, str]], dict[str, int]]:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    pairs: list[tuple[str, str]] = []
    stats: Counter[str] = Counter()
    with data_path.open("r", encoding="utf-8") as f:
        for line_num, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_num}: {exc}") from exc

            source = obj.get(input_key)
            target = obj.get(target_key)
            if not isinstance(source, str) or not isinstance(target, str):
                continue

            stats["examples_seen"] += 1
            source = source.strip()
            target = target.strip()
            if not source or not target:
                continue

            stats["source_words_before"] += text_word_count(source)
            stats["source_lines_before"] += len([line for line in source.splitlines() if line.strip()])

            if clean_chat_input:
                source, distill_stats = distill_chat_text(source, options=chat_options)
                stats["removed_system_lines"] += distill_stats.removed_system_lines
                stats["removed_empty_lines"] += distill_stats.removed_empty_lines
                stats["collapsed_duplicate_lines"] += distill_stats.collapsed_duplicate_lines
                stats["collapsed_duplicate_groups"] += distill_stats.collapsed_duplicate_groups
                stats["collapsed_token_flood_runs"] += distill_stats.collapsed_token_flood_runs
                stats["truncated_lines"] += distill_stats.truncated_lines

            source = source.strip()
            if not source:
                stats["dropped_empty_source_after_cleaning"] += 1
                continue

            if drop_low_info_targets and is_low_info_target(target):
                stats["dropped_low_info_targets"] += 1
                continue

            stats["source_words_after"] += text_word_count(source)
            stats["source_lines_after"] += len([line for line in source.splitlines() if line.strip()])
            pairs.append((source, target))
            stats["kept_pairs"] += 1

    if not pairs:
        raise ValueError(
            f"No valid pairs found in {data_path}. Expected keys: '{input_key}' and '{target_key}'."
        )
    return pairs, dict(stats)


def split_pairs(
    pairs: list[tuple[str, str]], val_ratio: float, seed: int
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be in [0, 1).")
    if len(pairs) < 2 or val_ratio == 0:
        return pairs, []

    indices = list(range(len(pairs)))
    random.Random(seed).shuffle(indices)

    val_size = int(len(indices) * val_ratio)
    if val_size <= 0:
        val_size = 1
    if val_size >= len(indices):
        val_size = len(indices) - 1

    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    train_pairs = [pairs[i] for i in train_idx]
    val_pairs = [pairs[i] for i in val_idx]
    return train_pairs, val_pairs


def cap_train_pairs_by_target(
    train_pairs: list[tuple[str, str]],
    max_examples_per_target: int,
    seed: int,
) -> list[tuple[str, str]]:
    if max_examples_per_target <= 0:
        return train_pairs

    grouped: dict[str, list[tuple[str, str]]] = {}
    for pair in train_pairs:
        grouped.setdefault(pair[1], []).append(pair)

    rng = random.Random(seed)
    capped: list[tuple[str, str]] = []
    for target, group in grouped.items():
        rng.shuffle(group)
        capped.extend(group[:max_examples_per_target])

    rng.shuffle(capped)
    return capped


def cap_train_pairs_by_target_prefix(
    train_pairs: list[tuple[str, str]],
    max_examples_per_prefix: int,
    prefix_words: int,
    seed: int,
) -> list[tuple[str, str]]:
    if max_examples_per_prefix <= 0:
        return train_pairs

    grouped: dict[str, list[tuple[str, str]]] = {}
    for pair in train_pairs:
        grouped.setdefault(target_prefix_key(pair[1], prefix_words), []).append(pair)

    rng = random.Random(seed)
    capped: list[tuple[str, str]] = []
    for _, group in grouped.items():
        rng.shuffle(group)
        capped.extend(group[:max_examples_per_prefix])

    rng.shuffle(capped)
    return capped


def duplicate_target_count(pairs: list[tuple[str, str]]) -> int:
    target_counts = Counter(target for _, target in pairs)
    return sum(count - 1 for count in target_counts.values() if count > 1)


class ChatSummaryDataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[str, str]],
        tokenizer,
        max_source_length: int,
        max_target_length: int,
        source_prefix: str = "",
    ) -> None:
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.source_prefix = source_prefix

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        source, target = self.pairs[idx]
        source = f"{self.source_prefix}{source}"
        model_inputs = self.tokenizer(
            source,
            truncation=True,
            max_length=self.max_source_length,
        )
        labels = self.tokenizer(
            text_target=target,
            truncation=True,
            max_length=self.max_target_length,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    print(f"Loading tokenizer: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        local_files_only=args.local_files_only,
        use_fast=True,
    )

    print(f"Loading model: {args.model_name_or_path}")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name_or_path,
        local_files_only=args.local_files_only,
    )

    max_positions = getattr(model.config, "max_position_embeddings", None)
    if isinstance(max_positions, int) and args.max_source_length > max_positions:
        print(
            f"max_source_length ({args.max_source_length}) exceeds model limit "
            f"({max_positions}). Clamping to {max_positions}."
        )
        args.max_source_length = max_positions

    chat_options = ChatDistillOptions(
        strip_metadata=args.strip_metadata,
        drop_system_lines=args.drop_system_lines,
        collapse_duplicate_messages=args.collapse_duplicate_messages,
        duplicate_count_threshold=args.duplicate_count_threshold,
        collapse_token_floods=args.collapse_token_floods,
        token_flood_threshold=args.token_flood_threshold,
        min_message_chars=args.min_message_chars,
        max_messages=args.max_messages_per_example,
    )

    pairs, load_stats = load_pairs(
        args.data_path,
        args.input_key,
        args.target_key,
        clean_chat_input=args.clean_chat_input,
        chat_options=chat_options,
        drop_low_info_targets=args.drop_low_info_targets,
    )
    train_pairs, val_pairs = split_pairs(pairs, args.val_ratio, args.seed)

    if args.clean_chat_input:
        processed_examples = (
            load_stats.get("kept_pairs", 0)
            + load_stats.get("dropped_low_info_targets", 0)
            + load_stats.get("dropped_empty_source_after_cleaning", 0)
        )
        processed_examples = max(1, processed_examples)
        avg_words_before = load_stats.get("source_words_before", 0) / processed_examples
        avg_words_after = load_stats.get("source_words_after", 0) / max(
            1, load_stats.get("kept_pairs", 0)
        )
        avg_lines_before = load_stats.get("source_lines_before", 0) / processed_examples
        avg_lines_after = load_stats.get("source_lines_after", 0) / max(
            1, load_stats.get("kept_pairs", 0)
        )
        print(
            "Input distillation stats: "
            f"avg words {avg_words_before:.1f} -> {avg_words_after:.1f}, "
            f"avg lines {avg_lines_before:.1f} -> {avg_lines_after:.1f}, "
            f"system lines removed={load_stats.get('removed_system_lines', 0)}, "
            f"duplicate lines collapsed={load_stats.get('collapsed_duplicate_lines', 0)}, "
            f"flood runs collapsed={load_stats.get('collapsed_token_flood_runs', 0)}."
        )
        if load_stats.get("truncated_lines", 0) > 0:
            print(f"Truncated distilled lines due to max_messages_per_example: {load_stats['truncated_lines']}.")
    if args.drop_low_info_targets and load_stats.get("dropped_low_info_targets", 0) > 0:
        print(f"Dropped low-info targets: {load_stats['dropped_low_info_targets']}.")

    before_train_size = len(train_pairs)
    before_train_duplicates = duplicate_target_count(train_pairs)
    train_pairs = cap_train_pairs_by_target(
        train_pairs=train_pairs,
        max_examples_per_target=args.max_examples_per_target,
        seed=args.seed,
    )
    after_train_size = len(train_pairs)
    after_train_duplicates = duplicate_target_count(train_pairs)
    before_prefix_cap_size = len(train_pairs)
    train_pairs = cap_train_pairs_by_target_prefix(
        train_pairs=train_pairs,
        max_examples_per_prefix=args.max_examples_per_target_prefix,
        prefix_words=args.target_prefix_words,
        seed=args.seed,
    )
    after_prefix_cap_size = len(train_pairs)
    print(
        f"Loaded {len(pairs)} valid pairs from {args.data_path}. "
        f"Train: {len(train_pairs)} | Val: {len(val_pairs)}"
    )
    if args.max_examples_per_target > 0:
        print(
            "Applied training target cap "
            f"(max_examples_per_target={args.max_examples_per_target}). "
            f"Train size: {before_train_size} -> {after_train_size}. "
            f"Duplicate target entries: {before_train_duplicates} -> {after_train_duplicates}."
        )
    if args.max_examples_per_target_prefix > 0:
        print(
            "Applied training target-prefix cap "
            f"(max_examples_per_target_prefix={args.max_examples_per_target_prefix}, "
            f"target_prefix_words={args.target_prefix_words}). "
            f"Train size: {before_prefix_cap_size} -> {after_prefix_cap_size}."
        )

    train_dataset = ChatSummaryDataset(
        train_pairs,
        tokenizer=tokenizer,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        source_prefix=args.source_prefix,
    )
    eval_dataset = (
        ChatSummaryDataset(
            val_pairs,
            tokenizer=tokenizer,
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length,
            source_prefix=args.source_prefix,
        )
        if val_pairs
        else None
    )

    use_cuda = torch.cuda.is_available()
    use_bf16 = bool(args.bf16 and use_cuda and torch.cuda.is_bf16_supported())
    use_fp16 = bool(args.fp16 and use_cuda and not use_bf16)
    if args.fp16 and not use_fp16:
        print("fp16 requested but CUDA fp16 is unavailable; continuing without fp16.")
    if args.bf16 and not use_bf16:
        print("bf16 requested but CUDA bf16 is unavailable; continuing without bf16.")

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        eval_strategy="epoch" if eval_dataset is not None else "no",
        save_strategy="epoch" if eval_dataset is not None else "steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="eval_loss" if eval_dataset is not None else None,
        greater_is_better=False if eval_dataset is not None else None,
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        fp16=use_fp16,
        bf16=use_bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=0,
        dataloader_pin_memory=use_cuda,
        use_cpu=not use_cuda,
        predict_with_generate=False,
        generation_max_length=args.max_target_length,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if use_cuda else None,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    print("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    if eval_dataset is not None:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    final_dir = args.output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Saved model and tokenizer to: {final_dir}")


if __name__ == "__main__":
    main()
