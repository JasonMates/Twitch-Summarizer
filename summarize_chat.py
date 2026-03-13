import argparse
import math
import re
from pathlib import Path

import torch
from chat_preprocess import DistillOpts, distill_text
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize a chat-log .txt file using a fine-tuned BART model."
    )
    parser.add_argument(
        "--input_txt",
        type=Path,
        required=True,
        help="Path to input chat-log text file.",
    )
    parser.add_argument(
        "--output_txt",
        type=Path,
        default=None,
        help="Path for summary output text. Defaults to <input_stem>.summary.txt",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="bart_chat_model_retrained/final",
        help="Local model directory or Hugging Face model id.",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=1024,
        help="Encoder max input length used during generation.",
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=64,
        help="Maximum generated summary length.",
    )
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument(
        "--max_output_sentences",
        type=int,
        default=2,
        help="Maximum number of final summary sentences. Use 0 to keep all.",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Load model/tokenizer only from local files.",
    )
    parser.add_argument(
        "--show_chunk_info",
        action="store_true",
        help="Print chunk and reduction details.",
    )
    parser.add_argument(
        "--clean_input",
        dest="clean_input",
        action="store_true",
        help="Distill input chat before summarization (recommended).",
    )
    parser.add_argument(
        "--no_clean_input",
        dest="clean_input",
        action="store_false",
        help="Disable input distillation.",
    )
    parser.set_defaults(clean_input=True)
    parser.add_argument(
        "--strip_metadata",
        dest="strip_metadata",
        action="store_true",
        help="Drop timestamps/usernames during input distillation.",
    )
    parser.add_argument(
        "--keep_metadata",
        dest="strip_metadata",
        action="store_false",
        help="Keep timestamps/usernames during input distillation.",
    )
    parser.set_defaults(strip_metadata=True)
    parser.add_argument(
        "--drop_system_lines",
        dest="drop_system_lines",
        action="store_true",
        help="Drop bot/system lines during input distillation.",
    )
    parser.add_argument(
        "--keep_system_lines",
        dest="drop_system_lines",
        action="store_false",
        help="Keep bot/system lines during input distillation.",
    )
    parser.set_defaults(drop_system_lines=True)
    parser.add_argument(
        "--collapse_duplicate_messages",
        dest="collapse_duplicate_messages",
        action="store_true",
        help="Collapse repeated chat lines into one line with a count suffix.",
    )
    parser.add_argument(
        "--no_collapse_duplicate_messages",
        dest="collapse_duplicate_messages",
        action="store_false",
        help="Do not collapse repeated chat lines.",
    )
    parser.set_defaults(collapse_duplicate_messages=True)
    parser.add_argument(
        "--duplicate_count_threshold",
        type=int,
        default=3,
        help="Minimum repeat count to collapse duplicated lines.",
    )
    parser.add_argument(
        "--collapse_token_floods",
        dest="collapse_token_floods",
        action="store_true",
        help="Collapse repeated-token floods inside each line.",
    )
    parser.add_argument(
        "--no_collapse_token_floods",
        dest="collapse_token_floods",
        action="store_false",
        help="Do not collapse repeated-token floods.",
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
        help="Minimum characters required for a line after cleanup.",
    )
    parser.add_argument(
        "--max_messages",
        type=int,
        default=0,
        help="Optional cap on distilled lines before chunking (0 disables cap).",
    )
    return parser.parse_args()


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        raise ValueError(f"Input file is empty: {path}")
    return text


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}.summary.txt")


def split_chunks(text: str, tokenizer, max_source_length: int) -> list[str]:
    token_budget = max(16, max_source_length - 2)
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        lines = [text]

    chunks: list[str] = []
    current_lines: list[str] = []
    current_tokens = 0

    def flush_current() -> None:
        nonlocal current_lines, current_tokens
        if current_lines:
            chunks.append("\n".join(current_lines))
            current_lines = []
            current_tokens = 0

    for line in lines:
        line_ids = tokenizer.encode(line, add_special_tokens=False)
        if not line_ids:
            continue

        if len(line_ids) > token_budget:
            flush_current()
            num_pieces = math.ceil(len(line_ids) / token_budget)
            for i in range(num_pieces):
                start = i * token_budget
                end = start + token_budget
                piece = tokenizer.decode(line_ids[start:end], skip_special_tokens=True).strip()
                if piece:
                    chunks.append(piece)
            continue

        needed = len(line_ids) + 1
        if current_lines and current_tokens + needed > token_budget:
            flush_current()
        current_lines.append(line)
        current_tokens += needed

    flush_current()
    return chunks


def pack_items(items: list[str], tokenizer, token_budget: int) -> list[list[str]]:
    groups: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0

    for item in items:
        item = item.strip()
        if not item:
            continue

        item_ids = tokenizer.encode(item, add_special_tokens=False)
        if not item_ids:
            continue
        if len(item_ids) > token_budget:
            item_ids = item_ids[:token_budget]
            item = tokenizer.decode(item_ids, skip_special_tokens=True).strip()
        needed = len(item_ids) + 1

        if current and current_tokens + needed > token_budget:
            groups.append(current)
            current = [item]
            current_tokens = needed
        else:
            current.append(item)
            current_tokens += needed

    if current:
        groups.append(current)
    return groups


def generate_summary(
    model,
    tokenizer,
    text: str,
    device: torch.device,
    max_source_length: int,
    max_target_length: int,
    num_beams: int,
    no_repeat_ngram_size: int,
    length_penalty: float,
) -> str:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_source_length,
    ).to(device)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_length=max_target_length,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=length_penalty,
            early_stopping=True,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def sentence_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def looks_like_duplicate(a: str, b: str) -> bool:
    if a.lower() == b.lower():
        return True
    a_tokens = sentence_tokens(a)
    b_tokens = sentence_tokens(b)
    if not a_tokens or not b_tokens:
        return False
    overlap = a_tokens & b_tokens
    union = a_tokens | b_tokens
    jaccard = len(overlap) / len(union)
    containment = len(overlap) / min(len(a_tokens), len(b_tokens))
    return jaccard >= 0.86 or containment >= 0.92


def split_sentences(text: str) -> list[str]:
    cleaned = re.sub(r"^\s*summary\s*:\s*", "", text.strip(), flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    out = []
    for part in parts:
        sentence = part.strip()
        if not sentence:
            continue
        if sentence[-1] not in ".!?":
            sentence += "."
        out.append(sentence)
    return out


def clean_sum(text: str, max_output_sentences: int) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return ""

    kept: list[str] = []
    for sentence in sentences:
        if any(looks_like_duplicate(sentence, prior) for prior in kept):
            continue
        kept.append(sentence)

    if max_output_sentences > 0:
        kept = kept[:max_output_sentences]
    return " ".join(kept).strip()


def hier_sum(
    chunks: list[str],
    model,
    tokenizer,
    device: torch.device,
    max_source_length: int,
    max_target_length: int,
    num_beams: int,
    no_repeat_ngram_size: int,
    length_penalty: float,
    show_chunk_info: bool,
) -> str:
    if not chunks:
        return ""

    summaries = []
    for idx, chunk in enumerate(chunks, start=1):
        summary = generate_summary(
            model=model,
            tokenizer=tokenizer,
            text=chunk,
            device=device,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=length_penalty,
        )
        summaries.append(summary)
        if show_chunk_info:
            print(f"Chunk {idx}/{len(chunks)} summary: {summary}")

    if len(summaries) == 1:
        return summaries[0]

    unique_summaries = {s for s in summaries if s}
    if len(unique_summaries) == 1:
        return next(iter(unique_summaries))

    reduction_budget = max(32, max_source_length - 24)
    round_num = 1
    while len(summaries) > 1:
        grouped = pack_items(summaries, tokenizer, reduction_budget)
        next_summaries = []
        for group in grouped:
            prompt = " ".join(group)
            reduced = generate_summary(
                model=model,
                tokenizer=tokenizer,
                text=prompt,
                device=device,
                max_source_length=max_source_length,
                max_target_length=max_target_length,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
                length_penalty=length_penalty,
            )
            next_summaries.append(reduced)
        summaries = next_summaries
        if show_chunk_info:
            print(f"Reduction round {round_num}: {len(summaries)} summary block(s) remaining.")
        round_num += 1

    return summaries[0]


# Backward-compatible aliases.
split_chat_into_chunks = split_chunks
clean_summary = clean_sum
hierarchical_summarize = hier_sum


def main() -> None:
    args = parse_args()
    input_text = read_text(args.input_txt)
    output_path = args.output_txt if args.output_txt is not None else default_output_path(args.input_txt)
    raw_line_count = len([line for line in input_text.splitlines() if line.strip()])

    if args.clean_input:
        distill_options = DistillOpts(
            strip_metadata=args.strip_metadata,
            drop_system_lines=args.drop_system_lines,
            collapse_duplicate_messages=args.collapse_duplicate_messages,
            duplicate_count_threshold=args.duplicate_count_threshold,
            collapse_token_floods=args.collapse_token_floods,
            token_flood_threshold=args.token_flood_threshold,
            min_message_chars=args.min_message_chars,
            max_messages=args.max_messages,
        )
        input_text, distill_stats = distill_text(input_text, options=distill_options)
        input_text = input_text.strip()
        if not input_text:
            raise ValueError("Input became empty after cleanup. Relax cleanup flags and retry.")
        if args.show_chunk_info:
            print(
                "Input distillation: "
                f"lines {raw_line_count} -> {distill_stats.output_lines}, "
                f"system removed={distill_stats.removed_system_lines}, "
                f"duplicates collapsed={distill_stats.collapsed_duplicate_lines}, "
                f"flood runs collapsed={distill_stats.collapsed_token_flood_runs}."
            )

    print(f"Loading tokenizer from: {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        local_files_only=args.local_files_only,
        use_fast=True,
    )
    print(f"Loading model from: {args.model_dir}")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_dir,
        local_files_only=args.local_files_only,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Using device: {device}")

    chunks = split_chunks(
        text=input_text,
        tokenizer=tokenizer,
        max_source_length=args.max_source_length,
    )
    if args.show_chunk_info:
        print(f"Prepared {len(chunks)} chunk(s) for summarization.")

    summary = hier_sum(
        chunks=chunks,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        num_beams=args.num_beams,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        length_penalty=args.length_penalty,
        show_chunk_info=args.show_chunk_info,
    )
    summary = clean_sum(summary, args.max_output_sentences)

    output_path.write_text(summary + "\n", encoding="utf-8")
    print(f"Summary written to: {output_path}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
