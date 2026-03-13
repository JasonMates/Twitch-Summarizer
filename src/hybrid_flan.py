import argparse
import math
import re
from pathlib import Path

import torch
from chat_preprocess import ChatDistillOptions, distill_chat_lines
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Hybrid summarizer: distill chat lines with MiniLM embeddings, "
            "then rewrite with FLAN."
        )
    )
    parser.add_argument("--input_txt", type=Path, required=True, help="Input chat-log text file.")
    parser.add_argument(
        "--output_txt",
        type=Path,
        default=None,
        help="Output summary path. Defaults to <input_stem>.hybrid.summary.txt",
    )
    parser.add_argument(
        "--flan_model_dir",
        type=str,
        default="flan_t5_small_model_epoch1/final",
        help="Local FLAN model directory or model id.",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model id/path for semantic distillation.",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Load embedding/FLAN artifacts only from local files.",
    )
    parser.add_argument("--max_source_length", type=int, default=384)
    parser.add_argument("--max_target_length", type=int, default=64)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--max_output_sentences", type=int, default=2)
    parser.add_argument(
        "--max_embedding_lines",
        type=int,
        default=320,
        help="Cap on distilled lines passed into embedding stage.",
    )
    parser.add_argument(
        "--max_selected_lines",
        type=int,
        default=80,
        help="Maximum representative lines selected from embedding stage.",
    )
    parser.add_argument(
        "--line_similarity_threshold",
        type=float,
        default=0.82,
        help="Cosine threshold above which a candidate line is treated as redundant.",
    )
    parser.add_argument(
        "--source_prefix",
        type=str,
        default="summarize: ",
        help="Optional source prefix for FLAN generation.",
    )
    parser.add_argument("--show_info", action="store_true", help="Print distillation details.")

    parser.add_argument("--strip_metadata", dest="strip_metadata", action="store_true")
    parser.add_argument("--keep_metadata", dest="strip_metadata", action="store_false")
    parser.set_defaults(strip_metadata=True)
    parser.add_argument("--drop_system_lines", dest="drop_system_lines", action="store_true")
    parser.add_argument("--keep_system_lines", dest="drop_system_lines", action="store_false")
    parser.set_defaults(drop_system_lines=True)
    parser.add_argument(
        "--collapse_duplicate_messages",
        dest="collapse_duplicate_messages",
        action="store_true",
    )
    parser.add_argument(
        "--no_collapse_duplicate_messages",
        dest="collapse_duplicate_messages",
        action="store_false",
    )
    parser.set_defaults(collapse_duplicate_messages=True)
    parser.add_argument("--duplicate_count_threshold", type=int, default=3)
    parser.add_argument("--collapse_token_floods", dest="collapse_token_floods", action="store_true")
    parser.add_argument(
        "--no_collapse_token_floods",
        dest="collapse_token_floods",
        action="store_false",
    )
    parser.set_defaults(collapse_token_floods=True)
    parser.add_argument("--token_flood_threshold", type=int, default=4)
    parser.add_argument("--min_message_chars", type=int, default=2)
    parser.add_argument("--max_messages", type=int, default=0)

    return parser.parse_args()


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        raise ValueError(f"Input file is empty: {path}")
    return text


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}.hybrid.summary.txt")


def split_chat_into_chunks(text: str, tokenizer, max_source_length: int) -> list[str]:
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
            for idx in range(num_pieces):
                start = idx * token_budget
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
    source_prefix: str,
    max_source_length: int,
    max_target_length: int,
    num_beams: int,
    no_repeat_ngram_size: int,
    length_penalty: float,
) -> str:
    prompt = f"{source_prefix}{text}" if source_prefix else text
    inputs = tokenizer(
        prompt,
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


def clean_summary(text: str, max_output_sentences: int) -> str:
    cleaned = re.sub(r"^\s*summary\s*:\s*", "", text.strip(), flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return ""

    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    sentences = []
    for part in parts:
        sentence = part.strip()
        if not sentence:
            continue
        if sentence[-1] not in ".!?":
            sentence += "."
        if any(looks_like_duplicate(sentence, prior) for prior in sentences):
            continue
        sentences.append(sentence)

    if max_output_sentences > 0:
        sentences = sentences[:max_output_sentences]
    return " ".join(sentences).strip()


def hierarchical_summarize(
    text: str,
    model,
    tokenizer,
    device: torch.device,
    source_prefix: str,
    max_source_length: int,
    max_target_length: int,
    num_beams: int,
    no_repeat_ngram_size: int,
    length_penalty: float,
) -> str:
    chunks = split_chat_into_chunks(text, tokenizer, max_source_length)
    if not chunks:
        return ""
    summaries = [
        generate_summary(
            model=model,
            tokenizer=tokenizer,
            text=chunk,
            device=device,
            source_prefix=source_prefix,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=length_penalty,
        )
        for chunk in chunks
    ]
    if len(summaries) == 1:
        return summaries[0]

    unique_summaries = {s for s in summaries if s}
    if len(unique_summaries) == 1:
        return next(iter(unique_summaries))

    reduction_budget = max(32, max_source_length - 24)
    while len(summaries) > 1:
        grouped = pack_items(summaries, tokenizer, reduction_budget)
        summaries = [
            generate_summary(
                model=model,
                tokenizer=tokenizer,
                text=" ".join(group),
                device=device,
                source_prefix=source_prefix,
                max_source_length=max_source_length,
                max_target_length=max_target_length,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
                length_penalty=length_penalty,
            )
            for group in grouped
        ]
    return summaries[0]


def select_semantic_lines(
    lines: list[str],
    embedding_model: str,
    local_files_only: bool,
    max_embedding_lines: int,
    max_selected_lines: int,
    similarity_threshold: float,
) -> list[str]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for hybrid mode. "
            "Install it with: pip install sentence-transformers"
        ) from exc

    if not lines:
        return []
    if max_embedding_lines > 0:
        lines = lines[:max_embedding_lines]
    if len(lines) <= max_selected_lines:
        return lines

    embedder = SentenceTransformer(embedding_model, local_files_only=local_files_only)
    embeddings = embedder.encode(
        lines,
        batch_size=64,
        show_progress_bar=False,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    if embeddings.ndim != 2:
        return lines[:max_selected_lines]

    centroid = embeddings.mean(dim=0, keepdim=True)
    centroid = centroid / centroid.norm(dim=1, keepdim=True).clamp_min(1e-8)
    scores = torch.matmul(embeddings, centroid.T).squeeze(1)
    ranked = torch.argsort(scores, descending=True).tolist()

    selected_indices: list[int] = []
    selected_vectors: list[torch.Tensor] = []
    for idx in ranked:
        candidate = embeddings[idx]
        if selected_vectors:
            sims = torch.stack([torch.dot(candidate, chosen) for chosen in selected_vectors])
            if float(torch.max(sims)) >= similarity_threshold:
                continue
        selected_indices.append(idx)
        selected_vectors.append(candidate)
        if len(selected_indices) >= max_selected_lines:
            break

    if not selected_indices:
        selected_indices = [ranked[0]]
    selected_indices.sort()
    return [lines[idx] for idx in selected_indices]


def main() -> None:
    args = parse_args()
    input_text = read_text(args.input_txt)
    output_path = args.output_txt if args.output_txt is not None else default_output_path(args.input_txt)

    distill_options = ChatDistillOptions(
        strip_metadata=args.strip_metadata,
        drop_system_lines=args.drop_system_lines,
        collapse_duplicate_messages=args.collapse_duplicate_messages,
        duplicate_count_threshold=args.duplicate_count_threshold,
        collapse_token_floods=args.collapse_token_floods,
        token_flood_threshold=args.token_flood_threshold,
        min_message_chars=args.min_message_chars,
        max_messages=args.max_messages,
    )
    distilled_lines, distill_stats = distill_chat_lines(input_text, options=distill_options)
    if not distilled_lines:
        raise ValueError("No usable lines remain after input cleanup.")

    semantic_lines = select_semantic_lines(
        lines=distilled_lines,
        embedding_model=args.embedding_model,
        local_files_only=args.local_files_only,
        max_embedding_lines=args.max_embedding_lines,
        max_selected_lines=args.max_selected_lines,
        similarity_threshold=args.line_similarity_threshold,
    )
    semantic_text = "\n".join(semantic_lines).strip()
    if not semantic_text:
        raise ValueError("No semantic lines selected after embedding distillation.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.flan_model_dir,
        local_files_only=args.local_files_only,
        use_fast=True,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.flan_model_dir,
        local_files_only=args.local_files_only,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    summary = hierarchical_summarize(
        text=semantic_text,
        model=model,
        tokenizer=tokenizer,
        device=device,
        source_prefix=args.source_prefix,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        num_beams=args.num_beams,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        length_penalty=args.length_penalty,
    )
    summary = clean_summary(summary, args.max_output_sentences)

    output_path.write_text(summary + "\n", encoding="utf-8")
    if args.show_info:
        print(
            f"Distillation lines: {distill_stats.input_lines} -> {distill_stats.output_lines} -> {len(semantic_lines)}"
        )
        print(
            "Cleanup details: "
            f"system_removed={distill_stats.removed_system_lines}, "
            f"duplicates_collapsed={distill_stats.collapsed_duplicate_lines}, "
            f"flood_runs_collapsed={distill_stats.collapsed_token_flood_runs}"
        )
    print(f"Summary written to: {output_path}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
