#!/usr/bin/env python3
"""
Few-shot chat summarizer (retrieval + template generation).

Uses labeled moments (with `abstractive_summary.summary`) as demonstrations,
retrieves nearest examples for each unlabeled moment, and generates a concise
summary without external LLM/API calls.

Usage:
  python few_shot_summarizer.py --target moment_data/jasontheween.json

Optional:
  python few_shot_summarizer.py --target moment_data/jasontheween.json \
      --train moment_data/Jerma1.json moment_data/Northernlion1.json moment_data/Squeex.json \
      --k 3 --output moment_data/jasontheween_fewshot.json --preview 5
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has", "he",
    "in", "is", "it", "its", "of", "on", "or", "that", "the", "to", "was", "were", "will",
    "with", "you", "your", "we", "they", "this", "those", "these", "i", "me", "my", "our",
}

EMOTION_LEXICON = {
    "hype": {"pog", "poggers", "letsgo", "letsg", "let's", "hype", "insane", "holy", "w", "w+", "clutch"},
    "laughter": {"lol", "lmao", "lmfao", "lul", "lulw", "omegalul", "xd", "rofl", "haha"},
    "frustration": {"wtf", "bruh", "sadge", "trash", "bad", "throw", "choke", "rigged"},
    "surprise": {"no", "way", "what", "crazy", "actually", "omg"},
}


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z0-9']+", text.lower())
    out: List[str] = []
    for t in tokens:
        if t in STOPWORDS:
            continue
        if len(t) < 2:
            continue
        out.append(t)
    return out


def extract_message_text(raw_line: str) -> str:
    m = re.match(r"\[[^\]]+\]\s+[^:]+:\s*(.*)", raw_line)
    return m.group(1).strip() if m else raw_line.strip()


def get_moment_text(moment: Dict) -> str:
    msgs = moment.get("messages", [])
    key = ((moment.get("extractive_summary") or {}).get("key_messages") or [])
    merged = []
    # Weight extractive key messages heavily for better grounding.
    merged.extend(key)
    merged.extend(key)
    merged.extend(extract_message_text(x) for x in msgs[:40])
    return "\n".join(merged)


def tfidf_vectors(docs: List[List[str]]) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    n_docs = len(docs)
    df = Counter()
    for d in docs:
        for t in set(d):
            df[t] += 1

    idf: Dict[str, float] = {}
    for t, c in df.items():
        idf[t] = math.log((1 + n_docs) / (1 + c)) + 1.0

    vectors: List[Dict[str, float]] = []
    for d in docs:
        tf = Counter(d)
        norm = 0.0
        vec: Dict[str, float] = {}
        for t, c in tf.items():
            val = (1 + math.log(c)) * idf.get(t, 1.0)
            vec[t] = val
            norm += val * val
        norm = math.sqrt(norm) if norm > 0 else 1.0
        for t in list(vec.keys()):
            vec[t] /= norm
        vectors.append(vec)
    return vectors, idf


def cosine_sparse(a: Dict[str, float], b: Dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())


def build_query_vector(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    tf = Counter(tokens)
    vec: Dict[str, float] = {}
    norm = 0.0
    for t, c in tf.items():
        w = (1 + math.log(c)) * idf.get(t, 1.0)
        vec[t] = w
        norm += w * w
    norm = math.sqrt(norm) if norm > 0 else 1.0
    for t in list(vec.keys()):
        vec[t] /= norm
    return vec


def dominant_emotion(text: str) -> str:
    toks = tokenize(text)
    counts = {k: 0 for k in EMOTION_LEXICON}
    for t in toks:
        for k, words in EMOTION_LEXICON.items():
            if t in words:
                counts[k] += 1
    best = max(counts.items(), key=lambda x: x[1])
    return best[0] if best[1] > 0 else "reaction"


def top_phrases(text: str, n: int = 3) -> List[str]:
    toks = tokenize(text)
    freq = Counter(toks)
    return [t for t, _ in freq.most_common(n)]


def summary_from_examples(moment: Dict, nearest_examples: List[Dict], single_sentence: bool = True) -> str:
    text = get_moment_text(moment)
    emotion = dominant_emotion(text)
    phrases = top_phrases(text, n=3)

    example_words = Counter()
    for ex in nearest_examples:
        s = ((ex.get("abstractive_summary") or {}).get("summary") or "").lower()
        for t in tokenize(s):
            example_words[t] += 1

    # Canonical ruleset output:
    # Chat reaction to [TOPIC] showed [SENTIMENT], with repeated messages indicating [INTERPRETATION].
    topic = phrases[0] if phrases else "the moment"
    sentiment_map = {
        "laughter": "laughter",
        "hype": "hype",
        "frustration": "disapproval",
        "surprise": "confusion",
        "reaction": "hype",
    }
    sentiment = sentiment_map.get(emotion, "hype")
    interp_map = {
        "laughter": "a shared joke across chat",
        "hype": "a hype spike in chat",
        "disapproval": "disagreement and pushback",
        "confusion": "confusion about what happened",
        "approval": "strong agreement with the point",
    }
    interpretation = interp_map.get(sentiment, "a clear shared reaction")

    if single_sentence:
        return (
            f"Chat reaction to {topic} showed {sentiment}, with repeated messages indicating {interpretation}."
        )

    return (
        f"Chat reaction to {topic} showed {sentiment}, with repeated messages indicating {interpretation}."
    )


def load_moments(path: Path) -> List[Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return data


def has_summary(moment: Dict) -> bool:
    s = (moment.get("abstractive_summary") or {}).get("summary")
    return bool(isinstance(s, str) and s.strip())


def simple_overlap_proxy(moment: Dict, summary: str) -> float:
    msg_toks = set(tokenize(get_moment_text(moment)))
    sum_toks = set(tokenize(summary))
    if not sum_toks:
        return 0.0
    return len(msg_toks & sum_toks) / len(sum_toks)


def run(args: argparse.Namespace) -> None:
    target_path = Path(args.target)
    output_path = Path(args.output) if args.output else target_path.with_name(f"{target_path.stem}_fewshot.json")

    train_paths = [Path(p) for p in args.train]
    train_examples: List[Dict] = []
    for p in train_paths:
        moments = load_moments(p)
        train_examples.extend([m for m in moments if has_summary(m)])

    if not train_examples:
        raise RuntimeError("No labeled training examples found. Provide at least one JSON with abstractive summaries.")

    train_docs = [tokenize(get_moment_text(m)) for m in train_examples]
    train_vecs, idf = tfidf_vectors(train_docs)

    target = load_moments(target_path)

    generated = 0
    overlap_scores: List[float] = []

    for moment in target:
        if has_summary(moment) and not args.overwrite:
            continue

        q_vec = build_query_vector(tokenize(get_moment_text(moment)), idf)
        sims = [(idx, cosine_sparse(q_vec, train_vecs[idx])) for idx in range(len(train_examples))]
        sims.sort(key=lambda x: x[1], reverse=True)

        nearest = [train_examples[i] for i, _ in sims[: args.k]]
        summary = summary_from_examples(moment, nearest, single_sentence=args.single_sentence)

        moment["abstractive_summary"] = {"summary": summary}

        generated += 1
        overlap_scores.append(simple_overlap_proxy(moment, summary))

    output_path.write_text(json.dumps(target, indent=2, ensure_ascii=False), encoding="utf-8")

    coverage = generated / max(len(target), 1)
    avg_len = 0.0
    if generated:
        lengths = [len((m.get("abstractive_summary") or {}).get("summary", "")) for m in target if has_summary(m)]
        avg_len = sum(lengths) / len(lengths) if lengths else 0.0

    print("=" * 70)
    print("Few-shot summarization complete")
    print("=" * 70)
    print(f"Target file: {target_path}")
    print(f"Output file: {output_path}")
    print(f"Training examples: {len(train_examples)} from {len(train_paths)} files")
    print(f"Generated summaries: {generated}/{len(target)} (coverage={coverage:.1%})")
    print(f"Average summary length: {avg_len:.1f} chars")
    if overlap_scores:
        print(f"Lexical overlap proxy (avg): {sum(overlap_scores)/len(overlap_scores):.3f}")

    if args.preview > 0:
        print("\nPreview:")
        shown = 0
        for m in target:
            s = (m.get("abstractive_summary") or {}).get("summary")
            if not s:
                continue
            print(f"- {m.get('window_id', 'unknown')}: {s[:180]}{'...' if len(s) > 180 else ''}")
            shown += 1
            if shown >= args.preview:
                break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Few-shot summarizer for Twitch chat moments")
    parser.add_argument("--target", required=True, help="Target moments JSON file to summarize")
    parser.add_argument(
        "--train",
        nargs="+",
        default=[
            "moment_data/Jerma1.json",
            "moment_data/Northernlion1.json",
            "moment_data/Northernlion2.json",
            "moment_data/Squeex.json",
        ],
        help="Labeled training JSON files with abstractive summaries",
    )
    parser.add_argument("--k", type=int, default=3, help="Number of nearest few-shot examples")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing summaries in target")
    parser.add_argument("--single-sentence", action="store_true", default=True, help="Generate exactly one sentence")
    parser.add_argument("--preview", type=int, default=5, help="Print N summary previews")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
