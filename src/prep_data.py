from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

BOT_PATTERNS = [
    re.compile(r"\bsubscribed\b", re.IGNORECASE),
    re.compile(r"\bsubscription\b", re.IGNORECASE),
    re.compile(r"\bgift(?:ed)?\s*sub(?:scription)?s?\b", re.IGNORECASE),
    re.compile(r"\b\d+\s*gift\s*sub(?:scription)?s?\b", re.IGNORECASE),
    re.compile(r"\braid(?:ed)?\b", re.IGNORECASE),
    re.compile(r"\bcheer\s*\d+\b", re.IGNORECASE),
    re.compile(r"\bprime\s+gaming\b", re.IGNORECASE),
    re.compile(r"\bprime\s+sub\b", re.IGNORECASE),
    re.compile(r"\b\d+[- ]?month(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bwelcome to the chat room\b", re.IGNORECASE),
]

NOISE_LINE_PATTERNS = [
    re.compile(r"^!\w+"),
    re.compile(r"^@\w+\s*->"),
]

NOISE_TOKEN_PREFIXES = (
    "caseoh", "jynxzi", "twitchcon", "robloxcheer", "goldplz", "subprise", "dailydoodle", "thorfork",
    "kittyyy", "kappu", "dinodance", "powerup", "squid", "pogchamp", "kekw", "lul", "omegalul",
)

NOISE_TOKEN_EXACT = {
    "kappa", "pog", "lul", "lol", "lmao", "kekw", "ww", "www", "w", "l", "fr", "bro", "yo",
    "case", "caseoh", "chat", "stream", "live", "hello", "hiii", "hi", "hey", "pls", "please",
}

STOPWORDS = {
    "the", "and", "for", "that", "with", "this", "from", "were", "was", "are", "you", "your", "chat",
    "about", "what", "when", "where", "they", "them", "have", "just", "like", "dont", "don't", "its",
    "it's", "had", "out", "into", "then", "than", "there", "here", "play", "showed", "reaction",
    "response", "messages", "around", "activity", "watch", "today", "tonight", "really", "very",
}

LAUGHTER_TERMS = {"lol", "lmao", "rofl", "omegalul", "kek", "lmfao", "lul", "haha", "hahaha"}
CONFUSION_TERMS = {"what", "huh", "why", "confused", "wtf", "idk", "how"}
POS_TERMS = {"w", "goat", "fire", "peak", "based", "good", "great", "love", "pog", "nice"}
NEG_TERMS = {"l", "trash", "bad", "mid", "awful", "boring", "hate", "terrible"}

LEADS = [
    "Chat response to {topic} showed {sentiment}.",
    "Chat activity around {topic} was {sentiment}.",
    "In chat, {topic} drew {sentiment}.",
    "Chat's reaction to {topic} was {sentiment}.",
    "Messages about {topic} in chat showed {sentiment}.",
    "The chat around {topic} turned {sentiment}.",
]

EXPLANATIONS = {
    "approval": "suggested chat was broadly supportive.",
    "disapproval": "suggested chat was mostly critical.",
    "confusion": "suggested chat was reacting to unclear context.",
    "laughter": "suggested chat found the moment funny.",
    "mixed": "suggested chat had mixed reactions.",
}


def strip_prefixes(line: str) -> str:
    line = re.sub(r"^\[\d{1,2}:\d{2}:\d{2}\]\s*", "", line).strip()

    if ":" in line:
        left, right = line.split(":", 1)
        left_l = left.lower()
        if (
            len(left) <= 160
            and (
                re.search(r"\b\d{1,2}\s?[ap]m\b", left_l)
                or any(k in left_l for k in ["subscriber", "turbo", "prime", "gift", "cheer", "badge", "recap", "runner"])
            )
        ):
            line = right.strip()

    line = re.sub(r"^[^:]{1,40}:\s*", "", line).strip()
    return line


def squash_repeated_chars(text: str) -> str:
    return re.sub(r"(.)\1{3,}", r"\1\1\1", text)


def squash_repeated_tokens(text: str, max_repeat: int = 2) -> str:
    toks = text.split()
    if not toks:
        return text
    out = []
    prev = None
    run = 0
    for t in toks:
        tl = t.lower()
        if tl == prev:
            run += 1
        else:
            prev = tl
            run = 1
        if run <= max_repeat:
            out.append(t)
    return " ".join(out)


def is_noise_line(line: str) -> bool:
    lower = line.lower()
    if any(p.search(lower) for p in BOT_PATTERNS):
        return True
    if any(p.search(lower) for p in NOISE_LINE_PATTERNS):
        return True

    toks = lower.split()
    if len(toks) >= 5 and len(set(toks)) == 1:
        return True

    alpha = sum(ch.isalpha() for ch in line)
    if alpha == 0:
        return True
    return False


def normalize_line(raw: str) -> str:
    line = strip_prefixes(raw.strip())
    line = squash_repeated_chars(line)
    line = squash_repeated_tokens(line, max_repeat=2)
    line = re.sub(r"\s+", " ", line).strip()
    return line


def clean_chat(raw_text: str, max_lines: int) -> list[str]:
    out = []
    seen = set()
    for raw in raw_text.splitlines():
        line = normalize_line(raw)
        if not line or is_noise_line(line):
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
        if len(out) >= max_lines:
            break
    return out


def tokenize_words(line: str) -> list[str]:
    return re.findall(r"[a-zA-Z]{3,}", line.lower())


def is_noise_token(tok: str) -> bool:
    if tok in STOPWORDS or tok in NOISE_TOKEN_EXACT:
        return True
    if len(tok) > 12:
        return True
    if any(tok.startswith(p) for p in NOISE_TOKEN_PREFIXES):
        return True
    return False


def topic_candidates(lines: list[str]) -> Counter[str]:
    count = Counter()
    line_occ = defaultdict(set)

    for i, line in enumerate(lines):
        toks = [t for t in tokenize_words(line) if not is_noise_token(t)]
        # de-dup token hits per line for stability
        for t in set(toks):
            line_occ[t].add(i)
        for a, b in zip(toks, toks[1:]):
            bg = f"{a} {b}"
            line_occ[bg].add(i)

    for k, rows in line_occ.items():
        count[k] = len(rows)

    return count


def infer_topic(lines: list[str], min_line_occ: int) -> str:
    c = topic_candidates(lines)
    if not c:
        return "the ongoing discussion"

    bigrams = [(k, v) for k, v in c.items() if " " in k and v >= min_line_occ]
    if bigrams:
        bigrams.sort(key=lambda kv: (kv[1], len(kv[0])), reverse=True)
        return bigrams[0][0]

    unigrams = [(k, v) for k, v in c.items() if " " not in k and v >= min_line_occ]
    if unigrams:
        unigrams.sort(key=lambda kv: (kv[1], -len(kv[0])), reverse=True)
        return unigrams[0][0]

    return "the ongoing discussion"


def infer_sentiment(lines: list[str]) -> str:
    toks = re.findall(r"[a-zA-Z']+", "\n".join(lines).lower())
    c = Counter(toks)
    laugh = sum(c[t] for t in LAUGHTER_TERMS)
    conf = sum(c[t] for t in CONFUSION_TERMS)
    pos = sum(c[t] for t in POS_TERMS)
    neg = sum(c[t] for t in NEG_TERMS)

    if laugh >= 2 and laugh >= max(conf, pos, neg):
        return "laughter"
    if conf >= 2 and conf >= max(laugh, pos, neg):
        return "confusion"
    if pos - neg >= 2:
        return "approval"
    if neg - pos >= 2:
        return "disapproval"
    return "mixed"


def is_human_readable_line(line: str) -> bool:
    toks = line.split()
    if len(toks) < 4:
        return False
    words = [t for t in toks if re.search(r"[A-Za-z]", t)]
    if len(words) < 4:
        return False
    # reject lines dominated by emote-like tokens
    emoteish = 0
    for w in words:
        wl = re.sub(r"[^a-z]", "", w.lower())
        if not wl:
            continue
        if is_noise_token(wl):
            emoteish += 1
    if emoteish / max(1, len(words)) > 0.5:
        return False
    return True


def quote_score(line: str, topic: str) -> int:
    s = 0
    ll = line.lower()
    if topic != "the ongoing discussion":
        s += 2 if topic in ll else 0
        for t in topic.split():
            if t in ll:
                s += 1
    if any(w in ll for w in LAUGHTER_TERMS | CONFUSION_TERMS | POS_TERMS | NEG_TERMS):
        s += 1
    if is_human_readable_line(line):
        s += 2
    return s


def shorten_quote(line: str, max_words: int = 12) -> str:
    words = line.split()
    if len(words) <= max_words:
        return line
    return " ".join(words[:max_words])


def choose_quotes(lines: list[str], topic: str) -> tuple[str, str]:
    candidates = [ln for ln in lines if is_human_readable_line(ln)]
    if len(candidates) < 2:
        candidates = lines[:]

    ranked = sorted(candidates, key=lambda ln: quote_score(ln, topic), reverse=True)
    if not ranked:
        return "chat was active", "people kept reacting"

    q1 = shorten_quote(ranked[0])
    q2 = q1
    for cand in ranked[1:]:
        if cand.lower() != ranked[0].lower():
            q2 = shorten_quote(cand)
            break
    return q1, q2


def build_summary(row_id: str, lines: list[str], topic_min_line_occ: int) -> str:
    topic = infer_topic(lines, topic_min_line_occ)
    sentiment = infer_sentiment(lines)
    q1, q2 = choose_quotes(lines, topic)

    h = int(hashlib.md5(row_id.encode("utf-8")).hexdigest(), 16)
    lead = LEADS[h % len(LEADS)].format(topic=topic, sentiment=sentiment)
    return f"{lead} Messages like '{q1}' and '{q2}' {EXPLANATIONS[sentiment]}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build improved training pairs (v3).")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--topic-min-line-occ", type=int, default=3)
    ap.add_argument("--min-lines", type=int, default=4)
    ap.add_argument("--max-lines", type=int, default=140)
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows_in = rows_out = dropped = fallback = 0
    sent = Counter()

    with inp.open("r", encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
        for raw in fin:
            raw = raw.strip()
            if not raw:
                continue
            rows_in += 1
            row = json.loads(raw)

            lines = clean_chat(row.get("input_text", ""), max_lines=args.max_lines)
            if len(lines) < args.min_lines:
                dropped += 1
                continue

            topic = infer_topic(lines, args.topic_min_line_occ)
            if topic == "the ongoing discussion":
                fallback += 1

            summary = build_summary(row.get("id", f"row_{rows_in}"), lines, args.topic_min_line_occ)
            m = re.search(r"\b(approval|disapproval|laughter|confusion|mixed)\b", summary)
            if m:
                sent[m.group(1)] += 1

            fout.write(json.dumps({
                "id": row.get("id", f"row_{rows_in}"),
                "input_text": "\n".join(lines),
                "target_text": summary,
            }, ensure_ascii=False) + "\n")
            rows_out += 1

    print(f"rows_in={rows_in}")
    print(f"rows_out={rows_out}")
    print(f"dropped_short={dropped}")
    print(f"topic_fallback={fallback}")
    print("sentiment_counts=" + json.dumps(sent, ensure_ascii=False))


if __name__ == "__main__":
    main()
