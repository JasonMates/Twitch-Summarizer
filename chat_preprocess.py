import re
from collections import Counter
from dataclasses import dataclass


LINE_RE = re.compile(r"^\[[^\]]*\]\s*([^:]+):\s*(.*)$")
PLAIN_LINE_RE = re.compile(r"^\s*([^:]+):\s*(.*)$")
WORD_RE = re.compile(r"[A-Za-z0-9']+")
URL_RE = re.compile(r"https?://\S+", re.I)
MENTION_RE = re.compile(r"@\w+")
SPACE_RE = re.compile(r"\s+")
MOMENTS_BADGE_RE = re.compile(
    r"(?:\d+\s*-\s*year\s*subscriber\s*)?moments\s+badge\s*-\s*tier\s*\d+\s*",
    re.I,
)

BOT_USERS = {
    "nightbot",
    "streamelements",
    "moobot",
    "streamlabs",
    "fossabot",
    "wizebot",
    "deepbot",
}

SYSTEM_PATTERNS = [
    re.compile(r"\bsubscribed(?:\s+at\s+tier)?\b", re.I),
    re.compile(r"\bsubscribed\s+with\s+prime\b", re.I),
    re.compile(r"\bresubscribed\b", re.I),
    re.compile(r"\bgift(?:ed|ing)\b", re.I),
    re.compile(r"\bwatch streak\b", re.I),
    re.compile(r"\bconsecutive streams\b", re.I),
    re.compile(r"\bfirst time chatter\b", re.I),
    re.compile(r"\bcommunity sub\b", re.I),
    re.compile(r"\bprime gaming\b", re.I),
    re.compile(r"\bcheer(?:ed|ing)?\b", re.I),
]


@dataclass
class DistillOpts:
    strip_metadata: bool = True
    drop_system_lines: bool = True
    collapse_duplicate_messages: bool = True
    duplicate_count_threshold: int = 3
    collapse_token_floods: bool = True
    token_flood_threshold: int = 4
    min_message_chars: int = 2
    max_messages: int = 0


@dataclass
class DistillStats:
    input_lines: int = 0
    output_lines: int = 0
    removed_empty_lines: int = 0
    removed_system_lines: int = 0
    collapsed_duplicate_lines: int = 0
    collapsed_duplicate_groups: int = 0
    collapsed_token_flood_runs: int = 0
    removed_noise_lines: int = 0
    truncated_lines: int = 0


def parse_line(raw: str) -> tuple[str, str]:
    line = raw.strip()
    if not line:
        return "", ""
    match = LINE_RE.match(line)
    if match:
        user = MOMENTS_BADGE_RE.sub("", match.group(1)).strip()
        return user, match.group(2).strip()
    match = PLAIN_LINE_RE.match(line)
    if match:
        user = MOMENTS_BADGE_RE.sub("", match.group(1)).strip()
        text = match.group(2).strip()
        return user, text
    return "", line


def is_system_line(user: str, text: str) -> bool:
    user_l = user.lower().strip()
    text_l = text.lower().strip()
    if not text_l:
        return True
    if user_l in BOT_USERS:
        return True
    return any(pattern.search(text_l) for pattern in SYSTEM_PATTERNS)


def normalize_message(text: str) -> str:
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def canonicalize_for_dedup(text: str) -> str:
    lowered = text.lower()
    tokens = WORD_RE.findall(lowered)
    return " ".join(tokens).strip()


def strip_known_noise_phrases(text: str) -> str:
    cleaned = MOMENTS_BADGE_RE.sub(" ", text)
    return SPACE_RE.sub(" ", cleaned).strip()


def compress_token_floods(text: str, threshold: int) -> tuple[str, int]:
    tokens = text.split()
    if len(tokens) < max(2, threshold):
        return text, 0

    out_tokens: list[str] = []
    collapsed_runs = 0
    idx = 0
    while idx < len(tokens):
        run_end = idx + 1
        while run_end < len(tokens) and tokens[run_end].lower() == tokens[idx].lower():
            run_end += 1
        run_len = run_end - idx
        if run_len >= threshold:
            out_tokens.append(f"{tokens[idx]} x{run_len}")
            collapsed_runs += 1
        else:
            out_tokens.extend(tokens[idx:run_end])
        idx = run_end

    if collapsed_runs == 0:
        low_tokens = [token.lower() for token in tokens]
        counts = Counter(low_tokens)
        if len(counts) <= 2 and len(tokens) >= threshold:
            parts = []
            for token, count in counts.most_common():
                if count >= threshold:
                    parts.append(f"{token} x{count}")
                else:
                    parts.extend([token] * count)
            if parts:
                return " ".join(parts), 1
        return text, 0

    return " ".join(out_tokens), collapsed_runs


def distill_lines(
    chat_text: str, options: DistillOpts | None = None
) -> tuple[list[str], DistillStats]:
    opts = options or DistillOpts()
    stats = DistillStats()

    raw_lines = [line for line in chat_text.splitlines() if line.strip()]
    stats.input_lines = len(raw_lines)
    if not raw_lines:
        return [], stats

    dedup_order: list[str] = []
    dedup_message: dict[str, str] = {}
    dedup_count: Counter[str] = Counter()
    kept_lines: list[str] = []

    for raw in raw_lines:
        user, text = parse_line(raw)
        payload = text if opts.strip_metadata else raw.strip()
        payload = strip_known_noise_phrases(payload)
        payload = normalize_message(payload)

        if opts.drop_system_lines and is_system_line(user, text):
            stats.removed_system_lines += 1
            continue
        if len(payload) < opts.min_message_chars:
            if MOMENTS_BADGE_RE.search(raw):
                stats.removed_noise_lines += 1
            else:
                stats.removed_empty_lines += 1
            continue

        if opts.collapse_token_floods:
            payload, collapsed_runs = compress_token_floods(
                payload,
                threshold=max(2, opts.token_flood_threshold),
            )
            stats.collapsed_token_flood_runs += collapsed_runs

        if opts.collapse_duplicate_messages:
            key = canonicalize_for_dedup(payload)
            if not key:
                stats.removed_empty_lines += 1
                continue
            if key not in dedup_message:
                dedup_order.append(key)
                dedup_message[key] = payload
            dedup_count[key] += 1
        else:
            kept_lines.append(payload)

    if opts.collapse_duplicate_messages:
        threshold = max(2, opts.duplicate_count_threshold)
        for key in dedup_order:
            line = dedup_message[key]
            count = dedup_count[key]
            if count >= threshold:
                kept_lines.append(f"{line} x{count}")
                stats.collapsed_duplicate_groups += 1
                stats.collapsed_duplicate_lines += count - 1
            else:
                kept_lines.extend([line] * count)

    if opts.max_messages > 0 and len(kept_lines) > opts.max_messages:
        stats.truncated_lines = len(kept_lines) - opts.max_messages
        kept_lines = kept_lines[: opts.max_messages]

    stats.output_lines = len(kept_lines)
    return kept_lines, stats


def distill_text(
    chat_text: str, options: DistillOpts | None = None
) -> tuple[str, DistillStats]:
    lines, stats = distill_lines(chat_text, options=options)
    return "\n".join(lines), stats


# Backward-compatible aliases.
ChatDistillOptions = DistillOpts
ChatDistillStats = DistillStats
distill_chat_lines = distill_lines
distill_chat_text = distill_text
