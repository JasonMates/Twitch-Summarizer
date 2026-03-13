import glob
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict


BASE_DIR = os.path.join(os.path.dirname(__file__), "moment_data")

LINE_RE = re.compile(r"^\[[^\]]*\]\s*([^:]+):\s*(.*)$")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9']+")
URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")

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
    "subscribed at tier",
    "subscribed with prime",
    "gifted",
    "watch streak",
    "consecutive streams",
    "they've subscribed",
    "has subscribed",
    "resubscribed",
    "month streak",
    "first time chatter",
    "community sub",
    "is gifting",
    "prime gaming",
    "sparked a watch streak",
]

STREAMER_ALIASES = {
    "Caseoh.json": {"caseoh", "casey", "case", "caso"},
    "Jerma1.json": {"jerma", "jerma985"},
    "Northernlion1.json": {"northernlion", "nl", "ryan"},
    "Northernlion2.json": {"northernlion", "nl", "ryan"},
    "Squeex.json": {"squeex"},
    "jasontheween.json": {"jason", "jasontheween", "theween"},
    "vinesauce.json": {"vinesauce", "vinny", "vine"},
}

STOPWORDS = {
    "a",
    "about",
    "after",
    "all",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "bro",
    "but",
    "by",
    "can",
    "could",
    "chat",
    "do",
    "dude",
    "for",
    "from",
    "get",
    "getting",
    "go",
    "going",
    "good",
    "guy",
    "guys",
    "had",
    "has",
    "have",
    "he",
    "her",
    "here",
    "him",
    "his",
    "how",
    "i",
    "if",
    "im",
    "in",
    "into",
    "is",
    "it",
    "its",
    "just",
    "like",
    "lol",
    "look",
    "looks",
    "lmao",
    "me",
    "moment",
    "my",
    "no",
    "not",
    "now",
    "of",
    "oh",
    "on",
    "one",
    "or",
    "our",
    "out",
    "people",
    "pls",
    "please",
    "really",
    "right",
    "so",
    "some",
    "someone",
    "something",
    "still",
    "stream",
    "stuff",
    "t",
    "that",
    "the",
    "their",
    "them",
    "there",
    "they",
    "thing",
    "this",
    "tho",
    "to",
    "too",
    "u",
    "up",
    "ur",
    "was",
    "we",
    "what",
    "when",
    "where",
    "who",
    "why",
    "with",
    "would",
    "yeah",
    "you",
    "your",
}

NOISE_TOKENS = {
    "awoo",
    "clap",
    "cmonbruh",
    "derp",
    "emote",
    "fr",
    "gg",
    "glorp",
    "glorpsi",
    "huh",
    "icant",
    "kekw",
    "kkonaw",
    "lolll",
    "lul",
    "monkas",
    "omegalul",
    "oop",
    "pearto",
    "pog",
    "poggers",
    "saj",
    "xdd",
    "xd",
    "ww",
}

LAUGHTER_RE = re.compile(r"\b(lol|lmao|lmfao|lul|omegalul|kekw|haha+|hehe+|xd+)\b", re.I)
HYPE_RE = re.compile(r"\b(let'?s\s*go+|letsgo+|pog+|hype|gooo+|we up|we are so back|w\b)\b", re.I)
APPROVAL_RE = re.compile(r"\b(good|nice|great|love|based|fire|clean|solid|awesome|valid|goated|cute)\b", re.I)
CONFUSION_RE = re.compile(r"\b(what|wtf|huh|why|how|wait|confused|unclear|hello)\b", re.I)
FRUSTRATION_RE = re.compile(r"\b(bad|trash|awful|hate|boring|annoying|stupid|worst|lame|mid|sucks?|terrible|painful|bullshit)\b", re.I)
MOCKERY_RE = re.compile(r"\b(clown|bozo|cooked|washed|cringe|fraud|deadass|skill issue|roast|joke|funny)\b", re.I)

TOPIC_PATTERNS = [
    ("fall guys", re.compile(r"\bfall guys\b", re.I), "Fall Guys"),
    ("ocean man", re.compile(r"\bocean man\b", re.I), "the Ocean Man bit"),
    ("new vegas", re.compile(r"\bnew vegas\b", re.I), "New Vegas"),
    ("roblox", re.compile(r"\broblox\b", re.I), "Roblox"),
    ("bart", re.compile(r"\bbart\b", re.I), "Bart"),
    ("neon", re.compile(r"\bneon\b", re.I), "Neon"),
    ("christmas tree", re.compile(r"\bchristmas tree\b", re.I), "the Christmas tree"),
    ("kitty", re.compile(r"\b(kitty|cat)\b", re.I), "the cat"),
    ("bazooka", re.compile(r"\bbazooka\b", re.I), "the bazooka"),
    ("bullshit", re.compile(r"\bbull ?shit\b", re.I), "the bullshit chant"),
    ("clown car", re.compile(r"\bclown car\b", re.I), "the clown-car bit"),
    ("throop", re.compile(r"\bthroop\b|\bthroughp\b", re.I), "the Throop town-name riff"),
    ("office", re.compile(r"\bthe office\b|\bscranton\b", re.I), "the Scranton and Office jokes"),
    ("geography", re.compile(r"\bgeoguessr\b|\bromania\b|\bbulgaria\b|\bturkey\b|\bturkiye\b|\bbalkans\b|\beast\b", re.I), "the geography guess"),
    ("driving", re.compile(r"\bdriv\w*\b|\bsteering wheel\b|\bgas pedal\b|\bturning\b|\bboost\b|\bcar\b", re.I), "the driving"),
    ("gibberish", re.compile(r"\bglue\b|\bcereal\b|\bmarf\w*\b|\bjaming\b|\bmonoxide check\b|\bfruit farming\b", re.I), "the weird gibberish"),
    ("vibes", re.compile(r"\bvibes?\b|\btoxic\b|\brancid\b", re.I), "how rancid the vibes are"),
    ("slur alert", re.compile(r"\balert\b.*\bslur\b|\bslur\b.*\balert\b|\bone slur\b", re.I), "the ALERT slur joke"),
    ("spoilers", re.compile(r"\bleak\w*\b|\bspoiler\w*\b", re.I), "the spoilers"),
    ("copypasta", re.compile(r"\bcopy ?pasta\b|\bspam\b", re.I), "the spam"),
    ("goodnight", re.compile(r"\bgoodnight\b|\bnightnight\b|\bbye\b|\blove you\b", re.I), "the stream ending"),
    ("list", re.compile(r"\blist\b|\branking\b|\bhonorable mentions?\b|\bwhat about\b", re.I), "the game list"),
    ("bah bah", re.compile(r"\bbah\b", re.I), "the bah-bah spam"),
    ("heavy", re.compile(r"\bheavy\b", re.I), "the heavy line"),
]

DEATH_WORDS = {"dead", "died", "dies", "kill", "killed", "shot", "rip", "o7", "execute", "murdered"}
ARGUMENT_WORDS = {"liar", "lying", "cap", "fraud", "accountability", "story", "both sides", "arguing"}
TEAMPLAY_WORDS = {"teammate", "comms", "baiting", "throw", "self promo", "modz"}
LOOK_WORDS = {"look", "looks", "cute", "ugly", "weird", "fish", "chopped", "derpy"}

SHORT_OK = {"gta", "nba", "osu", "cod", "cs2", "lcs", "wow", "tf2"}

GENERIC_PATTERNS = [
    "topic is unclear",
    "main topic is unclear",
    "shared topic stays unclear",
    "dominant reaction is consistent across the window",
    "window keeps reinforcing that overall reaction",
    "tone shows up repeatedly across the messages",
]

GENERIC_CALL_NAMES = {"someone", "somebody", "him", "her", "them", "bro", "guy", "girl", "chat"}


def parse_line(raw):
    match = LINE_RE.match(raw)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return "", raw.strip()


def is_system_line(user, text):
    user_l = user.lower()
    text_l = text.lower()
    if user_l in BOT_USERS:
        return True
    return any(pattern in text_l for pattern in SYSTEM_PATTERNS)


def normalize_text(text):
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text):
    return [token.lower() for token in WORD_RE.findall(normalize_text(text))]


def streamerize(text):
    replacements = [
        r"\bcaseoh\b",
        r"\bcasey\b",
        r"\bjerma985?\b",
        r"\bjerma\b",
        r"\bnorthernlion\b",
        r"\bnl\b",
        r"\bsqueex\b",
        r"\bjasontheween\b",
        r"\bjason\b",
        r"\bvinesauce\b",
        r"\bvinny\b",
    ]
    out = text
    for pattern in replacements:
        out = re.sub(pattern, "{streamer}", out, flags=re.I)
    out = re.sub(r"\{streamer\}(?:\s+\{streamer\})+", "{streamer}", out)
    return out


def word_count(text):
    return len(re.findall(r"\{streamer\}|[A-Za-z0-9']+", text))


def filtered_messages(obj):
    out = []
    for raw in obj.get("messages", []):
        user, text = parse_line(str(raw))
        if not text:
            continue
        if is_system_line(user, text):
            continue
        out.append(normalize_text(text))
    return out


def key_messages(obj):
    out = []
    data = obj.get("extractive_summary", {})
    if not isinstance(data, dict):
        return out
    for item in data.get("key_messages", []):
        _, text = parse_line(str(item))
        text = normalize_text(text)
        if text and not is_system_line("", text):
            out.append(text)
    return out


def tone_scores(messages):
    scores = Counter()
    for text in messages:
        lower = text.lower()
        scores["laughter"] += len(LAUGHTER_RE.findall(lower))
        scores["hype"] += len(HYPE_RE.findall(lower))
        scores["approval"] += len(APPROVAL_RE.findall(lower))
        scores["confusion"] += len(CONFUSION_RE.findall(lower)) + lower.count("?") * 0.35
        scores["frustration"] += len(FRUSTRATION_RE.findall(lower))
        scores["mockery"] += len(MOCKERY_RE.findall(lower))
    scores["laughter"] += 0.65 * scores["mockery"]
    return scores


def choose_tone(messages):
    scores = tone_scores(messages)
    combined = [
        ("laughter", scores["laughter"]),
        ("hype", scores["hype"]),
        ("approval", scores["approval"]),
        ("confusion", scores["confusion"]),
        ("frustration", scores["frustration"]),
    ]
    combined.sort(key=lambda item: item[1], reverse=True)
    top_tone, top_score = combined[0]
    second_tone, second_score = combined[1]

    if top_score < 2.2:
        return "chaotic", None
    if second_score >= 2.4 and second_score >= top_score * 0.78:
        return "mixed", (top_tone, second_tone)
    if second_score >= 2.8 and second_score >= top_score * 0.6:
        return top_tone, second_tone
    return top_tone, None


def phrase_interpretable(tokens, aliases):
    if not tokens:
        return False
    if all(token in STOPWORDS for token in tokens):
        return False
    if all(token in NOISE_TOKENS for token in tokens):
        return False
    if len(tokens) == 1:
        token = tokens[0]
        if token in aliases:
            return True
        if token in STOPWORDS or token in NOISE_TOKENS:
            return False
        if len(token) < 4 and token not in SHORT_OK:
            return False
    strong = [token for token in tokens if token not in STOPWORDS and token not in NOISE_TOKENS]
    if len(tokens) > 1:
        if len(set(tokens)) == 1:
            return False
        if len(strong) < 2:
            return False
        if tokens[0] in STOPWORDS or tokens[-1] in STOPWORDS:
            return False
    return bool(strong)


def collect_surface_phrases(messages, aliases):
    surface_counts = defaultdict(Counter)
    exact_counts = Counter()
    for text in messages:
        raw_tokens = [token.lower() for token in WORD_RE.findall(normalize_text(text))]
        if not raw_tokens:
            continue
        seen = set()
        for n in (3, 2, 1):
            for i in range(len(raw_tokens) - n + 1):
                ngram = raw_tokens[i : i + n]
                joined = " ".join(ngram)
                if joined in seen:
                    continue
                seen.add(joined)
                if not phrase_interpretable(ngram, aliases):
                    continue
                exact_counts[joined] += 1
                surface = " ".join(raw_tokens[i : i + n])
                surface_counts[joined][surface] += 1

    candidates = []
    for phrase, count in exact_counts.items():
        tokens = phrase.split()
        if len(tokens) >= 2 and count >= 2:
            score = count * (2.4 if len(tokens) == 2 else 3.0)
            candidates.append((score, phrase, count))

    candidates.sort(reverse=True)
    if not candidates:
        return None, 0

    _, phrase, count = candidates[0]
    surface = surface_counts[phrase].most_common(1)[0][0]
    return surface, count


def detect_concepts(messages):
    joined = "\n".join(messages)
    lower = joined.lower()
    counts = Counter()
    details = {}

    for key, pattern, label in TOPIC_PATTERNS:
        hits = len(pattern.findall(joined))
        if hits:
            counts[key] = hits
            details[key] = label

    tokens = [token.lower() for token in WORD_RE.findall(joined)]
    token_counts = Counter(tokens)

    death_hits = sum(token_counts[word] for word in DEATH_WORDS if word in token_counts)
    if death_hits:
        counts["death"] = death_hits
        details["death"] = "someone getting killed"

    if any(word in lower for word in TEAMPLAY_WORDS):
        counts["teamplay"] = sum(lower.count(word) for word in TEAMPLAY_WORDS)
        details["teamplay"] = "the bad teamplay"

    if any(word in lower for word in LOOK_WORDS):
        counts["looks"] = sum(lower.count(word) for word in LOOK_WORDS)
        details["looks"] = "how someone looks"

    if any(word in lower for word in ARGUMENT_WORDS):
        counts["argument"] = sum(lower.count(word) for word in ARGUMENT_WORDS)
        details["argument"] = "the argument"

    call_matches = re.findall(r"\bcall\s+([A-Za-z][A-Za-z0-9_']+)\b", joined, flags=re.I)
    if call_matches:
        filtered = [
            match.lower()
            for match in call_matches
            if match.lower() not in GENERIC_CALL_NAMES and len(match) >= 4
        ]
        if filtered:
            name_counts = Counter(filtered)
            call_name, call_count = name_counts.most_common(1)[0]
            counts["call"] = call_count + 1
            details["call"] = f"the call with {call_name.capitalize()}"

    person_candidates = []
    for token, count in token_counts.items():
        if token in STOPWORDS or token in NOISE_TOKENS:
            continue
        if token in {"alert", "bullshit", "kitty", "throop", "glue"}:
            continue
        if len(token) < 4:
            continue
        if count < 4:
            continue
        person_candidates.append((count, token))
    person_candidates.sort(reverse=True)
    if person_candidates:
        person = person_candidates[0][1]
        details["person"] = person.capitalize()
        counts["person"] = person_candidates[0][0]

    return counts, details


def choose_topic(file_name, messages, keys):
    aliases = STREAMER_ALIASES.get(file_name, set())
    counts, details = detect_concepts(messages + keys)

    if counts["kitty"] >= 2 and counts["christmas tree"] >= 2:
        return "the cat and Christmas tree bit", "bit"
    if counts["kitty"] >= 3:
        return "the cat", "entity"
    if counts["christmas tree"] >= 2:
        return "the Christmas tree", "entity"
    if counts["bullshit"] >= 5:
        return "the BULLSHIT chant", "bit"
    if counts["clown car"] >= 1:
        return "the clown-car visual", "bit"
    if counts["slur alert"] >= 1:
        return "the ALERT slur joke", "bit"
    if counts["vibes"] >= 2:
        return "how rancid the vibes are", "mood"
    if counts["throop"] >= 3:
        return "the Throop town-name riff", "bit"
    if counts["office"] >= 2 and counts["throop"] >= 1:
        return "the Scranton and Office jokes", "bit"
    if counts["geography"] >= 3:
        return "the geography guess", "event"
    if counts["gibberish"] >= 3:
        return "the weird gibberish bit", "bit"
    if counts["list"] >= 2:
        return "the game list", "event"
    if counts["goodnight"] >= 6:
        return "the stream ending", "event"
    if counts["spoilers"] >= 3:
        return "the spoilers", "event"
    if counts["driving"] >= 4:
        return "the driving", "event"
    if counts["death"] >= 4 and counts["bazooka"] >= 1:
        return "the bazooka kill", "event"
    if counts["death"] >= 5:
        return "someone getting killed", "event"
    if counts["call"] >= 2:
        return details["call"], "event"
    if counts["teamplay"] >= 2:
        return "the bad teamplay", "event"
    if counts["argument"] >= 2:
        return "the argument", "event"
    phrase, phrase_count = collect_surface_phrases(messages + keys, aliases)
    if phrase:
        phrase = " ".join(word.capitalize() if len(word) > 3 else word.upper() if word in SHORT_OK else word for word in phrase.split())
        if phrase_count >= 2 and phrase.lower() not in aliases:
            return phrase, "phrase"

    if any(alias in " ".join(messages).lower() for alias in aliases):
        return "{streamer}", "entity"
    return None, None


def pair_tone(a, b):
    pair = {a, b}
    if pair == {"laughter", "confusion"}:
        return "a mix of disbelief and laughter"
    if pair == {"laughter", "frustration"}:
        return "mockery with a frustrated edge"
    if pair == {"hype", "confusion"}:
        return "hype mixed with confusion"
    if pair == {"approval", "confusion"}:
        return "support mixed with confusion"
    if pair == {"approval", "frustration"}:
        return "a split between approval and criticism"
    return f"{a} mixed with {b}"


def build_specific_summary(topic, topic_kind, tone, secondary, messages, keys, seed):
    rng = random.Random(seed)
    joined = " ".join(messages).lower()

    if topic == "the cat and Christmas tree bit":
        options = [
            "Chat is bouncing between yelling about the cat and roasting {streamer} for still having the Christmas tree up.",
            "Chat splits its attention between the cat and the still-up Christmas tree, with a messy mix of jokes and confusion.",
        ]
        return rng.choice(options)

    if topic == "the bazooka kill":
        options = [
            "Chat is freaking out over the bazooka kill and reacting with disbelief, jokes, and a lot of loud laughter.",
            "Chat is losing it over the bazooka moment, treating the instant kill like a ridiculous punchline that keeps escalating.",
            "Chat treats the bazooka kill like a completely absurd payoff and reacts with panic, laughs, and shouting.",
        ]
        return rng.choice(options)

    if topic == "the BULLSHIT chant":
        options = [
            "Chat is spamming BULLSHIT and treating the whole scene like obvious nonsense, with a lot of laughing pile-on.",
            "Chat turns the repeated BULLSHIT chant into the main joke and reacts like the moment is blatantly ridiculous.",
            "Chat keeps yelling BULLSHIT at the scene and mostly treats it like ridiculous nonsense worth piling onto.",
        ]
        return rng.choice(options)

    if topic == "someone getting killed":
        options = [
            "Chat is freaking out over someone getting instantly killed and reacting with disbelief and laughter.",
            "Chat treats the sudden death like a ridiculous shock moment, with panic, disbelief, and laughing spam all mixed together.",
            "Chat reacts to someone getting wiped out with loud shock, panic, and a lot of laughing disbelief.",
        ]
        return rng.choice(options)

    if topic == "the clown-car visual":
        options = [
            "Chat is losing it over the clown-car visual and piling on with jokes about how absurd it looks.",
            "Chat turns the tiny clown-car image into the joke of the moment and keeps piling on with laughter.",
        ]
        return rng.choice(options)

    if topic == "the ALERT slur joke":
        options = [
            "Chat turns the possible slur into an ALERT running joke and keeps spamming it with mocking laughter.",
            "Chat is riffing on the ALERT slur bit and treating it like a dumb running joke that everybody piles onto.",
        ]
        return rng.choice(options)

    if topic == "the weird gibberish bit":
        options = [
            "Chat is reacting to the weird gibberish and wordplay with a mix of confusion, spam, and amused clowning.",
            "Chat keeps riffing on the weird gibberish bit, sounding confused but still very ready to joke about it.",
            "Chat is latching onto the weird gibberish and mostly reacting with confused laughter and spammy riffs.",
        ]
        return rng.choice(options)

    if topic == "how rancid the vibes are":
        options = [
            "Chat is roasting how rancid and toxic the vibes feel, with mockery and criticism mixed together.",
            "Chat keeps calling out the rancid vibes and reacts like the whole interaction is painfully toxic.",
        ]
        return rng.choice(options)

    if topic == "the Throop town-name riff":
        options = [
            "Chat is riffing on the town name Throop and treating it like a goofy little joke.",
            "Chat keeps playing with the name Throop and turns it into a playful running bit across the window.",
        ]
        return rng.choice(options)

    if topic == "the Scranton and Office jokes":
        options = [
            "Chat is making Scranton and Office jokes, with a mostly playful tone built around the place-name references.",
            "Chat latches onto the Scranton and Office references and treats them like an easy joke thread.",
        ]
        return rng.choice(options)

    if topic == "the geography guess":
        options = [
            "Chat is reacting to the geography guess with confusion, corrections, and a lot of clowning.",
            "Chat keeps arguing over the geography guess and mixes map-nerd corrections with mocking jokes.",
        ]
        return rng.choice(options)

    if topic == "the game list":
        options = [
            "Chat is arguing over the game list, with complaints, side picks, and a lot of second-guessing.",
            "Chat treats the game list like a debate topic and keeps throwing in objections and rival picks.",
            "Chat turns the game list into an argument, with backseat rankings and complaints taking over the window.",
        ]
        return rng.choice(options)

    if topic == "the stream ending":
        options = [
            "Chat is saying goodnight and reacting warmly to the stream ending, with a mostly affectionate tone.",
            "Chat feels soft and supportive around the stream ending, with goodnights and love-you messages taking over.",
        ]
        return rng.choice(options)

    if topic == "the spoilers":
        options = [
            "Chat is reacting to spoilers with a mix of annoyance, calls to stop, and amused side comments.",
            "Chat sounds split on the spoilers, with some people laughing and others clearly fed up with the leaks.",
        ]
        return rng.choice(options)

    if topic == "the driving":
        options = [
            "Chat is roasting the driving and reacting like the whole sequence is painful to watch.",
            "Chat treats the driving like a disaster and piles on with criticism and jokes from every angle.",
            "Chat is clowning on the driving and reacting like the gameplay is a complete mess.",
        ]
        return rng.choice(options)

    if topic == "the bad teamplay":
        options = [
            "Chat is calling out the bad teamplay and piling on with criticism about the comms and baiting.",
            "Chat reacts to the bad teamplay with annoyance, blame, and some mocking side jokes.",
        ]
        return rng.choice(options)

    if topic == "the argument":
        options = [
            "Chat feels messy and argumentative, with people piling into the back-and-forth instead of sharing one clear reaction.",
            "Chat is split by the argument and keeps bouncing between bait, jokes, and criticism.",
            "Chat gets dragged into the argument and mostly reacts with bait, jokes, and side-taking.",
        ]
        return rng.choice(options)

    if topic_kind == "entity":
        if tone == "laughter":
            return f"Chat is clowning on {topic} and turning it into the punchline of the moment."
        if tone == "approval":
            return f"Chat is reacting positively to {topic}, with a mostly supportive and playful tone."
        if tone == "hype":
            return f"Chat is visibly hyped about {topic} and keeps pushing the energy higher across the whole window."
        if tone == "frustration":
            return f"Chat sounds annoyed about {topic} and keeps responding with criticism."
        if tone == "confusion":
            return f"Chat seems confused about {topic} and keeps trying to figure out what exactly is going on."

    if topic_kind == "phrase":
        if tone == "laughter":
            return f"Chat keeps repeating {topic} like a running joke and mostly reacts with laughter instead of taking it seriously."
        if tone == "hype":
            return f"Chat is spamming {topic} with a lot of excitement and celebratory energy across the window."
        if tone == "confusion":
            return f"Chat keeps circling around {topic}, but the overall reaction still feels confused rather than settled."
        if tone == "frustration":
            return f"Chat latches onto {topic} while sounding mostly annoyed and critical about the whole thing."

    if tone == "mixed" and isinstance(secondary, tuple):
        a, b = secondary
        if topic:
            return f"Chat is split around {topic}, with {pair_tone(a, b)} rather than one clean reaction."
        return f"Chat feels split, with {pair_tone(a, b)} and no clearly shared topic."

    if topic:
        if tone == "laughter":
            return f"Chat is joking about {topic} and mostly treating it like a ridiculous bit."
        if tone == "hype":
            return f"Chat seems excited about {topic} and keeps reacting with hype instead of hesitation."
        if tone == "approval":
            return f"Chat sounds broadly positive about {topic}, with a mostly supportive tone across the messages."
        if tone == "confusion":
            return f"Chat seems confused about {topic} and is trying to make sense of it in real time as the messages pile up."
        if tone == "frustration":
            return f"Chat is mostly negative about {topic} and keeps piling on with criticism instead of support."
        if tone == "chaotic":
            return f"Chat is all over the place around {topic}, mixing jokes, side comments, and scattered reactions without one clear mood."

    if tone == "laughter":
        options = [
            "Chat is mostly cracking up at the moment and piling on with jokes.",
            "Chat treats whatever just happened like a ridiculous joke that landed.",
        ]
        return rng.choice(options)
    if tone == "hype":
        options = [
            "Chat is clearly hyped, even if the exact topic is hard to pin down from this window alone.",
            "Chat feels loud and excited, with hype carrying the window more than one specific topic.",
        ]
        return rng.choice(options)
    if tone == "approval":
        options = [
            "Chat reads as mostly supportive, even if the exact topic stays a little fuzzy.",
            "Chat sounds broadly positive here, with a generally approving tone.",
        ]
        return rng.choice(options)
    if tone == "confusion":
        options = [
            "Chat seems confused and scattered, with people trying to figure out what just happened.",
            "Chat is mostly reacting with confusion and uncertainty rather than one clear take.",
        ]
        return rng.choice(options)
    if tone == "frustration":
        options = [
            "Chat sounds annoyed and critical, though the exact topic is still a little unclear.",
            "Chat is mostly negative here, with frustration outweighing everything else.",
        ]
        return rng.choice(options)
    return "Chat is all over the place, mixing jokes, side chatter, and scattered reactions."


def finalize_summary(summary, tone, secondary):
    summary = streamerize(summary)
    summary = re.sub(r"\s+", " ", summary).strip()
    seed = int(hashlib.sha1(summary.encode("utf-8")).hexdigest(), 16)
    rng = random.Random(seed)

    if word_count(summary) < 20 and secondary and not summary.endswith("."):
        summary += "."
    if word_count(summary) < 20 and secondary and isinstance(secondary, str):
        additions = {
            "laughter": [
                " A few messages are still more confused than amused.",
                " Some of the window is laughing through the confusion.",
            ],
            "confusion": [
                " A lot of the confusion comes with laughing disbelief.",
                " People sound baffled more than certain.",
            ],
            "frustration": [
                " The annoyance is hard to miss.",
                " The complaints cut through pretty clearly.",
            ],
            "approval": [
                " The mood still leans more positive than skeptical.",
                " There is still a supportive streak underneath it.",
            ],
            "hype": [
                " The energy is clearly tilted upward.",
                " Excitement still carries a lot of the window.",
            ],
        }
        summary += rng.choice(additions.get(secondary, [""]))

    if word_count(summary) < 20:
        extras = {
            "laughter": [
                " Most of chat is just piling onto the joke.",
                " The mood is mostly amused and a little mean.",
            ],
            "confusion": [
                " Nobody sounds fully sure what they just watched.",
                " The mood is more baffled than confident.",
            ],
            "frustration": [
                " The complaints come through pretty clearly.",
                " The negativity is doing most of the work here.",
            ],
            "approval": [
                " The mood leans more supportive than doubtful.",
                " The overall reaction stays mostly warm.",
            ],
            "hype": [
                " The energy is clearly pointed upward.",
                " The window stays loud and excited.",
            ],
            "chaotic": [
                " The reaction never really settles into one lane.",
                " Nobody sticks to one clean reaction for long.",
            ],
        }
        summary += rng.choice(extras.get(tone, [""]))

    if word_count(summary) > 55 and ". " in summary:
        summary = summary.split(". ")[0].strip()
        if not summary.endswith("."):
            summary += "."

    if word_count(summary) > 55:
        parts = re.findall(r"\{streamer\}|[A-Za-z0-9']+|[^\w\s{}]+", summary)
        out = []
        count = 0
        for part in parts:
            if re.match(r"\{streamer\}|[A-Za-z0-9']+", part):
                if count >= 55:
                    break
                count += 1
            out.append(part)
        summary = "".join(out).strip()
        if summary[-1] not in ".!?":
            summary += "."

    return summary


def rewrite_entry(file_name, obj):
    messages = filtered_messages(obj)
    keys = key_messages(obj)
    combined = messages + keys
    seed = int(hashlib.sha1(f"{file_name}:{obj.get('window_id','x')}".encode("utf-8")).hexdigest(), 16)
    tone, secondary = choose_tone(combined or messages or keys)
    topic, topic_kind = choose_topic(file_name, messages, keys)

    if topic in {"the Throop town-name riff", "the Scranton and Office jokes"} and tone in {"frustration", "confusion"}:
        secondary = tone
        tone = "laughter"

    if tone == "mixed":
        summary = build_specific_summary(topic, topic_kind, tone, secondary, messages, keys, seed)
        return finalize_summary(summary, "chaotic", None)

    summary = build_specific_summary(topic, topic_kind, tone, secondary, messages, keys, seed)

    if secondary and topic and topic_kind not in {"bit", "event", "mood"}:
        if tone == "laughter" and secondary == "confusion":
            summary += " There is also a noticeable thread of confusion."
        elif tone == "confusion" and secondary == "laughter":
            summary += " A lot of the confusion comes with laughing disbelief."
        elif tone == "frustration" and secondary == "laughter":
            summary += " Some of the criticism is clearly joking rather than purely angry."
        elif tone == "approval" and secondary == "confusion":
            summary += " Some viewers still sound unsure about what they are reacting to."
        elif tone == "hype" and secondary == "confusion":
            summary += " A few messages are excited while still sounding a little lost."

    return finalize_summary(summary, tone, secondary)


def rewrite_file(path):
    file_name = os.path.basename(path)
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    for obj in data:
        obj.setdefault("abstractive_summary", {})
        obj["abstractive_summary"]["summary"] = rewrite_entry(file_name, obj)

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    return len(data)


def main():
    total = 0
    for path in sorted(glob.glob(os.path.join(BASE_DIR, "*.json"))):
        count = rewrite_file(path)
        total += count
        print(f"{os.path.basename(path)}: rewritten {count}")
    print(f"TOTAL_REWRITTEN={total}")


if __name__ == "__main__":
    main()
