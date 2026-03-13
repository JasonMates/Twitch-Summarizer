# Abstractive Summary Ruleset v2
## Chat-Only Training Targets

## Purpose
Write short, grounded summaries of a chat window.

The model sees only chat messages. The summary should help a late viewer understand what chat is reacting to right now and how the overall tone feels.

## Core Goal
Each summary should answer two questions:
1. What is chat mainly talking about?
2. How is chat reacting to it?

A strong summary is brief, readable, and clearly supported by the visible messages.

## Input Assumptions
- Input is a raw window of chat messages only.
- Do not assume access to stream audio, video context, prior windows, or external metadata.
- Ignore bot/system lines when choosing topic, tone, and evidence.

Ignore examples such as:
- subscription alerts
- gifted sub notifications
- streak messages
- prime/tier notices
- automated bot posts

## Output Requirements
- Write 1-2 sentences.
- Target 20-45 words.
- Hard cap: 55 words unless absolutely necessary.

Include:
- the main topic or event chat is reacting to
- the dominant tone or sentiment
- a secondary tone only if clearly present

Exclude:
- usernames
- timestamps
- message counts
- sentiment counts
- references to information outside the current chat window

## Grounding Rules
- Use only information directly supported by the current chat window.
- Do not invent what happened on stream.
- Do not explain why something happened unless chat clearly states it.
- Do not infer specific off-screen events from vague reactions alone.

If signal is weak, noisy, or split, use cautious wording:
- seems
- appears
- mostly
- mixed
- unclear

If topic is unclear, summarize the reaction honestly without faking context.

## Topic Selection
Choose one primary topic for the window. Prioritize:
1. A clearly repeated game title, subject, or named topic.
2. A repeated phrase or recurring idea.
3. A dominant interpretable subject implied by many related messages.

If no clear topic exists, summarize the chat reaction itself.

Prefer:
- interpretable phrases
- recurring ideas
- concrete subjects

Avoid:
- isolated one-off phrases
- repeated emote spam as the topic
- vague placeholders such as `one`, `good`, `that`, `clear`, `aga`
- slash-joined fragments or token garbage

Treat a game title as the main topic only when it is clearly central in the window.

## Sentiment and Tone
Identify the dominant tone of the window.

Common tones:
- laughter or mockery
- hype or excitement
- approval or support
- confusion
- frustration or criticism
- mixed reaction
- chaotic/off-topic joking

Use the dominant tone in the summary. Include a secondary tone only when it is genuinely visible across the window.

Do not force a single clean sentiment when chat is clearly split or noisy.
Never include raw counts in the final summary.

## Streamer and Entity Handling
- Replace streamer-name mentions with `{streamer}` when the summary refers to the streamer.
- Mention a game, person, or entity only if it is clearly present in the chat window and relevant to the main reaction.
- Do not elevate a minor mention into the main topic.

## Quotes
- Do not include quotes by default.
- Include a short quote only when it materially improves grounding and clarity.
- In most cases, a clean paraphrase is better.

If a quote is used:
- keep it short
- remove @mentions
- exclude bot/system lines
- prefer readable, contentful chat lines

Never build summaries around a quote-template scaffold.

## Writing Style
Write naturally. Summaries should sound like concise human descriptions, not annotation reports.

Preferred style:
- `Chat is joking about the bad driving and mostly reacting with mockery.`
- `Chat seems excited about Fall Guys and is spamming hype messages.`
- `Chat is split between laughing at the moment and being confused about what happened.`

Avoid repetitive scaffolds such as:
- `Messages like X and Y suggested...`
- `Chat response to X showed...`
- `In chat, X drew...`

Do not rotate fixed openers mechanically. Vary wording naturally.

## Canonical Rewrite Examples (Original -> Better)
Use these examples as calibration references when rewriting noisy targets. The "better" version shows the preferred abstraction style.

1. Loud/Fish Mockery
- Original target: `In chat, loud looks drew laughter. Messages like "he's loud and looks like a fish LMAO" and "hes loud and looks like a fish LOL" suggested chat found the moment funny.`
- Better target: `Chat is clowning on how loud and weird he looks, and the tone is mostly mocking laughter.`
- Why better: Preserves reaction and topic without forcing evidence-reporting scaffolds.

2. "Bah Bah" Running Bit
- Original target: `Chat response to bah bah showed laughter. Messages like "glorpsi BAH BAH glorpsi" and "BAH pearto BAH pearto" suggested chat found the moment funny.`
- Better target: `Chat is spamming "bah bah" as a joke and treating the moment like a dumb running bit.`
- Why better: Names the behavior directly instead of pseudo-citation language.

3. Neon Roast
- Original target: `Messages about neon in chat showed laughter. Messages like "you know it's bad when neon isn't scared of you" and "NEON BEWEN ON SINCE 14" suggested chat found the moment funny.`
- Better target: `Chat is roasting Neon and turning him into the punchline of the moment.`
- Why better: Cleaner, more natural, and less brittle.

4. Unclear/Random Reaction
- Original target: `Chat response to aga showed confusion. Messages like "pb that bad huh?" and "is it more shameful to watch Hassan instead of squeex?" suggested chat was reacting to unclear context.`
- Better target: `Chat seems confused and is throwing out random comments without a clear shared reaction.`
- Why better: Admits unclear context instead of forcing fake specificity.

5. Bad Driving Pile-On
- Original target: `In chat, ass drew confusion. Messages like "U BOOST WHILE TURNING SHARP ARE U DEADASS LOL" and "deadass lost all ur gaming skills sob" suggested chat was reacting to unclear context.`
- Better target: `Chat is roasting the terrible driving and reacting like the gameplay is painful to watch.`
- Why better: Reflects the visible behavior more accurately than generic confusion.

6. Fall Guys Hype (Label Fix)
- Original target: `Chat's reaction to fall guys was confusion. Messages like "FALL GUYS LETS GO" and "FALL GUYS!!! caseohLife caseohLife" suggested chat was reacting to unclear context.`
- Better target: `Chat is excited that Fall Guys is coming up and reacts with hype rather than confusion.`
- Why better: Corrects a likely wrong sentiment label.

7. Spam/Copypasta Annoyance
- Original target: `Chat response to good showed approval. Messages like "just mucked a teen burger and fries, it was good" and "Oh god, please don't start this copy pasta shit again. Dude grow" suggested chat was broadly supportive.`
- Better target: `Chat is messy and split, but the dominant reaction is annoyance at spam and copypasta.`
- Why better: Better matches the actual dominant tone.

8. Bart Meme Play
- Original target: `In chat, bart drew approval. Messages like "Bart's Army official version" and "bart dissolved in acid" suggested chat was broadly supportive.`
- Better target: `Chat is riffing on Bart jokes and references, with a chaotic but mostly playful tone.`
- Why better: Captures meme-play instead of flattening to approval.

9. Ocean Man Vibe
- Original target: `Chat response to ocean man showed approval. Messages like "caseohWiijams ocean man caseohWiijams caseohWiijams HSWP" and "someone really said copyright like he doesn't know ocean man is copyrighted" suggested chat was broadly supportive.`
- Better target: `Chat is vibing with the "Ocean Man" moment and spamming emotes in a supportive, playful way.`
- Why better: Keeps positivity while sounding like a real summary.

10. Chaotic/Split Window
- Original target: `The chat around clear turned mixed. Messages like "im sending screenshots of each frame to my friend max cause he" and "he really is trade" suggested chat had mixed reactions.`
- Better target: `Chat is chaotic and split between joking, arguing, and trying to make sense of what they are seeing.`
- Why better: Replaces vague topic token with interpretable behavior.

11. Diffuse "Aga" Window
- Original target: `In chat, aga drew mixed. Messages like "Will you speak out against hacks?" and "aga i wont cry" suggested chat had mixed reactions.`
- Better target: `Chat is all over the place, mixing jokes, bait, and random side comments rather than focusing on one reaction.`
- Why better: More faithful for diffuse, weak-topic windows.

12. Leak/Spoiler Reactions
- Original target: `Chat's reaction to leak was mixed. Messages like "ray leak u god" and "NO KNE LEAKED. THE DETAILS" suggested chat had mixed reactions.`
- Better target: `Chat is reacting to spoilers with a mix of amusement, frustration, and calls for people to stop leaking details.`
- Why better: Teaches a richer mixed-reaction summary.

13. Squeex Ironic Praise
- Original target: `Chat's reaction to squeex was approval. Messages like "Is Squeex in his dark woke era?" and "@Squeex the files released were to cover up the Clavicular frame mog" suggested chat was broadly supportive.`
- Better target: `Chat is joking about Squeex with ironic praise and absurd meme accusations, but the tone stays playful.`
- Why better: Distinguishes playful irony from literal approval.

14. "Heavy" Running Joke
- Original target: `Chat's reaction to heavy was laughter. Messages like ""make sure it looks heavy"" and "MAKE IT REAL HEAVY" suggested chat found the moment funny.`
- Better target: `Chat is losing it over the repeated "heavy" line and turns it into a running joke.`
- Why better: Short, direct, and anchored to the central bit.

15. Vague "One" Token
- Original target: `Chat activity around one was laughter. Messages like "thats a good one" and "i liked that one" suggested chat found the moment funny.`
- Better target: `Chat is reacting with loud laughter and shock, treating the moment like a ridiculous joke that landed.`
- Why better: Avoids vague placeholder topic words and prioritizes reaction clarity.

## Mixed or Weak-Signal Windows
If multiple reactions are present:
- summarize the dominant one first
- mention secondary tone only if it materially affects interpretation

If topic is unclear but tone is obvious:
- summarize the tone directly
- state uncertainty honestly

Examples:
- `Chat is mostly laughing and piling on jokes, though the exact topic is unclear.`
- `Chat seems divided, with some viewers hyped and others annoyed.`

Do not fake certainty when the signal is weak.

## Prohibited Behavior
Do not:
- hallucinate stream events
- use prior windows as hidden context
- include usernames or personal callouts
- fabricate game context
- over-explain
- force every summary into one rigid structure
- treat emote repetition alone as a fully sufficient topic

## Quality Check
Before finalizing, confirm:
- the summary is supported by the current chat window
- the main topic is real and interpretable
- the tone matches the dominant reaction
- mixed sentiment is included only when visible
- streamer names are replaced with `{streamer}` when relevant
- no outside context was added
- output is concise and natural

## Dataset Health Guidance
During batch review, monitor:
- wording diversity across summaries
- reaction balance across labels
- unclear-topic frequency
- hallucination rate
- bot/system leakage
- overuse of identical sentence structures

No single summary pattern should dominate the dataset.
