# Twitch Chat Summarization

Twitch Chat Summarization is a standalone pipeline for turning noisy Twitch chat windows into short plain-language summaries. It focuses on describing what chat is reacting to in 1-2 sentences, while keeping the output grounded in the visible messages instead of outside stream context.

The repository includes extracted chat-moment JSON files, cleaned chat-summary training pairs, a reusable chat preprocessing module, seq2seq training and inference scripts for BART and FLAN-style models, a hybrid FLAN summarizer, and a comparison notebook that can also evaluate optional zero-shot and few-shot LLM baselines through OpenRouter.


## Demo and Links

Companion sentiment repo: https://github.com/JasonMates/Twitch-Sentiment

Hugging Face BART: https://huggingface.co/JDMates/TwitchBART

Hugging Face FLAN-T5 Small: https://huggingface.co/JDMates/TwitchFlanSmall-v2

Notebook demo: `project_summaries.ipynb`


## Project Layout

```text
.
|-- datasets/
|   |-- summary_pairs.jsonl
|   |-- chat_summary_pairs.jsonl
|   |-- bart_vs_flan.json
|   `-- flan_summary.txt
|-- moment_data/
|   |-- Caseoh.json
|   |-- Jerma1.json
|   |-- Northernlion1.json
|   |-- Northernlion2.json
|   |-- Squeex.json
|   |-- Vinesauce.json
|   `-- jasontheween.json
|-- src/
|   |-- few_shot_sum.py
|   |-- hybrid_flan.py
|   |-- prep_data.py
|   |-- rewrite_abstractive_summaries.py
|   |-- split_dataset.py
|   |-- summarize_chat_txt.py
|   `-- train_bart.py
|-- chat_preprocess.py
|-- summarize_chat.py
|-- project_summaries.ipynb
|-- ABSTRACTIVE_SUMMARY_RULESET.md
|-- test.txt
`-- requirements.txt
```


## What Each Part Does

**Dataset**
`moment_data/` stores extracted chat windows from several streamers used during summarization development. `datasets/summary_pairs.jsonl` stores the final `{chat, summary}` pairs used by the notebook and training scripts, while the other files in `datasets/` are intermediate or comparison artifacts kept for inspection.

**Preprocessing**
`chat_preprocess.py` is the shared cleanup layer used by training and inference. It can strip timestamps and usernames, drop bot and system lines, collapse duplicate messages, collapse token floods, and report what was removed.

**Model Inference**
`summarize_chat.py` is the main CLI summarizer. It reads a chat-log `.txt` file, distills the input, chunks long windows, runs hierarchical seq2seq summarization, and writes a short output summary.

`src/hybrid_flan.py` is an alternate summarizer that first uses MiniLM embeddings to select representative chat lines, then rewrites them with FLAN-T5. It is useful when a window is large and very repetitive.

**Training and Data Prep**
`src/train_bart.py` is the seq2seq trainer used for chat-to-summary pairs. Despite the filename, it can fine-tune BART or FLAN/T5-style checkpoints by changing `--model_name_or_path` and, for T5-style models, `--source_prefix`.

`src/prep_data.py`, `src/rewrite_abstractive_summaries.py`, and `src/split_dataset.py` are the data-building utilities for generating, rewriting, and splitting training pairs.

`src/few_shot_sum.py` is a lightweight retrieval baseline that builds summaries from nearby labeled examples without calling an external API.

**Notebook**
`project_summaries.ipynb` compares fine-tuned BART, fine-tuned FLAN-T5 Small, and optional OpenRouter zero-shot and few-shot prompting on a contiguous slice from `test.txt`.


## Setup

1. Create a virtual environment
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install dependencies
   ```powershell
   pip install -r requirements.txt
   ```

3. Optional for the notebook: set an OpenRouter API key

   If you want to run the zero-shot and few-shot OpenRouter baselines in `project_summaries.ipynb`, fill in `OR_KEY` in the notebook config cell. If you leave it blank, the notebook still runs the local BART and FLAN comparisons.


## Running the Summarizer

**BART summarizer**
```powershell
python summarize_chat.py --input_txt test.txt --model_dir JDMates/TwitchBART
```

This writes `test.summary.txt` by default.

**Hybrid FLAN summarizer**
```powershell
python src/hybrid_flan.py --input_txt test.txt --flan_model_dir JDMates/TwitchFlanSmall-v2
```

This writes `test.hybrid.summary.txt` by default.

If you already have local checkpoints, pass the local directory instead of a Hugging Face model id. `summarize_chat.py` also supports `--local_files_only` for offline loading.


## Training
The training pairs live under `datasets/`, so pass that path explicitly.

**BART**
```powershell
python src/train_bart.py --data_path datasets/summary_pairs.jsonl --input_key chat --target_key summary --model_name_or_path facebook/bart-base --output_dir bart_chat_model_retrained
```

**FLAN-T5 Small**
```powershell
python src/train_bart.py --data_path datasets/summary_pairs.jsonl --input_key chat --target_key summary --model_name_or_path google/flan-t5-small --source_prefix "summarize: " --output_dir flan_t5_small_model_epoch1
```

Thus saves the final model and tokenizer under `<output_dir>/final`, which can then be passed back into the inference scripts as a local model directory.


## Notebook
`project_summaries.ipynb`.

One small path note: the notebook config currently expects a summary pair file path in `PAIR_PATH`. In this repo, that file lives at `datasets/summary_pairs.jsonl`, so update the config cell before running if needed.


## A Few Notes

- This repo is the summarization half of the larger Twitch analysis project. The live sentiment classifier and dashboard live in the companion repository linked above.
- The current codebase is file-based and standalone. The project report treats live summarization as future integration work rather than something already wired into the real-time dashboard.
- The summary style is intentionally conservative and chat-only. `ABSTRACTIVE_SUMMARY_RULESET.md` captures the output rules the data pipeline was aiming for.
- The first run may take longer if model weights need to be downloaded from Hugging Face.
