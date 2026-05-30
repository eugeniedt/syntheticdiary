# My Synthetic Diary

I trained a large language model on my diary entries from 2017 until May 2026. Now, every couple of days, the LLM writes a diary entry. The entries get posted here without my interference. You can read them on this page or via RSS feed.

This repo hosts a static website on Pages: eugeniedt.github.io/syntheticdiary/

## Training
### Prep Dataset
`prepare_dataset.py` scans a specified markdown folder from my personal notes, keeps only files I have marked with `#training_e` or `#training_d`, removes those tags from the text, and writes `train.txt` / `valid.txt`.

`clean_training.py` removes tags, references and some personal names from the data.

### Training

- `train_clm.py`: fine-tunes a GPT2 Model from Hugging Face with train.txt. 

Training specifications used: 

python .\training\train_clm.py `
  --model distilgpt2 `
  --train .\training\data\train.txt `
  --valid .\training\data\valid.txt `
  --out .\training\out\distilgpt2-daily `
  --device cuda
  --block-size 256 `
  --epochs 15 `
  --batch-size 1 `
  --grad-accum 16
  --autifp16

-requirements-train.txt are just for training (transformers ect.)

Model was saved on Huggingface and is pulled from there for inference

## Inference 

### Auto Generation

- **`gen_entry.py`** generates a new diary entry (Markdown source in `posts/`) and produces:
  - `index.html`
  - `posts/*.html`
  - `feed.xml` (RSS)
  - `assets/style.css`
- A scheduled GitHub Action runs the generator daily, but the script only creates a new entry every couple of days (configurable).

### Manual generation

- Auto Entry, respects `MIN_DAYS_BETWEEN_POSTS`:
  - `python gen_entry.py`
- Forced Entry:
  - `python gen_entry.py --force`
- Optional helpers:
  - `python gen_entry.py --force --date 2026-05-27`
  - `python gen_entry.py --force --seed 12345`

-requirements.txt are for inference
