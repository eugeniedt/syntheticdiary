# Training (fine-tune GPT‑2 on your tagged markdown)

This folder contains two scripts:

- `prepare_dataset.py`: scans your markdown folder, keeps only files containing `#training_e` or `#training_d`, removes those tags from the text, and writes `train.txt` / `valid.txt`.
- `train_clm.py`: fine-tunes a GPT‑2-family causal LM (default: `distilgpt2`) on CPU and saves the model.

## 1) Create a virtualenv (PowerShell)

From the repo root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## 2) Install training dependencies

```powershell
pip install -r requirements-train.txt
```

Notes:
- `torch` is installed as a CPU build from PyPI by default.
- Training on CPU can be slow; `distilgpt2` is recommended.

## 3) Prepare dataset (filter + remove tags)

```powershell
python .\training\prepare_dataset.py `
  --input "C:\Users\AK128613\Documents\Me\Personal\Daily" `
  --tags "#training_e" "#training_d" `
  --out .\training\data
```

Outputs:
- `training\data\train.txt`
- `training\data\valid.txt`
- `training\data\manifest.jsonl` (file list + split + char counts)
- `training\data\summary.json` (counts and settings)

## 4) Fine-tune (CPU-friendly defaults)

```powershell
python .\training\train_clm.py `
  --model distilgpt2 `
  --train .\training\data\train.txt `
  --valid .\training\data\valid.txt `
  --out .\training\out\distilgpt2-daily `
  --block-size 256 `
  --epochs 2 `
  --batch-size 1 `
  --grad-accum 16
```

If it’s too slow, reduce work:
- set `--epochs 1`
- set `--block-size 128`
- set `--grad-accum 8`

## 5) Use the fine-tuned model in `gen_entry.py`

`gen_entry.py` uses `MODEL_NAME` as the Hugging Face model identifier or a local folder path.

Example:

```powershell
$env:MODEL_NAME="C:\Users\AK128613\syntheticdiary\training\out\distilgpt2-daily"
python .\gen_entry.py --force
```

