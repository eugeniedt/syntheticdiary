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

## 2b) (Recommended) Enable NVIDIA GPU (CUDA) on Windows

If you have an NVIDIA GPU, you typically do **not** need to install the full CUDA Toolkit to train with PyTorch.
You do need a working NVIDIA driver (so `nvidia-smi` works), and you must install a CUDA-enabled PyTorch wheel.

### Step 1: Verify the driver sees your GPU

```powershell

```

If that command isn’t found or shows no GPU, install/update the NVIDIA driver first (Game Ready or Studio).

### Step 2: Install CUDA-enabled PyTorch

From your activated venv, install CUDA PyTorch from the official PyTorch wheel index (example uses CUDA 12.4):

```powershell
pip uninstall -y torch
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
```

If you don’t want the extra packages, you can omit `torchvision`/`torchaudio`, but they’re harmless.

### Step 3: Sanity check (Python)

```powershell
python -c "import torch; print('torch', torch.__version__); print('cuda_runtime', torch.version.cuda); print('cuda_is_available', torch.cuda.is_available()); print('gpu', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'))"
```

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

## 4b) Fine-tune (GPU-friendly defaults)

If CUDA is enabled (see section 2b), try:

```powershell
python .\training\train_clm.py `
  --model distilgpt2 `
  --train .\training\data\train.txt `
  --valid .\training\data\valid.txt `
  --out .\training\out\distilgpt2-daily-gpu `
  --device cuda `
  --block-size 256 `
  --epochs 2 `
  --batch-size 1 `
  --grad-accum 16 `
  --auto-fp16
```

If you hit OOM (out of memory):

- lower `--block-size` to 128
- lower `--grad-accum` (less total work per update) or keep it but reduce `--epochs`
- keep `--batch-size 1` (usually best on laptops)

## 5) Use the fine-tuned model in `gen_entry.py`

`gen_entry.py` uses `MODEL_NAME` as the Hugging Face model identifier or a local folder path.

Example:

```powershell
$env:MODEL_NAME="eugenieee/synthdiary"
$env:HF_TOKEN="hf_..."  # required if the Hub repo is private
python .\gen_entry.py --force
```

