# Comprehensive Guide: Installation, Inference, Testing & Training

This document covers everything you need to get MeloTTS running — from installation to training a custom voice model. Supports both native (local) execution and Docker.

---

## 1. Installation

### Requirements

- Python ≥ 3.9, ≤ 3.12 (PyTorch does **not** support Python 3.13 yet)
- CUDA ≥ 11.8 (for GPU support)
- [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (for Docker GPU training)

### Install from source

```bash
git clone https://github.com/myshell-ai/MeloTTS.git
cd MeloTTS
pip install -e .
```

> **Using conda?** Make sure you activate the correct environment first:
> ```bash
> conda activate melotts   # or whichever env you set up for this project
> pip install -e .
> ```
> Running `pip install -e .` from the wrong environment (e.g., the `base` env with Python 3.13) will fail because PyTorch has no wheel for Python 3.13.

> **Already installed, then refactored files?** Re-run `pip install -e .` (in the correct env) to update the CLI entry points. Without this step, commands like `melo-ui` will continue to reference the old file paths.

### Post-install: download Japanese tokenizer

```bash
python -m unidic download
```

### Verify CLI commands are available

After installation, the following commands should be available system-wide:

| Command | Description |
|---|---|
| `melo` / `melotts` | Synthesise a text string directly to a WAV file |
| `melo-infer` | Load a local checkpoint and synthesise per-speaker WAV files |
| `melo-ui` | Launch the Gradio WebUI |

```bash
melo --help
melo-infer --help
melo-ui --help
```

---

## 2. Inference (Speech Synthesis)

### Via Web UI

```bash
melo-ui
# or equivalently:
python -m melo.ui.app
```

The Gradio interface launches at `http://127.0.0.1:7860` by default.

Additional options:

```bash
melo-ui --host 0.0.0.0 --port 8888   # bind to a specific host/port
melo-ui --share                        # create a public Gradio tunnel URL
```

### Via CLI (`melo` command)

Synthesise a single text string to a WAV file:

```bash
melo "Hello world" output.wav --language EN --speaker EN-Default
```

Supported languages: `EN`, `ES`, `FR`, `ZH`, `JP`, `KR`

Supported English speakers: `EN-Default`, `EN-US`, `EN-BR`, `EN_INDIA`, `EN-AU`

```bash
melo "Le ciel est bleu." output.wav --language FR
melo "日本語のテスト" output.wav --language JP --speed 0.9
```

To synthesise from a text file, use the `--file` / `-f` flag:

```bash
melo -f mytext.txt output.wav --language EN
```

### Via `melo-infer` (checkpoint-based)

Load a custom checkpoint and generate one WAV per speaker defined in the checkpoint config:

```bash
melo-infer \
  --ckpt_path logs/my_model/G_10000.pth \
  --text "Hello from my custom model." \
  --language EN \
  --output_dir outputs/my_model
```

### Via Python API

```python
from melo.api import TTS

model = TTS(language="EN", device="auto")
speaker_id = model.hps.data.spk2id["EN-Default"]
model.tts_to_file("Hello world!", speaker_id, output_path="output.wav", speed=1.0)
```

Custom checkpoint:

```python
model = TTS(
    language="EN",
    config_path="logs/my_model/config.json",
    ckpt_path="logs/my_model/G_10000.pth",
    device="cuda:0",
)
```

### Inference via Docker

```bash
cd docker
docker-compose up -d melo-ui
```

The WebUI is accessible at `http://localhost:8888`.

---

## 3. Data Preparation (Preprocessing)

Before training a custom model, prepare a pipe-delimited `metadata.list` file:

```
audio_path|speaker_name|language|text
```

Example (`data/example/metadata.list`):

```text
data/wavs/001.wav|Speaker_1|EN|Hello world.
data/wavs/002.wav|Speaker_1|EN|How are you doing today?
data/wavs/003.wav|Speaker_2|ZH|你好世界。
```

Run preprocessing to clean text, extract BERT features, and create train/val splits:

```bash
python melo/scripts/preprocess_text.py \
    --metadata data/example/metadata.list \
    --config_path configs/base.json \
    --val-per-spk 4 \
    --max-val-total 8
```

This generates:
- `data/example/metadata.list.cleaned` — cleaned phoneme/tone file
- `data/example/train.list` — training split
- `data/example/val.list` — validation split
- `data/example/config.json` — updated config with `spk2id`, `num_languages`, `symbols`

---

## 4. Training

### Method 1: Native Training (Local)

**Requirements:** CUDA, PyTorch, and all dependencies installed (`pip install -e .`).

1. Run preprocessing (see Section 3) to generate the `config.json` and split files.
2. Start distributed training with `torchrun` via the provided shell script:

```bash
# Syntax: bash melo/scripts/train.sh <config_path> <num_gpus>
bash melo/scripts/train.sh data/example/config.json 1    # single GPU
bash melo/scripts/train.sh data/example/config.json 4    # 4 GPUs
```

Checkpoints (`G_*.pth`, `D_*.pth`) and logs are saved to `logs/<model_name>/`.

The script includes an **auto-restart loop** to recover from occasional `gloo` crashes on certain GPU configurations.

### Method 2: Training via Docker (Recommended)

Docker isolates the PyTorch environment and native libraries (e.g., `libsndfile`). The `melo-train` service in `docker/docker-compose.yml` mounts your local `data/`, `logs/`, and `configs/` directories into the container and enables full GPU pass-through via the Nvidia Container Toolkit.

**Step 1:** Build the image and start the training container.

```bash
cd docker
docker-compose up -d melo-train
```

**Step 2:** Open a shell inside the container and run the pipeline.

```bash
docker exec -it melotts_train /bin/bash
```

Inside the container:

```bash
# 1. Preprocess data
python melo/scripts/preprocess_text.py \
    --metadata data/example/metadata.list \
    --config_path configs/base.json

# 2. Start training (1 GPU)
bash melo/scripts/train.sh data/example/config.json 1
```

All checkpoints are written to `/app/logs/` inside the container, which is mounted to `../logs/` on the host — so your progress is immediately accessible outside Docker.

---

## 5. Evaluation & Monitoring

Evaluation runs automatically during training at the interval specified by `eval_interval` in `config.json` (default: every 1000 steps). The model synthesises a set of utterances from `val.list` and logs the generated audio and loss metrics.

### TensorBoard

```bash
tensorboard --logdir logs/
```

Visit `http://localhost:6006` to view loss curves, spectrograms, and synthesised audio samples.

To expose TensorBoard while training inside Docker, add the following to the `melo-train` service in `docker/docker-compose.yml`:

```yaml
ports:
  - "6006:6006"
```

Then launch TensorBoard inside the container:

```bash
docker exec -it melotts_train tensorboard --logdir /app/logs --host 0.0.0.0
```
