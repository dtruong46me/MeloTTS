# Comprehensive Guide: Inference, Testing & Training (MeloTTS)

This document provides detailed instructions on how to run Inference (speech synthesis), Testing (evaluation), and Training for MeloTTS. It supports running directly on the host machine (Local) or via Docker.

---

## 1. Inference (Speech Synthesis)

Inference is the process of using the trained model to synthesize speech from text.

### Native Execution (Local)
**Via Web UI:**
```bash
python -m melo.ui.app
```
*The Gradio interface will launch at `http://127.0.0.1:8888`.*

**Via Command Line (CLI):**
```bash
melo-infer --text "Hello world" --output_path output.wav --language EN --speaker EN-Default
```

**Via Python API:**
```python
from melo.api import TTS

model = TTS(language='EN', device='cuda:0')
speaker_id = model.hps.data.spk2id['EN-Default']
model.tts_to_file("Hello world!", speaker_id, output_path="output.wav")
```

### Execution via Docker
By default, `docker-compose` has configured the `melo-ui` service with GPU support.
```bash
# Build and run the WebUI
cd docker
docker-compose up -d melo-ui
```

---

## 2. Data Preparation (Preprocessing)

Before Training or Testing, you need to prepare a `metadata.list` file formatted with the following columns per line:
`audio_path|speaker_name|language|text`

Example (`data/example/metadata.list`):
```text
data/wavs/001.wav|Speaker_1|EN|Hello world.
```

**Preprocessing:** Run the following script to normalize the text, extract BERT features, and split the dataset into Train/Val sets:
```bash
python melo/scripts/preprocess_text.py \
    --metadata data/example/metadata.list \
    --config_path configs/base.json
```
This generates:
- A cleaned metadata file `data/example/metadata.list.cleaned`
- Data splits `train.list` and `val.list`
- An updated `config.json` containing the derived `spk2id` and `symbols`.

---

## 3. Training

You can run the training process in two ways: **Locally** or **Via Docker**.

### Method 1: Native Training (Local)
Requires a fully configured environment, CUDA, and dependencies (via `pip install -e .`).

1. Edit `configs/config.json` to customize hyperparameters like batch size or learning rate.
2. Run the distributed training script using `torchrun`:
```bash
# Syntax: bash melo/scripts/train.sh <config_path> <num_gpus>
bash melo/scripts/train.sh configs/config.json 1
```

Logs and checkpoints (`G_*.pth`, `D_*.pth`) will automatically be saved in the `logs/<model_name>/` directory.

### Method 2: Training via Docker (Recommended)
Using Docker isolates the PyTorch environment and C++ libraries (e.g., libsndfile). A `melo-train` service is pre-configured in `docker-compose.yml` with direct access to host GPUs (via the Nvidia Container Toolkit).

**Step 1:** Launch the `melo-train` container in the background.
```bash
cd docker
docker-compose up -d melo-train
```
*(Note: External directories like `data/`, `logs/`, and `configs/` are automatically mounted inside the container).*

**Step 2:** Access the container and begin data processing & training:
```bash
docker exec -it melotts_train /bin/bash

# Inside the container:
python melo/scripts/preprocess_text.py --metadata data/example/metadata.list --config_path configs/base.json

# Start training (e.g., using 1 GPU):
bash melo/scripts/train.sh configs/config.json 1
```

You can safely monitor the training progress. All checkpoints are saved in `logs/` and will immediately sync back to the host machine.

---

## 4. Testing (Evaluation)

Model testing is performed automatically during the Training process. At every `eval_interval` step defined in your `config.json` (typically 1000 steps), the model evaluates itself against the `val.list` dataset.

To view the loss graphs or listen to the generated audio samples during evaluation:
```bash
tensorboard --logdir logs/
```
Visit `http://localhost:6006` in your browser for visual graphs. (You can expose this port in Docker if you are training via Docker).
