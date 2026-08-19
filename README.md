# YOLO Training Pipeline & Automation Framework

Modular training orchestration framework for Ultralytics YOLO models (YOLOv8, YOLO11, etc.), with clean separation of concerns, dynamic artifact management, and structured logging.

---

## Key Features

- **Modular architecture** — decoupled dataset validation, model init, config, and logging
- **Dynamic artifact routing** — outputs organized under `models/trained/{dataset}-{date}-{model}/`
- **Pretrained weights management** — stored in `models/base/`
- **Custom logging** — Loguru-based, intercepts Ultralytics logs, writes to `logs/`
- **TOML-based configuration** with CLI overrides
- **Docker-ready** for reproducible runs

---

## Project Structure

```text
.
├── configs/
│   ├── config.local.toml    # Training and environment configuration
│   └── data.yaml             # YOLO dataset structure definition
├── core/
│   ├── base_trainer.py       # Abstract trainer interface
│   ├── dataset.py            # Dataset validation & metadata handler
│   └── trainer.py            # YOLO training engine & logger interceptor
├── datasets/                 # Dataset directory (symlinked or local)
├── logs/                      # Application and training log outputs
├── models/
│   ├── base/                 # Downloaded pretrained weights (e.g., yolo11n.pt)
│   └── trained/               # Timestamped training runs & metrics
├── utils/
│   ├── config.py              # Configuration loader
│   ├── logger.py              # Custom Loguru logger & notification hooks
│   └── sender.py              # Notification dispatcher (Email/Slack hooks)
├── main.py                    # Pipeline entrypoint
├── requirements.txt           # Core dependencies
└── README.md
```

---

## Environment

| Component    | Version                              |
|--------------|----------------------------------------|
| OS           | Ubuntu 24.04 LTS (x86_64)             |
| GPU          | NVIDIA RTX 4050 Laptop (6GB)          |
| CUDA         | 12.1                                    |
| Python       | 3.12.x                                  |
| PyTorch      | 2.3.1+cu121                             |

---

## Installation

```bash
git clone https://github.com/your-username/yolo-trainer.git
cd yolo-trainer

conda create -n yolo_trainer python=3.12 -y
conda activate yolo_trainer

pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

---

## Usage

```bash
python main.py
python main.py --config configs/config.prod.toml   # custom config
```

---

## Outputs

- Pretrained weights → `models/base/<model_name>.pt`
- Checkpoints, metrics, `best.pt` → `models/trained/<dataset_name>-<YYYYMMDD>-<model_name>/`
- Logs → `logs/training.log`

---

## Docker

```bash
docker compose up --build
```