# Trained Model Checkpoints

All LoRA adapters are stored on Google Drive:

[https://drive.google.com/drive/folders/1nZKTGHxoEC6jBfQKXZQksT1GgvcQFgAE](https://drive.google.com/drive/folders/1nZKTGHxoEC6jBfQKXZQksT1GgvcQFgAE?usp=sharing)

## Contents

| Folder | Model | Training Data | Epochs | Test EM | Test CER |
|--------|-------|---------------|--------|---------|----------|
| `smolvlm_lora_1dataset` | SmolVLM-256M-Instruct | LaTeX_OCR (1,164 ex) | 3 | 70.0% | 8.1% |
| `smolvlm_lora_2datasets` | SmolVLM-256M-Instruct | LaTeX_OCR + MathWriting (6,164 ex) | 3 | 80.0% | 4.1% |
| `qwen3vl_lora_1dataset` | Qwen3-VL-2B-Instruct | LaTeX_OCR (1,164 ex) | 3 | 94.3% | 0.4% |
| `qwen3vl_lora_2datasets` | Qwen3-VL-2B-Instruct | LaTeX_OCR + MathWriting (1,864 ex) | 2 | 85.7% | 3.3% |

## Usage

Download the trained LoRA adapter folder `smolvlm_lora_2datasets` and place it next to `app.py`:

```
your-project/
├── app.py
└── models/
    └── smolvlm_lora_2datasets/
        ├── adapter_config.json
        └── adapter_model.safetensors
```

Each folder contains two files required by `PeftModel.from_pretrained()`:
- `adapter_config.json` — LoRA configuration
- `adapter_model.safetensors` — adapter weights
