# Technical Report: Comparative Evaluation of Vision-Language Models for Handwritten LaTeX OCR

## 1. Task and Models

The task is **image-to-LaTeX conversion**: given a handwritten mathematical formula image, produce the corresponding LaTeX source string. Two vision-language models (VLMs) are compared under identical evaluation conditions:

| Property | SmolVLM-256M-Instruct | Qwen3-VL-2B-Instruct |
|----------|-----------------------|---------------------|
| HuggingFace ID | `HuggingFaceTB/SmolVLM-256M-Instruct` | `Qwen/Qwen3-VL-2B-Instruct` |
| Parameters | 256M | 2B |
| LoRA trainable | 2,884,608 (1.11%) | 8,716,288 (0.41%) |
| Baseline inference | Zero-shot & one-shot | Zero-shot & one-shot |

## 2. Datasets

### 2.1. Primary: `linxy/LaTeX_OCR` (human_handwrite)

- **Splits:** 1,200 train → 1,164 after deduplication, 68 validation, 70 test (~90/5/5)
- **Format:** RGBA PNG images, space-tokenized LaTeX (e.g., `\frac { a } { b }`)
- **Image size:** ~515 × variable height (40–389 px), resized to longest_edge=512
- **Quality:** 36 pixel-level duplicate groups removed from train (different annotations → label noise). 23 texts shared across all splits retained (visually distinct handwriting). No missing values.

### 2.2. Supplementary: `deepcopy/MathWriting-human`

- **Full size:** 229,864 train, 15,674 val, 7,644 test. Only train split is used.
- **Subsample:** 5,000 examples (SmolVLM combined), 700 examples (Qwen3-VL combined, reduced due to T4 time constraints)
- **Format:** RGB JPG images, compact LaTeX (75%, e.g., `\frac{a}{b}`) and spaced LaTeX (25%)
- **Image size:** ~312 × 148 px (mean), resized to longest_edge=512
- **Quality:** No pixel-level duplicates. Subsample verified representative of full dataset.

### 2.3. Combined Training Sets

| Model | LaTeX_OCR | MathWriting | Total |
|-------|-----------|-------------|-------|
| SmolVLM | 1,164 | 5,000 | 6,164 |
| Qwen3-VL | 1,164 | 700 | 1,864 |

## 3. Evaluation Metrics

All metrics are computed on the LaTeX_OCR **test split (70 examples)**. Metrics are implemented in a shared `metrics.py` module.

| Metric | Level | Description |
|--------|-------|-------------|
| Exact Match (EM) | String | Fraction of predictions identical to reference (whitespace-trimmed) |
| EM (normalised) | String | Same after removing all spaces (handles format differences) |
| CER | Character | Levenshtein edit distance / reference length, averaged over examples |
| BLEU | Corpus | SacreBLEU corpus-level score |

## 4. Experimental Setup

### 4.1. Hardware and Environment

- **GPU:** Google Colab T4 (16 GB VRAM)
- SmolVLM: `transformers==4.46.0`; Qwen3-VL: `transformers>=4.57.0` (from git)

### 4.2. LoRA Configuration (identical for all fine-tuning runs)

| Parameter | Value |
|-----------|-------|
| `r` | 8 |
| `lora_alpha` | 8 |
| `lora_dropout` | 0.05 |
| `target_modules` | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` (7 modules) |
| `init_lora_weights` | `"gaussian"` |

### 4.3. Training Hyperparameters

| Parameter | SmolVLM (1 dataset) | SmolVLM (combined) | Qwen3-VL (1 dataset) | Qwen3-VL (combined) |
|-----------|--------------------|--------------------|-----------------------|----------------------|
| Training examples | 1,164 | 6,164 | 1,164 | 1,864 |
| Epochs | 3 | 3 | 3 | 2 |
| Batch size | 2 | 2 | 2 | 2 |
| Gradient accumulation | 8 | 8 | 8 | 8 |
| Effective batch size | 16 | 16 | 16 | 16 |
| Learning rate | 1e-4 | 1e-4 | 1e-4 | 1e-4 |
| Weight decay | 0.01 | 0.01 | 0.01 | 0.01 |
| Warmup | ratio 0.1 | ratio 0.1 | ratio 0.1 | ratio 0.1 |
| Optimiser | AdamW | AdamW | AdamW | AdamW |
| Precision | bf16 | bf16 | bf16 | bf16 |
| Gradient checkpointing | Yes | Yes | Yes | Yes |
| Seed | 42 | 42 | 42 | 42 |

### 4.4. Checkpoint Selection

All epoch checkpoints were evaluated on the 68-example validation set. The checkpoint with the highest normalised EM was selected (CER as tiebreaker). Three checkpoints were evaluated for SmolVLM (3 epochs) and Qwen3-VL single-dataset (3 epochs); two checkpoints for Qwen3-VL combined (2 epochs). Best checkpoint was the last epoch in all cases.

## 5. Evaluation Results

### 5.1. Complete Results (LaTeX_OCR test set, 70 examples)

| Setup | EM | EM (norm.) | CER | BLEU | Time (s/ex) |
|-------|-----|-----------|------|------|-------------|
| SmolVLM — Zero-shot | 0.0% | 20.0% | 60.5% | 49.7 | 1.1 |
| SmolVLM — One-shot | 0.0% | 15.7% | 45.6% | 42.4 | 1.1 |
| SmolVLM — SFT (LaTeX_OCR) | 70.0% | 70.0% | 8.1% | 91.4 | 2.6 |
| **SmolVLM — SFT (combined)** | **80.0%** | **80.0%** | **4.1%** | **95.0** | **2.4** |
| Qwen3-VL — Zero-shot | 0.0% | 0.0% | 24.1% | 73.6 | 4.9 |
| Qwen3-VL — One-shot | 4.3% | 10.0% | 21.0% | 81.3 | 7.3 |
| Qwen3-VL — SFT (LaTeX_OCR) | 94.3% | 94.3% | 0.4% | 98.9 | 5.7 |
| Qwen3-VL — SFT (combined) | 85.7% | 85.7% | 3.3% | 95.1 | 5.8 |

### 5.2. Key Observations

**Baseline capability.** Neither model produces exact matches in the zero-shot setting. Qwen3-VL-2B shows substantially stronger pre-trained understanding (CER 24.1% vs 60.5%, BLEU 73.6 vs 49.7), consistent with its 8× larger parameter count. Notably, Qwen3-VL wraps zero-shot outputs in `$$...$$` delimiters, causing 0% raw EM despite often correct mathematical content.

**Effect of fine-tuning.** LoRA SFT transforms both models from unusable (0% EM) to practical (70–94% EM). All SFT configurations show EM = EM (normalised), meaning every correct prediction is character-perfect — no partial matches.

**Combined dataset effect.** The two models respond differently to combined training. SmolVLM gains +10 pp EM (70→80%) and −4.0 pp CER (8.1→4.1%) when trained on the combined 6,164-example set. Qwen3-VL shows the opposite pattern: −8.6 pp EM (94.3→85.7%) on combined data. This asymmetry is confounded by differences in auxiliary dataset size (5,000 vs 700 MathWriting examples) and epoch count (3 vs 2). Additionally, Qwen3-VL's 94.3% on the single-dataset setting may partly reflect exploitation of the LaTeX_OCR distribution (same source for train and test, shared background style and notation conventions) rather than genuine generalisation. The combined-dataset setting trades some same-distribution accuracy for broader robustness.

**Inference speed.** SmolVLM is 2.4× faster than Qwen3-VL in the SFT setting (2.4 vs 5.7–5.8 s/ex on a T4 GPU), making it more suitable for batch processing and real-time applications.

### 5.3. Model Selection for Application Deployment

SmolVLM-256M-Instruct fine-tuned on the combined dataset is selected as the deployment model. It achieves 80% EM / 4.1% CER / 95.0 BLEU with 2.4× faster inference and a smaller memory footprint. The combined training exposes the model to formulas with varied backgrounds and notation styles (MathWriting), which is likely to improve robustness on diverse real-world inputs compared to single-dataset training.