"""
Handwriting Formula → LaTeX Converter
Streamlit application for image-to-LaTeX recognition
using fine-tuned SmolVLM-256M-Instruct with LoRA.
"""

import streamlit as st
import torch
import sys
import types as _types

if hasattr(torch, "distributed") and not hasattr(torch.distributed, "tensor"):
    _dtensor = _types.ModuleType("torch.distributed.tensor")
    _dtensor.DTensor = type("DTensor", (), {})
    torch.distributed.tensor = _dtensor
    sys.modules["torch.distributed.tensor"] = _dtensor

# Prevent transformers from importing the real (broken) FSDP module chain
# during model.generate() — it checks is_fsdp_managed_module which triggers
# import torch.distributed.fsdp → torch.distributed.tensor → crash.
if hasattr(torch, "distributed") and "torch.distributed.fsdp" not in sys.modules:
    _fsdp = _types.ModuleType("torch.distributed.fsdp")
    _fsdp.is_fsdp_managed_module = lambda _: False
    _fsdp.FullyShardedDataParallel = type("FullyShardedDataParallel", (), {})
    sys.modules["torch.distributed.fsdp"] = _fsdp
    torch.distributed.fsdp = _fsdp

from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import io
import time
import os

# ═══════════════════════════════════════════════════════════════
# Configuration — change these to switch models later
# ═══════════════════════════════════════════════════════════════
MODEL_NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"
ADAPTER_PATH = os.path.join(os.path.dirname(__file__), "models", "smolvlm_lora_2datasets")
MAX_NEW_TOKENS = 256

# Auto-detect device: GPU with compute capability >= 8 → bfloat16, else → CPU float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if (DEVICE == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else (torch.float16 if DEVICE == "cuda" else torch.float32)

# ═══════════════════════════════════════════════════════════════
# Model loading (cached — runs once per session)
# ═══════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    """Load base model + LoRA adapter + processor."""
    try:
        processor = AutoProcessor.from_pretrained(
            MODEL_NAME,
            size={"longest_edge": 512}
        )

        base_model = AutoModelForVision2Seq.from_pretrained(
            MODEL_NAME,
            torch_dtype=DTYPE,
            attn_implementation="eager",
        ).to(DEVICE)

        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        model.eval()

        trainable, total = model.get_nb_trainable_parameters()
        st.session_state["model_loaded"] = True
        st.session_state["trainable_params"] = trainable
        st.session_state["total_params"] = total
        st.session_state["device"] = DEVICE

        return model, processor
    except Exception as e:
        st.error(
            f"Failed to load model. Ensure the adapter directory exists:\n"
            f"`{ADAPTER_PATH}`\n\nError: {e}"
        )
        st.stop()


# ═══════════════════════════════════════════════════════════════
# Inference
# ═══════════════════════════════════════════════════════════════
def predict(image: Image.Image, model, processor) -> str:
    """Run inference: image → LaTeX string."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        "Convert this handwritten mathematical formula "
                        "to LaTeX format. Output only the LaTeX code, nothing else."
                    ),
                },
            ],
        }
    ]

    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(text=prompt_text, images=[image], return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.2,
        )

    input_len = inputs["input_ids"].shape[1]
    new_tokens = generated_ids[:, input_len:]
    latex = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    return latex


# ═══════════════════════════════════════════════════════════════
# LaTeX rendering to image
# ═══════════════════════════════════════════════════════════════
def render_latex_to_image(latex_str: str, dpi=200) -> Image.Image:
    """Render a LaTeX string into a PNG image using matplotlib.

    Tries full LaTeX (requires MiKTeX/TeX Live installed).
    Falls back to matplotlib's built-in mathtext if unavailable.
    """
    latex_display = f"${latex_str}$"

    # Try with system LaTeX first (higher quality, full LaTeX support)
    try:
        plt.rcParams.update({
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath,amssymb}",
        })
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(
            0.5, 0.5, latex_display,
            transform=ax.transAxes, fontsize=36,
            ha="center", va="center",
        )
        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.3)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        return img
    except Exception:
        plt.close("all")

    # Fallback: matplotlib mathtext (no external LaTeX needed)
    plt.rcParams.update({"text.usetex": False})
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.text(
        0.5, 0.5, latex_display,
        transform=ax.transAxes, fontsize=36,
        ha="center", va="center",
    )
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).copy()
    return img


# ═══════════════════════════════════════════════════════════════
# Streamlit UI
# ═══════════════════════════════════════════════════════════════
st.set_page_config(page_title="Handwriting Formula → LaTeX", layout="centered")

st.title("Handwriting Formula → LaTeX Converter")
st.markdown(
    "Upload an image of a handwritten mathematical formula. "
    "The model will convert it to LaTeX and render the result."
)

# --- Load model ---
with st.spinner("Loading model (this may take a minute on first run)..."):
    model, processor = load_model()

if st.session_state.get("model_loaded"):
    device_label = st.session_state.get("device", "cpu").upper()
    st.caption(
        f"Model: {MODEL_NAME} + LoRA adapter | "
        f"Trainable: {st.session_state['trainable_params']:,} / {st.session_state['total_params']:,} params | "
        f"Device: {device_label}"
    )

# --- File upload ---
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["png", "jpg", "jpeg", "bmp", "webp"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        st.image(image, width='stretch')

    # --- Run inference ---
    with st.spinner("Recognizing formula..."):
        start_time = time.time()
        latex = predict(image, model, processor)
        elapsed = time.time() - start_time

    # --- Show LaTeX text ---
    st.success(f"Recognized LaTeX ({elapsed:.1f}s):")
    st.code(latex, language="latex")

    # --- Render and show LaTeX image ---
    with col2:
        st.subheader("Rendered LaTeX")
        try:
            rendered = render_latex_to_image(latex)
            st.image(rendered, width='stretch')
        except Exception as e:
            st.warning(f"LaTeX rendering failed: {e}")
            st.caption("The recognized text is shown above. Rendering requires valid LaTeX syntax.")
