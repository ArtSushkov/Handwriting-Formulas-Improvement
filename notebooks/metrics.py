"""metrics.py — Model-agnostic evaluation metrics for LaTeX OCR project.

Provides normalize_latex() and compute_metrics() used by both model notebooks
(2_smolvlm.ipynb, 3_qwen3vl.ipynb) and the comparison notebook (4_comparison.ipynb).
"""

import editdistance
import sacrebleu


def normalize_latex(s: str) -> str:
    """Remove all spaces for fairer comparison (dataset uses space-tokenized LaTeX)."""
    return s.replace(" ", "").strip()


def compute_metrics(predictions, references):
    """Compute EM (raw + normalized), CER, and BLEU.

    Args:
        predictions: list of predicted LaTeX strings.
        references: list of ground-truth LaTeX strings.

    Returns:
        dict with keys: exact_match, exact_match_normalized, cer, bleu.
    """
    n = len(references)

    # Exact Match (raw)
    em_raw = sum(1 for p, r in zip(predictions, references)
                 if p.strip() == r.strip()) / n

    # Exact Match (normalized — spaces removed)
    em_norm = sum(1 for p, r in zip(predictions, references)
                  if normalize_latex(p) == normalize_latex(r)) / n

    # Character Error Rate (on normalized strings)
    total_dist = 0
    total_ref_len = 0
    for p, r in zip(predictions, references):
        p_n, r_n = normalize_latex(p), normalize_latex(r)
        total_dist += editdistance.eval(p_n, r_n)
        total_ref_len += len(r_n)
    cer = total_dist / total_ref_len if total_ref_len > 0 else 0.0

    # BLEU (corpus-level, for comparison with CER)
    bleu = sacrebleu.corpus_bleu(predictions, [references]).score

    return {
        "exact_match": em_raw,
        "exact_match_normalized": em_norm,
        "cer": cer,
        "bleu": bleu,
    }
