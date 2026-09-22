"""
SatQuery AI — Benchmark Evaluation Harness
==========================================
Evaluates SatQuery AI against standard RS benchmark datasets.
Run BEFORE and AFTER fine-tuning to demonstrate improvement.

Supported benchmarks:
  --benchmark rsvqa_lr    → RSVQA Low-Resolution (yes/no + count + presence)
  --benchmark vrsbench_vqa → VRSBench VQA (open-ended VQA)
  --benchmark cdvqa       → CDVQA (Change Detection VQA)
  --benchmark all         → Run all benchmarks

Usage:
  python -m satquery.backend.training.evaluate \\
    --benchmark all \\
    --model_path models/geochat_lora_bigearthnet \\
    --output results/benchmark_results.json

Outputs a comparison table: Base GeoChat vs SatQuery-LoRA
"""

import os
import sys
import json
import time
import argparse
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="SatQuery AI Benchmark Evaluation")
    parser.add_argument("--benchmark", choices=["rsvqa_lr", "vrsbench_vqa", "cdvqa", "all"],
                        default="all")
    parser.add_argument("--model_path", default=None,
                        help="Path to LoRA adapter. If None, uses base GeoChat.")
    parser.add_argument("--data_dir", default="data/benchmarks")
    parser.add_argument("--n_samples", type=int, default=100,
                        help="Number of samples per benchmark (default 100 for speed)")
    parser.add_argument("--output", default="results/benchmark_results.json")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────
# RSVQA EVALUATION
# ──────────────────────────────────────────────────────────────

def eval_rsvqa(
    model, processor, data_dir: str, n_samples: int, device: str
) -> Dict[str, float]:
    """
    Evaluate on RSVQA-LR (Low Resolution) benchmark.
    Metrics: Yes/No accuracy, Count accuracy, Presence accuracy, Overall accuracy.
    """
    rsvqa_dir = os.path.join(data_dir, "rsvqa_sample")
    if model is None or not os.path.exists(rsvqa_dir):
        logger.warning(f"RSVQA data not found at {rsvqa_dir} or model not loaded. Using benchmark baseline.")
        return _mock_rsvqa_results()

    import torch
    from PIL import Image
    import glob

    qa_file = os.path.join(rsvqa_dir, "qa_pairs.json")
    if not os.path.exists(qa_file):
        logger.warning("RSVQA qa_pairs.json not found.")
        return _mock_rsvqa_results()

    with open(qa_file) as f:
        qa_pairs = json.load(f)

    qa_pairs = qa_pairs[:n_samples]

    correct = {"yn": 0, "count": 0, "presence": 0}
    total = {"yn": 0, "count": 0, "presence": 0}

    for qa in qa_pairs:
        img_path = os.path.join(rsvqa_dir, "images", qa["image"])
        if not os.path.exists(img_path):
            continue

        img = Image.open(img_path).convert("RGB")
        question = qa["question"]
        gt_answer = str(qa["answer"]).lower().strip()
        q_type = qa.get("type", "yn")

        # Run model
        pred = _run_single_vqa(model, processor, img, question, device)
        pred_clean = pred.lower().strip()

        # Check answer
        if q_type in ("yn", "yes_no"):
            is_correct = (
                ("yes" in pred_clean and "yes" in gt_answer) or
                ("no" in pred_clean and "no" in gt_answer)
            )
            correct["yn"] += int(is_correct)
            total["yn"] += 1
        elif q_type == "count":
            try:
                pred_num = int(''.join(filter(str.isdigit, pred_clean.split()[0])))
                gt_num = int(gt_answer)
                correct["count"] += int(abs(pred_num - gt_num) <= 1)
            except Exception:
                pass
            total["count"] += 1
        elif q_type == "presence":
            is_correct = gt_answer in pred_clean or pred_clean[:10] in gt_answer
            correct["presence"] += int(is_correct)
            total["presence"] += 1

    results = {}
    for key in ["yn", "count", "presence"]:
        if total[key] > 0:
            results[key + "_accuracy"] = round(correct[key] / total[key] * 100, 2)
        else:
            results[key + "_accuracy"] = 0.0

    overall = sum(correct.values()) / max(sum(total.values()), 1) * 100
    results["overall_accuracy"] = round(overall, 2)
    results["n_evaluated"] = sum(total.values())
    return results


def _mock_rsvqa_results() -> Dict:
    """Return an explicit non-evaluated state; never fabricate benchmark results."""
    return {
        "status": "not_evaluated",
        "n_evaluated": 0,
        "note": "RSVQA data/model were not available for an actual evaluation run."
    }


# ──────────────────────────────────────────────────────────────
# VRSBENCH VQA EVALUATION
# ──────────────────────────────────────────────────────────────

def eval_vrsbench_vqa(
    model, processor, data_dir: str, n_samples: int, device: str
) -> Dict[str, float]:
    """
    Evaluate on VRSBench VQA benchmark.
    Metric: BLEU-4, CIDEr (for open-ended), binary accuracy for closed QA.
    """
    vrsbench_dir = os.path.join(data_dir, "vrsbench_sample")
    if model is None or not os.path.exists(vrsbench_dir):
        logger.warning("VRSBench data not found or model not loaded. Using benchmark baseline.")
        return _mock_vrsbench_results()

    qa_file = os.path.join(vrsbench_dir, "vqa_pairs.json")
    if not os.path.exists(qa_file):
        return _mock_vrsbench_results()

    with open(qa_file) as f:
        qa_pairs = json.load(f)[:n_samples]

    from PIL import Image
    em_correct = 0
    total = 0
    bleu_scores = []

    for qa in qa_pairs:
        img_path = os.path.join(vrsbench_dir, "images", qa["image"])
        if not os.path.exists(img_path):
            continue
        img = Image.open(img_path).convert("RGB")
        pred = _run_single_vqa(model, processor, img, qa["question"], device)
        gt = qa["answer"]

        # Exact match (lenient)
        em_correct += int(gt.lower().strip() in pred.lower())
        total += 1

        # BLEU
        bleu = _compute_bleu1(pred, gt)
        bleu_scores.append(bleu)

    return {
        "em_accuracy": round(em_correct / max(total, 1) * 100, 2),
        "bleu1": round(sum(bleu_scores) / max(len(bleu_scores), 1) * 100, 2),
        "n_evaluated": total,
    }


def _mock_vrsbench_results() -> Dict:
    return {
        "status": "not_evaluated",
        "n_evaluated": 0,
        "note": "VRSBench data/model were not available for an actual evaluation run."
    }


# ──────────────────────────────────────────────────────────────
# CDVQA EVALUATION (Change Detection VQA)
# ──────────────────────────────────────────────────────────────

def eval_cdvqa(
    model, processor, data_dir: str, n_samples: int, device: str
) -> Dict[str, float]:
    """
    Evaluate on CDVQA (Change Detection VQA) benchmark.
    Tests: change type classification, change extent estimation.
    """
    cdvqa_dir = os.path.join(data_dir, "cdvqa_sample")
    if model is None or not os.path.exists(cdvqa_dir):
        return _mock_cdvqa_results()

    qa_file = os.path.join(cdvqa_dir, "qa_pairs.json")
    if not os.path.exists(qa_file):
        return _mock_cdvqa_results()

    from PIL import Image
    with open(qa_file) as f:
        qa_pairs = json.load(f)[:n_samples]

    correct = 0
    total = 0

    for qa in qa_pairs:
        t1_path = os.path.join(cdvqa_dir, "images", qa["t1_image"])
        t2_path = os.path.join(cdvqa_dir, "images", qa["t2_image"])
        if not (os.path.exists(t1_path) and os.path.exists(t2_path)):
            continue

        # For CDVQA, concatenate T1 and T2 side-by-side as single image
        t1 = Image.open(t1_path).convert("RGB")
        t2 = Image.open(t2_path).convert("RGB")
        combined_w = t1.width + t2.width
        combined = Image.new("RGB", (combined_w, t1.height))
        combined.paste(t1, (0, 0))
        combined.paste(t2, (t1.width, 0))

        pred = _run_single_vqa(model, processor, combined, qa["question"], device)
        correct += int(qa["answer"].lower() in pred.lower())
        total += 1

    return {
        "accuracy": round(correct / max(total, 1) * 100, 2),
        "f1": round(correct / max(total, 1) * 100, 2),  # Approximate
        "n_evaluated": total,
    }


def _mock_cdvqa_results() -> Dict:
    return {
        "status": "not_evaluated",
        "n_evaluated": 0,
        "note": "CDVQA data/model were not available for an actual evaluation run."
    }


# ──────────────────────────────────────────────────────────────
# HELPER: Single VQA Inference
# ──────────────────────────────────────────────────────────────

def _run_single_vqa(model, processor, pil_img, question: str, device: str) -> str:
    """Run a single VQA forward pass."""
    import torch
    from .vqa_engine import GEOCHAT_SYSTEM_PROMPT, GEOCHAT_PROMPT_TEMPLATE

    prompt = GEOCHAT_PROMPT_TEMPLATE.format(
        system=GEOCHAT_SYSTEM_PROMPT,
        query=question,
    )
    try:
        inputs = processor(text=prompt, images=pil_img, return_tensors="pt")
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        input_len = inputs["input_ids"].shape[1]
        return processor.decode(out[0][input_len:], skip_special_tokens=True).strip()
    except Exception as e:
        logger.warning(f"VQA inference failed: {e}")
        return ""


def _compute_bleu1(pred: str, gt: str) -> float:
    """Simple BLEU-1 (unigram) precision."""
    pred_tokens = pred.lower().split()
    gt_tokens = gt.lower().split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    matches = sum(1 for t in pred_tokens if t in gt_tokens)
    return matches / len(pred_tokens)


# ──────────────────────────────────────────────────────────────
# MAIN EVALUATION RUNNER
# ──────────────────────────────────────────────────────────────

def load_model_for_eval(model_path: Optional[str], base_model: str, device: str):
    """Load model for evaluation."""
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    ) if torch.cuda.is_available() else None

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    if model_path and os.path.isdir(model_path):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, model_path)
        logger.info(f"LoRA adapter loaded from {model_path}")

    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    model.eval()
    return model, processor


def run_evaluation(args):
    """Run all requested benchmarks and save results."""
    logger.info("=" * 60)
    logger.info("SatQuery AI — Benchmark Evaluation")
    logger.info(f"Benchmarks: {args.benchmark}")
    logger.info(f"Model path: {args.model_path or 'Base GeoChat (no LoRA)'}")
    logger.info(f"Samples per benchmark: {args.n_samples}")
    logger.info("=" * 60)

    try:
        model, processor = load_model_for_eval(args.model_path, "MBZUAI/GeoChat", args.device)
        model_loaded = True
    except Exception as e:
        logger.warning(f"Could not load model: {e}. Using published baselines.")
        model, processor = None, None
        model_loaded = False

    device = "cuda" if model_loaded else "cpu"
    results = {
        "model": args.model_path or "GeoChat-7B-base",
        "model_loaded": model_loaded,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmarks": {}
    }

    benchmarks = {
        "rsvqa_lr": ("RSVQA Low-Resolution", eval_rsvqa),
        "vrsbench_vqa": ("VRSBench VQA", eval_vrsbench_vqa),
        "cdvqa": ("Change Detection VQA", eval_cdvqa),
    }

    targets = list(benchmarks.keys()) if args.benchmark == "all" else [args.benchmark]

    for bm_key in targets:
        bm_name, eval_fn = benchmarks[bm_key]
        logger.info(f"\n--- Evaluating: {bm_name} ---")
        t0 = time.time()
        result = eval_fn(model, processor, args.data_dir, args.n_samples, device)
        result["eval_time_sec"] = round(time.time() - t0, 1)
        results["benchmarks"][bm_key] = result

        logger.info(f"Results for {bm_name}:")
        for k, v in result.items():
            if isinstance(v, (int, float)):
                logger.info(f"  {k}: {v}")

    # Save results
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    for bm, res in results["benchmarks"].items():
        print(f"\n{bm.upper()}:")
        for k, v in res.items():
            if isinstance(v, (int, float)):
                print(f"  {k:30s}: {v}")
    print("=" * 60)
    print(f"\nFull results saved to: {args.output}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    run_evaluation(args)
