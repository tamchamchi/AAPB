"""GEPA optimization of AAPB's Rare-to-Frequent anchor-generation instruction.

The evaluator generates one SD3 image from the proposed anchor, then scores
target identity/attributes/relations and visual coherence with a fixed judge.
Validation examples select candidates; held-out examples are scored only after
search. This is a research pilot, not an automatic novelty determination.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from openai import OpenAI
from gepa.optimize_anything import GEPAConfig, EngineConfig, ReflectionConfig, optimize_anything


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROMPT_DIR = ROOT / "src/gpt/template"
# Both the system instruction and the user template (few-shot examples, output
# format) are optimized jointly: the user template carries most of the actual
# task-specific rules (BREAK/AND format, worked examples), so leaving it fixed
# would hide that content from GEPA's reflection.
SEED_CANDIDATE = {
    "system_prompt": (PROMPT_DIR / "template_r2f_system.txt").read_text(encoding="utf-8"),
    "user_prompt": (PROMPT_DIR / "template_r2f_user.txt").read_text(encoding="utf-8"),
}
MODEL_DIR = Path(
    os.getenv(
        "AAPB_MODEL_DIR",
        "/mnt/mmlab2024nas/anhndt/cache/huggingface/hub/models--stabilityai--stable-diffusion-3-medium/snapshots/84a9ff37a0a30f7252e21daae69cfd0134198d27",
    )
)
CUDA_DEVICE = os.getenv("AAPB_DEVICE", "cuda:0")
SCREEN_SIZE = int(os.getenv("AAPB_GEPA_IMAGE_SIZE", "512"))
SCREEN_STEPS = int(os.getenv("AAPB_GEPA_STEPS", "28"))
BASE_SEED = int(os.getenv("AAPB_GEPA_SEED", "42"))
RUN_DIR = Path(os.getenv("AAPB_GEPA_RUN_DIR", "output/gepa_aapb_anchor_prompt"))

_client: OpenAI | None = None
_pipe: Any = None
_score_cache: dict[str, tuple[float, dict[str, Any]]] = {}


def load_project_env() -> None:
    """Load project .env without printing values or overriding shell vars."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.removeprefix("export ").strip()
        value = value.strip().strip("\"'")
        if name:
            os.environ.setdefault(name, value)


def load_pipeline(device="cuda:0") -> Any:
    global _pipe
    if _pipe is not None:
        return _pipe

    print(f"Loading SD3 from {MODEL_DIR} onto {device}...", flush=True)
    import torch
    from pipelines.pipeline_aapb_sd3 import AAPBDiffusion3Pipeline

    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(f"Cached SD3 model directory not found: {MODEL_DIR}")

    _pipe = AAPBDiffusion3Pipeline.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=torch.float16,
        local_files_only=True,
        use_safetensors=True,
    ).to(device)
    print("SD3 pipeline loaded.", flush=True)
    return _pipe

def build_examples() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build only train/selection/audit examples; never load final-test rows."""
    manifest = json.loads((ROOT / "research/split_manifest.json").read_text(encoding="utf-8"))
    visible_rows = (
        manifest["gepa_train"]
        + manifest["gepa_selection_validation"]
        + manifest["full_validation_audit"]
    )
    visible_categories = {str(row["category"]) for row in visible_rows}
    category_data: dict[str, dict[str, Any]] = {}
    for filename, expected_hash in manifest["input_sha256"].items():
        path = ROOT / "dataset/r2f_prompt/rarebench" / filename
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeError(f"Dataset changed after split freeze: {filename}")
        if path.stem in visible_categories:
            category_data[path.stem] = json.loads(path.read_text(encoding="utf-8"))

    def materialize(row: dict[str, Any]) -> dict[str, Any]:
        category = str(row["category"])
        index = int(row["index"])
        data = category_data[category]
        targets = list(data.keys())
        if index >= len(targets) or targets[index] != row["prompt"]:
            raise RuntimeError(f"Manifest/data mismatch at {category}[{index}]")
        target = targets[index]
        value = data[target]
        r2f = value.get("r2f_prompt", [[target]])
        pair = r2f[0] if r2f else [target]
        return {
            "name": f"{category}:{index}",
            "category": category,
            "group_id": row["group_id"],
            "target": target,
            "existing_anchor": pair[0] if pair else target,
            "existing_target": pair[-1] if pair else target,
        }

    # Four simple-category training families (64 rows) plus a small, fixed
    # sample from each multi-object/complex category (16 rows) so GEPA sees
    # multi-concept AND/BREAK structure during search too.
    train_rows = manifest["gepa_train"]
    if len(train_rows) != 80:
        raise RuntimeError(f"Expected 80 GEPA training rows, found {len(train_rows)}")
    train = [materialize(row) for row in train_rows]
    val = [materialize(row) for row in manifest["gepa_selection_validation"]]
    audit = [materialize(row) for row in manifest["full_validation_audit"]]
    return train, val, audit


def generate_anchor(candidate: dict[str, str], target: str) -> tuple[str, str]:
    global _client
    if _client is None:
        _client = OpenAI()
    user_template = candidate["user_prompt"]
    if "{prompt}" not in user_template:
        raise ValueError("Candidate user_prompt dropped the required '{prompt}' placeholder")
    user_prompt = user_template.replace("{prompt}", target)
    response = _client.chat.completions.create(
        model=os.getenv("AAPB_ANCHOR_MODEL", "gpt-6-luna"),
        # temperature=0,
        # max_tokens=700,
        messages=[
            {"role": "system", "content": candidate["system_prompt"]},
            {"role": "user", "content": user_prompt},
        ],
    )
    raw = response.choices[0].message.content or ""
    match = re.search(r"Final Prompt Sequence\s*:\s*(.+)", raw, flags=re.IGNORECASE)
    sequence = match.group(1).strip() if match else ""
    if "BREAK" not in sequence:
        return target, raw
    parts = sequence.split("BREAK", maxsplit=1)
    anchor = parts[0].strip().strip("`\"'")
    if not anchor:
        return target, raw
    return anchor, raw


def judge_image(image_path: Path, target: str) -> tuple[float, dict[str, Any]]:
    global _client
    if _client is None:
        _client = OpenAI()
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    response = _client.chat.completions.create(
        model=os.getenv("AAPB_JUDGE_MODEL", "gpt-6-luna"),
        # temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Evaluate the image against the exact target prompt. Score each dimension from 1 to 5: "
                    "object_identity, rare_attribute_fidelity, relations_and_composition, visual_coherence. "
                    "A 5 means clearly present and correct; 1 means absent or wrong. Judge only visible evidence. "
                    "Do not reward a generic attractive image that misses the rare concept. Return JSON with integer "
                    "scores for all dimensions and a short explanation."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Target prompt: {target}"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded}"},
                    },
                ],
            },
        ],
    )
    raw = response.choices[0].message.content or "{}"
    grade = json.loads(raw)
    names = [
        "object_identity",
        "rare_attribute_fidelity",
        "relations_and_composition",
        "visual_coherence",
    ]
    scores = {name: max(1, min(5, int(grade[name]))) for name in names}
    weights = {
        "object_identity": 0.35,
        "rare_attribute_fidelity": 0.35,
        "relations_and_composition": 0.20,
        "visual_coherence": 0.10,
    }
    normalized = sum(weights[name] * (scores[name] - 1) / 4 for name in names)
    return normalized, {"scores_1_to_5": scores, "explanation": str(grade.get("explanation", ""))}


def _candidate_fingerprint(candidate: dict[str, str]) -> str:
    serialized = json.dumps(candidate, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode()).hexdigest()


def evaluate(candidate: dict[str, str], example: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    candidate_hash = _candidate_fingerprint(candidate)
    cache_key = hashlib.sha256((candidate_hash + "\0" + example["name"]).encode()).hexdigest()
    if cache_key in _score_cache:
        return _score_cache[cache_key]
    anchor, raw_anchor_response = generate_anchor(candidate, example["target"])
    prompt_key = candidate_hash[:12]
    item_key = hashlib.sha256(example["name"].encode()).hexdigest()[:12]
    image_path = RUN_DIR / "images" / prompt_key / f"{item_key}_seed{BASE_SEED}.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if not image_path.exists():
        pipe = load_pipeline()
        result = pipe(
            r2f_prompts={"r2f_prompt": [[anchor, example["target"]]]},
            batch_size=1,
            height=SCREEN_SIZE,
            width=SCREEN_SIZE,
            num_inference_steps=SCREEN_STEPS,
            guidance_scale=7.0,
            seed=BASE_SEED,
        )
        result.images[0].save(image_path)
    score, judge_info = judge_image(image_path, example["target"])
    info = {
        "example": example["name"],
        "target": example["target"],
        "generated_anchor": anchor,
        "anchor_response": raw_anchor_response,
        "image_path": str(image_path),
        **judge_info,
    }
    _score_cache[cache_key] = (score, info)
    return score, info


def main() -> None:
    load_project_env()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY in the environment or project .env")
    train, val, audit = build_examples()
    max_metric_calls = int(os.getenv("AAPB_GEPA_MAX_METRIC_CALLS", "120"))
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    result = optimize_anything(
        seed_candidate=SEED_CANDIDATE,
        evaluator=evaluate,
        dataset=train,
        valset=val,
        objective=(
            "Improve Rare-to-Frequent anchor generation for AAPB so rare object identity and attributes are "
            "preserved while composition remains coherent on unseen RareBench prompts."
        ),
        background=(
            "The candidate has two components: 'system_prompt' (the R2F system instruction) and 'user_prompt' "
            "(the task template with few-shot examples and output format, containing a literal '{prompt}' "
            "placeholder that is substituted with the target text at call time -- this placeholder must always be "
            "kept, verbatim, somewhere in user_prompt). Each candidate is judged after generating an SD3 image with "
            "AAPB. Optimize the rare concept, not general image attractiveness. Keep anchor prompts short and "
            "semantically close. Avoid unsupported claims and do not change the target prompt. Most training and "
            "validation examples are single-object prompts, but a smaller fixed subset uses multi-object prompts "
            "combining concepts with AND/BREAK; the instructions and examples should handle both."
        ),
        config=GEPAConfig(
            engine=EngineConfig(
                run_dir=str(RUN_DIR / "engine"),
                seed=17,
                max_metric_calls=max_metric_calls,
                max_reflection_cost=float(os.getenv("AAPB_GEPA_REFLECTION_BUDGET_USD", "5")),
                max_workers=1,
                parallel=False,
                display_progress_bar=True,
            ),
            reflection=ReflectionConfig(
                reflection_lm=os.getenv("GEPA_REFLECTION_MODEL", "openai/gpt-6-luna"),
                reflection_minibatch_size=2,
            ),
        ),
    )
    best = result.best_candidate
    # Validation audit is disjoint from GEPA's 8-row selection set. Final-test
    # rows are not loaded by this runner and must be evaluated separately only
    # after the candidate is frozen and registered.
    audit_results: dict[str, list[dict[str, Any]]] = {}
    for label, candidate in (("seed", SEED_CANDIDATE), ("best", best)):
        audit_results[label] = []
        for example in audit:
            score, info = evaluate(candidate, example)
            audit_results[label].append({"score": score, "info": info})
    audit_means = {
        label: sum(row["score"] for row in rows) / len(rows)
        for label, rows in audit_results.items()
    }
    validation_score = result.val_aggregate_scores[result.best_idx]
    report = {
        "objective": "GEPA optimization of R2F system+user prompt for AAPB",
        "screening": {"image_size": SCREEN_SIZE, "steps": SCREEN_STEPS, "seed": BASE_SEED},
        "split": {
            "train": [x["name"] for x in train],
            "validation": [x["name"] for x in val],
            "validation_audit": [x["name"] for x in audit],
        },
        "best_candidate": best,
        "best_validation_score": validation_score,
        "validation_audit_mean": audit_means,
        "validation_audit_details": audit_results,
        "validation_audit_delta_best_minus_seed": audit_means["best"] - audit_means["seed"],
        "total_metric_calls": result.total_metric_calls,
        "num_candidates": result.num_candidates,
        "engine_run_dir": result.run_dir,
    }
    (RUN_DIR / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (RUN_DIR / "best_anchor_system_prompt.txt").write_text(best["system_prompt"], encoding="utf-8")
    (RUN_DIR / "best_anchor_user_prompt.txt").write_text(best["user_prompt"], encoding="utf-8")
    print(f"Best validation score: {validation_score:.3f}")
    print(f"Validation audit mean: seed={audit_means['seed']:.3f}, best={audit_means['best']:.3f}")
    print(f"Validation audit delta: {report['validation_audit_delta_best_minus_seed']:+.3f}")
    print(f"GEPA metric calls={result.total_metric_calls}; candidates={result.num_candidates}")
    print(f"Saved report to {RUN_DIR / 'result.json'}")


if __name__ == "__main__":
    main()
