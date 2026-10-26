"""Freeze a concept-group split for the AAPB GEPA anchor study."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset/r2f_prompt/rarebench"
OUT = ROOT / "research/split_manifest.json"
SIMPLE_CATEGORIES = [
    "rarebench_single_1property_gpt4",
    "rarebench_single_2shape_gpt4",
    "rarebench_single_3texture_gpt4",
    "rarebench_single_4action_gpt4",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def item(category: str, index: int, group_id: str, prompt: str) -> dict[str, object]:
    return {"category": category, "index": index, "group_id": group_id, "prompt": prompt}


def main() -> None:
    paths = sorted(DATA_DIR.glob("*_gpt4.txt"))
    data: dict[str, list[str]] = {}
    hashes: dict[str, str] = {}
    for path in paths:
        category = path.stem
        prompts = list(json.loads(path.read_text(encoding="utf-8")))
        if len(prompts) != 40:
            raise ValueError(f"Expected 40 prompts in {path.name}; found {len(prompts)}")
        data[category] = prompts
        hashes[path.name] = sha256(path)

    train: list[dict[str, object]] = []
    val_search: list[dict[str, object]] = []
    val_audit: list[dict[str, object]] = []
    test: list[dict[str, object]] = []
    group_audit: dict[str, list[list[str]]] = {}

    for category in SIMPLE_CATEGORIES:
        prompts = data[category]
        groups = [prompts[i : i + 4] for i in range(0, 40, 4)]
        if len(groups) != 10 or any(len(group) != 4 for group in groups):
            raise ValueError(f"Unexpected prompt-family layout in {category}")
        # Verify that the four prompts in each family share a salient lexical
        # feature (an attribute, shape, material or action). This catches
        # accidental ordering/schema changes without assuming it is a prefix.
        family_prefixes = []
        for group_index, group in enumerate(groups):
            token_sets = [
                {
                    token
                    for token in re.findall(r"[a-z]+", prompt.lower())
                    if token not in {"a", "an", "the", "of", "made", "with", "on", "is", "and", "two"}
                }
                for prompt in group
            ]
            shared_tokens = sorted(set.intersection(*token_sets))
            if not shared_tokens:
                raise ValueError(
                    f"Prompts {category}[{group_index}] no longer form one four-item concept family: {group}"
                )
            family_prefixes.append({"shared_tokens": shared_tokens, "prompts": group})
            group_id = f"{category}:family_{group_index:02d}"
            entries = [item(category, group_index * 4 + offset, group_id, prompt) for offset, prompt in enumerate(group)]
            if group_index <= 3:
                train.extend(entries)
            elif group_index == 4:
                # Keep distinct instances from the same held-out family for a
                # post-search audit, never shown during GEPA selection.
                val_audit.extend([entries[1], entries[3]])
                # Two fixed, position-balanced examples per category for GEPA
                # candidate selection. The other two remain a post-search audit.
                val_search.extend([entries[0], entries[2]])
            else:
                test.extend(entries)
        group_audit[category] = family_prefixes

    # Composition-heavy categories are mostly held out from prompt evolution
    # (transfer test), but each contributes a small, evenly spaced, fixed
    # sample to train/val so GEPA sees the multi-concept AND/BREAK structure
    # during search too. 8 of 40 rows per category (20%) go to train/val;
    # the remaining 32 (80%) stay test-only.
    complex_categories = sorted(set(data) - set(SIMPLE_CATEGORIES))
    COMPLEX_HOLD_INDICES = list(range(0, 40, 5))  # [0, 5, 10, 15, 20, 25, 30, 35]
    COMPLEX_TRAIN_INDICES = {COMPLEX_HOLD_INDICES[i] for i in (0, 2, 4, 6)}  # 0, 10, 20, 30
    COMPLEX_VAL_SEARCH_INDICES = {COMPLEX_HOLD_INDICES[i] for i in (1, 5)}  # 5, 25
    COMPLEX_VAL_AUDIT_INDICES = {COMPLEX_HOLD_INDICES[i] for i in (3, 7)}  # 15, 35
    for category in complex_categories:
        for i, prompt in enumerate(data[category]):
            group_id = f"{category}:row_{i:02d}"
            entry = item(category, i, group_id, prompt)
            if i in COMPLEX_TRAIN_INDICES:
                train.append(entry)
            elif i in COMPLEX_VAL_SEARCH_INDICES:
                val_search.append(entry)
            elif i in COMPLEX_VAL_AUDIT_INDICES:
                val_audit.append(entry)
            else:
                test.append(entry)

    split_ids = {
        name: {(entry["category"], entry["index"]) for entry in entries}
        for name, entries in (("train", train), ("val_search", val_search), ("val_audit", val_audit), ("test", test))
    }
    if split_ids["train"] & split_ids["val_search"] or split_ids["train"] & split_ids["test"] or split_ids["val_search"] & split_ids["test"]:
        raise ValueError("Prompt-level leakage detected between search splits")
    expected_train = 64 + len(complex_categories) * len(COMPLEX_TRAIN_INDICES)
    expected_val_search = 8 + len(complex_categories) * len(COMPLEX_VAL_SEARCH_INDICES)
    expected_val_audit = 8 + len(complex_categories) * len(COMPLEX_VAL_AUDIT_INDICES)
    expected_test = 240 - len(complex_categories) * len(COMPLEX_HOLD_INDICES)
    if (
        len(train) != expected_train
        or len(val_search) != expected_val_search
        or len(val_audit) != expected_val_audit
        or len(test) != expected_test
    ):
        raise ValueError(
            f"Unexpected split sizes train={len(train)}, val={len(val_search)}, "
            f"audit={len(val_audit)}, test={len(test)}"
        )

    manifest = {
        "version": 2,
        "purpose": "GEPA anchor instruction search; no test data enters optimization or candidate selection",
        "group_rule": (
            "For single-property/shape/texture/action files, each consecutive block of four prompts shares one "
            "rare-concept family. Families 0-3 are train, family 4 is validation, families 5-9 are test. "
            "Single-complex and multi-prompt categories (no 4-row family structure) contribute a small, fixed, "
            "evenly spaced sample per category to train/val (rows 0,10,20,30 train; 5,25 GEPA-selection val; "
            "15,35 validation audit) so GEPA sees multi-concept AND/BREAK structure during search; the other 32 "
            "of 40 rows per category (80%) remain test-only. This hybrid still tests most transfer from simple to "
            "composition-heavy prompts; it does not claim disjoint primitive attributes across splits."
        ),
        "input_sha256": hashes,
        "family_audit": group_audit,
        "gepa_train": train,
        "gepa_selection_validation": val_search,
        "full_validation_audit": val_audit,
        "final_test": test,
    }
    OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")
    print(f"train={len(train)}, GEPA validation={len(val_search)}, full validation audit={len(val_audit)}, test={len(test)}")
    print(f"test categories held out: {', '.join(complex_categories)} plus unseen simple concept families")


if __name__ == "__main__":
    main()
