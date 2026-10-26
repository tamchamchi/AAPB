# AAPB research loop

This folder turns literature triage into a repeatable step before changing the sampler.

The current detailed experiment plan is [`PROPOSAL_GEPA_AAPB_ANCHORS.md`](PROPOSAL_GEPA_AAPB_ANCHORS.md). Treat it as the source of truth for research questions, splits, metrics, baselines and go/no-go criteria.

## 1. Check prior work

The Semantic Scholar key belongs in an environment variable; do not commit it or put it on a command line:

```bash
export SEMANTIC_SCHOLAR_API_KEY="..."
python research/novelty_check.py \
  --claim "Per-concept AAPB coefficients estimated from denoising score discrepancies for rare compositions" \
  --query "per-concept adaptive prompt blending text-to-image diffusion rare concepts" \
  --query "semantic-aware region-specific classifier-free guidance rare concepts" \
  --out output/novelty_candidate.md
```

The checker retries rate limits, deduplicates returned records, and writes abstracts and direct paper links. Treat this as candidate retrieval, then read the closest papers and citation chains before making any novelty claim. Search several formulations and include adjacent terms such as CFG, prompt scheduling, regional guidance, rare concepts, and score interpolation.

## 2. Turn a plausible gap into a falsifiable experiment

For each idea, record (a) the code location and observed behavior, (b) a precise claim, (c) closest prior work, (d) a matched baseline, (e) metrics for rare-concept identity and composition separately, (f) seeds and held-out prompts, and (g) an explicit result that would falsify the idea. The repo's initial candidates and ordering are in [`NOVELTY_CHECK_2026-10-25.md`](NOVELTY_CHECK_2026-10-25.md).

## 3. Optimize anchor generation with GEPA

`gepa_anchor_prompt.py` is an unvalidated pilot harness for applying `optimize_anything` to the R2F system instruction. It asks the anchor LLM for a frequent-concept anchor, renders one SD3+AAPB image at 512px/16 steps, and uses a fixed vision judge to score target object identity, rare-attribute fidelity, relations/composition, and visual coherence. Its current tiny split is for bringing up the pipeline only; it is not the paper split specified in the proposal.

This uses OpenAI for anchor generation and vision judging plus GEPA's reflection model. The runner loads project-root `.env` without printing values and does not override variables already set in the process. Optional model overrides are `AAPB_ANCHOR_MODEL`, `AAPB_JUDGE_MODEL`, and `GEPA_REFLECTION_MODEL`; hardware/search overrides are `AAPB_DEVICE`, `AAPB_GEPA_IMAGE_SIZE`, `AAPB_GEPA_STEPS`, and `AAPB_GEPA_MAX_METRIC_CALLS`. GEPA search was stopped before producing an image or result; do not cite the harness as experimental evidence. Follow the proposal's group split, screening budget, independent evaluation and full-resolution confirmatory runs before drawing conclusions.

The available GEPA version in `ab_3` uses `GEPAConfig`/`EngineConfig` and requires LiteLLM for reflection; install `litellm` explicitly. Do not optimize directly against the repo's existing single GPT image score: `src/eval/eval_by_GPT.py` collapses outcomes and has rubric/scaling risks recorded in the run log. GEPA search feedback remains an LLM-judge proxy and cannot establish novelty; compare the selected prompt against R2F, ADAPT, and score-switching methods with independent metrics and human review.

## Suggested next experiment

The proposed study tests whether GEPA-driven anchor instructions improve AAPB's RareBench outputs. The novelty search found close prior work on T2I prompt optimization, regional guidance, attention-driven schedules, score-discrepancy switching, and denoising-time feedback. Any positive pilot is only a screening signal: the proposal requires full held-out testing, multiple seeds, dimension-specific independent scoring and a blind human subset before a method claim.
