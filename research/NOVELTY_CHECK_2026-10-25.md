# AAPB follow-up: novelty triage

**Date:** 2026-10-25  
**Method:** Semantic Scholar Graph API search (using the key supplied for this task), followed by primary-source arXiv checks.  
**Scope:** Existing code-inspection hypotheses in `experiements/aapb_agent_run_2026-10-25.md`, especially global scalar gamma, multi-concept anchors, and robustness.  
**Status:** Preliminary literature triage, not a novelty determination.

## Search findings

| Area | Nearest work found | Consequence |
|---|---|---|
| Rare-concept generation with frequent-concept guidance | R2F plans and applies rare-to-frequent prompt guidance during diffusion. | Anchor selection and use of frequent semantic neighbors already have substantial prior art. |
| Per-region / per-semantic-unit guidance | S-CFG adapts CFG scales for semantic regions at each denoising step. | A proposal framed only as replacing global scalar gamma with region-specific gamma has high overlap risk. |
| Rare-concept prompt scheduling | ADAPT uses attention to plan prompt schedules and orthogonal components for rare concepts; it reports CVPR 2026 Findings acceptance. Its author list substantially overlaps AAPB's. | Closely inspect ADAPT and compare algorithms/claims before extending AAPB along prompt scheduling or attention-guided interpolation. |
| Adaptive rare-prompt switching / score discrepancy | *Score Replacement with Bounded Deviation* triggers a prompt-score switch when proxy/target deviation crosses a threshold. | A reliability gate based on denoising-time score discrepancy is not a clean gap by itself; this is a particularly close algorithmic neighbor. |
| Closed-loop rare-concept feedback | RADIANCE monitors compositional similarity during denoising and adjusts guidance scales. | A generic denoising-time monitor/controller also has direct recent overlap; any AAPB variant needs a sharply different signal/objective and direct baseline comparison. |
| AAPB itself | The arXiv record for AAPB reports CVPR 2026 main-track acceptance and describes the closed-form adaptive auxiliary/target coefficient. | Treat AAPB as established prior work in any new submission; the checked-in README's conference note and arXiv metadata are consistent. |

## Candidate assessment

### 1. Per-region or per-concept adaptive gamma — **high overlap risk**

This follows hypothesis C1 in the run log. S-CFG already adapts guidance across semantic regions, and ADAPT is much closer to AAPB's rare-composition setting. A genuinely distinct method would need a different technical mechanism and a measured advantage over both, not just a finer coefficient map. Keep it as an ablation/diagnostic, not the lead claim yet.

### 2. Automatically detect unreliable anchors and fall back or reselect — **high overlap risk; narrow the claim**

This follows A1/A3/A4 in the run log. But *Score Replacement with Bounded Deviation* already uses proxy/target score deviation to trigger switching, and RADIANCE uses denoising-time similarity feedback to adjust guidance. A new anchor gate would need a distinct reliability signal/objective and a demonstrated advantage over those methods, R2F, and ADAPT. This search did not establish a gap.

### 3. Batch-wise scalar coupling and clamp/fallback behavior — **engineering first**

The `.sum()` across the full tensor may couple batch examples; clamp and near-zero-denominator behavior may introduce edge cases. Test and repair correctness first. Those observations alone are not a research novelty claim.

## Recommended next steps

1. Read the full ADAPT, Score Replacement, and RADIANCE papers and inspect code/supplements where available; map each method against `src/pipelines/pipeline_aapb_sd3.py`.
2. Run a paired-seed benchmark on the same RareBench prompts for R2F, AAPB, ADAPT, and Score Replacement. Score rare object/attribute identity and relations separately from structure and image quality.
3. Instrument raw gamma, denominator, clamp-hit rate, score discrepancy, and latency. First test batch independence and coefficient stability as correctness/robustness work.
4. Hold out prompts and seeds. Report per-category results and uncertainty; do not select the method on the held-out set.
5. Re-run novelty search only after a candidate mechanism is defined precisely. Then use GEPA to optimize the research-planning prompt against its versioned evaluator. GEPA has not been run yet because its package and proposer-model credentials are missing in this environment.

## Sources

- Lee, K., et al. (2026). *Adaptive Auxiliary Prompt Blending for Target-Faithful Diffusion Generation*. arXiv:2603.19158. https://arxiv.org/abs/2603.19158
- Lee, K., et al. (2026). *ADAPT: Attention Driven Adaptive Prompt Scheduling and InTerpolating Orthogonal Complements for Rare Concepts Generation*. arXiv:2603.19157. https://arxiv.org/abs/2603.19157
- Ruan, B.-K., et al. (2025). *Score Replacement with Bounded Deviation for Rare Prompt Generation*. arXiv:2505.20808. https://arxiv.org/abs/2505.20808
- Ni, Z.-X., et al. (2026). *RADIANCE: Relative Adaptive Denoising with IP-Adapter for Novel Concept Enhancement*. arXiv:2607.05088. https://arxiv.org/abs/2607.05088
- Park, D., et al. (2025). *Rare-to-Frequent: Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with LLM Guidance*. ICLR 2025. https://arxiv.org/abs/2410.22376
- Shen, D., et al. (2024). *Rethinking the Spatial Inconsistency in Classifier-Free Diffusion Guidance*. CVPR 2024. https://arxiv.org/abs/2404.05384
