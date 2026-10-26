# Proposal: GEPA-Evolved Auxiliary Anchors for Rare-Concept Diffusion

**Status:** Research proposal; no GEPA image-search result has been completed.  
**Date:** 2026-10-25  
**Decision requested:** Use this as the experiment plan for the next AAPB study.

## 1. Proposed title

**Learning Better Auxiliary Anchors for Rare-Concept Diffusion with Pareto-Guided Prompt Evolution**

Alternative, more conservative title: **When Do Better Anchors Help Adaptive Auxiliary Prompt Blending? A Controlled Study of Rare-Concept Generation**.

## 2. Core idea

AAPB keeps the diffusion sampler fixed and blends the target-conditioned prediction with an auxiliary-anchor prediction using its closed-form, per-step coefficient. Its usefulness therefore depends on the anchor being helpful: an anchor that is too generic may discard the rare attribute, while an anchor that is too close to the target may provide little structural support.

Use GEPA to evolve the **R2F anchor-generation system instruction** against image-level feedback. The optimized component is the rule that turns a target prompt into a frequent, structurally useful anchor. Keep the target prompt, SD3 checkpoint, scheduler, guidance scale, AAPB equation and denoising settings fixed during search. GEPA receives examples of the target, the generated anchor, the resulting image, decomposed scores and failure notes; its reflective mutations try revised anchor instructions, while Pareto selection retains candidates that help different examples or quality dimensions.

The paper question is:

> Can an anchor-generation instruction evolved against AAPB outcomes improve rare-object and rare-attribute fidelity on unseen concept groups, while preserving composition and image coherence, compared with the current R2F instruction under the same image-generation budget?

GEPA is the search method, not the contribution by itself. A paper claim requires a distinct AAPB-specific finding and held-out evidence.

## 3. Working hypotheses

| ID | Hypothesis | Evidence that would support it | Falsifier |
|---|---|---|---|
| H1 | GEPA-evolved anchors improve rare-object identity and attribute fidelity over the current R2F instruction when both use adaptive AAPB. | Positive paired difference on held-out prompt groups and seeds; confidence interval excludes zero. | No reliable gain, or gains appear only on GEPA's search examples. |
| H2 | Better anchors provide a benefit specifically through AAPB's blending, beyond prompt rewriting alone. | A positive interaction in a 2×2 comparison: original/evolved anchor instruction × target-only/adaptive AAPB. | Evolved anchors help target-only generation just as much, or adaptive AAPB erases the gain. |
| H3 | The gain does not trade away structure or general image coherence. | Relation/composition and coherence are non-inferior under pre-registered margins. | Rare-attribute gains coincide with worse relations, layout or coherence. |
| H4 | GEPA's reflection is more efficient than an equal-budget search baseline. | Better held-out score than random/best-of-N prompt variants under matched image and API calls. | The same-budget baseline matches or exceeds GEPA. |

These are hypotheses, not established findings.

## 4. Candidate method

### 4.1 GEPA candidate

The candidate is one text artifact: the R2F system instruction. It must produce a structured anchor record, for example:

```json
{
  "target_prompt": "A hairy frog",
  "anchor_prompt": "A hairy animal",
  "replacements": [{"target_span": "frog", "anchor_span": "animal"}],
  "retained_attributes": ["hairy"],
  "retained_relations": []
}
```

The adapter must reject or repair outputs that alter the target prompt, omit the anchor, or fail to preserve prompt constraints. The anchor is allowed to replace a rare noun with a frequent, semantically compatible concept; it must not silently remove the rare attribute or change the target text.

### 4.2 Fixed generation system

During GEPA search, hold constant:

- Stable Diffusion 3 Medium checkpoint and revision already used by AAPB.
- The AAPB implementation and closed-form adaptive coefficient.
- Scheduler, classifier-free guidance, resolution, denoising steps and VAE.
- Target prompt, random seed and evaluator version for every paired comparison.
- Anchor-generation model and temperature; record exact model identifier and date.

This isolates anchor-instruction changes from sampler changes. A joint GEPA search over the anchor rule and gamma policy is a later ablation only if this experiment identifies a robust anchor effect.

### 4.3 Feedback and objective

Use a decomposed vision evaluation with four separately retained scores, each normalized to 0–1:

1. **Target object identity** — is the requested object present and recognizable?
2. **Rare attribute fidelity** — is the unusual property bound to the requested object?
3. **Relations and composition** — are requested actions, counts and spatial relations correct?
4. **Visual coherence** — is the result coherent without rewarding generic attractiveness over prompt fidelity?

Suggested scalar used for an initial screening run: `0.35*object + 0.35*attribute + 0.20*relations + 0.10*coherence`. Also retain the component scores and failure descriptions as GEPA feedback. Before committing to multi-objective GEPA configuration, verify that the installed GEPA version actually consumes the configured score vector; do not assume that fields in side information become optimization objectives automatically.

The search judge is a proxy, not ground truth. The final evaluator must use a different model or scoring process, plus a blinded human subset, to reduce reward hacking and evaluator overfit.

## 5. Experimental design

### 5.1 Data split

RareBench has 320 target prompts across eight categories in this checkout. Split by **concept/template groups**, not randomly by individual row, so closely related prompts do not leak from GEPA search into validation or test. First inspect each file and define groups (for example, a rare attribute paired with its four object instances). Freeze the split manifest before running GEPA.

Proposed structure:

- **GEPA search data:** 24 prompts, balanced across the four simple categories (single-property, shape, texture, action): six per category, drawn from training families.
- **GEPA selection set:** 8 fixed prompts, two per simple category, from a disjoint validation family.
- **Full validation audit:** 8 other prompts from those same four validation families, scored after search and separately from GEPA candidate selection.
- **Final test:** 240 prompts: the remaining five simple concept families per simple category, plus all prompts from the four complex/multi categories. This measures transfer to unseen simple families and composition-heavy categories; primitive concepts may overlap, so do not call it a fully attribute-disjoint test.

Keep prompt text, concept-group IDs and split assignment in a versioned JSON/CSV manifest. No test score may influence GEPA mutations, candidate selection, thresholds or prompt wording.

### 5.2 GEPA screening run

- Search budget: start at 120 metric calls, which permits roughly 15 candidate evaluations on an eight-example selection set in the installed GEPA version. Cap GEPA reflection-model cost separately from image/evaluator API cost.
- Screening render: 512×512, 16 denoising steps, one fixed seed per prompt. This is only for inexpensive candidate search.
- Record every candidate, parent, anchor text, image, dimension score, evaluator explanation, failed parser output, API model/version, time and cost.
- Re-evaluate the seed instruction and selected candidate on the full validation audit before opening the test set.
- Compare GEPA with an equal-budget best-of-N or random-edit search. Count image generations and judge calls, not just GEPA proposals.

### 5.3 Confirmatory comparison

Run a paired 2×2 design on the final test prompts:

| Anchor instruction | Target-only CFG (`γ=0`) | Adaptive AAPB (`γ=γ*`) |
|---|---:|---:|
| Current R2F instruction | Control A | Main AAPB baseline |
| GEPA-evolved instruction | Anchor-only effect | Proposed combined condition |

Generate three fixed, pre-registered seeds per prompt and condition. Keep all four cells paired by target prompt and seed. This separates the effect of changing anchors from the effect of applying AAPB's blend. Add fixed-γ baselines on a smaller, pre-registered subset; choose γ only on training/validation data.

For a direct related-work comparison, implement or obtain reproducible versions of R2F, ADAPT and Score Replacement with Bounded Deviation. Start with an 80-prompt stratified subset if full comparison is too expensive, but report that subset comparison separately from the primary all-test result. Do not treat an unavailable baseline as an equivalent reproduction.

### 5.4 Evaluation

**Primary endpoints**

- Rare object identity score.
- Rare attribute-binding score.

**Guardrails and secondary endpoints**

- Relation/composition score.
- Visual coherence and human preference.
- CLIP or another image-text similarity score, clearly labeled as a proxy.
- Latency, peak VRAM, anchor-generation tokens/cost and total inference cost.

Use a final vision judge that was not used to evolve prompts. Blind the method labels for human ratings; sample across all categories, including failures. Report per-category means and confidence intervals, not only one pooled score.

Use paired analysis with bootstrap resampling clustered by concept group and prompt, and account for seed variation. Report absolute effect sizes and 95% intervals; correct secondary comparisons for multiple testing. Estimate variance on the pilot and perform a power check before fixing the final sample size.

## 6. Baselines and ablations

1. **Original R2F + AAPB** — current checked-in system prompt and adaptive coefficient.
2. **GEPA-R2F + AAPB** — proposed system.
3. **Original R2F + target-only CFG** — removes anchor contribution.
4. **GEPA-R2F + target-only CFG** — tests whether evolved anchors help without blending.
5. **Equal-budget best-of-N prompt search** — tests whether GEPA reflection is useful beyond spending more evaluation calls.
6. **Fixed-γ sweep** — confirms whether any gain comes from anchor quality rather than a coefficient change.
7. **External baselines on a shared subset:** R2F's native schedule, ADAPT, and Score Replacement, subject to faithful reproduction.

Optional ablations after the main result: single-object versus multi-concept prompts; GEPA scalar score versus Pareto/multi-objective selection; 16-step screening versus 50-step verification; one versus three search seeds.

## 7. Novelty and publication risk

The broad claim “use GEPA to optimize prompts for text-to-image generation” is **not novel enough**: RATTPO already performs reward-guided test-time prompt optimization, and CVPR 2025 work directly studies prompt optimization for minority/low-density image generation. GEPA itself is a general reflective prompt optimizer, not an AAPB method contribution. ([RATTPO](https://arxiv.org/abs/2506.16853), [Minority-Focused T2I Prompt Optimization](https://openaccess.thecvf.com/content/CVPR2025/html/Um_Minority-Focused_Text-to-Image_Generation_via_Prompt_Optimization_CVPR_2025_paper.html), [GEPA](https://arxiv.org/abs/2507.19457))

The narrower candidate gap is **optimizing the auxiliary anchor instruction specifically for the frozen AAPB score-blending mechanism**, with decomposed rare-concept feedback, Pareto selection across prompts, and a factorial test that isolates anchor effects from gamma effects. This is a literature-search hypothesis only. It must be compared directly with existing R2F prompting, generic T2I prompt optimizers and related AAPB extensions before claiming novelty.

Prior work makes overlap risk substantial:

- AAPB already contributes closed-form adaptive blending of target and auxiliary anchor predictions for rare generation/editing. ([AAPB](https://arxiv.org/abs/2603.19158))
- R2F already uses LLM-planned rare-to-frequent concept guidance. ([R2F](https://arxiv.org/abs/2410.22376))
- ADAPT already targets rare concepts with attention-driven prompt scheduling and orthogonal components; five authors overlap with AAPB. ([ADAPT](https://arxiv.org/abs/2603.19157))
- Score Replacement already switches between proxy and target score according to score deviation. ([Score Replacement](https://arxiv.org/abs/2505.20808))
- RADIANCE already monitors denoising-time compositional similarity and reacts by adjusting guidance. ([RADIANCE](https://arxiv.org/abs/2607.05088))
- Semantic-aware CFG already varies guidance over semantic regions. ([S-CFG](https://arxiv.org/abs/2404.05384))

**Publication decision:** do not promise a method paper yet. A positive result must beat an equal-budget prompt-search baseline and improve the final held-out primary endpoints without degrading structure. If gains are small, judge-specific or confined to seen concept groups, the more defensible output is an AAPB reliability/benchmark study rather than a new-method claim.

## 8. Proposed go/no-go criteria

Before unlocking test results, register these provisional thresholds and revise only using pilot variance:

- **Go:** GEPA+ AAPB improves the primary rare-attribute score over original-R2F+AAPB by at least 3 points on a 0–100 scale, with the paired 95% interval above zero; object identity is not worse; relation/coherence do not fall by more than 1 point; and the equal-budget search baseline is weaker.
- Human preference on the blinded subset should favor GEPA anchors with an interval excluding 50%, or the paper must clearly limit its claim to the automatic endpoint.
- The effect should appear in several categories, not one category driving the aggregate.
- **No-go for a method claim:** any primary endpoint fails these conditions, the effect disappears on the independent judge, or the held-out prompts show substantial regression. Continue with diagnostic analysis or report a negative result instead of broadening the claim post hoc.

## 9. Execution stages

1. **Audit (before runs):** read full AAPB, ADAPT, RATTPO, minority-prompt optimization and score-replacement papers; inspect their code; confirm the actual RareBench concept-group structure; record model/checkpoint hashes and evaluator versions.
2. **Instrument and freeze:** add robust JSON anchor parsing, cache each generated image/judgment by candidate-prompt hash + example + seed, and write the split manifest. Verify the scoring rubric on a small blind set.
3. **Pilot:** time 10 paired images, measure VRAM and API cost, calibrate the evaluator against human ratings, and estimate variance. This sets a realistic full-run budget.
4. **GEPA search:** optimize only the R2F system instruction on training examples and the fixed validation set; save the full Pareto frontier and traces.
5. **Validation audit:** compare seed prompt, GEPA candidate and equal-budget search baseline across all validation groups. Freeze the winning candidate before test.
6. **Confirmatory run:** generate all paired test images at 1024px/50 steps for three seeds, run independent metrics and blinded human review.
7. **Write-up:** report methods, exact prompts, checkpoint/version identifiers, total compute/API cost, confidence intervals, failure cases, novelty limitations and released evaluation manifests.

## 10. Current project evidence and boundaries

- The working tree has 320 existing AAPB images at seed 42 across eight categories and a partial second-seed run. These are useful for smoke checks but are not a paired multi-seed comparison.
- The SD3 Medium checkpoint and requested `ab_3` environment are present on the shared machine. The first GEPA image-run attempts were stopped before producing a generated image or an optimization result; do not cite them as experimental evidence.
- The current evaluation script gives one GPT score and has prompt-extraction/rubric issues. It is not sufficient as the sole paper endpoint.
- The earlier novelty search used Semantic Scholar plus primary paper pages. It is not exhaustive prior-art review.

## References

- Agrawal, L. A., et al. (2026). *GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning*. ICLR 2026. https://arxiv.org/abs/2507.19457
- Kim, S., et al. (2025). *Reward-Agnostic Prompt Optimization for Text-to-Image Diffusion Models*. https://arxiv.org/abs/2506.16853
- Um, S., & Ye, J. C. (2025). *Minority-Focused Text-to-Image Generation via Prompt Optimization*. CVPR 2025. https://openaccess.thecvf.com/content/CVPR2025/html/Um_Minority-Focused_Text-to-Image_Generation_via_Prompt_Optimization_CVPR_2025_paper.html
- Park, D., et al. (2025). *Rare-to-Frequent: Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with LLM Guidance*. ICLR 2025. https://arxiv.org/abs/2410.22376
- Lee, K., et al. (2026). *Adaptive Auxiliary Prompt Blending for Target-Faithful Diffusion Generation*. CVPR 2026. https://arxiv.org/abs/2603.19158
- Lee, K., et al. (2026). *ADAPT: Attention Driven Adaptive Prompt Scheduling and InTerpolating Orthogonal Complements for Rare Concepts Generation*. CVPR 2026 Findings. https://arxiv.org/abs/2603.19157
- Ruan, B.-K., et al. (2025). *Score Replacement with Bounded Deviation for Rare Prompt Generation*. https://arxiv.org/abs/2505.20808
- Ni, Z.-X., et al. (2026). *RADIANCE: Relative Adaptive Denoising with IP-Adapter for Novel Concept Enhancement*. ECCV 2026. https://arxiv.org/abs/2607.05088
- Shen, D., et al. (2024). *Rethinking the Spatial Inconsistency in Classifier-Free Diffusion Guidance*. CVPR 2024. https://arxiv.org/abs/2404.05384
