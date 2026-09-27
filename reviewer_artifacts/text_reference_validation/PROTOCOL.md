# Blinded two-author text-reference audit protocol

## Purpose

This audit evaluates the 294 references generated for the fresh matched
reference-information experiment: two references for each of 147
VisualWebArena tasks. It evaluates reference content directly and is separate
from the downstream verifier-performance analysis.

## Independence and blinding

The two authors review every reference independently and do not discuss labels
or view each other's exports until both final exports are complete. Each author
receives a different randomized record order. The interface hides generation
arm, job ID, verifier outputs, verifier correctness, agent outcome label and
the paired reference for the same task.

The stored benchmark criteria are shown as audit evidence. They may or may not
have been supplied to the reference generator. Their presence in the audit
interface must not be used to infer the hidden generation arm. A task-only
reference is not defective merely because it omits a detail found only in the
criteria; the relevant question is whether it contradicts or misrepresents the
available task and criteria.

## Required ratings

Each item is rated **Yes**, **No** or **Uncertain** on six questions:

1. **Task faithful:** The reference is consistent with the user task.
2. **Criteria consistent:** The reference does not contradict or misrepresent
   the stored benchmark criteria.
3. **Free of unsupported details:** Every material requirement in the reference
   is supported by the task or stored criteria.
4. **Observable final state:** The reference describes evidence that could be
   judged from a final webpage screenshot.
5. **Sufficiently specific:** The reference is specific enough to guide a
   verifier without inventing requirements.
6. **Overall usable:** The reference is suitable as an expected-outcome
   description for the tested post-hoc verification setup.

Notes are optional but encouraged for every **No** or **Uncertain** rating.

## Analysis

The analysis reports per-author and pooled rating distributions by hidden arm
and domain, raw agreement and unweighted Cohen's kappa for each question, and a
disagreement queue. The primary approval measure is **overall usable**.

A conservative sensitivity subset contains only task pairs for which both
authors answer **Yes** to all six questions for both hidden references. The
fresh criteria-enriched versus task-only verifier comparison is recomputed on
that subset. This strict subset is a sensitivity analysis, not an adjudicated
replacement dataset. Any final adjudicated dataset must preserve both original
author labels and record the adjudication rationale separately.

## Historical boundary

This audit validates the new 294-reference experiment. It does not reconstruct
the exact historical primary references or identify the four originally
missing reference inputs.
