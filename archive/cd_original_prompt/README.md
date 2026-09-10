# Conditions C and D under the original prompt

These runs used a prompt that described the first screenshot as
"the reference state (what the page looked like before the agent started)".

That description was wrong. The reference image is the final page state of a
*human demonstration that completed the task successfully*, extracted from the
VisualWebArena human Playwright traces (see scripts/extract_human_references.py).
The 147-instance visual subset is defined by human-trace availability: all 147
appear among the 233 extracted human references, and 13 of them do not exist in
the agent's first-frame set at all.

Condition C additionally asked for a verdict "based on the changes between the
reference and final state", which presumes a before/after pair rather than a
target to match.

The runs in the repository root supersede these: same data, same models, with the
prompt corrected to describe the reference as a successful human end state.
These files are retained so the effect of the correction can be measured.
