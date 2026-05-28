# 04 Numpy Data Logic

## Starting Issues

- `trial_data` is a Python list, but the function uses numpy-style slicing with two dimensions.
- The baseline has one value per trial, but subtracting it directly from a two-dimensional array will not align the way intended.
- The intended calculation is to subtract each trial's own baseline from every timepoint in that trial.

## Learning Goal

Practice asking Codex to explain and fix a small numpy shape or broadcasting problem.

## Suggested Codex Prompt

```text
Please inspect only agentic-coding-small-edits/04_numpy_data_logic/numpy_shape_bug.py.

Goal:
Fix the numpy data-shape problem so the script correctly baseline-subtracts each trial.

Before editing:
Explain the shape problem in plain language. Identify whether the issue is the data type, the array shape, the axis choice, or broadcasting.

Scope limits:
- Keep the change small and focused on the stated goal.
- Preserve the existing behavior unless the goal requires changing it.
- Avoid unrelated cleanup or restructuring.
- Do not add extra files, tools, or dependencies unless they are necessary.
- Keep the code easy for a beginner to review.

After editing:
Summarize what changed and why the numpy operation now matches the intended data shape.
```
