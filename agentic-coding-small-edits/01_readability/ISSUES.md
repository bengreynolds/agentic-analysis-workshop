# 01 Readability

## Starting Issues

- Variable names like `x`, `y`, `z`, `a`, `b`, and `v` hide what the values mean.
- Spacing is cramped and inconsistent.
- Several loops are harder to read than necessary.
- The script works, but it is not beginner-friendly.

## Learning Goal

Practice asking Codex to make code more readable while preserving the same analysis result.

## Suggested Codex Prompt

```text
Please work only in agentic-coding-small-edits/01_readability/ugly_analysis.py.

Goal:
Make this script easier for a beginner to read without changing the analysis result.

Before editing:
Briefly identify the main readability issues in the file.

Scope limits:
- Keep the change small and focused on the stated goal.
- Preserve the existing behavior unless the goal requires changing it.
- Avoid unrelated cleanup or restructuring.
- Do not add extra files, tools, or dependencies unless they are necessary.
- Keep the code easy for a beginner to review.

After editing:
Report the file changed, what changed, whether the printed results should be the same, and any assumptions.
```
