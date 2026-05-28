# 03 Variable Names

## Starting Issues

- Variable names use mixed styles such as `Data`, `AvgResponse`, `finalOutput`, and `ITEM`.
- Some names are too vague, such as `x1` and `x2`.
- The code works, but inconsistent naming makes it harder to follow.
- The dictionary keys are output labels and should usually be preserved unless the user asks to change output format.

## Learning Goal

Practice asking Codex to rename variables consistently without changing the logic.

## Suggested Codex Prompt

```text
Please work only in agentic-coding-small-edits/03_variable_names/naming_cleanup.py.

Goal:
Clean up inconsistent variable names so the script follows normal Python snake_case style.

Before editing:
Briefly identify the naming issues that make the code harder to read.

Scope limits:
- Keep the change small and focused on the stated goal.
- Preserve the existing behavior unless the goal requires changing it.
- Avoid unrelated cleanup or restructuring.
- Do not add extra files, tools, or dependencies unless they are necessary.
- Keep the code easy for a beginner to review.

After editing:
Report the file changed, what names changed, whether behavior should be the same, and any assumptions.
```
