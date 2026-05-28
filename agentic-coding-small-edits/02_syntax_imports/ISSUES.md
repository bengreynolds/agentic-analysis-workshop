# 02 Syntax And Imports

## Starting Issues

- `broken_helpers.py` uses `np.random.normal(...)` without importing numpy.
- `run_analysis.py` imports a helper function name that does not exist.
- `run_analysis.py` calls the same incorrect function name.
- The `for` loop in `run_analysis.py` is missing required Python syntax.

## Learning Goal

Practice asking Codex to fix obvious syntax and import errors across a very small pair of files.

## Suggested Codex Prompt

```text
Please inspect only these files:
- agentic-coding-small-edits/02_syntax_imports/run_analysis.py
- agentic-coding-small-edits/02_syntax_imports/broken_helpers.py

Goal:
Fix the obvious syntax and import errors needed for the analysis script to run.

Before editing:
Briefly list the errors you see and which file each one is in.

Scope limits:
- Keep the change small and focused on the stated goal.
- Preserve the existing behavior unless the goal requires changing it.
- Avoid unrelated cleanup or restructuring.
- Do not add extra files, tools, or dependencies unless they are necessary.
- Keep the code easy for a beginner to review.

After editing:
Report the files changed, what changed, and the command I could run if I want to verify it myself.
```
