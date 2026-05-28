# Agentic Coding Small Edits

This folder contains small, intentionally flawed Python examples for practicing basic Codex-assisted coding.

The goal is not to build a full project. The goal is to practice asking Codex for small, isolated edits that are easy to review.

## How to Use This Folder

1. Pick one exercise folder.
2. Read the `ISSUES.md` file for that exercise.
3. Open the source file listed in the exercise.
4. Give Codex one focused prompt.
5. Review the diff before asking for another change.

For best results, ask Codex to work on one file or one small problem at a time.

## Recommended Prompt Structure

Use prompts with these parts:

```text
Please work only in [file path].

Goal:
[State the specific change you want.]

Before editing:
[Ask Codex to briefly identify the relevant issue.]

Scope limits:
- Keep the change small and focused on the stated goal.
- Preserve the existing behavior unless the goal requires changing it.
- Avoid unrelated cleanup or restructuring.
- Do not add extra files, tools, or dependencies unless they are necessary.
- Keep the code easy for a beginner to review.

After editing:
[Ask Codex to summarize what changed, what file changed, and any assumptions.]
```

For optimization exercises, use this version of the scope limits:

```text
Scope limits:
- Keep the change small and focused on the stated goal.
- Preserve the intended analysis result.
- Avoid unrelated cleanup or restructuring.
- Do not add extra files, tools, or dependencies unless they are necessary.
- Keep the optimized code easy for a beginner to review.
```

## Exercises

| Folder | Main skill | Source file |
| --- | --- | --- |
| `01_readability` | Make ugly code readable | `ugly_analysis.py` |
| `02_syntax_imports` | Fix syntax and import errors across tiny files | `run_analysis.py`, `broken_helpers.py` |
| `03_variable_names` | Clean up variable naming conventions | `naming_cleanup.py` |
| `04_numpy_data_logic` | Fix a numpy data-shape logic bug | `numpy_shape_bug.py` |
| `05_numpy_speed` | Optimize a slow numpy loop | `slow_numpy_loop.py` |
| `06_matplotlib_subplots` | Combine repeated figures into subplots | `four_separate_figures.py` |

## What Codex Is Good At In These Exercises

- Making messy code easier to read.
- Fixing obvious syntax, import, and naming errors.
- Renaming variables consistently.
- Explaining simple numpy shape problems.
- Replacing slow loops with readable numpy operations.
- Reducing repeated plotting code.
- Making small edits while preserving the original goal.

## Suggested Workflow

Start by asking Codex to inspect only one file and make a plan. Then ask it to edit that file. Finally, ask it to explain what changed.

This keeps each change small enough for a beginner to review.
