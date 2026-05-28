# Example Prompts For Codex

These prompts are intentionally more detailed than a normal chat message. They teach Codex the file to edit, the goal, the scope limits, and the expected summary.

## Planning Only Prompt

```text
Please inspect only [file path].

Goal:
I want to understand what small code issue should be fixed first.

Before making any changes:
- Identify the main issue in the file.
- Suggest the smallest useful edit.
- Tell me whether the edit should preserve behavior or intentionally change behavior.

Scope limits:
- Do not edit files yet.
- Keep the suggestion focused on this file only.
- Avoid unrelated cleanup or restructuring.
- Keep the explanation beginner-friendly.

After inspection:
Report back with a short plan and wait for my approval before editing.
```

## Basic Editing Prompt Template

```text
Please work only in [file path].

Goal:
[Describe the one specific change you want.]

Before editing:
Briefly identify the relevant issue you see in the code.

Scope limits:
- Keep the change small and focused on the stated goal.
- Preserve the existing behavior unless the goal requires changing it.
- Avoid unrelated cleanup or restructuring.
- Do not add extra files, tools, or dependencies unless they are necessary.
- Keep the code easy for a beginner to review.

After editing:
Report the file changed, what changed, whether behavior should be the same, and any assumptions.
```

## 01 Readability Prompt

```text
Please work only in agentic-coding-small-edits/01_readability/ugly_analysis.py.

Goal:
Make this script easier for a beginner to read without changing the analysis result.

Before editing:
Briefly identify the main readability issues in the file.

Requested edit:
- Rename unclear variables to descriptive names.
- Improve spacing and formatting.
- Add short comments for the major analysis steps if helpful.
- Keep the script simple and readable from top to bottom.

Scope limits:
- Keep the change small and focused on the stated goal.
- Preserve the existing behavior unless the goal requires changing it.
- Avoid unrelated cleanup or restructuring.
- Do not add extra files, tools, or dependencies unless they are necessary.
- Keep the code easy for a beginner to review.

After editing:
Report the file changed, what changed, whether the printed results should be the same, and any assumptions.
```

## 02 Syntax And Imports Prompt

```text
Please inspect only these files:
- agentic-coding-small-edits/02_syntax_imports/run_analysis.py
- agentic-coding-small-edits/02_syntax_imports/broken_helpers.py

Goal:
Fix the obvious syntax and import errors needed for the analysis script to run.

Before editing:
Briefly list the errors you see and which file each one is in.

Requested edit:
- Fix missing imports.
- Fix the mismatched helper function name.
- Fix the syntax error in the loop.
- Keep the existing toy analysis structure.

Scope limits:
- Keep the change small and focused on the stated goal.
- Preserve the existing behavior unless the goal requires changing it.
- Avoid unrelated cleanup or restructuring.
- Do not add extra files, tools, or dependencies unless they are necessary.
- Keep the code easy for a beginner to review.

After editing:
Report the files changed, what changed, and the command I could run if I want to verify it myself.
```

## 03 Variable Names Prompt

```text
Please work only in agentic-coding-small-edits/03_variable_names/naming_cleanup.py.

Goal:
Clean up inconsistent variable names so the script follows normal Python snake_case style.

Before editing:
Briefly identify the naming issues that make the code harder to read.

Requested edit:
- Rename variables to clear snake_case names.
- Keep dictionary output keys unchanged unless changing them is necessary.
- Do not change the calculations.

Scope limits:
- Keep the change small and focused on the stated goal.
- Preserve the existing behavior unless the goal requires changing it.
- Avoid unrelated cleanup or restructuring.
- Do not add extra files, tools, or dependencies unless they are necessary.
- Keep the code easy for a beginner to review.

After editing:
Report the file changed, what names changed, whether behavior should be the same, and any assumptions.
```

## 04 Numpy Data Logic Prompt

```text
Please inspect only agentic-coding-small-edits/04_numpy_data_logic/numpy_shape_bug.py.

Goal:
Fix the numpy data-shape problem so the script correctly baseline-subtracts each trial.

Before editing:
Explain the shape problem in plain language. Identify whether the issue is the data type, the array shape, the axis choice, or broadcasting.

Requested edit:
- Make sure the data passed into the numpy calculation is appropriate for numpy indexing.
- Fix the baseline subtraction so each trial is corrected by its own baseline.
- Preserve the intended per-trial output.

Scope limits:
- Keep the change small and focused on the stated goal.
- Preserve the existing behavior unless the goal requires changing it.
- Avoid unrelated cleanup or restructuring.
- Do not add extra files, tools, or dependencies unless they are necessary.
- Keep the code easy for a beginner to review.

After editing:
Summarize what changed and why the numpy operation now matches the intended data shape.
```

## 05 Numpy Speed Prompt

```text
Please work only in agentic-coding-small-edits/05_numpy_speed/slow_numpy_loop.py.

Goal:
Optimize the slow loop by using numpy directly, while keeping the code readable.

Before editing:
Briefly identify why the current loop is slower than necessary for numpy data.

Requested edit:
- Replace the manual loop with a clear numpy operation.
- Preserve the same thresholding logic.
- Keep the printed output meaning the same thing.

Scope limits:
- Keep the change small and focused on the stated goal.
- Preserve the intended analysis result.
- Avoid unrelated cleanup or restructuring.
- Do not add extra files, tools, or dependencies unless they are necessary.
- Keep the optimized code easy for a beginner to review.

After editing:
Report the file changed, what changed, and why the new version should be faster.
```

## 06 Matplotlib Subplots Prompt

```text
Please edit only agentic-coding-small-edits/06_matplotlib_subplots/four_separate_figures.py.

Goal:
Change the script from making four separate matplotlib figures to making one figure with four subplots.

Before editing:
Briefly identify the repeated plotting pattern in the current script.

Requested edit:
- Use a 2 by 2 subplot layout.
- Keep the same four signals and titles.
- Keep axis labels understandable.
- Use shared code only if it makes the script easier to read.

Scope limits:
- Keep the change small and focused on the stated goal.
- Preserve the existing behavior unless the goal requires changing it.
- Avoid unrelated cleanup or restructuring.
- Do not add extra files, tools, or dependencies unless they are necessary.
- Keep the code easy for a beginner to review.

After editing:
Report the file changed, what changed, and how I can run the script to view the combined figure.
```

## Review Prompt After Codex Edits

```text
Please review only the changes you just made.

Goal:
Check whether the edit stayed within the requested scope and preserved the intended behavior.

Review focus:
- Did you change only the requested file or files?
- Did the change solve the stated issue?
- Did you avoid unrelated cleanup or restructuring?
- Is the code still easy for a beginner to review?

After reviewing:
Give me a short summary of any remaining concerns. Do not make more edits unless I ask.
```

## Too Broad Prompt Rewritten As A Better Prompt

Too broad:

```text
Fix this folder and make all the code better.
```

Better:

```text
Please inspect only agentic-coding-small-edits/01_readability/ugly_analysis.py.

Goal:
Suggest one small readability improvement that would make this file easier for a beginner to understand.

Scope limits:
- Do not edit yet.
- Focus only on this file.
- Avoid unrelated cleanup or restructuring.
- Keep the recommendation beginner-friendly.

After inspection:
Tell me the smallest useful edit and wait for approval.
```
