# Example Prompts For The Agentic Worksheet

Use these prompts during the workshop to model effective agentic coding behavior. The structure follows the numbered cells in `example-worksheet/reachx_agentic_worksheet.ipynb`.

The presenter version can be more explicit than the learner-facing worksheet, but still avoid asking the agent to solve the whole notebook in one pass. The best workshop flow is: run one cell, inspect the failure, prompt the agent narrowly, review the patch, then continue.

## Always Start With This Boundary

```text
Work only inside example-worksheet/. Do not inspect or reference neighboring completed examples, solution notebooks, or presenter materials. Use the worksheet notebook and the errors from running cells as your context.
```

```text
Before editing, explain what the current cell appears to be trying to do and what the smallest useful change would be.
```

## Cell 1: Imports And Constants

This cell should run as-is. Use it to model a read-only orientation prompt.

```text
Read Cell 1 and summarize the constants and packages the notebook expects to use. Do not edit anything yet.
```

## Cell 2: User Settings

Goal: fix the setup values enough that later cells can run.

```text
Cell 2 has an intentionally incomplete setup value. Patch only the minimum needed so the notebook can continue. Keep the setting easy for a beginner to change later.
```

```text
Instead of hard-coding my machine path, make Cell 2 accept a clear user setting or a simple safe fallback. Stay inside this worksheet folder.
```

## Cell 3: Load Data

Goal: create the variables later cells expect. This can be done with real data or with a small demo dataset if real data are unavailable.

```text
Cell 3 is the first real missing step. Read the later cells to infer which variables it must create, then complete Cell 3 with simple, linear code.
```

```text
I do not have real data ready. Add a tiny demo dataset inside Cell 3 so the worksheet can be practiced end to end. Make the demo data obviously synthetic.
```

```text
I have real session data. Update Cell 3 so it loads from SESSION_FOLDER and fails clearly if the folder does not contain the expected inputs.
```

## Cell 4: Validate Curated Reach Rows

Goal: add the missing validation check without rewriting the cell.

```text
Cell 4 is mostly complete but is missing one validation check. Find the gap from the surrounding code and add only that check.
```

## Cell 5: Sort Reaches And Check For Overlap

This cell should usually be read-only after earlier fixes. Use it to show restraint.

```text
Review Cell 5 and tell me whether it needs an edit after the previous fixes. If it does not, say so and leave it unchanged.
```

## Cell 6: Validate The Trajectory Array

This cell should usually be read-only if Cell 3 created the right object.

```text
Cell 6 fails. Determine whether the problem belongs in Cell 6 or in the earlier data-loading cell. Patch the source of the problem, not the symptom.
```

## Cell 7: Summarize Loaded Data

Goal: add one useful checkpoint line.

```text
Cell 7 asks for one more summary line. Add a concise line that helps a user confirm the data loaded correctly.
```

## Cell 8: Add Reach Details

Goal: diagnose a subtle variable/key mismatch.

```text
Cell 8 throws an error even though the earlier reach table looks valid. Use the traceback to find the mismatch and patch only that line.
```

## Cell 9: Select Reaches To Plot

Goal: complete the single-reach branch.

```text
Cell 9 needs the single-reach selection branch finished. Implement it in the simplest readable way using the existing variables.
```

```text
Make Cell 9 support both a displayed reach index and a reach start frame, but keep the code beginner-readable.
```

## Cell 10: Extract Reach Segments

This cell should usually be read-only after Cell 9 is fixed.

```text
Review Cell 10 after Cell 9 works. Confirm whether it correctly checks frame ranges before slicing trajectory data.
```

## Cell 11: Calculate A Simple Reach Metric

Goal: calculate path length from the selected trajectory segment.

```text
Cell 11 is missing one simple metric calculation. Add the path-length calculation with explicit NumPy steps, not a clever one-liner.
```

## Cell 12: Plot Reach Paths

Goal: fix a plotting configuration issue, and optionally continue into the 3D path if that is part of the lesson.

```text
Cell 12 fails while building the 2D plot. Diagnose whether the error is caused by the data, the plotting configuration, or the plotting library call. Patch only the failing setting.
```

```text
Now switch PLOT_VIEW to 3d and run Cell 12. Fix the import error first, then identify the next intentionally unfinished part without solving unrelated cells.
```

## Cell 13: Save The Figure

Goal: improve the output filename while preserving the output folder.

```text
Cell 13 saves files, but the filename is too generic. Make the filename include enough context to identify the run without changing the output folder.
```

## Cell 14: Write Run Notes

Goal: produce a short Markdown note summarizing the run.

```text
Cell 14 is missing the run note text. Write a concise Markdown summary using variables already created by the notebook.
```

```text
Keep the run note brief: session, plot mode, selected reaches, metrics, and saved files.
```

## Final Review Prompts

```text
Review the worksheet after these edits. List which numbered cells were changed and which were intentionally left unchanged.
```

```text
Check that no files outside example-worksheet/ were changed.
```

```text
Explain where the user's prompt gave enough context and where you had to infer intent from notebook errors.
```
