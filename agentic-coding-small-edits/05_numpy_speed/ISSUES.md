# 05 Numpy Speed

## Starting Issues

- The script loops over a numpy array one value at a time.
- It builds a Python list before converting back to a numpy array.
- The thresholding operation can be expressed directly with numpy.

## Learning Goal

Practice asking Codex to make one small speed improvement while keeping the analysis readable.

## Suggested Codex Prompt

```text
Please work only in agentic-coding-small-edits/05_numpy_speed/slow_numpy_loop.py.

Goal:
Optimize the slow loop by using numpy directly, while keeping the code readable.

Before editing:
Briefly identify why the current loop is slower than necessary for numpy data.

Scope limits:
- Keep the change small and focused on the stated goal.
- Preserve the intended analysis result.
- Avoid unrelated cleanup or restructuring.
- Do not add extra files, tools, or dependencies unless they are necessary.
- Keep the optimized code easy for a beginner to review.

After editing:
Report the file changed, what changed, and why the new version should be faster.
```
