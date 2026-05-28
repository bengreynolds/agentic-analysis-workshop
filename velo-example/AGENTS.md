# Codex Instructions for ReachX Trajectory-Speed Figures

## Scope

- Work only inside this standalone `example-project/` folder unless the user explicitly asks otherwise.
- Do not edit ReachX source code.
- Do not edit ReachX output files.
- Do not modify raw data files.
- Use `temp/` only as a read-only reference for ReachX source code and example outputs.
- Keep `temp/` ignored by git and untracked.

## Change Discipline

- Make minimal, bounded changes that directly satisfy the user request.
- Do not broaden scope into ReachX development, pipeline refactoring, or data cleanup.
- Do not perform broad refactors, formatting sweeps, dependency upgrades, or file reorganization.
- Preserve existing files and behavior unless the requested change requires otherwise.
- Prefer small in-place edits over full-file rewrites.

## Implementation Style

- Use plain Python.
- Keep code beginner-readable and easy to review.
- Prefer explicit variable names and step-by-step logic.
- Use short helper functions only when they reduce repeated logic.
- Do not use classes unless there is no simple alternative.
- Add comments only for major steps or non-obvious decisions.
- Avoid unnecessary dependencies.
- Do not introduce package structure unless it is required.

## ReachX Data Safety

- Load curated reaches only.
- Never fall back to algorithm-scored reaches.
- Stop and report the issue if curated data are missing, ambiguous, malformed, or inconsistent with the planned method.
- Stop for review if fixed-cam or legacy output structure is unclear.
- Do not silently infer data meaning when the source files do not provide enough evidence.

## Validation

- Use the smallest useful validation after changes.
- Confirm expected files exist.
- Confirm `temp/` remains untracked.
- Confirm raw data, ReachX source code, and ReachX outputs were not modified.
- Re-read changed logic for consistency with `PLANNING.md`.
- Do not add tests, temporary files, or generated outputs unless the user asks or they are necessary for the requested change.

## Failure Behavior

- Do not force completion around flawed assumptions.
- Do not create fake outputs or placeholder logic that appears complete.
- If required inputs, formats, columns, or curated-reach rules are unclear, pause and request user review.
