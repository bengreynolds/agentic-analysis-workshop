# ReachX Trajectory-Speed Figure Project Plan

## Current Status

Implementation may proceed.

The ReachX reference files are available in `temp/`. The source documentation and example outputs show that curated reaches are stored in the session-level `<session_prefix>_reaches.txt` file, while algorithm-scored reaches are stored as `detected_reaches.txt` inside scorer folders. This project must read only the session-level curated reach file.

## Intended Scope

This standalone project will generate trajectory-speed figures from ReachX session outputs without modifying ReachX source code, raw data, or ReachX output files.

In scope:

- Open a file dialog so the user can select a ReachX session folder.
- Support both fixed-cam and legacy ReachX workspace/output formats after those formats are verified.
- Load curated reaches only.
- Allow plotting one selected curated reach or all curated reaches in a selected session.
- In all-reach mode, plot all curated reaches together on one combined figure.
- Support a 2D Y-vs-Z view and a live Plotly 3D X/Y/Z view with axes shown.
- Save 2D PNG/PDF outputs and 3D interactive HTML outputs to a local `figures/` folder inside this standalone project.
- Stop with a clear message if the workspace format, curated-reach source, or reach data are missing, ambiguous, or inconsistent.

Out of scope:

- Editing ReachX source code.
- Editing ReachX output files.
- Modifying raw data.
- Falling back to algorithm-scored reaches.
- Building a package, GUI application, or test suite.
- Refactoring any code outside this standalone example project.

## Intended File Structure

```text
example-project/
  AGENTS.md
  PLANNING.md
  README.md
  reachx_trajectory_speed_figures.py
  figures/
```

Notes:

- `figures/` will be created by the script when needed.
- No ReachX files should be copied into this project.
- `temp/` is only a reference location and must remain ignored by git.

## Workspace Detection Strategy

The script should start from the session folder selected in a file dialog.

Planned behavior:

1. Inspect the selected folder for known fixed-cam workspace markers.
2. Inspect the selected folder for known legacy workspace markers.
3. If exactly one workspace format is detected, use it automatically.
4. If both formats are detected in auto mode, stop and ask the user to rerun with manual selection.
5. If no known format is detected, stop and report the missing markers.
6. If multiple candidate curated-reach files or scorer trajectory folders exist for the same session, stop and ask for review instead of guessing.

Manual workspace selection should be available as a simple prompt after folder selection:

- auto-detect
- fixed-cam
- legacy

Auto-detection should only proceed when there is one clear match.

## Curated-Reach Loading Strategy

The script must use curated reaches only.

Verified loading behavior:

1. Locate exactly one session-level `*_reaches.txt` file directly inside the selected session folder.
2. Reject `detected_reaches.txt` and any reach file inside a scorer folder.
3. Load only curated reach rows from that session-level file.
4. Confirm each selected reach has enough trajectory information to plot x position, y position, and speed.
5. Stop if curated data are missing, ambiguous, malformed, overlapping, out of frame range, or inconsistent with the verified ReachX format.

ReachX reach files contain whitespace-separated integer rows with either four columns from older versions or five columns from current versions:

```text
frame  max_delta  dur  result  hand_pos
```

For older four-column rows, `hand_pos` is treated as unspecified (`0`).

The script must not:

- Read algorithm-scored reach files as a fallback.
- Infer curated status from filenames alone if ReachX has a more reliable marker.
- Mix curated and scored reach tables.
- Silently skip malformed curated reaches when plotting all reaches.

## Plotting Approach

Single-reach mode should show one reach trajectory. All-reach mode should show all curated reach trajectories together on one shared figure.

Planned figure content:

- 2D view: Y position on the x-axis and Z position on the y-axis.
- 3D view: live Plotly X, Y, and Z position with axes shown and browser controls for rotate, pan, zoom, and editing.
- Path color-coded by speed using a continuous heatmap-style gradient.
- Labeled speed colorbar with units.
- x-axis and y-axis labels with units.
- Title containing session identifier, workspace format, plot view, and reach identifier or all-reach count.

Planned output behavior:

- Save one PNG and one PDF for a single selected 2D reach.
- Save one PNG and one PDF for the combined 2D all-reach figure.
- Save one interactive HTML file for a selected 3D reach or combined 3D all-reach figure.
- Write outputs to `figures/`.
- Use clear filenames that include the session identifier, workspace format, reach identifier or all-reach marker, and plot view.
- Do not overwrite ReachX output files.

Verified trajectory sources:

- Fixed-cam: one scorer folder containing `trajectories.npz`; load key `R_Hand`; use columns `0` = X mm, `1` = Y mm, `2` = Z mm, and `6` = filtered speed in mm/ms.
- Legacy-cam: one scorer folder containing `hand.npy`; use columns `6` = filtered X mm, `1` = filtered Y mm, `3` = filtered Z mm, and `10` = filtered speed in mm/s per ReachX session file documentation.

The plotting stack is:

- `numpy` for numeric validation and speed/path arrays.
- `matplotlib` for 2D trajectory plotting and static PNG/PDF output.
- `plotly` for live 3D trajectory plotting and interactive HTML output.
- `tkinter` for the session folder picker and simple dialogs.

## Validation Checks

Before implementation resumes:

- Confirm `temp/` exists and is ignored by git. Completed.
- Inspect the fixed-cam ReachX source/output format. Completed.
- Inspect the legacy ReachX source/output format. Completed.
- Identify the authoritative curated-reach source for each format. Completed.
- Identify required trajectory columns/fields and their units. Completed.
- Identify how reach IDs or reach indices are represented. Completed; reach files have no separate ID, so the script uses the curated reach index and start frame.

After implementation:

- Confirm the expected project files exist.
- Confirm `temp/` remains untracked.
- Confirm no raw data, ReachX source code, or ReachX outputs were modified.
- Re-read the script and verify it follows this plan.
- Run only the smallest useful syntax or import check if local dependencies are available.

## Assumptions To Verify

These are now implementation assumptions:

- Fixed-cam and legacy outputs both contain a session-level curated-reach artifact named `*_reaches.txt`.
- ReachX does not store a separate stable reach ID, so the script identifies reaches by 1-based curated order and start frame.
- Curated reaches segment the right-hand trajectory.
- Speed is already present in ReachX trajectory output and should not be recomputed here.
- A selected session folder may not follow the full ReachX root/date/rig/session tree, so the script derives the session label from the curated reach filename when needed.

## Risks

- Curated reach data may be stored separately from trajectory samples, requiring a validated join strategy.
- Fixed-cam and legacy outputs may use different column names, file layouts, or units.
- Scored and curated outputs may have similar filenames, creating a risk of accidental scored-data use.
- Some sessions may contain both fixed-cam and legacy outputs, requiring manual selection.
- Some sessions may contain multiple scorer folders; the script will stop rather than guessing which trajectory output to use.

## Staged Implementation Steps

1. Obtain or restore the `temp/` reference folder. Completed.
2. Inspect ReachX source and example output files for fixed-cam and legacy formats. Completed.
3. Update this planning document with the verified curated-reach file locations, required columns, and units. Completed.
4. Implement workspace detection and manual selection.
5. Implement curated-reach loading with strict failure behavior.
6. Implement single-reach and all-reach selection flow.
7. Implement trajectory-speed plotting and PNG/PDF saving.
8. Add setup, usage, assumptions, validation, and troubleshooting documentation in `README.md`.
9. Perform the smallest useful validation described above.
