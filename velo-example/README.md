# ReachX Trajectory-Speed Figures

This small standalone project generates trajectory-speed figures from manually curated ReachX reaches. It reads existing ReachX session outputs and saves new figure files locally. It does not edit ReachX source code, raw data, or ReachX output files.

## Files

- `reachx_trajectory_speed_figures.py`: standalone script.
- `PLANNING.md`: implementation plan, assumptions, risks, and validation notes.
- `AGENTS.md`: Codex operating rules for this project.
- `figures/`: created by the script when figures are saved.
- `temp/`: ignored reference-only ReachX source/output folder.

## Setup

Create a conda env before running the script:

```powershell
conda create -n reachx-figures python=3.11 numpy matplotlib plotly
conda activate reachx-figures
```

If your conda installation does not include Tk support, install it too:

```powershell
conda install tk
```

## Run

From this project folder:

```powershell
python reachx_trajectory_speed_figures.py
```

The script opens a folder picker. Select one ReachX session folder, not the whole workspace root.

Example reference session folders in this repository are:

```text
temp/sessions/reachx/fixedcam/session001
temp/sessions/reachx/legacycam/session003
```

## Session Picker And Workspace Options

After selecting a session folder, the script asks for the workspace format:

- `auto`: use this when the selected session has only one clear fixed-cam or legacy trajectory output.
- `fixed`: require fixed-cam output.
- `legacy`: require legacy-cam output.

Auto-detection is conservative. If both fixed-cam and legacy markers are present, or if no known trajectory output is present, the script stops and reports the issue.

## Curated Reaches Only

The script reads exactly one session-level curated reach file:

```text
<session_prefix>_reaches.txt
```

It does not read scorer-level `detected_reaches.txt` files. If the curated reach file is missing, ambiguous, malformed, overlapping, or out of range for the trajectory data, the script stops instead of falling back to algorithm-scored reaches.

## Trajectory Sources

Fixed-cam sessions:

- Expected trajectory file: `<scorer_folder>/trajectories.npz`
- Required key: `R_Hand`
- X column: `0`, in mm
- Y column: `1`, in mm
- Z column: `2`, in mm
- Speed column: `6`, filtered speed in mm/ms

Legacy-cam sessions:

- Expected trajectory file: `<scorer_folder>/hand.npy`
- X column: `6`, filtered X in mm
- Y column: `1`, filtered Y in mm
- Z column: `3`, filtered Z in mm
- Speed column: `10`, filtered speed in mm/s

If more than one scorer folder has the required trajectory output, the script stops because it cannot safely guess which scorer to use.

## Single-Reach And All-Reach Plotting

After loading curated reaches, the script prints a numbered table with each reach start frame, end frame, result, and hand position.

Choose:

- `one`: plot one curated reach. Enter either the displayed 1-based reach index or the reach start frame.
- `all`: plot every curated reach from the selected session together on one combined figure.

After choosing one or all, choose the plot view:

- `2d`: plot Y position on the horizontal axis and Z position on the vertical axis.
- `3d`: create a live Plotly X/Y/Z figure that can be rotated, panned, zoomed, and edited in the browser.

## Outputs

Figures are saved in:

```text
figures/
```

2D single-reach mode produces one PNG and one PDF for the selected reach. 2D all-reach mode produces one combined PNG and one combined PDF.

- PNG
- PDF

3D mode produces one interactive HTML file:

- HTML

Each figure includes:

- Right-hand trajectory.
- Speed color-coded along the path with a continuous heatmap-style gradient.
- Labeled speed colorbar.
- Axis labels with units.
- Session, workspace format, plot view, and reach information in the title.

Open the 3D HTML output in a browser to rotate the view, pan, zoom, and edit the plot interactively.

## Validation Checks

The script performs these checks before saving figures:

- Exactly one curated `*_reaches.txt` file is present directly in the selected session folder.
- No scorer-level `detected_reaches.txt` file is used.
- Reach rows have 4 or 5 integer columns.
- Reach result codes are valid ReachX result codes.
- Reach frame ranges are valid and non-overlapping.
- Exactly one matching trajectory scorer folder is present.
- Trajectory arrays have the expected shape and finite values.
- Each selected reach fits inside the trajectory frame range.

## Assumptions

- A ReachX session folder contains one curated session-level reach file.
- A session has one clear scorer folder to use for trajectory data.
- ReachX curated reach files do not contain a separate stable reach ID, so this script uses reach order and start frame.
- The right-hand trajectory is the desired reach trajectory.
- Speed is taken from ReachX output and is not recomputed.

## Troubleshooting

`No session-level curated reach file was found`

Select the actual ReachX session folder that contains `<session_prefix>_reaches.txt`.

`Multiple possible curated reach files were found`

The session folder has more than one possible curated reach file. Review the folder and remove ambiguity before running.

`Both fixed-cam and legacy trajectory outputs were detected`

Rerun the script and choose `fixed` or `legacy` manually.

`Multiple scorer folders with trajectory output were found`

The script cannot safely choose a scorer. Use a session folder with one clear trajectory output or move unrelated scorer outputs out of the selected folder.

`Curated reach ... ends at frame ... but the trajectory has only ... frames`

The curated reach file and trajectory output do not match the same session output. Review the selected folder before plotting.

`Plotly is required for live 3D plots`

Install Plotly in the active environment:

```powershell
conda install plotly
```
