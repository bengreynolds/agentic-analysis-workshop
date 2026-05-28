"""Generate trajectory-speed figures from curated ReachX reaches.

This script is intentionally standalone. It reads ReachX output files but
does not import ReachX, modify ReachX source code, or edit session data.
"""

from pathlib import Path
import re
import sys
import tkinter as tk
from tkinter import filedialog

import matplotlib

matplotlib.use("Agg")

from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np


RESULT_NAMES = {
    2: "grabbed",
    3: "missed",
    4: "dropped",
    5: "stalled",
}

HAND_POS_NAMES = {
    0: "unspecified",
    1: "left",
    2: "right",
    3: "above",
    4: "below",
}

FIXED_TRAJ_COLUMNS = {
    "x": 0,
    "y": 1,
    "z": 2,
    "speed": 6,
}

LEGACY_TRAJ_COLUMNS = {
    "x": 6,
    "y": 1,
    "z": 3,
    "speed": 10,
}

POINTS_PER_FRAME_PAIR = 8


def main():
    print("ReachX curated trajectory-speed figure generator")
    session_folder = choose_session_folder()
    if session_folder is None:
        print("No session folder selected. Nothing to do.")
        return 1

    try:
        workspace_choice = ask_workspace_choice()
        workspace_format = resolve_workspace_format(session_folder, workspace_choice)
        reach_file = find_curated_reach_file(session_folder)
        reaches = load_curated_reaches(reach_file)
        scorer_folder = find_scorer_folder(session_folder, workspace_format)
        trajectory, speed_units = load_right_hand_trajectory(scorer_folder, workspace_format)
        plot_scope, selected_reaches = choose_reaches(reaches)
        plot_view = choose_plot_view()
        output_files = save_reach_figures(
            session_folder=session_folder,
            reach_file=reach_file,
            workspace_format=workspace_format,
            trajectory=trajectory,
            speed_units=speed_units,
            plot_scope=plot_scope,
            plot_view=plot_view,
            reaches=selected_reaches,
        )
    except SystemExit as exc:
        print(f"\nStopped for review: {exc}")
        return 1

    print("\nCreated figure files:")
    for output_file in output_files:
        print(f"  {output_file}")
    return 0


def choose_session_folder():
    root = tk.Tk()
    root.withdraw()
    root.update()
    selected = filedialog.askdirectory(title="Select a ReachX session folder")
    root.destroy()
    if not selected:
        return None
    return Path(selected)


def ask_workspace_choice():
    valid_choices = {
        "": "auto",
        "a": "auto",
        "auto": "auto",
        "f": "fixed",
        "fixed": "fixed",
        "fixed-cam": "fixed",
        "l": "legacy",
        "legacy": "legacy",
        "legacy-cam": "legacy",
    }
    print("\nWorkspace format options:")
    print("  auto   - detect fixed-cam or legacy when only one format is clear")
    print("  fixed  - require fixed-cam ReachX output")
    print("  legacy - require legacy-cam ReachX output")
    answer = input("Choose workspace format [auto]: ").strip().lower()
    if answer not in valid_choices:
        stop(f"Unknown workspace option: {answer}")
    return valid_choices[answer]


def resolve_workspace_format(session_folder, choice):
    detected = detect_workspace_formats(session_folder)
    if choice == "auto":
        if len(detected) == 1:
            workspace_format = detected[0]
            print(f"Auto-detected {workspace_format} workspace output.")
            return workspace_format
        if len(detected) == 0:
            stop(
                "Could not detect fixed-cam or legacy ReachX trajectory output. "
                "Expected one scorer folder with trajectories.npz for fixed-cam "
                "or hand.npy for legacy-cam."
            )
        stop(
            "Both fixed-cam and legacy trajectory outputs were detected. "
            "Rerun the script and choose fixed or legacy manually."
        )

    if choice not in detected:
        stop(
            f"Manual selection requested {choice}, but the selected session folder "
            f"has detected output formats: {', '.join(detected) if detected else 'none'}."
        )
    return choice


def detect_workspace_formats(session_folder):
    formats = []
    if list_scorer_folders(session_folder, "fixed"):
        formats.append("fixed")
    if list_scorer_folders(session_folder, "legacy"):
        formats.append("legacy")
    return formats


def list_scorer_folders(session_folder, workspace_format):
    if workspace_format == "fixed":
        return sorted(
            folder
            for folder in session_folder.iterdir()
            if folder.is_dir() and Path(folder, "trajectories.npz").is_file()
        )
    if workspace_format == "legacy":
        return sorted(
            folder
            for folder in session_folder.iterdir()
            if folder.is_dir() and Path(folder, "hand.npy").is_file()
        )
    stop(f"Unsupported workspace format: {workspace_format}")


def find_curated_reach_file(session_folder):
    candidates = []
    for path in session_folder.glob("*_reaches.txt"):
        if path.name == "detected_reaches.txt":
            continue
        if "detected" in path.name.lower():
            continue
        if path.is_file():
            candidates.append(path)

    if len(candidates) == 0:
        stop(
            "No session-level curated reach file was found. "
            "Expected exactly one *_reaches.txt file directly inside the session folder."
        )
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        stop(f"Multiple possible curated reach files were found: {names}")
    return candidates[0]


def load_curated_reaches(reach_file):
    reaches = []
    with reach_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            values = parse_reach_row(stripped, reach_file, line_number)
            reach = make_reach(values, reach_file, line_number)
            reaches.append(reach)

    if not reaches:
        stop(f"Curated reach file contains no reach rows: {reach_file}")

    reaches = sorted(reaches, key=lambda reach: reach["frame"])
    validate_no_overlaps(reaches, reach_file)
    for index, reach in enumerate(reaches, start=1):
        reach["index"] = index
    return reaches


def parse_reach_row(line, reach_file, line_number):
    parts = line.split()
    if len(parts) not in (4, 5):
        stop(
            f"{reach_file.name} line {line_number} has {len(parts)} columns; "
            "expected 4 or 5 integer columns."
        )
    try:
        values = [int(part) for part in parts]
    except ValueError:
        stop(f"{reach_file.name} line {line_number} contains a non-integer value.")
    if len(values) == 4:
        values.append(0)
    return values


def make_reach(values, reach_file, line_number):
    frame, max_delta, duration, result, hand_pos = values
    if frame < 0:
        stop(f"{reach_file.name} line {line_number} has a negative frame number.")
    if max_delta < 0:
        stop(f"{reach_file.name} line {line_number} has a negative max_delta.")
    if duration <= 0:
        stop(f"{reach_file.name} line {line_number} has a non-positive duration.")
    if max_delta > duration:
        stop(f"{reach_file.name} line {line_number} has max_delta greater than duration.")
    if result not in RESULT_NAMES:
        stop(
            f"{reach_file.name} line {line_number} has result {result}; "
            "expected a reach result code from 2 through 5."
        )
    if hand_pos not in HAND_POS_NAMES:
        stop(
            f"{reach_file.name} line {line_number} has hand_pos {hand_pos}; "
            "expected 0 through 4."
        )

    return {
        "frame": frame,
        "max_delta": max_delta,
        "duration": duration,
        "result": result,
        "hand_pos": hand_pos,
        "line_number": line_number,
    }


def validate_no_overlaps(reaches, reach_file):
    previous = None
    for reach in reaches:
        if previous is not None and reach["frame"] <= reach_end_frame(previous):
            stop(
                f"Curated reaches overlap in {reach_file.name}: "
                f"line {previous['line_number']} and line {reach['line_number']}."
            )
        previous = reach


def find_scorer_folder(session_folder, workspace_format):
    candidates = list_scorer_folders(session_folder, workspace_format)
    if len(candidates) == 0:
        stop(f"No {workspace_format} scorer folder with trajectory output was found.")
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        stop(
            f"Multiple {workspace_format} scorer folders with trajectory output were found: {names}. "
            "Choose a session folder with one clear trajectory output or remove ambiguity before running."
        )
    return candidates[0]


def load_right_hand_trajectory(scorer_folder, workspace_format):
    if workspace_format == "fixed":
        return load_fixed_cam_trajectory(scorer_folder)
    if workspace_format == "legacy":
        return load_legacy_cam_trajectory(scorer_folder)
    stop(f"Unsupported workspace format: {workspace_format}")


def load_fixed_cam_trajectory(scorer_folder):
    trajectory_file = Path(scorer_folder, "trajectories.npz")
    try:
        with np.load(trajectory_file) as data:
            if "R_Hand" not in data:
                stop(f"{trajectory_file.name} does not contain the fixed-cam R_Hand trajectory.")
            raw = data["R_Hand"]
            validate_raw_trajectory(raw, 7, trajectory_file)
            trajectory = raw[:, [
                FIXED_TRAJ_COLUMNS["x"],
                FIXED_TRAJ_COLUMNS["y"],
                FIXED_TRAJ_COLUMNS["z"],
                FIXED_TRAJ_COLUMNS["speed"],
            ]]
    except OSError as exc:
        stop(f"Could not load {trajectory_file}: {exc}")

    validate_plot_trajectory(trajectory, trajectory_file)
    return trajectory, "mm/ms"


def load_legacy_cam_trajectory(scorer_folder):
    trajectory_file = Path(scorer_folder, "hand.npy")
    try:
        raw = np.load(trajectory_file)
    except OSError as exc:
        stop(f"Could not load {trajectory_file}: {exc}")

    validate_raw_trajectory(raw, 11, trajectory_file)
    trajectory = raw[:, [
        LEGACY_TRAJ_COLUMNS["x"],
        LEGACY_TRAJ_COLUMNS["y"],
        LEGACY_TRAJ_COLUMNS["z"],
        LEGACY_TRAJ_COLUMNS["speed"],
    ]]
    validate_plot_trajectory(trajectory, trajectory_file)
    return trajectory, "mm/s"


def validate_raw_trajectory(raw, expected_columns, trajectory_file):
    if raw.ndim != 2:
        stop(f"{trajectory_file.name} is not a 2D trajectory array.")
    if raw.shape[1] < expected_columns:
        stop(
            f"{trajectory_file.name} has {raw.shape[1]} columns; "
            f"expected at least {expected_columns}."
        )


def validate_plot_trajectory(trajectory, trajectory_file):
    if trajectory.shape[0] < 2:
        stop(f"{trajectory_file.name} has fewer than two trajectory frames.")
    if not np.isfinite(trajectory).all():
        stop(f"{trajectory_file.name} contains non-finite trajectory values.")


def choose_reaches(reaches):
    print(f"\nLoaded {len(reaches)} curated reaches.")
    print_reach_table(reaches)
    print("\nPlot options:")
    print("  one - plot one curated reach")
    print("  all - plot every curated reach")
    answer = input("Choose plot mode [one]: ").strip().lower()
    if answer in ("", "one", "1", "single"):
        return "single", [choose_one_reach(reaches)]
    if answer in ("all", "a"):
        return "all", reaches
    stop(f"Unknown plot mode: {answer}")


def choose_plot_view():
    valid_choices = {
        "": "2d",
        "2": "2d",
        "2d": "2d",
        "3": "3d",
        "3d": "3d",
    }
    print("\nPlot view options:")
    print("  2d - plot Y position vs Z position")
    print("  3d - plot X/Y/Z trajectory with axes shown")
    answer = input("Choose plot view [2d]: ").strip().lower()
    if answer not in valid_choices:
        stop(f"Unknown plot view: {answer}")
    return valid_choices[answer]


def print_reach_table(reaches):
    print("\nCurated reaches:")
    for index, reach in enumerate(reaches, start=1):
        print(
            f"  {index:>3}. frame={reach['frame']} "
            f"end={reach_end_frame(reach)} "
            f"result={RESULT_NAMES[reach['result']]} "
            f"hand_pos={HAND_POS_NAMES[reach['hand_pos']]}"
        )


def choose_one_reach(reaches):
    answer = input("Enter reach index or start frame: ").strip()
    if not answer:
        stop("No reach index or start frame was entered.")
    try:
        selected_value = int(answer)
    except ValueError:
        stop("Reach selection must be an integer index or start frame.")

    if 1 <= selected_value <= len(reaches):
        return reaches[selected_value - 1]

    matches = [reach for reach in reaches if reach["frame"] == selected_value]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        stop(f"More than one reach starts at frame {selected_value}.")
    stop(f"No curated reach matched index or start frame: {selected_value}")


def save_reach_figures(
        session_folder, reach_file, workspace_format, trajectory, speed_units, plot_scope, plot_view, reaches):
    output_dir = Path(__file__).resolve().parent / "figures"
    output_dir.mkdir(exist_ok=True)

    session_label = session_label_from_reach_file(reach_file, session_folder)
    for reach in reaches:
        validate_reach_with_trajectory(reach, trajectory)

    if plot_scope == "all":
        if plot_view == "3d":
            html_file = output_path_3d(output_dir, session_label, workspace_format, None)
            plot_reaches_3d_html(
                output_html=html_file,
                session_label=session_label,
                workspace_format=workspace_format,
                reaches=reaches,
                trajectory=trajectory,
                speed_units=speed_units,
            )
            return [html_file]

        png_file, pdf_file = output_paths_2d(output_dir, session_label, workspace_format, None)
        plot_reaches_2d_file(
            output_png=png_file,
            output_pdf=pdf_file,
            session_label=session_label,
            workspace_format=workspace_format,
            reaches=reaches,
            trajectory=trajectory,
            speed_units=speed_units,
        )
        return [png_file, pdf_file]

    output_files = []
    for reach in reaches:
        if plot_view == "3d":
            html_file = output_path_3d(output_dir, session_label, workspace_format, reach)
            plot_reaches_3d_html(
                output_html=html_file,
                session_label=session_label,
                workspace_format=workspace_format,
                reaches=[reach],
                trajectory=trajectory,
                speed_units=speed_units,
            )
            output_files.append(html_file)
        else:
            png_file, pdf_file = output_paths_2d(output_dir, session_label, workspace_format, reach)
            plot_reaches_2d_file(
                output_png=png_file,
                output_pdf=pdf_file,
                session_label=session_label,
                workspace_format=workspace_format,
                reaches=[reach],
                trajectory=trajectory,
                speed_units=speed_units,
            )
            output_files.extend([png_file, pdf_file])
    return output_files


def session_label_from_reach_file(reach_file, session_folder):
    suffix = "_reaches.txt"
    if reach_file.name.endswith(suffix):
        return reach_file.name[:-len(suffix)]
    return session_folder.name


def output_paths_2d(output_dir, session_label, workspace_format, reach):
    safe_session = safe_filename(session_label)
    if reach is None:
        stem = f"{safe_session}_{workspace_format}_all_curated_reaches_2d"
    else:
        stem = (
            f"{safe_session}_{workspace_format}_"
            f"reach{reach['index']:03d}_frame{reach['frame']}_2d"
        )
    return output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"


def output_path_3d(output_dir, session_label, workspace_format, reach):
    safe_session = safe_filename(session_label)
    if reach is None:
        stem = f"{safe_session}_{workspace_format}_all_curated_reaches_3d_interactive"
    else:
        stem = (
            f"{safe_session}_{workspace_format}_"
            f"reach{reach['index']:03d}_frame{reach['frame']}_3d_interactive"
        )
    return output_dir / f"{stem}.html"


def safe_filename(text):
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return cleaned.strip("_") or "reachx_session"


def validate_reach_with_trajectory(reach, trajectory):
    end_frame = reach_end_frame(reach)
    if end_frame >= trajectory.shape[0]:
        stop(
            f"Curated reach starting at frame {reach['frame']} ends at frame {end_frame}, "
            f"but the trajectory has only {trajectory.shape[0]} frames."
        )

    excerpt = trajectory[reach["frame"]:end_frame + 1, :]
    if excerpt.shape[0] < 2:
        stop(f"Curated reach starting at frame {reach['frame']} has fewer than two trajectory samples.")
    if not np.isfinite(excerpt).all():
        stop(f"Curated reach starting at frame {reach['frame']} contains non-finite trajectory values.")


def plot_reaches_2d_file(output_png, output_pdf, session_label, workspace_format, reaches, trajectory, speed_units):
    speed_norm = speed_normalizer_for_reaches(reaches, trajectory)
    fig, axis = plt.subplots(figsize=(7, 6))
    color_source = plot_reaches_2d(axis, reaches, trajectory, speed_norm)
    axis.set_title(build_title(session_label, workspace_format, reaches, "2d"))
    colorbar = fig.colorbar(color_source, ax=axis, shrink=0.8)
    colorbar.set_label(f"Speed ({speed_units})")
    axis.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_png, dpi=300)
    fig.savefig(output_pdf)
    plt.close(fig)


def plot_reaches_3d_html(output_html, session_label, workspace_format, reaches, trajectory, speed_units):
    try:
        import plotly.graph_objects as go
    except ImportError:
        stop("Plotly is required for live 3D plots. Install it with: conda install plotly")

    speed_min, speed_max = speed_range_for_reaches(reaches, trajectory)
    fig = go.Figure()
    for reach_index, reach in enumerate(reaches):
        excerpt = reach_excerpt(reach, trajectory)
        points, speeds = interpolate_path(excerpt[:, :3], excerpt[:, 3])
        line_settings = {
            "color": speeds,
            "colorscale": "Viridis",
            "cmin": speed_min,
            "cmax": speed_max,
            "width": 6,
            "showscale": reach_index == 0,
        }
        if reach_index == 0:
            line_settings["colorbar"] = {"title": f"Speed ({speed_units})"}
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="lines",
                name=reach_trace_name(reach),
                line=line_settings,
                hovertemplate=(
                    f"{reach_trace_name(reach)}<br>"
                    "X: %{x:.3f} mm<br>"
                    "Y: %{y:.3f} mm<br>"
                    "Z: %{z:.3f} mm<br>"
                    "Speed: %{customdata:.5f}<extra></extra>"
                ),
                customdata=speeds,
            )
        )
        add_plotly_endpoints(fig, reach, points, reach_index)

    fig.update_layout(
        title=build_title(session_label, workspace_format, reaches, "3d interactive"),
        scene={
            "xaxis_title": "X position (mm)",
            "yaxis_title": "Y position (mm)",
            "zaxis_title": "Z position (mm)",
            "aspectmode": "data",
        },
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
    )
    fig.write_html(
        str(output_html),
        include_plotlyjs="cdn",
        full_html=True,
        config={
            "displaylogo": False,
            "editable": True,
            "responsive": True,
            "scrollZoom": True,
        },
    )


def plot_reaches_2d(axis, reaches, trajectory, speed_norm):
    color_source = None
    for reach_index, reach in enumerate(reaches):
        excerpt = reach_excerpt(reach, trajectory)
        y_values = excerpt[:, 1]
        z_values = excerpt[:, 2]
        speed_values = excerpt[:, 3]
        segments, segment_speeds = smooth_2d_segments(y_values, z_values, speed_values)

        collection = LineCollection(segments, cmap="viridis", norm=speed_norm, linewidth=2.0)
        collection.set_array(segment_speeds)
        axis.add_collection(collection)
        add_2d_endpoints(axis, y_values, z_values, reach_index)
        color_source = collection

    axis.autoscale()
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xlabel("Y position (mm)")
    axis.set_ylabel("Z position (mm)")
    return color_source


def add_2d_endpoints(axis, y_values, z_values, reach_index):
    start_label = "start" if reach_index == 0 else None
    end_label = "end" if reach_index == 0 else None
    axis.scatter(y_values[0], z_values[0], color="black", s=24, marker="o", label=start_label, zorder=3)
    axis.scatter(y_values[-1], z_values[-1], color="black", s=24, marker="s", label=end_label, zorder=3)


def add_plotly_endpoints(fig, reach, points, reach_index):
    show_legend = reach_index == 0
    fig.add_trace(
        {
            "type": "scatter3d",
            "mode": "markers",
            "x": [points[0, 0]],
            "y": [points[0, 1]],
            "z": [points[0, 2]],
            "name": "start",
            "showlegend": show_legend,
            "marker": {"color": "black", "size": 4, "symbol": "circle"},
            "hovertemplate": f"{reach_trace_name(reach)} start<extra></extra>",
        }
    )
    fig.add_trace(
        {
            "type": "scatter3d",
            "mode": "markers",
            "x": [points[-1, 0]],
            "y": [points[-1, 1]],
            "z": [points[-1, 2]],
            "name": "end",
            "showlegend": show_legend,
            "marker": {"color": "black", "size": 4, "symbol": "square"},
            "hovertemplate": f"{reach_trace_name(reach)} end<extra></extra>",
        }
    )


def smooth_2d_segments(horizontal_values, vertical_values, speed_values):
    points, speeds = interpolate_path(
        np.column_stack([horizontal_values, vertical_values]),
        speed_values,
    )
    segments = np.stack([points[:-1], points[1:]], axis=1)
    segment_speeds = (speeds[:-1] + speeds[1:]) / 2.0
    return segments, segment_speeds


def interpolate_path(points, speeds):
    smooth_points = []
    smooth_speeds = []
    for index in range(points.shape[0] - 1):
        for step in range(POINTS_PER_FRAME_PAIR):
            fraction = step / POINTS_PER_FRAME_PAIR
            smooth_points.append(points[index] + fraction * (points[index + 1] - points[index]))
            smooth_speeds.append(speeds[index] + fraction * (speeds[index + 1] - speeds[index]))
    smooth_points.append(points[-1])
    smooth_speeds.append(speeds[-1])
    return np.asarray(smooth_points), np.asarray(smooth_speeds)


def speed_normalizer_for_reaches(reaches, trajectory):
    speed_min, speed_max = speed_range_for_reaches(reaches, trajectory)
    return plt.Normalize(vmin=speed_min, vmax=speed_max)


def speed_range_for_reaches(reaches, trajectory):
    speeds = []
    for reach in reaches:
        speeds.append(reach_excerpt(reach, trajectory)[:, 3])
    all_speeds = np.concatenate(speeds)
    speed_min = float(np.min(all_speeds))
    speed_max = float(np.max(all_speeds))
    if speed_min == speed_max:
        speed_max = speed_min + 1.0
    return speed_min, speed_max


def reach_excerpt(reach, trajectory):
    start = reach["frame"]
    end = reach_end_frame(reach)
    return trajectory[start:end + 1, :]


def reach_trace_name(reach):
    return f"reach {reach['index']} frame {reach['frame']} ({RESULT_NAMES[reach['result']]})"


def build_title(session_label, workspace_format, reaches, plot_view):
    if len(reaches) == 1:
        reach = reaches[0]
        reach_text = f"reach {reach['index']} frame {reach['frame']} ({RESULT_NAMES[reach['result']]})"
    else:
        reach_text = f"all curated reaches (n={len(reaches)})"
    return f"{session_label} | {workspace_format} | {plot_view.upper()} | {reach_text}"


def reach_end_frame(reach):
    return reach["frame"] + reach["duration"]


def stop(message):
    raise SystemExit(message)


if __name__ == "__main__":
    sys.exit(main())
