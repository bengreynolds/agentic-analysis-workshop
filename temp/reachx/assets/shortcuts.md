## List of keyboard shortcuts supported in ReachX

### Navigating the session video timeline

 - **Right Arrow**: Increase forward playback speed (or reduce reverse playback speed), starting timed playback if
necessary.
 - **Left Arrow**: Decrease forward playback speed (or increase reverse playback speed), starting timed playback if
necessary.
 - **Space Bar**: Stop timed playback.
 - **Right Bracket (']')**: Step forward by the # of frames specified in the "step size" combo box.
 - **Left Bracket ('[')**: Step backward by the # of frames specified in the combo box. If timed playback is in 
progress, stepping forward or backward will stop playback before making the step.
 - **Backslash ('\\')**: Select the next step size in the combo box, with wrap-around. The supported step sizes are
1, 5, 10, 50, 100, and 200 frames.
 - **Up Arrow**: Jump forward to the next "reach epoch", defined as the frame number for the next "pellet delivery"
event in the session timeline. If video playback is in progress, playback continues after the jump.
 - **Down Arrow**: Jump back to the previous "reach epoch". If video playback is in progress, playback continues after
the jump.
 - **Page Up**: Go to the next "marked frame" in the "training data" for the current ReachX body part detection model. A
"marked frame" is a session video frame for which the user has specified the locations of one or more body parts. 
If no model is loaded or the current model has no defined training data, this accelerator has no effect. If video 
playback is in progress, playback continues after the jump.
 - **Page Down**: Go the previous "marked frame" in the "training data" for the current ReachX body part detection
model. If video playback is in progress, playback continues after the jump.

Note that all of the above are global application shortcut keys.

### Editing the set of curated reach segments for the current session
These hot keys are active only when the mouse cursor is *inside* the **TimeLine** widget; they are not global shortcut 
keys.
- The **A** key: Press this to toggle the widget between normal operation (crosshair-shaped cursor) and *add reach*
mode (pin-shaped cursor). Note that ReachX remembers which mode was in effect when the mouse leaves the widget and will
return to that mode when the mouse reenters.
- The **C** key: Add a scored reach to the curated set (if vertical line currently intersects a scored reach).
- The **E** key: Raise modal dialog to change non-keyframe parameters of a curated reach.
- The **X**, **Delete**, or **Backspace** key: Delete the curated reach intersected by the vertical line cursor.

### Trajectory Analysis Window
- **Comma (',')** When **Playback** is enabled, press this key to advance the displayed reaches to the next frame.
- **Period ('.')**: When **Playback** is enabled, press this key to rewind displayed reaches to the previous frame.