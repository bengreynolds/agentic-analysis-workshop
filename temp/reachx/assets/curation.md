## Reach segment curation in ReachX

### Curated versus scored reaches

If you've analyzed an experiment session with a trained body part detection model and run ReachX's automatic 
segmentation algorithm on it, the set of auto-generated reach segments are ***scored*** reaches; the model used to
produce them is the "scorer". Any given session could have more than one set of scored reaches; one for each unique
model iteration used to analyze that session.

The ***curated*** set of reach segments are those reaches you explicitly assign to the session; it is the curated set
that is typically used in downstream analysis. You may choose to ignore any scored reaches and define the curated set 
manually, add all scored reaches to the curated set then edit them to your liking, or some combination of these two 
approaches.

The **TimeLine** widget graphically displays both the curated and scored sets of reaches. In both cases, a filled
horizontal bar spans the duration of the reach segment, with a vertical line marking the location of the *reachMax*
keyframe. The fill color indicates the result of the reach: *grabbed* (green), 
*missed* (yellow-brown), *dropped* (orange), or *stalled* (red). The two sets of reaches are distinguised by their
position -- the scored reaches appear above the curated set --, and the bar outline color -- blue for scored reaches
and white for curated ones.

### Manually defining reaches using the TimeLine widget

Load the desired experiment session. Navigate to each reach epoch in the session using the **Reach Epoch** push buttons; 
alternatively, pressing the **Up Arrow** key jumps forward to the next reach epoch (defined as the frame for the next 
"pellet delivery" event in the session timeline), while pressing **Down Arrow** jumps back to the previous epoch.

At this point, move the mouse cursor inside the **TimeLine** widget and press the **A** key to switch to *add reach*
mode. The vertical line following the mouse turns green, and the cursor switches from a crosshair to a pin shape. More
importantly, notice that the current frame follows the cursor -- as you pan back and forth within the time line, the
current frame indicator (a white arrow with frame number below) is attached to the green line, and the displayed
camera frame changes as you move the mouse. This interactive feature allows you to fine-tune the placement of a reach 
segment:

1. When you locate the frame at which the reach begins, left-click to mark that location. The vertical line cursor
turns yellow to indicate that you've begun defining the reach segment.
2. Now move the mouse to the right (forward in time) while you watch the camera views to locate the *reachMax* keyframe.
Observe that a yellow bar animates the reach segment being defined. Left-click again to mark *reachMax*; the vertical 
line turns blue, the yellow bar is now fixed, and a blue bar anchored at the *reachMax* keyframe now follows the cursor.
3. Move the mouse further forward, locate the reach segment's end, and left-click once more. The yellow-and-blue bar
animating the reach segment definition disappears and a modal dialog pops up so that you can specify two additional 
paramaters of the reach: the result code, and the hand's position relative to the pellet at the *reachMax* keyframe 
(*left*, *right*, *above*, *below*, or *unspecified*). Press **Cancel** to discard the newly defined reach, or **Ok** to
confirm your choices and add the reach segment to the session's ***curated*** set. In the latter case, the new reach 
segment will now appear in the **TimeLine** widget -- as well as in the list of curated reaches in the **SessionView**.

If you move the mouse outside the time line bounds while in the process of defining a reach segment, the partially
defined reach is discarded.

### Curating results from model-based reach segmenation

If you have a set of scored reaches for a session, you may elect to add any number of these to the session's curated
set, then edit the curated reaches as described in the next section (scored reaches are read-only; they cannot be
changed). 

To add the entire scored set to the curated set, go to the **Reaches** tab in the **SessionView** panel. Press the 
plus-sign icon button above the auto-generated list of reach segments. ReachX will add all of the scored reaches to 
the curated set (skipping any scored reach that overlaps an already existing reach in the curated set). 

Alternatively, you may want to use some of the scored reaches on a case-by-case basis. In this scenario, use the **Up**
and **Down** arrow keys as before to advance to each reach epoch and locate the scored reach within that epoch. To add 
that scored reach to the curated set, move the mouse into the **TimeLine** widget so that the vertical line cursor
intersects the horizontal bar representing the scored reach, then press the **C** key. The scored reach is added to
the curated set (unless it overlaps an existing reach).

### Using the TimeLine widget to edit or delete curated reaches

- To delete a curated reach, simply move the mouse cursor inside the **TimeLine** widget so that the vertical line 
intersects the horizontal bar for the reach segment, then press the **X**, **Delete**, or **Backspace** key.
- To change the result code for a curated reach -- or the pellet-relative hand position at the *reachMax* keyframe --,
move the vertical line cursor so that it intersects the target reach segment and press the **E** key. The modal dialog
described above appears; make the desired changes and press **Ok**.
- To adjust any of the three keyframes -- *reachStart*, *reachMax*, *reachEnd* -- that define a reach segment, move
the mouse inside the time line and press **A** to put it in *add reach* mode. Left-click close to the keyframe you 
wish to adjust; the yellow-and-blue horizontal bar reappears and the keyframe selected moves as you move the mouse. 
Move the cursor to the desired location and left-click again to set the new value for that keyframe.

### Summary of TimeLine widget "hot keys"
These are active only when the mouse cursor is inside the widget; they are not global shortcut keys.
- The **A** key: Press this to toggle the widget between normal operation (crosshair-shaped cursor) and *add reach*
mode (pin-shaped cursor). Note that ReachX remembers which mode was in effect when the mouse leaves the widget and will
return to that mode when the mouse reenters.
- The **C** key: Add a scored reach to the curated set (if vertical line currently intersects a scored reach).
- The **E** key: Raise modal dialog to change non-keyframe parameters of a curated reach.
- The **X**, **Delete**, or **Backspace** key: Delete the curated reach intersected by the vertical line cursor.

