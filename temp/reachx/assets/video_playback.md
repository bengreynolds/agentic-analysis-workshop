## Reviewing session video

### Loading a session
To load an experiment session into ***ReachX***, click on the pushbutton at the top of the **Session View**. Assuming 
the active workspace has some sessions in it, a modal dialog appears with a tree-like control for selecting a 
session. The sessions are organized first by rig, then recording date, then session number. 

Once you select a session, click **OK** to extinguish the dialog. Observe that the session selector button's title
reflects the recording date, session number, and rig for the session you selected. The session's primary camera
videos are loaded, along with session events and any defined reach segments if the session has been previously
analyzed. 

In the **Main Cameras View**, the first frame (frame 0) from each video -- left/right for a fixed-cam session, and 
side/front for a legacy-cam session -- is displayed in the side-by-side **CamFrame** widgets. The frame number indicator
centered near the bottom of the **TimeLine** widget should read "0".

If a session fails to load, be sure to open the **Log Messages View** to check for any error messages.

#### NOTE: Different camera frame image size for fixed-cam vs legacy-cam sessions
If you happen to switch from a legacy-cam to a fixed-cam workspace or vice versa, bear in mind that the frame size
is 320x200 for a legacy-cam video and 256x256 for a fixed-cam video. You may need to adjust the size of the **Main
Cameras View** to ensure you're seeing the entire frame when you make the switch.

### Using playback and other controls in the Main Cameras View

Underneath the **TimeLine** widget is a single line of playback controls, from left to right:
 - *Reverse*: Increase reverse playback speed (or reduce speed during forward playback).
 - *Stop*: Stop playback.
 - *Forward*: Increase forward playback speed (or reduce speed during reverse playback).
 - *Playback speed readout*: Displays the current playback speed or "\[stopped\]".
 - *Slider*: Drag the thumb to quickly pan forward or backward to any point in the session timeline.
 - *Current frame number readout*
 - *Step back*: Jump back in the session timeline in accordance with the current step size.
 - *Step size selector*: A combo box to select the step size: 1, 5, 10, 50, 100, or 200 frames per step.
 - *Step forward*: Jump forward in the session timeline.

Try out all of these controls and observe how the contents of the **CamFrame** and **TimeLine** widgets are updated.
As you get used to controlling playback, you'll likely prefer keyboard accelerators to pressing button controls with
the mouse -- see the *Keyboard Shortcuts* chapter for a full list of the relevant "hot keys". Alternatively, hover the
mouse over a control until the tooltip appears; the tooltip description includes the associated hot key.

Below the line of playback controls is a second line of widgets for configuring the appearance of the **TimeLine** 
and **CamFrame** widgets.
 - Uncheck the **Show** checkbox if you wish to hide the **TimeLine**.
 - Check the **Legend** checkbox to display a legend of the colored markers representing the different types of session
events. The events and their timestamps are listed in full in the **Events** tab of the **Session View**.
 - Use the combo box to set the +/- span of the TimeLine in frames (50, 100, 200, 500, 1000, or 5000).
 - Check the **Show Marker Labels** checkbox to display body part nicknames next to the corresponding body part markers
that may appear on top of the image in each **CamFrame**. You probably don't want to leave it checked, as the labels
tend to "get in the way".
 - Once a session has been analyzed with a body part detection model, predicted body part locations (those with a
confidence score > 0.9) are available for display over the images in the **CamFrame** widgets. Check the 
**Predictions** box to see all predicted body part locations. If you're only interested in predicted locations for the 
right hand and food pellet, also check the **Hand/pellet only** box.

Finally, the left and right arrow buttons labeled **Reach Epochs** let you jump forward to the next "pellet delivery"
event, or jump backward to the previous pellet delivery. These may come in handy when you just want to focus on finding
reach segments in each reach epoch.

### Navigate to a particular session event or reach segment in the Session View

Open the **Events** tab in **Session View**. It contains two side-by-side tables. The left-hand table indicates how many
events of each type occurred during the session. The right-hand table lists the event frames for all events (or one 
type of event, selected via the combo box) in chronological order. Click on a row in this second table, and observe 
that ReachX immediately
jumps to the corresponding frame number in the session timeline. The **CamFrame** and **TimeLine** widgets in the 
**Main Cameras View** are updated accordingly.

Similarly to jump to the start of a curated or auto-generated reach segment in the session, open the **Reaches** tab and
click on the row corresponding to the reach segment of interest.
