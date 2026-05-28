## The ReachX User Interface

The ***ReachX*** main application window has a customizable layout. The primary view is anchored to the top-left corner 
of the window. It shows the current frame from the two primary cameras in the experiment rig, along with a custom 
widget depicting session events and reach segments (if any) in the vicinity of the current frame number. 
This view is fundamental to the application, so it cannot be undocked or hidden. Most other user interface elements are 
housed in dockable, reconfigurable views.

Each of these dockable views may be "floated" as a top-level window, docked to the right or bottom edge of the main 
application window, or hidden entirely. In a multi-monitor setup, you should be able to move any top-level window onto a 
second screen. You can also organize several docked views within a single tabbed panel. ***ReachX*** persists the current
layout (size, docking location, visibility) of the main window and its views in a user preferences file on shutdown and 
restores that layout the next time you launch the application.

When docked, each view includes a title bar with two pushbuttons -- to either hide or undock the view. Once hidden,
you can show the view by checking the corresponding item in the **View** menu. To move a docked view, "grab" its title
bar and drag it to a new location. As you drag the view around the main window, transient animations hint at where the
view will end up if you release the mouse (it may take a little practice to get the hang of it). Drag the view on top of
another docked view to create a tabbed panel containing both views. 

Individual docked views are separated from each other and the primary view by horizontal or vertical "splitters".
Grab and drag the splitter to change how much horizontal or vertical space gets allocated to the docked components.

### Main Cameras View
The primary view houses two side-by-side `CamFrame` widgets displaying the current video frame on the *left* and *right* 
cameras for a fixed-cam session, or the *side* and *front* cameras for a legacy-cam session. The widgets are
interactive, allowing the user to place body part markers on selected video frames -- this is how training data is
specified for an object detection model. The model, in turn, is used to analyze session videos, generating predicted
locations (with a confidence score) of the individual body parts included in the model's training set. Those predicted
locations can also be displayed on the two `CamFrame` widgets when an already analyzed session is loaded into 
***ReachX***.

For details on how to navigate to any frame in the session videos, see the chapter on *Reviewing Session Video*. For 
instructions on how to use the interactive `CamFrame` to define the training data for a body part detection model, 
see the *Modeling* chapter.

Below the camera frames is another interactive plot, the `TimeLine` widget. It shows any session events and any defined
reach segments within an interval around the current frame. This widget is essential when reviewing the results of 
model-based reach segmentation for a session, and when manually adding, editing or deleting reach segments. For details,
see the chapter on *Curating Reaches*.

### Workspaces View
Use this view to create and manage your ***ReachX*** workspaces. A workspace is simply a collection of key configuration
settings that tell the program where to find the experiment sessions, model files, and calibration information that 
"belong" to that workspace. You can define any number of workspaces -- a convenient way to silo your work: a distinct 
workspace for each research study or for each test subject, for example. 

Press the **New** button to create a new workspace. Specify a unique name and the workspace type: *fixed-cam* or 
*legacy-cam*. By design, the program does not allow "mixing" of sessions recorded on the older legacy-cam rigs vs
the newer fixed-cam setups. This separation is enforced at the workspace level.

The name of the currently *active workspace* is displayed in the combo box. To switch to a different one, simply select 
its name from the dropdown combo box. The workspace's root directories for locating **Sessions**, **Models**, and 
**Calibrations** are shown. Edit any file path by clicking on the corresponding **Change** button. Note that calibration
information pertains only to fixed-cam workspaces; this field is disabled for legacy-cam workspaces.

You can delete the active workspace by pressing the **Delete** button. This merely removes the workspace from 
***ReachX***; it does not destroy any of the workspace content located under the session, model, and calibration roots.
The "default" workspace always exists and cannot be deleted.

For more information on workspaces and other application settings in ***ReachX***, review the *Configuration* chapter.

### Session View
You will likely want to have this view visible at all times. Here is where you load an experiment session into 
***ReachX*** and review its timestamped event list and any defined reach segments. Launch tasks to analyze the current
session -- or a batch of sessions -- with a trained body part detection model and/or perform automated
reach segmentation based on projected hand and pellet trajectories.

For more details, see the chapters on *Reviewing Video* and *Modeling*.

### Model View
Here is where you create an object detection model to predict the per-frame locations of one or more "body parts" 
during a ***ReachX*** experiment session. You can create any number of models, each of which has one or more 
*iterations*. For each model iteration, you must annotate one or more session videos -- on both of the primary
cameras -- with a minimum number of marked locations for each kind of body part you wish to track with the model. The
collection of body part markers serves as the *training data set* for the model.

For details on creating, managing, and training your models, see the *Modeling* chapter.

### Trajectory Analysis Window
The **Trajectory Analysis** window offers interactive plotting of reach trajectories for the current loaded session. 
Unlike other ***ReachX*** views, it is housed in its own dedicated top-level application window. See the *Trajectory 
Analysis* chapter for further details.

### Additional Cameras View
In a legacy-cam rig, video may be recorded from one or two additional cameras (*stimCam* and *fastCam*). If any
additional camera videos are found for the current session, their current frames are displayed in this view. Unlike the
`CamFrame` widgets in the **Main Cameras** view, these widgets are not interactive. You cannot place body part
markers on the videos displayed here.

The view is likely to be rarely used, especially as the lab transitions to the newer fixed-cam rigs. It is hidden by
default.

### Log Messages View
If something appears to go wrong while using ***ReachX***, be sure to open this view. It displays a straightforward log
message history; scroll to the bottom to see the most recent log messages. Examining the log history may offer a clue
about what happened.

Use the combo box above the message history to select the logging level. The default is **INFO**, the recommended
setting. Set it to **WARNING** if you want to see only warnings and error messages; set it to **DEBUG** to see every
log message. (Note that changing the log level does not filter what's already in the message history; it only changes
what level of messages get posted to the history going forward).

### Help View
If you're reading this in ***ReachX***, then you're looking at this view. It provides a small, in-app user guide for the
program. Use the combo box to select a chapter within the guide -- see the *Overview* chapter for a table of
contents.
