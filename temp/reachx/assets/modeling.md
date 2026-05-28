## Modeling and Session Analysis Tasks

A primary purpose for ***ReachX*** is to track the locations of various "body parts" and other reference features
during an experiment session, particularly the animal's right hand and food pellet.

To this end, the program uses a machine learning-based object detection model to predict the locations of body parts
in each of the two primary camera videos (left/right for a fixed-cam and side/front for a legacy-cam session). The
model must, of course, be trained on relevant data: a sampling of body part locations marked manually by the user on
one or more session videos. The trained model then can be applied to any number of sessions, predicting the location
of each body part in the training set during every frame in the recorded timeline. Each pair of 2D trajectories are 
then triangulated and transformed to form a single trajectory in the 3D rig space. The 3D body part 
trajectories, in turn, are used to find those segments in a session's timeline where the animal reaches for the
food pellet.

### Creating and training a body part detection model in Model View

To create a brand-new model project, open the **Model View** and click on the green "+" button to the right of the 
combo box at the top of the view. Specify a name for the model, 5-20 alphanumeric characters long, starting with a
letter. Duplicate model names are rejected. The new model is added to the active workspace.

Each model is really just a container for one or more numbered *iterations*. You might train an initial iteration, 
then analyze
a few sessions with the trained model, then decide to augment the training data set and retrain. Rather than modify the
first iteration, simply create a second one by pressing the green "+" button to the right of the combo box that
selects the current iteration. *When you create a new iteration, **ReachX** copies the training set of the current
iteration -- so you don't have to start all over.*

To prepare a training data set for the currently loaded model iteration, start by loading a representative 
session in **Session View**. 
Navigate to various frames in the session timeline, and use the mouse to place markers on the frame images displayed
in the camera frame widgets in the **Main Cameras View**. Center the mouse over the body part or feature you want to
"label" on the frame image, then right-click to raise a context menu. Select the appropriate body part label. The 
context menu disappears, and a colored symbol is placed at the location of the right-click. If you need to fine-tune
the position of an existing marker, left-click on it; the crosshair lines are hidden and the marker symbol is
highlighted. Use the up/down/left/right arrows to nudge the marker in the corresponding direction, and press `Enter` to
save the change and restore the crosshairs. Alternatively, delete the highlighted marker by pressing the
`X`, `Del`, or `Backspace` key. 

Repeat this process to define a sufficient number of "samples" for each type of body part you wish to track.
As you place body part markers on selected frames in one or more sessions, observe that the *Training Data** panel 
in the **Model View** is updated to reflect the 
markers you've added. The marked session(s) appears in the combo box above the left-hand table, which lists all the
individual markers attached to that session. The right-hand table is a summary showing how many markers have been placed
for each type of body part/feature, across all marked sessions.

To train the current model iteration, select the residual neural network model upon which the trained model will be
based, then press the **Train** button. That button will NOT be enabled until the training data set meets certain
minimum requirements:
 - At least 50 right-hand poses are marked.
 - At least 50 pellet markers have been placed.
 - For a fixed-cam model, at least 50 left-hand poses are marked.

Once the training data is sufficient, press the **Train** button. A modal blocking dialog is raised, and a training task
is launched in the background. As that task proceeds, progress messages are posted to the blocking dialog. The 
training task typically takes several hours, and users will typically run it overnight. If you wish, you can cancel
the task at any time by pressing the **Cancel** button. Once the job finishes, successfully or 
not, the dialog remains visible so that you can review the message history. 

You can delete the current model iteration or the entire model project by clicking on the appropriate red "trash can"
icon in **Model View**. Removing a model or one of its iterations is not a recoverable operation, and ***ReachX*** will
confirm your intent before removing the underlying files/folders.

### Analyzing sessions with a trained model

If the currently selected model iteration in **Model View** has been trained and a session is loaded in 
**Session View**, then
the buttons **Analyze this session** and **Multisession Analysis** are enabled. The former launches a background task
to analyze the current session using the current model iteration; the latter configures a "batch operation" to 
analyze a selected set of sessions and, optionally, perform automated reach segmentation on each successfully
analyzed session. As with model training, session analysis takes a while, so the modal dialog is raised to block input
to the main application window and report progress until the job has finished.

Once a session has been analyzed, predicted body part locations are displayed in the left/right (or side/front)
camera frame images in the **Main Cameras View**. (Be aware that the "predicted" body part marker symbols are 
translucent
rather than opaque and somewhat larger than the markers you placed manually when building the training set for a
model.) The computed 3D body part trajectories are stored in files within the session folder. If you need to load
these files to perform your own analyses, be sure to consult the *Session Files* chapter in this user guide.

When a previously analyzed session is loaded into **Session View**, the **Analyzed By** combo box displays the 
name of the
current "scorer", i.e., the particular model iteration that was used to analyze the session. You can analyze a 
session with any number of different model iterations; the analysis results for each scorer are stored separately. To
see the results for a different scorer, simply select its name from the combo box.

If you want to run the automated reach segmentation algorithm on a session using the results from a particular 
scorer, select the desired scorer from the combo box dropdown list, then press the **Find Reaches** button. Again, the
modal blocking dialog appears, but reach segmentation should only take a minute or less. The resulting "scored reaches"
are displayed in a table in the **Reaches** tab of the **Session View**, alongside a tabulation of any manually curated
reaches. For more information on reach segment curation, see the *Curating Reaches* chapter.



