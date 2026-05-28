## Configuration in ReachX
A key design goal for ***ReachX*** was to eliminate the "configuration spaghetti" that plagued its predecessor: 
 - YAML configuration files were stored in multiple locations in the file system.
 - It was often difficult to determine what settings in these YAML files were really necessary, and to which
application component(s) they applied.
 - The user had to manually edit these YAML files in various usage scenarios, often introducing inadvertent errors
that sometimes crashed the application.
 - The old application was "installed" by cloning the relevant repository from the lab's GitHub, and at least two
YAML files inside the repository tree were modified by the user and read by the program at runtime.

### The application settings file
The ***ReachX*** approach to configuration is to minimize the amount of it and shield you
from the details of reading and writing configuration parameters. All application settings are persisted in a single
file located in the application's home directory at `$USER_HOME/.ReachX/settings.ini`. You need not be concerned with
its content. The first time it is launched after installation, ***ReachX*** will create a default
settings
file along with other application internals within the home directory. The entire contents are read at
launch and written at shutdown, preserving any configuration changes made during runtime. 

As of v0.7.0, the following application configuration data are persisted in `settings.ini`:
 - The size, geometry and layout of the application window(s). Thus, the next
time you run the program, it appears exactly as it did when it was last shutdown.
 - The most recently active application workspace (see next section), as well as the most recently loaded 
experiment session, and body part detection model from that workspace. Again, these are restored the next time 
***ReachX*** launches.
 - Workspace configuration parameters for every defined workspace.
 - User preferences explicitly set in the **Preferences** dialog (**File | Preferences...**). 

### Preferences Dialog
The **Preferences** dialog exposes two performance-related flags pertaining to the analysis of a session's videos to
infer per-frame body part locations using a previously trained  model:
 - **Use GPU for session analysis**. Always check this box for best performance. It is primarily 
for testing purposes, to assess the speed-up that GPU-supported inference provides.
 - **Analyze a session videos in parallel**. The only reason to uncheck this box is if your machine has a GPU with 
insufficient RAM to support simultaneous analysis of two videos in separate processes.

Future releases may add additional user preferences exposed in this dialog.

### Workspaces

A ***ReachX*** *workspace* is collection of file system paths that tell the application where to find the 
sessions, object detection models, and calibration information belonging to that workspace, as well a number of other
workspace-specific user preferences.

Users can create any number of workspaces to logically separate their research data and results into distinct "silos"
-- individual studies, experiments on different animal subjects, and so on.

As of v0.6.5, workspace settings include:
 - The *session root*: File system root directory under which all ***ReachX*** experiment sessions are stored.
 - The *model root*: File system root directory under which all body part detection models are stored.
 - The *calibration root*: File system root directory under which all fixed-cam calibration files are stored. This is
   unique to a fixed-cam workspace.
 - *Reach segmentation control parameters*: These govern the behavior of the reach segmentation algorithm that
   analyzes predicted hand and pellet trajectories to find and categorize reaches during a session.

The workspace concept is helpful when multiple individuals run ***ReachX*** on a shared 
workstation and use the same login credentials (a zombie or guest username and password) for that machine. In this
scenario, each user would create their own workspace(s) in ***ReachX*** to keep their work separate from that of the 
other users. (Note that this is not a perfect solution. The application window size/geometry/layout configuration 
are not stored "per workspace". Also, if someone else was last using the program, their workspace will be active when 
you launch the program. Thus, you'll need to switch to your own workspace and perhaps tweak the application layout
before continuing with your work.)
