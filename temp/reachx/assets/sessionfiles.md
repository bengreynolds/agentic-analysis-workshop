## Reading session analysis results

While analyzing a session with a trained object detection model, ***ReachX*** stores analysis results in several files
within the session folder. You may wish to load some of these results to perform additional analysis of your own. This
chapter describes the folder contents of a fully analyzed experiment session, along with code snippets for loading and
using selected analysis results.

### Structure of the session tree

All sessions belonging to a named workspace are stored under a session root directory. ***ReachX*** finds all 
session folders under that directory conforming to the following structure:

```
<session_root>/
    20260409/ 
        rig_A/
            session001/
            session002/
            ...
        rig_B/
            session001/
            ....
        ...
    20251230/
    ...
```
<br></br>
Sessions are stored first by date recorded in the standard `YYYYMMDD` format, then by the name of the rig on which the
session was recorded, and then by session number. All files content pertaining to a particular session are found 
within the session folder `<root>/<YYYYMMDD>/rig/sessionNNN`.

### Contents of a session folder

Many of the files in the session folder are written by a separate application that controls the experiment and are read
when ***ReachX*** "loads" the session. These include the video camera recordings (MP4 files), associated timestamps 
files (ending in `_timestamps.txt`), the session events file (ending in `_events.txt`), and some YAML files that are 
generally ignored. 

***ReachX*** stores your manually curated reaches for the session in `<session_prefix>_reaches.txt`. All other
analysis results are placed in a subfolder named for the particular model iteration -- called 
the *scorer* -- on which the analysis was based. That way, ***ReachX*** maintains the analysis results for any number
of different scorers. Here are two examples, for a fixed-cam and a legacy-cam session. Only content created by 
***ReachX*** is shown.

#### Fixed-cam session
```
<root>/20260409/christielab09/session001/
    20260409_christielab09_session001_reaches.txt
    Rx_resnet50_fctestmodel1-20260324-fix_1_snapshot-1030000/
        detected_reaches.txt
        pelletHistory.pickle
        trajectories.npz
        detected_markers.npy
        left.npy
        right.npy
    ...
```
<br></br>
#### Legacy-cam session
```
<root>/20260409/christie2P/session001/
    20260409_christie2P_session001_reaches.txt
    Rx_resnet50_reachxtest_0_snapshot-1030000/
        detected_reaches.txt
        hand.npy
        pellet.npy
        detected_markers.npy
        frontCam.npy
        sideCam.npy
```
<br></br>
The files `left.npy` and `right.npy` contain the model-predicted body part locations during a fixed-cam session on the 
two primary videos; `sideCam.npy` and `frontCam.npy` for a legacy-cam session. Each file contains a single `np.float32` 
Numpy array of shape `(N*P, 5)`, for `N` frames and `P` 
body parts, where each row represents a single body part location prediction: `[frame_num, bp_value, x, y, likelihood]`.
Here `int(bp_value)` is an integer identifying the body part, `(x,y)` is the predicted location in 
pixels, and `likelihood` is a confidence score in \[0..1\].

The file `detected_markers.npy` is a compilation of all body part locations, across both
camera videos, for which `likelihood > 0.9`. ***ReachX*** uses this information to display high-confidence body part
location predictions in the **Main Cameras** view. The file contains a single Numpy array: `np.array((K, 6), 
dtype=np.int32)`, where each row = ``[frame_num, cam_value, bp_value, x, y, likelihood*10000]``. The likelihood value is
scaled to preserve 4 decimal digits when we round the array contents and cast to ``np.int32``.

***ReachX*** combines the model output on the two primary cameras to calculate body part trajectories in the cameras'
3D coordinate space. Here is where the analysis diverges for a fixed-cam vs legacy-cam session. 

For a legacy-cam session, ***ReachX*** only calculates 3D trajectories for the animal's right hand and the food pellet.
The trajectory data for the hand is saved as a single Numpy array `np.array((N, 11), dtype=np.float32)` in `hand.npy`. 
Here `N` is the number of frames in the session, and each column corresponds to a different trajectory parameter:
```
     0       Y-coordinate in millimeters
     1       Low-pass filtered version of Y
     2       Z-coordinate in millimeters
     3       Low-pass filtered version of Z
     4       YZ likelihood (confidence score on sideCam)
     5       X-coordinate in millimeters
     6       Low-pass filtered version of X
     7       X likelihood (confidence score on frontCam)
     8       Distance from system origin, in millimeters
     9       Speed in mm/second.
    10       Low-pass filtered version of speed.
```
<br></br>
Analogously for the pellet trajectory data, which is saved in `pellet.npy`.

For a fixed-cam session, ***ReachX*** uses FLIR camera calibration information to transform the predicted body part
locations on the left and right camera videos to trajectories in 3D space. All defined body part trajectories -- not 
just the right hand and food pellet -- are calculated and saved in `trajectories.npz`. Each body part's 3D trajectory
is stored as a *named* Numpy array -- `np.array((N, 7), dtype=np.float32)` -- where the array name matches the body 
part's nickname as it appears in ***ReachX***. Again, N is the number of frames recorded, and each column represents
a different trajectory parameter:
```
    0        X-coordinate (mm), after filtering, interpolation, triangulation, and recentering.
    1        Y-coordinate (mm), after filtering, interpolation, triangulation, and recentering.
    2        Z-coordinate (mm), after filtering, interpolation, triangulation, and recentering.
    3        P = Confidence flag per body part location: 1 for high-confidence, otherwise 0.
    4        D = Computed 3D distance from the origin (after recentering), in mm.
    5        S = Computed 3D speed WRT origin (after recentering), in mm/ms.
    6        Low-pass filtered version of S.
```
<br></br>
Finally, a reach segmentation algorithm uses the trajectory data to find reach segments over the course of the
session. The algorithmic details are different for the two kinds of sessions, but the set of reach segments detected
are stored in the same format in the file `detected_reaches.txt`. The set of manually curated reaches for the session
are saved to `<session folder>/<session_prefix>reaches.txt` in exactly the same format. Each segment is written to the 
file as a line of five whitespace-separated integers: `frame  max_delta  dur  result  hand_pos`. Here `frame` is the 
frame number at which the segment starts, `frame + max_delta` is the "reachMax" frame, `dur` is the segment duration, 
`result` indicates the outcome of the reach, and `hand_pos` indicates the hand position at "reachMax". For further
details, see the relevant code snippet below.

### Code Snippets

These snippets all take advantage of a number of simple enumerations and classes defined in the ReachX `common.py` 
module. You should be able to import this module into your scripts or a Jupyter notebook as long as you have Numpy
installed.

Error checking is omitted for brevity's sake.

#### Find all curated reaches in a session with a successful outcome
```
from pathlib import Path
from reachx.commmon import RxReachSegment, RxSessionID, RxReach

session_folder = Path("session_root/20260410/christie2P/session001")
sesh_id = RxSessionID.from_session_folder(session_folder)
reach_file = Path(session_folder, f"{sesh_id.session_file_prefix}reaches.txt")
all_segments = RxReachSegment.load_reach_segments(reach_file)

ok_segments = [seg for seg in all_segments if seg.result == RxReach.GRABBED]
```
<br></br>
#### Find all detected locations of a specified body part on a specified camera
```
from pathlib import Path
import numpy as np

from reachx.common import RxSessionID, RxBodyPart, RxCam

def find_detected_locations(sesh_id: RxSessionID, scorer: str, bp: RxBodyPart, cam: RxCam) -> np.ndarray:
    markers_file = Path(sesh_id.location, scorer, "detected_markers.py")
    markers = np.load(markers_file)                 # all "detected" marker locations across both cams
    markers = markers[markers[:, 1] == cam.value]   # restrict to specified cam
    markers = markers[markers[:, 2] == bp.value]    # then restrict to specified marker
    return markers
```
<br></br>
#### Compute distance between right-hand and pellet per frame for a legacy-cam session
```
from pathlib import Path
import numpy as np
from reachx.common import TrajCol

scorer_folder = Path("root/20260410/rigA/session002/Rx_resnet50_MyModel-20251031_0_snapshot-1030000")
hand = np.load(Path(scorer_folder, "hand.npy")
hand_x = hand[:, TrajCol.X_FILT.value]
hand_y = hand[:, TrajCol.Y_FILT.value]
hand_z = hand[:, TrajCol.Z_FILT.value]
pellet = np.load(Path(scorer_folder, "pellet.npy")
pellet_x = pellet[:, TrajCol.X_FILT.value]
pellet_y = pellet[:, TrajCol.Y_FILT.value]
pellet_z = pellet[:, TrajCol.Z_FILT.value]
distance = np.sqrt((hand_x - pellet_x) ** 2 + (hand_y - pellet_y) ** 2 + (hand_z - pellet_z) ** 2)
```
<br></br>
#### Extract the 3D filtered speed trajectory of a specified body part for a fixed-cam session
```
from typing import Optional
from pathlib import Path
import numpy as np

from reachx.common import RxSessionID, RxBodyPart, FixedCamTrajCol

def get_body_part_trajectory(sesh_id: RxSessionID, scorer: str, bp: RxBodyPart) -> Optional[np.ndarray]:
    traj_file = Path(sesh_id.location, scorer, "trajectories.npz")
    
    with np.load(traj_file) as data:
        if str(bp) in data:
            traj = data[str(bp)]
            return traj[:, FixedTrajCol.SPEED_FILT]
        else:
            return None
```
