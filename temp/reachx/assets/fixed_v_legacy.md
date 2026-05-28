## Fixed-cam vs Legacy-cam Rig Support

The lab has begun transitioning to a newer, more rigorously calibrated experiment rig -- the ***fixed-cam*** rig --
that also includes specially marked features which serve as reference points for video analysis. While ***ReachX***
supports reviewing and analyzing sessions recorded on either a fixed-cam rig or the older ***legacy-cam*** rig, it does 
not allow inter-mixing of the two types of experiment sessions. To enforce this separation, a ***ReachX*** workspace 
comes in two flavors: a fixed-cam workspace only processes sessions recorded on fixed-cam rigs and only uses fixed-cam 
models to analyze those sessions; analogously for a legacy-cam workspace.

This chapter summarizes the key differences between the two rig setups.

### Cameras
 - **Legacy-Cam**: The two primary cameras are the **Side** and **Front** Cameras. Other cameras sometimes used 
(but not for session analysis in ***ReachX***) include the **Stim** and **Fast** cameras. Frame size is 
320x200 pixels; frame rate, 150Hz.
 - **Fixed-Cam**: Only two cameras - **Left** and **Right**. Frame size is 256x256 pixels, 150Hz frame rate.

### Calibration
 - **Legacy-Cam**: Simple conversion from pixels on the **Side** and **Front** cameras to pseudo-3D coordinate space in 
millimeters using hard-coded calibration constants.
 - **Fixed-Cam**: Rigourous calibration procedure is performed regularly (annually?) on the FLIR cameras in this rig,
generating a stereo parameters file and other calibration information. This information is used to transform the 
**Left** and **Right** camera viewpoints into a 3D coordinate space.

### Set of defined body parts that may be tracked in ReachX
The body parts applicable to a fixed-cam session do not overlap much with those used for a legacy-cam session. The
fixed-cam set includes dedicated markers for labeling reference points in one or both camera views ("diamond", "star",
and "triangle"). Also, there are both left- and right-hand alternate poses, for tracking both hands in both camera 
views. In the legacy-cam setup, the various hand poses apply only to the animal's right hand; the "SdH" body parts are 
for tracking the right hand on the **Side** camera, while the "FtH" body parts are for tracking the right hand on the
**Front** camera.

#### Legacy-Cam
```
               Legacy-Cam
RxBodyPart                  Nickname in ReachX
----------                  ------------------
TONGUE = 0                  "tongue"
MOUTH = 1                   "mouth"
PELLET = 2                  "pellet"
SIDE_HAND_FLAT = 3          "SdH_Flat"
SIDE_HAND_SPREAD = 4        "SdH_Spread"
SIDE_HAND_GRAB = 5          "SdH_Grab"
FRONT_HAND_REACH = 6        "FtH_Reach"
FRONT_HAND_GRASP = 7        "FtH_Grasp"
EXCLUDE1 = 8                "exclude1"      # These can be used to tag distinctive image features
EXCLUDE2 = 9                "exclude2"      # that might be confused with a tracked body part.
NOSE = 10                   "nose"
```
#### Fixed-Cam
```
RxBodyPart                  Nickname in ReachX
----------                  ------------------
MOUTH = 1                   "mouth"
PELLET = 2                  "pellet"
EXCLUDE1 = 8                "exclude1"      # These can be used to tag distinctive image features
EXCLUDE2 = 9                "exclude2"      # that might be confused with a tracked body part.
NOSE = 10                   "nose"
RIGHT_HAND_FLAT = 11        "RH_flat"
RIGHT_HAND_SPREAD = 12      "RH_spread"
RIGHT_HAND_GRAB = 13        "RH_grab"
LEFT_HAND_FLAT = 14         "LH_flat"
LEFT_HAND_SPREAD = 15       "LH_spread"
LEFT_HAND_GRAB = 16         "LH_grab"
STAR = 17                   "Star"          # Star-shaped marker affixed to pellet cover arm
TONGUE_MID = 18             "Tongue_mid"
TONGUE_TIP = 19             "Tongue_tip"
TRIANGLE = 20               "Triangle"      # Triangle-shaped marker affixed to pellet holder arm
DIAMOND = 21                "Diamond"       # Fixed reference point ("the origin") in both views
```
