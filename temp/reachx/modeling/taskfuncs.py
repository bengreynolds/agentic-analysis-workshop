"""
taskfuncs.py - Time-consuming tasks in ReachX that must be performed on a background thread.

 - Train a neural network (NN) model to perform body part detection on ReachX experimenent session videos, using
   selected image frames augmented by the user with body part locations.
 - Use a trained neural network to detect body part locations (with confidence scores) on the two required videos
   from one or more ReachX sessions.
 - Use the neural net output to perform automated reach segmentation on those sessions.

Usage:
 - Call ``get_default_training_params()`` to get the dictionary of training parameters that will be needed by the
   underlying code that trains the neural net model. Set the parameters specific to the particular training session, as
   described in the function header.
 - Call ``train_network()`` to initiate training of a body part detection NN model, passing in the just-prepared
   dictionary of training parameters. Include the ``BackgroundTask`` object through which the training function
   reports progress, posts messages, and checks for cancellation of the task.
 - Call ``detect_body_parts()`` to use a previously trained model to detect body part locations in every frame (with
   confidence scores) of the front- and side-cam videos for one or more ReachX experiment sessions. Pass in the
   dictionary of training parameters for the trained model, a list of session identifiers, and the ``BackgroundTask``
   object.

--------------------------------------
CREDIT: Much of the code in this module is extracted/adapted from the DeepLabCut toolbox repository:

    DeepLabCut Toolbox (deeplabcut.org)
    © A. & M.W. Mathis Labs
    https://github.com/DeepLabCut/DeepLabCut

    Please see AUTHORS for contributors.
    https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS

    Adapted from DeeperCut by Eldar Insafutdinov
    https://github.com/eldar/pose-tensorflow

    Licensed under GNU Lesser General Public License v3.0
--------------------------------------
"""

import os
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional

import tensorflow as tf
import yaml

from reachx.modeling.train import train
from reachx.modeling.analyze import analyze_session, find_reach_segments
from reachx.common import RxSessionID, RxReachSegControls
from reachx.uicommon import BackgroundTask

# we never use eager execution with Tensorflow -- so disable it early on
tf.compat.v1.disable_eager_execution()

MAT_FILE_FOR_TRAINING: str = "train.mat"
""" Name of Matlab file that contains the ``imgaug``-formatted data for training a ReachX object detection model. """


def _is_tensorflow_using_gpu() -> bool:
    """ Does the ``tensorflow`` library have CUDA/GPU support on the host system? """
    with_cuda: bool = tf.test.is_built_with_cuda()
    gpus = tf.config.list_physical_devices('GPU')
    return with_cuda and (len(gpus) > 0)


def get_default_training_params() -> Dict[str, Any]:
    """
    Get default values for the various training parameters that the DeepLabCut toolkit looks for when training a neural
    net model. In DLC, these parameters are written to <train>/pose_cfg.yaml, where <train> is the directory in which
    the files generated during training are stored.

    Since ReachX is only using a small portion of the capabilities exposed in DLC, we only set the parameters that
    actually get used. ReachX only uses the **imgaug** dataset format and the **resnet_50** and **resnet_152**
    pre-trained neural nets.

    By design, ReachX uses these default training parameters as is -- except for the parameters that are unique to a
    particular training session:
     - "train_dir": (str) Absolute path to the directory containing the training data. The required MAT file is at
       ``<train_dir>/train.mat``, and the PNG image files referenced in the MAT file by filename only are stored at
       ``<train_dir>/images/***.png``.
     - "model_dir: (str) Absolute path to the directory into which model snapshots and any other files are written.
     - "num_joints": (int) The number of unique body parts actually labeled in the training images -- this
       will be less than the total number of body parts (``RxBodyPart`` enum) supported in ReachX.
     - "all_joints_names": (List[str]) The names of the unique body parts actually labeled in the training images. In
       ReachX, these must be set to the ``RxBodyPart`` nickname.

    **The order of the body parts list in "all_joints_names" is CRITICAL**. The 0-based index position of the body part
    in this list -- NOT its enum value -- is stored with its location on a given image in the ``train.mat`` file.
    Furthermore, when the trained model is used to predict body part locations in a video of ``N`` frames, the output is
    a Numpy array with the shape ``(N, 3*L)``. Here ``L`` = the number of body parts listed in "all_joints_names". Row
    ``M`` in this array contains the location predictions for frame ``M`` for all the body parts in
    "all_joints_names": ``[x0, y0, p0, x1, y1, p1, ... x(L-1), y(L-1), p(L-1)]``, where p is a confidence score between
    0 and 1. Thus, to map these location predictions to the right body part, we need to know what "all_joints_names"
    was set to when the model was trained!

    These DLC-specifie parameters that are unique to a training session are NOT needed in ReachX:
     - "project_path": Instead of the model project path, the ReachX-specific "train_dir" and "model_dir" parameters
       specify where the training data files are and where all files generated during training should be written.
     - "dataset": This is the MAT file path. ReachX expects it to be at ``<train_dir>/train.mat``.
     - "metadataset": ReachX does not write this file.
     - "all_joints": ReachX does not use this parameter.
     - "init_weights": This is the path to the pre-trained neural net. In ReachX, the training procedure downloads the
       pretrained neural net's checkpoint file (.cpkt) on first use and places it in the ReachX application home
       directory for all future trainings.

    :return: Dictionary containing the default-valued training parameters.
    """
    cfg = dict()

    # the following must be set before invoking the train() function
    # cfg["dataset"] = None -- NOT USED in ReachX
    # cfg["metadataset"]  -- NOT USED in ReachX
    cfg["num_joints"] = None   # (int) In ReachX, this is the number of different body parts supported
    # cfg["all_joints"] = None  -- NOT USED in ReachX
    cfg["all_joints_names"] = None   # (List[str]) Body part names
    # cfg["init_weights"] -- This gets set programmatically in ReachX, both for training and when analyzing videos
    # cfg["project_path"]  -- NOT USED in ReachX
    cfg["net_type"] = "resnet_50"   # or "resnet_152" -- only NNs supported in ReachX

    # Added specifically for ReachX (not used in original DLC code)
    cfg["train_dir"] = None  # (str) Absolute path to directory containing the training data
    cfg["model_dir"] = None  # (str) Absolute path to directory where all model snapshots and other files are written

    # anything below this line that's not commented out is kept from pose_estimation_tensorflow.default_config
    # because it is used by resnet or imgaug DLC implemenations
    cfg["stride"] = 8.0
    cfg["weigh_part_predictions"] = False
    # cfg["weigh_negatives"] = False
    # cfg["fg_fraction"] = 0.25
    cfg["mean_pixel"] = [123.68, 116.779, 103.939]   # imagenet mean for resnet pretraining
    # cfg["shuffle"] = True
    # cfg["snapshot_prefix"] = None  -- NOT USED in ReachX. Set programmatically as cfg["model_dir"] + "snapshot"
    cfg["log_dir"] = "log"   # TF will write an events file inside this subfolder of the current working directory
    cfg["global_scale"] = 0.8
    cfg["location_refinement"] = True
    cfg["locref_stdev"] = 7.2801
    cfg["locref_loss_weight"] = 0.05
    cfg["locref_huber_loss"] = True
    cfg["optimizer"] = "sgd"
    cfg["intermediate_supervision"] = False   # NOTE: DLC recommends setting this to True for resnet-152.
    cfg["intermediate_supervision_layer"] = 12
    # cfg["regularize"] = False
    cfg["weight_decay"] = 0.0001
    # cfg["crop_pad"] = 0
    # cfg["scoremap_dir"] = "test"
    cfg["dataset_type"] = "imgaug"
    cfg["deterministic"] = False
    cfg["mirror"] = False
    cfg["pairwise_huber_loss"] = False
    cfg["weigh_only_present_joints"] = False
    cfg["partaffinityfield_predict"] = False
    cfg["pairwise_predict"] = False

    # anything below this line I added because it is explicitly set in the default pose_config.yaml file. However, any
    # params in that file that are not used by resnet or imgaug are not included here!
    cfg["batch_size"] = 1
    cfg["multi_step"] = [[0.005, 10000], [0.02, 430000], [0.002, 730000], [0.001, 1030000]]  # for sgd/adam optimizer
    cfg["display_iters"] = 1000   # Loss written to log after every N iterations during training
    cfg["save_iters"] = 50000     # Iteration snapshot saved after every M iterations during training
    cfg["max_input_size"] = 1500   # in ReachX, images are 300 x 200 (legacy-cam) or 256 x 256 (fixed-cam)
    cfg["min_input_size"] = 64
    cfg["scale_jitter_lo"] = 0.5
    cfg["scale_jitter_up"] = 1.25
    cfg["rotation"] = 25
    cfg["rotatio"] = 0.4
    # covering, motion_blur, elastic_transform - Not set. Imgaug implementation uses all of these by default.
    # gaussian_noise, grayscale - Not set. Imgaug implementation assumes False for both
    cfg["contrast"] = {"clahe": True, "claheratio": 0.1, "histeq": True, "histeqratio": 0.1}
    cfg["convolution"] = {"sharpen": False, "sharpenratio": 0.3, "edge": False,
                          "emboss": {"alpha": [0.0, 1.0], "strength": [0.5, 1.5]}, "embossratio": 0.1}
    cfg["crop_by"] = 0.15   # not set in pose_config.yaml, but this is what imgaug implementation uses as default
    cfg["cropratio"] = 0.4
    cfg["pos_dist_thresh"] = 17

    return cfg


def train_network(config: Dict[str, Any], task: BackgroundTask) -> None:
    """
    Train a ReachX object detection model IAW specified training parameters.

    This method is an adaptation of a like-named method in the DeepLabCut toolbox's ``pose_estimaation_tensorflow``
    package. It dispenses with the `pose_config.yaml` file required by DLC, as it does not support the flexibility and
    all the capabilities of DLC. Only the ResNet-50 and ResNet-152 pre-trained neural nets are supported as the
    starting point for training the model, and the only supported training dataset format supported is ``imgaug``.

    ReachX does not expose most of the applicable training parameters to the user, instead using the default values
    specified in DLC. Parameters not applicable to ResNet or ImgAug are omitted.

    Keyword arguments in the DLC version of this method have been removed, as ReachX does not support changing the
    values assigned to these.
     - ``displayiters``, ``saveiters``, ``maxiters``: Use the values in ``config``.
     - ``allow_growth``: Always True
     - ``gputouse``: Always None
     - ``auto_tune``: Always False
     - ``keepdeconvweights``: Not needed, as ReachX never starts training from a previously trained snapshot. It only
       uses one of two pretrained NNs -- resnet_50_v1 and resnet_152_v1.
     - ``max_snapshots_to_keep``: 5

    :param config: Dictionary of training parameters. See `get_default_training_params()` for details.
    :param task: The task object that invoked this method on a background thread. Provides facilities for reporting
       progress, posting info/error/progress messages (for GUI display), as well as to check for task cancellation.
    """
    task.progress_updated.emit(0)
    task.message_logged.emit("Preparing to train network model...")
    if not _is_tensorflow_using_gpu():
        task.message_logged.emit("WARNING: GPU support unavailable -- training will take a very long time.")
    else:
        task.message_logged.emit(f"Tensorflow GPU support is enabled.")

    # allow_growth = True always
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    tf.compat.v1.reset_default_graph()

    # train() method will change the current working directory, so we need to restore it.
    start_path = os.getcwd()

    # autotune == False always
    os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
    # gputouse == None always
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gputouse)

    try:
        train(config, task)   # note defaults: max_keep = 5, allow_growth = True
    except BaseException as e:
        emsg = f"Training session FAILED: {str(e)}"
        task.message_logged.emit(emsg)
        traceback.print_exc()
    finally:
        os.chdir(str(start_path))


_BATCH_SIZE_FOR_DETECT = 32
""" 
Batch size for processing video image frames. This is configurable in DLC. Testing (on a machine w/o GPU support) found 
that performance did not improve with increasing batch size beyond this value.
"""


def analyze_sessions(model_cfg_file: Path, sesh_ids: List[RxSessionID], task: BackgroundTask,
                     batch_size: int = _BATCH_SIZE_FOR_DETECT, use_gpu: bool = True, enable_pp: bool = True,
                     calib_root: Optional[Path] = None, seg_parms: Optional[RxReachSegControls] = None) -> None:
    """
    Use a previously trained model -- the "scorer" -- to analyze the two required videos recorded during one or more
    ReachX experiment sessions, generating per-frame predicted body part locations on each video, and from that filtered
    hand and pellet trajectories. Optionally perform automated reach segmentation on each session based on those
    trajectories.

    For each session to be processed, the following tasks are performed:
     - Use model to infer body part locations on each frame of the two required videos: ``RxCam.SIDE/FRONT`` for
       legacy-cam sessions, ``RxCam.LEFT/RIGHT`` for fixed-cam. To improve performance and better utilize the GPU (when
       available), the videos are analyzed in separate spawned processes. The inference results for each video are
       stored as a Numpy array in ``<session_folder>/<rx_scorer>/<cam_nickname>.npy``.
     - Using the results from the first two steps, prepare a ``np.int32`` array holding the predicted body part marker
       locations across both videos **for which the confidence or likelihood score > 0.9**. Each row in this array has
       the form ``[frame_num, cam_value, bp_value, x, y, likelihood*10000]``, where the likelihood is scaled by 10000 to
       preserve 4 decimal digits when the array contents are rounded to ``np.int32``. This array is stored at
       ``<session_folder>/<rx_scorer>/detected_markers.npy``.
     - From previous results, prepare smmoothed (X, Y, Z) position and speed trajectories. For legacy-cam sessions,
       trajectories are generated only for one hand (animal's right hand) and the food pellet, stored as ``np.float``
       arrays in ``<session_folder>/<rx_scorer>/hand.npy, pellet.npy``. TODO: Complete for fixed-cam analysis.
     - Finally, if reach segmentation control parameters are specified, use the hand/pellet trajectories to perform
       automated reach segmentation for the session. The The reach segments found are stored in
       ``<session_folder>/<rx_scorer>/detected_reaches.txt``.
    See ``analyze.analyze_session()`` for full details.

    The string ``rx_scorer = "Rx_<nnet>_<model>_<iter_num>_snapshot-<snapshot_index>"`` serves to identify the
    particular trained model instance that generated the analysis results. It also serves to the name the subfolder
    within the session folder in which those results are stored, as indicated above.

    NOTE: Prior to v0.6.0, the scorer name component ``<model>`` was set to the model project name, which is not
    necessarily unique. As of v0.6.0, ``<model>`` is set to the full model project folder name: "modelName-YYYYMMDD"
    for legacy-cam models, and "modelName-YYYYMMDD-fix" for fixed-cam models.

    IMPORTANT: A model need not be trained to detect every ``RxBodyPart`` supported in ReachX. The training parameter
    ``cfg["all_joints_names"]`` is a list of the body parts for which the model was trained to detect. Their order in
    that list defines the order of the predicted body part locations in the model output.

    :param model_cfg_file: YAML file containing the training parameters dictionary for the previously trained model.
        This includes information identifying the model's neural net architecture, the location of the model snapshots,
        and the list of body parts the model was trained to detect. It also includes other parameters needed to use the
        model to predict body part locations in each frame of a session video.
    :param sesh_ids: List of session identifiers.
    :param task: The task object on the background thread from which this method is invoked. Provides facilities for
        reporting progress, posting info/error/progress messages (for GUI display), as well as to check for task
        cancellation.
    :param batch_size: Number of video frames to process in one batch. Increasing this MAY improve performance on a
        system with GPU support enabled. Default = 32.
    :param use_gpu: If True, GPU is used for inference if available. If False, inference is done on CPU (slower!)
        regardless if GPU is available. Default = True.
    :param enable_pp: If True, the required session videos (side/front or left/right) are analyzed in parallel using
        two separate processes. If False, the two videos are analyzed sequentially. Default = True.
    :param calib_root: Source directory for calibration data. Fixed-cam session analysis only; ignored for legacy-cam
        sessions. Directory should contain date-formated subfolders ("YYYYMMDD"), each of which contains FLIR camera
        calibration files generated on that date. Default = None
    :param seg_parms: If not None, perform reach segmentation on each analyzed session using these segmentation control
        parameters. Default = None.
    """
    if len(sesh_ids) == 0:
        task.message_logged.emit("No sessions specified - nothing to do!")
        return
    if sesh_ids[0].is_fixed_cam and not (isinstance(calib_root, Path) and calib_root.is_dir()):
        task.message_logged.emit("Missing calibration data root directory for fixed-cam session analysis!")
        return

    tf_gpu_inference = _is_tensorflow_using_gpu()
    if not tf_gpu_inference:
        task.message_logged.emit("WARNING: GPU support unavailable -- body part detection will take a very long time.")
    elif not use_gpu:
        task.message_logged.emit("WARNING: Tensorflow GPU support is available but is disabled by user setting!")
    else:
        task.message_logged.emit(f"Tensorflow GPU support will be used to analyze session videos.")

    # just in case, even though ReachX implementation of training does not set this
    if "TF_CUDNN_USE_AUTOTUNE" in os.environ:
        del os.environ["TF_CUDNN_USE_AUTOTUNE"]  # was potentially set during training

    tf.compat.v1.reset_default_graph()

    try:
        task.progress_updated.emit(0)
        task.message_logged.emit("Loading and checking trained model configuration...")

        model_cfg: Dict[str, Any]
        with open(model_cfg_file, 'r') as f:
            model_cfg = yaml.safe_load(f)

        # find the prefix for the files defining the last model snaphot (highest # iterations) in model folder. This
        # assumes the files have the form "snapshot-<iter>.<extension>
        model_dir = Path(model_cfg["model_dir"])
        if not model_dir.is_dir():
            raise Exception(f"Configuration error - Model snapshot folder not found: {str(model_dir)}")
        all_prefixes = [f.name.split(".")[0] for f in model_dir.iterdir() if ("index" in f.name)]
        if len(all_prefixes) == 0:
            raise Exception(f"Configuration error - No model snapshots found in: {str(model_dir)}")
        max_iter = max([int(pfx.split("-")[-1]) for pfx in all_prefixes])
        snapshot_pfx = next((pfx for pfx in all_prefixes if (str(max_iter) in pfx)))

        # ReachX version of "DLCScorer", which serves to identify the model snapshot used. It also serves as the name
        # of the session subfolder in which the analysis results are stored.
        net_name = str(model_cfg["net_type"]).replace("_", "")   # "resnet_50" --> "resnet50"
        iter_num = int(model_dir.parents[0].name.split('-')[-1])
        model_prj = model_dir.parents[1].name  # the full model project folder name
        rx_scorer = f"Rx_{net_name}_{model_prj}_{iter_num}_{snapshot_pfx}"

        task.message_logged.emit(
            f"Inference analysis using trained model: {rx_scorer}. Batch size = {batch_size}. "
            f"Using GPU={tf_gpu_inference and use_gpu}. Videos processed in parallel = {enable_pp}.")
        model_cfg["init_weights"] = str(Path(model_dir, snapshot_pfx))
        model_cfg["num_outputs"] = 1   # ReachX does not support more than one output per body part
        model_cfg["batch_size"] = batch_size
        model_cfg["use_gpu"] = tf_gpu_inference and use_gpu
        model_cfg["enable_pp"] = enable_pp

        # check for cancel before looping
        if task.was_canceled():
            return

        task.message_logged.emit(f"Analyzing {len(sesh_ids)} sessions...")
        progress_chunk_per_session = 100.0 / len(sesh_ids)
        total_progress = 0.0
        for sesh_id in sesh_ids:
            skip = False
            if not sesh_id.exists():
                task.message_logged.emit(f"!!! Skipping session {str(sesh_id)} -- session folder not found.")
                skip = True
            calib_path = RxSessionID.calibration_folder_for(sesh_id, calib_root)
            if sesh_id.is_fixed_cam and (calib_path is None):
                task.message_logged.emit(f"!!! Skipping session {str(sesh_id)} -- no fixed-cam calibration data found.")
                skip = True
            if skip:
                total_progress += progress_chunk_per_session
                task.progress_updated.emit(int(total_progress))
                continue

            task.message_logged.emit(f"Running body part detection model on session {str(sesh_id)}...")
            analyze_session(sesh_id, model_cfg, rx_scorer, calib_path, task)
            if isinstance(seg_parms, RxReachSegControls):
                find_reach_segments(sesh_id, rx_scorer, seg_parms, task)
            total_progress += progress_chunk_per_session
            task.progress_updated.emit(int(total_progress))

    except BaseException as e:
        emsg = f"Session analysis FAILED: {str(e)}"
        task.message_logged.emit(emsg)
        traceback.print_exc()
