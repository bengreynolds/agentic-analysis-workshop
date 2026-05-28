"""
analyze.py - Use a trained object detection model to detect body parts in ReachX session videos.

--------------------------------------
CREDIT: The code in this module is extracted/adapted from the DeepLabCut toolbox repository:

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
from __future__ import annotations

import pickle
import time
import traceback
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import multiprocessing as mp
import queue
from threading import Event
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

import numpy as np
import tensorflow as tf
from scipy.signal import butter, savgol_coeffs, filtfilt

from reachx.config.app_log import get_null_logger
from reachx.data.sessionmgr import CamSource, SessionManager
from reachx.modeling.calibration import FlirCalibration
from reachx.modeling.resnet import PoseResnet
from reachx.common import RxBodyPart, RxSessionID, RxCam, RxEvent, RxReachSegControls, RxReachSegment, \
    RxReach, RxHandPos, TrajCol, FixedCamTrajCol
from reachx.uicommon import BackgroundTask


# ********************************************************************************************************************
# Methods in this section taken pretty much verbatim from deeplabcut.pose_estimation_tensorflow.core.predict.py
# ********************************************************************************************************************
def setup_pose_prediction(cfg, allow_growth=False):
    tf.compat.v1.reset_default_graph()
    inputs = tf.compat.v1.placeholder(tf.float32, shape=[cfg["batch_size"], None, None, 3])
    net_heads = PoseResnet(cfg).test(inputs)

    outputs = [net_heads["part_prob"]]
    if cfg["location_refinement"]:
        outputs.append(net_heads["locref"])

    outputs.append(net_heads["peak_inds"])

    restorer = tf.compat.v1.train.Saver()

    if allow_growth:
        # Pycharm cannot find this, though it is in tensorflow._api.v2.compat.v1.__init__.py
        # noinspection PyUnresolvedReferences
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)
    else:
        sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(tf.compat.v1.local_variables_initializer())

    # Restore variables from disk.
    restorer.restore(sess, cfg["init_weights"])

    return sess, inputs, outputs


def _extract_cnn_outputmulti(outputs_np, cfg):
    """
    extract locref + scmap from network
    Dimensions: image batch x imagedim1 x imagedim2 x bodypart
    """
    scmap = outputs_np[0]
    locref = None
    if cfg["location_refinement"]:
        locref = outputs_np[1]
        shape = locref.shape
        locref = np.reshape(locref, (shape[0], shape[1], shape[2], -1, 2))
        locref *= cfg["locref_stdev"]
    if len(scmap.shape) == 2:  # for single body part!
        scmap = np.expand_dims(scmap, axis=2)
    return scmap, locref


# noinspection All
def get_top_values(scmap, n_top=5):
    batchsize, ny, nx, num_joints = scmap.shape
    scmap_flat = scmap.reshape(batchsize, nx * ny, num_joints)
    if n_top == 1:
        scmap_top = np.argmax(scmap_flat, axis=1)[None]
    else:
        scmap_top = np.argpartition(scmap_flat, -n_top, axis=1)[:, -n_top:]
        for ix in range(batchsize):
            vals = scmap_flat[ix, scmap_top[ix], np.arange(num_joints)]
            arg = np.argsort(-vals, axis=0)
            scmap_top[ix] = scmap_top[ix, arg, np.arange(num_joints)]
        scmap_top = scmap_top.swapaxes(0, 1)

    Y, X = np.unravel_index(scmap_top, (ny, nx))
    return Y, X


# noinspection All
def _get_pose_np(image, cfg, sess, inputs, outputs, outall=False):
    """
    Adapted from DeeperCut, performs numpy-based faster inference on batches.
    Introduced in https://www.biorxiv.org/content/10.1101/457242v1
    """

    num_outputs = 1  # ReachX only supports 1 for this: cfg.get("num_outputs", 1)
    outputs_np = sess.run(outputs, feed_dict={inputs: image})

    scmap, locref = _extract_cnn_outputmulti(outputs_np, cfg)   # processes image batch.
    batchsize, ny, nx, num_joints = scmap.shape

    Y, X = get_top_values(scmap, n_top=num_outputs)

    # Combine scoremat and offsets to the final pose.
    DZ = np.zeros((num_outputs, batchsize, num_joints, 3))
    for m in range(num_outputs):
        for l in range(batchsize):
            for k in range(num_joints):
                x = X[m, l, k]
                y = Y[m, l, k]
                DZ[m, l, k, :2] = locref[l, y, x, k, :]
                DZ[m, l, k, 2] = scmap[l, y, x, k]

    X = X.astype("float32") * cfg["stride"] + 0.5 * cfg["stride"] + DZ[:, :, :, 0]
    Y = Y.astype("float32") * cfg["stride"] + 0.5 * cfg["stride"] + DZ[:, :, :, 1]
    P = DZ[:, :, :, 2]

    Xs = X.swapaxes(0, 2).swapaxes(0, 1)
    Ys = Y.swapaxes(0, 2).swapaxes(0, 1)
    Ps = P.swapaxes(0, 2).swapaxes(0, 1)

    pose = np.empty(
        (cfg["batch_size"], num_outputs * cfg["num_joints"] * 3), dtype=X.dtype
    )
    pose[:, 0::3] = Xs.reshape(batchsize, -1)
    pose[:, 1::3] = Ys.reshape(batchsize, -1)
    pose[:, 2::3] = Ps.reshape(batchsize, -1)

    if outall:
        return scmap, locref, pose
    else:
        return pose


def setup_gpu_pose_prediction(cfg, allow_growth=False):
    tf.compat.v1.reset_default_graph()
    inputs = tf.compat.v1.placeholder(tf.float32, shape=[cfg["batch_size"], None, None, 3])
    net_heads = PoseResnet(cfg).inference(inputs)
    outputs = [net_heads["pose"]]

    restorer = tf.compat.v1.train.Saver()

    if allow_growth:
        # Pycharm cannot find this, though it is in tensorflow._api.v2.compat.v1.__init__.py
        # noinspection PyUnresolvedReferences
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)
    else:
        sess = tf.compat.v1.Session()

    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(tf.compat.v1.local_variables_initializer())

    # Restore variables from disk.
    restorer.restore(sess, cfg["init_weights"])

    return sess, inputs, outputs


def _extract_gpu_prediction(outputs):
    return outputs[0]

# ********************************************************************************************************************
# END: Methods in this section taken pretty much verbatim from deeplabcut.pose_estimation_tensorflow.core.predict.py
# ********************************************************************************************************************


def _get_pose_f(cfg: Dict[str, Any], video_src: CamSource, q_task: _QueueTask, sess, inputs, outputs):
    """
    Batchwise prediction of pose (body part locations) with inference on CPU rather than GPU.

    Based on DLC version ``GetPoseF``, but adapted to use ``CamSource`` instead of ``cv2.VideoCapture`` directly -- to
    handle dropped frames in the video. A dropped frame is returned as None, which we represent as an all-black frame.
    Also, ``CamSource`` puts each image frame in the format required for processing (ubyte, RGB, #rows, #cols).

    :param cfg: Params defining the trained model used to make body part location predictions. The model snapshot to
        be used is in "init_weights" key.
    :param video_src: A ReachX wrapper that handles the complexity introduced by any dropped frames in the video
        timeline.
    :param q_task: Process-safe task object for message logging, progress updates, and premature cancellation.
    :param sess: Tensorflow predictor infrastructure.
    :param inputs: Tensorflow predictor infrastructure.
    :param outputs: Tensorflow predictor infrastructure.
    :return: A Numpy array (N, 3*M) containing the results, where N is the video frame count (including dropped frames)
        and M is the number of body parts. Per frame, per body part is stored (x, y, likelihood).
    """
    n_frames = video_src.frame_count

    # NOTE: only one output per body part supported in ReachX.
    cfg["num_outputs"] = 1

    # use float32 rather than default float64 to save memory
    predicted_data = np.zeros(
        (n_frames, cfg["num_outputs"] * 3 * len(cfg["all_joints_names"])), dtype=np.float32
    )

    img_w, img_h = video_src.frame_size
    batch_sz = int(cfg["batch_size"])
    batch_frames = np.empty((batch_sz, img_h, img_w, 3), dtype=np.uint8)  # a batch's worth of video frames

    q_task.progress_update(0)
    next_disp = int(n_frames / 20)
    n_frames_done = 0
    while n_frames_done < n_frames:
        n_retrieved = video_src.grab_frame_block(n_frames_done, batch_sz, batch_frames)
        pose = _get_pose_np(batch_frames, cfg, sess, inputs, outputs)
        predicted_data[n_frames_done: n_frames_done + n_retrieved] = pose[:n_retrieved]
        n_frames_done += n_retrieved

        if q_task.was_canceled():
            return predicted_data

        pct = int(100.0 * n_frames_done / n_frames)
        q_task.progress_update(pct)
        if n_frames_done >= next_disp:
            next_disp = int(next_disp + n_frames / 20)
            q_task.log_message(f"{n_frames_done} of {n_frames} frames processed.")

    return predicted_data


def _get_pose_f_gtf(cfg: Dict[str, Any], video_src: CamSource, q_task: _QueueTask, sess, inputs, outputs):
    """
    Batchwise prediction of pose (body part locations) with inference on GPU using Tensorflow.

    Based on DLC version ``GetPoseF_GTF``, but adapted to use ``CamSource`` instead of ``cv2.VideoCapture`` directly --
    to handle dropped frames in the video. A dropped frame is returned as None, which we represent as an all-black
    frame. Also, ``CamSource`` puts each image frame in the format required for processing (ubyte, RGB, #rows, #cols).

    :param cfg: Params defining the trained model used to make body part location predictions. The model snapshot to
        be used is in "init_weights" key.
    :param video_src: A ReachX wrapper that handles the complexity introduced by any dropped frames in the video
        timeline.
    :param q_task: Process-safe task object for message logging, progress updates, and premature cancellation.
    :param sess: Tensorflow predictor infrastructure.
    :param inputs: Tensorflow predictor infrastructure.
    :param outputs: Tensorflow predictor infrastructure.
    :return: A Numpy array (N, 3*M) containing the results, where N is the video frame count (including dropped frames)
        and M is the number of body parts. Per frame, per body part is stored (x, y, likelihood).
    """
    n_frames = video_src.frame_count
    predicted_data = np.zeros(
        (n_frames, cfg["num_outputs"] * 3 * len(cfg["all_joints_names"])), dtype=np.float32
    )

    img_w, img_h = video_src.frame_size
    batch_sz = int(cfg["batch_size"])
    batch_frames = np.empty((batch_sz, img_h, img_w, 3), dtype=np.uint8)  # a batch's worth of video frames

    # Flip x, y, confidence and reshape
    pose_tensor = _extract_gpu_prediction(outputs)
    pose_tensor = tf.gather(pose_tensor, [1, 0, 2], axis=1)
    pose_tensor = tf.reshape(pose_tensor, (batch_sz, -1))

    q_task.progress_update(0)
    next_disp = int(n_frames / 20)
    n_frames_done = 0
    while n_frames_done < n_frames:
        n_retrieved = video_src.grab_frame_block(n_frames_done, batch_sz, batch_frames)
        pose = sess.run(pose_tensor, feed_dict={inputs: batch_frames})
        predicted_data[n_frames_done: n_frames_done + n_retrieved] = pose[:n_retrieved]
        n_frames_done += n_retrieved

        if q_task.was_canceled():
            return predicted_data

        pct = int(100.0 * n_frames_done / n_frames)
        q_task.progress_update(pct)
        if n_frames_done >= next_disp:
            next_disp = int(next_disp + n_frames / 20)
            q_task.log_message(f"{n_frames_done} of {n_frames} frames processed.")

    return predicted_data


class _QueueTask:
    """
    Handles the forwarding of log messages and progress updates, and checks for cancelation in a multi-process
    context, using a process-safe queue and event object. Intended only for use in the functions invoked in the
    spawned process that does model inference on a session video. See ``analyze_session()``.
    """
    def __init__(self, cam_value: int, event_queue: queue.Queue, cancel_event: Event):
        """
        Create the ``_QueueTask``.
        :param event_queue: The process-safe event queue.
        :param cam_value: Integer value of ``RxCam`` enumerant identifying the video camera analyzed in the
            communicating process.
        :param cancel_event: Process-safe event checked for premature cancelation.
        """
        self._event_queue = event_queue
        self._cam_value = cam_value
        self._cancel_event = cancel_event

    def log_message(self, msg: str) -> None:
        try:
            self._event_queue.put((self._cam_value, "log", msg), block=False)
        except queue.Full:
            pass

    def progress_update(self, pct: int) -> None:
        try:
            self._event_queue.put((self._cam_value, "progress", pct), block=False)
        except queue.Full:
            pass

    def error(self, emsg: str) -> None:
        try:
            self._event_queue.put((self._cam_value, "error", emsg), block=False)
        except queue.Full:
            pass

    def done(self) -> None:
        try:
            self._event_queue.put((self._cam_value, "done", None), block=False)
        except queue.Full:
            pass

    def was_canceled(self):
        return bool(self._cancel_event is not None and self._cancel_event.is_set())


def _detect_body_parts_in_video(cfg: Dict[str, Any], session_path: str, cam: int, q_task: _QueueTask) \
        -> Optional[np.ndarray]:
    """

    Predict body part locations in each frame from a video camera source.

    The Tensorflow-base model implementation generates a large Numpy ``float64`` array of shape ``(N, 3*M)``, where
    ``N`` is the number of video frames and ``M`` is the number of body parts included in the analysis. Each row
    contains ``(x, y, likelihood)`` per body part, where ``(x,y)`` is the predicted location on the frame image in
    pixels, and ``likelihood`` is a confidence score in ``[0..1]`` for that prediction. The order of the 3-tuples in the
    row is the same as the order in ``cfg['all_joints_names']``. **It is important to note that a model may be trained
    to detect only a subset of the body parts supported in ReachX. So this training configuration parameter is CRITICAL
    in order to map each prediction to the right body part!**

    For the benefit of downstream analysis, this array is reshaped to ``(N*M, 5)``, where each row is a distinct body
    part location prediction: ``[frame_num bp_index x y likelihood]``. Here ``bp_index`` identifies the specific body
    part: ``part = RxBodyPart(int(bp_index))``. While this increases the total number of elements in the array by 40%,
    we save the reshaped arrays as ``np.float32`` instead of ``np.float64``, so the overall memory and file space burden
    is less than the original array.

    **On progress updates**: This method and the underlying infrastructure resets progress to 0% and updates it
    regularly as the video frames are processed in batches.

    **Designed to run in a separate process**: To improve performance and better utilize the host GPU (when available),
    this method was redesigned to run in a separate process so that both camera videos can be analyzed in parallel.
    Thus, the method creates its own TensorFlow 1.x graph and``Session`` object to run inference on the specified video,
    and the GPU is configured to allow memory growth so that neither inference process uses all GPU memory. For more
    information, see ``analyze_session()``.

    :param cfg: Params defining the trained model used to make body part location predictions. The model snapshot to
        be used is in "init_weights" key. The model only locates body parts that were included in training! Expects
        added param "use_gpu" indicating whether to use the GPU for inference. If missing, does NOT use GPU.
    :param session_path: Full path to session folder containing recorded videos.
    :param cam: Integer value of the ``RxCam`` enumerant identifying the camera source to analyze. Cannot pass
        enum object over process boundary.
    :param q_task: A process-safe communication object used to check for premature cancelation and to log messages
        and progress updates.
    :return: As described above. On failure or premature cancelation, returns None.
    """
    tf.compat.v1.reset_default_graph()

    sesh_id = RxSessionID.from_session_folder(Path(session_path))
    if sesh_id is None:
        q_task.error("Invalid session ID!")
        return None

    cam_id = RxCam(cam)
    q_task.log_message(f"Loading {str(cam_id)} video for {str(sesh_id)}...")
    videos = CamSource.load_session_cams(sesh_id, [cam_id], get_null_logger())
    if len(videos) == 0:
        q_task.error(f"Failed to load master cam for session {str(sesh_id)}. Cannot continue.")
        return None
    if cam_id not in videos:
        q_task.error(f"Failed to load cam {str(cam_id)} for session {str(sesh_id)}. Cannot continue. ")
        return None
    video_src = videos[cam_id]

    use_gpu = cfg.get("use_gpu", False)
    if use_gpu:
        sess, inputs, outputs = setup_gpu_pose_prediction(cfg, allow_growth=True)
    else:
        sess, inputs, outputs = setup_pose_prediction(cfg, allow_growth=True)

    q_task.progress_update(0)
    n_frames = video_src.frame_count
    q_task.log_message(
        f"  {str(video_src.session_id)}, {str(video_src.id)}: Total frames = {n_frames}, "
        f"#dropped = {video_src.num_dropped_frames}")

    batch_size = int(cfg["batch_size"])
    if batch_size > 1:
        if use_gpu:
            predicted_data = _get_pose_f_gtf(cfg, video_src, q_task, sess, inputs, outputs)
        else:
            predicted_data = _get_pose_f(cfg, video_src, q_task, sess, inputs, outputs)
    else:
        q_task.error("Batch size for inference must be > 1!")
        return None

    if q_task.was_canceled():
        return None

    # need to map zero-based index used in model analysis to the actual RxBodyPart value (also an integer). Depends on
    # which body parts the model was trained to detect!
    body_parts = [RxBodyPart.from_nickname(nickname) for nickname in cfg["all_joints_names"]]
    n_bp = len(body_parts)
    bp_values = np.array([bp.value for bp in body_parts])

    # reshape model output as described above
    assert n_bp * 3 == int(predicted_data.shape[1])
    all_markers = predicted_data.reshape(n_frames * n_bp, 3)  # [x, y, likelihood]
    frame_nums_col = np.repeat(np.arange(n_frames), n_bp)
    frame_nums_col = frame_nums_col.reshape(-1, 1)
    bp_values_col = np.tile(bp_values, n_frames)
    bp_values_col = bp_values_col.reshape(-1, 1)
    reshaped_arr = np.hstack((frame_nums_col, bp_values_col, all_markers))  # row = [frame, bp.value, x, y, likelihood]

    return reshaped_arr.astype(np.float32)


def _dispatch_process_event(task: BackgroundTask, progress: Dict[RxCam, int], cam_value: int, event_type: str,
                            payload: Union[str, int]) -> None:
    """
    Helper method for ``analyze_session()`: Dispatch an interprocess event received from one of the spawned processes
    that analyze a session's videos in parallel.
    :param task: The background task object that posts log messages and progress updates for consumption back on the
        main GUI thread.
    :param progress: Dictionary holding current progress (integer % complete) per video camera analyzed. This is
        updated in response to a "progress" event. It is then used to estimate the total progress across all processes
        and emits a progress update via the background task object.
    :param cam_value: Integer value of the RxCam enumerant identifying the video camera analyzed in the process that
        sourced the event.
    :param event_type: The event type: "log", "progress", "error", or "done".
    :param payload: The event payload: A string for "log" or "error", an integer for "progress", None for "done".
    """
    cam_id = RxCam(cam_value)

    if event_type == "progress" and isinstance(payload, int) and (cam_id in progress):
        progress[cam_id] = int(payload)
        total_progress = int(sum(pct for pct in progress.values()) / len(progress))
        task.progress_updated.emit(total_progress)
        return

    if event_type == "log" and isinstance(payload, str):
        task.message_logged.emit(f"[{str(cam_id)}] {payload}")
        return

    if event_type == "done" and (cam_id in progress):
        progress[cam_id] = 100
        total_progress = sum(pct for pct in progress.values()) / len(progress)
        task.progress_updated.emit(total_progress)
        return

    if event_type == "error" and isinstance(payload, str):
        msg = f"ERROR processing {str(cam_id)}: {payload}"
        task.message_logged.emit(msg)


def _terminate_processes(executor: ProcessPoolExecutor) -> None:
    """
    Helper method for ``analyze_session()``: Ungraceful termination of the processes spawned from a process pool.
    :param executor: The process pool executor.
    """
    processes = getattr(executor, "_processes", None)
    if not processes:
        return
    for proc in processes.values():
        try:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1)
        except Exception:
            continue


CONF_THRESHOLD = 0.9
""" Object locations with a model confidence score at or above this level are considered 'good'. """


def analyze_session(sesh_id: RxSessionID, cfg: Dict[str, Any], rx_scorer: str, calib_path: Optional[Path],
                    task: BackgroundTask) -> None:
    """
    Run a trained body part detection model over the two required camera videos of the specified ReachX experiment
    session. Store the predicted per-frame body part locations for each video (Numpy array) in a subfolder within the
    session folder. In addition extract all body part location predictions across both videos with a confidence score
    exceeding 0.9 out of 1.0, and store the resulting Numpy array in the same subfolder.

    To improve performance and better utilize the host GPU (when available), this method spawns two separate processes,
    one to analyze each required video (``RxCam.SIDE/FRONT`` for legacy-cam, ``RxCam.LEFT/RIGHT`` for fixed-cam). The
    helper method ``_detect_body_parts_in_video()`` runs the model over a single video and is designed to run in a
    separate process. It uses process-safe queue and event objects -- encapsulated in a ``_QueueTask`` object -- to
    deliver progress updates and log messages and check for premature cancelation.

    ``_detect_body_parts_in_video()`` returns a ``np.float32`` array of shape ``(N*P, 5)``, for ``N`` frames and ``P``
    body parts, where each row represents a single body part location prediction: ``[frame_num, bp_value, x, y,
    likelihood]``, where ``int(bp_value)`` is an ``RxBodyPart`` enumerated value identifying the body part.

    The model output for each camera source is saved to ``<sesson_folder>/<rx_scorer/<cam>.py``, where ``<rx_scorer>``
    identifies the model instance that generated the output and ``<cam>`` identifies the camera source.

    Next, rows with ``likelihood > 0.9`` are extracted from the two arrays and combined into a single ``(K, 6)``
    ``np.int32`` array, where each row = ``[frame_num, cam_value, bp_value, x, y, likelihood*10000]``. The likelihood
    value is scaled to preserve 4 decimal digits when we round the array contents and cast to ``np.int32``. This array
    contains every "detected marker location" (confidence > 0.9) across the entire session, and it is saved at
    ``<session_folder>/<rx_scorer>/detected_markers.npy``.

    The final stage of processing is different for legacy-cam vs fixed-cam sessions. The latter requires calibration
    information found in the folder specified for by the calib_path argument. For full details, refer to:
     - ``_save_filtered_hand_and_pellet_trajectories()`` for a legacy-cam session.
     - ``_process_fixed_cam_object_trajectories()`` for a fixed-cam session.

    The ``cfg`` argument supplies parameters defining the trained model, including the model snapshot (``init_weights``
    key), the unique body part names labeled in the training data (``all_joints_names`` key), and others. In addition,
    it may include two performance flags:
     - ``use_gpu`` (``bool``, default is ``False``): If True and Tensorflow GPU support is available, the GPU is used
       for inference to improve performance.
     - ``enable_pp`` (``bool``, default is ``True``): If True, the left/right or front/side cam videos are analyzed
       in parallel using two spawned processes. If False, the two videos are analyzed sequentially.

    :param sesh_id: The session identifier.
    :param cfg: Params defining the trained model used to make body part location predictions, plus performance flags
        as described above.
    :param rx_scorer: Identifies the "scorer", ie, the particular model instance that generated the output. Used as the
        name of a subfolder of the session folder in which the scorer's analysis results are kept.
    :param calib_path: Folder containing the FLIR camera calibration data files to use when analyzing a fixed-cam
        session. Ignored if the specified session is a legacy-cam session.
    :param task: Background task object for message logging, progress updates, and premature cancellation.
    """
    # load videos to verify both are present and to get nominal frame rate (needed later)
    task.message_logged.emit(f"Verifying session videos...")
    videos = _load_required_videos_for_analysis(sesh_id)
    required_cams = [cam for cam in videos.keys()]
    frame_rate = videos[required_cams[0]].frame_rate

    # fixed-cam session only: validation the calibration folder contents
    flir_calib: Optional[FlirCalibration] = None
    if sesh_id.is_fixed_cam:
        emsg, flir_calib = FlirCalibration.create_flir_calibration(calib_path)
        if flir_calib is None:
            raise Exception(f"Bad calibration folder for fixed-cam session: {emsg}")
        else:
            task.message_logged.emit(f"Loaded calibration info from {calib_path.parent.name}/{calib_path.name}")

    enable_pp = cfg.get("enable_pp", True)   # process the two videos in parallel or sequentially

    task.message_logged.emit(f"Preparing infrastructure to analyze two session "
                             f"videos {'in parallel' if enable_pp else 'sequentially'}.")
    task.progress_updated.emit(0)

    # need separate configuration objects for each process!
    cfgs = [dict(cfg), dict(cfg)]

    # track progress for each process
    progress: Dict[RxCam, int] = {cam: 0 for cam in required_cams}
    results: Dict[RxCam, Optional[np.ndarray]] = {cam: None for cam in required_cams}
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    event_queue = manager.Queue(maxsize=2000)
    cancel_event = manager.Event()
    executor = ProcessPoolExecutor(max_workers=2, mp_context=ctx)
    canceled = False
    try:
        # use the existing MP infrastructure to analyze the two videos in parallel, or one after the other!
        for idx in range(2):
            if enable_pp:
                if idx == 0:
                    futures = {
                        executor.submit(
                            _detect_body_parts_in_video,
                            cfgs[0],
                            str(sesh_id.location),
                            required_cams[0].value,
                            _QueueTask(required_cams[0].value, event_queue, cancel_event),
                        ): required_cams[0],
                        executor.submit(
                            _detect_body_parts_in_video,
                            cfgs[1],
                            str(sesh_id.location),
                            required_cams[1].value,
                            _QueueTask(required_cams[1].value, event_queue, cancel_event)
                        ): required_cams[1],
                    }
                else:
                    futures = {}
            else:
                futures = {
                    executor.submit(
                        _detect_body_parts_in_video,
                        cfgs[idx],
                        str(sesh_id.location),
                        required_cams[idx].value,
                        _QueueTask(required_cams[idx].value, event_queue, cancel_event),
                    ): required_cams[idx]
                }

            while futures:
                if task.was_canceled():
                    canceled = True
                    task.message_logged.emit("Cancel requested. Stopping spawned process(es)...")
                    cancel_event.set()
                    for future in futures:
                        future.cancel()
                    _terminate_processes(executor)
                    break
                done, _ = wait(futures, timeout=0.2, return_when=FIRST_COMPLETED)
                while True:
                    try:
                        cam_value, event_type, payload = event_queue.get_nowait()
                    except queue.Empty:
                        break
                    _dispatch_process_event(task, progress, cam_value, event_type, payload)

                for future in done:
                    cam_id = futures.pop(future)
                    results[cam_id] = future.result()
                    task.message_logged.emit(f"Inference completed for {str(cam_id)} video.")
                    progress[cam_id] = 100
                    total_progress = int(sum(pct for pct in progress.values()) / len(progress))
                    task.progress_updated.emit(total_progress)

                    # if the completed process ended on an error (returns None), cancel the rest
                    if results[cam_id] is None:
                        cancel_event.set()

            while True:
                try:
                    cam_value, event_type, payload = event_queue.get_nowait()
                except queue.Empty:
                    break
                _dispatch_process_event(task, progress, cam_value, event_type, payload)
    finally:
        if canceled:
            executor.shutdown(wait=False, cancel_futures=True)
        else:
            executor.shutdown(wait=True, cancel_futures=False)
        manager.shutdown()

    if canceled or task.was_canceled():
        return

    if any(results[cam] is None for cam in required_cams):
        raise Exception("Inference analysis results missing for one or both cameras!")

    # save inference results
    out_dir = Path(sesh_id.location, rx_scorer)
    out_dir.mkdir(exist_ok=True)
    for cam in required_cams:
        out_file = Path(out_dir, f"{str(cam)}.npy")
        task.message_logged.emit(f"Saving inference results for {str(cam)} to {str(out_file)}")
        np.save(out_file, results[cam])

    task.progress_updated.emit(90)
    task.message_logged.emit(f"Extracting body part locations across both cams with "
                             f"confidence score > {CONF_THRESHOLD:0.1f}...")
    detected: Dict[RxCam, np.ndarray] = dict()
    for cam in required_cams:
        markers = results[cam]
        markers = markers[markers[:, 4] > CONF_THRESHOLD]
        n = int(markers.shape[0])
        cam_values = np.ones((n, 1)) * cam.value
        detected[cam] = np.column_stack((markers[:, 0:1], cam_values, markers[:, 1:]))

    if task.was_canceled():
        return

    # V-stack the detected markers from each cam, sort all markers by frame number (col 0), scale likelihood (col 5) by
    # 10000, then round and convert to int32; rounding doesn't affect frame number, body part value or cam value.
    cams = tuple(required_cams)
    detected_markers = np.vstack((detected[cams[0]], detected[cams[1]]))
    detected_markers[:, 5] = detected_markers[:, 5] * 10000
    detected_markers = detected_markers[detected_markers[:, 0].argsort()]
    detected_markers = np.round(detected_markers).astype(np.int32)

    out_file = Path(out_dir, "detected_markers.npy")
    task.message_logged.emit(f"  Saving results to: {str(out_file)}")
    np.save(out_file, detected_markers)

    if task.was_canceled():
        return

    task.progress_updated.emit(95)
    if not sesh_id.is_fixed_cam:
        _save_filtered_hand_and_pellet_trajectories(cfg, results[RxCam.SIDE], results[RxCam.FRONT], frame_rate,
                                                    out_dir, task)
    else:
        _process_fixed_cam_object_trajectories(cfg, results, frame_rate, flir_calib, out_dir, task)


def _load_required_videos_for_analysis(sesh_id: RxSessionID) -> Dict[RxCam, CamSource]:
    """
    Load the two camera videos that must be present to analyze a session.
    :param sesh_id: The session identifier.
    :return: A dictionary mapping camera identifier to a ``CamSource`` encapsulating the video.
    """
    required_cams = RxCam.required_cams(sesh_id.is_fixed_cam)
    assert len(required_cams) == 2
    videos = CamSource.load_session_cams(sesh_id, required_cams, get_null_logger())
    if len(videos) < 2:
        raise Exception(f"One or both required videos not found for {str(sesh_id)}")
    return videos


_PIX2MM_SIDECAM: float = 0.128865083
""" Multiplicative scale factor converts pixels to mm on sideCam. """
_PIX2MM_FRONTCAM: float = 0.2057297
""" Multiplicative scale factor converts pixels to mm on frontCam. """
_SG_WINDOW_LEN: int = 9
""" Window length for Savitz-Golay smoothing filter. """
_SG_POLYORDER: int = 3
""" Polynomial order for Savitz-Golay smoothing filter. """
_BW_CUTOFF_FREQ: float = 50.0
""" Cutoff frequency for Butterworth low-pass filter. """
_BW_FILTER_ORDER: int = 5
""" Filter order for Butterworth low-pass filter. """


def _save_filtered_hand_and_pellet_trajectories(
        cfg: Dict[str, Any], markers_side: np.ndarray, markers_front: np.ndarray, frame_rate: int, out_dir: Path,
        task: BackgroundTask) -> None:
    """
    Calculate the filtered trajectories of the hand and pellet in XYZ space for a legacy-cam session ONLY.

    Uses the inference output of a body part detection model, i.e., per-frame predicted locations (with confidence
    score) of various body parts in the side and front-cam session videos.

    The (X,Y) coordinates on the sideCam correspond to the (Y,Z) coordinates in this XYZ space, while the X-coordinate
    comes from the X-coordinate on the frontCam (Y-coordinate not used).

    The "hand" is really a composite of all the supported hand-like body parts. For each frame, the method looks at
    all hand-like body parts included in the model output and chooses the location with the highest likelihood as the
    location of the "hand" for that frame. The "pellet", of course, is ``RxBodyPart.PELLET``.

    After constructing the hand and pellet (X, Y, Z) trajectories from the raw model output and scaling the coordinates
    from pixels to millimeters, coordinate values with low confidence (ie, likelihood < 0.9) are interpolated from
    surrounding high-confidence values. Then a low-pass Butterworth filter is applied to each coordinate vector to
    form the filtered trajectories (X_filt, Y_filt, Z_filt) for hand and pellet, and from these the per-frame position
    change and speed. Finally, the speed vectors are passed through a Savitz-Golay smoothing filter.

    The trajectory data for the hand is saved as a single Numpy array ``np.array((N, 11), dtype=np.float32)`` in the
    file ``<out_dir>/hand.npy``. Here ``N`` is the number of frames in the recorded session, and each column corresponds
    to a different trajectory parameter in the following order: 'y', 'y_filt', 'z', 'z_filt', 'yz_likelihood', 'x',
    'x_filt', 'x_likelihood', 'distance', 'speed', 'speed_filt']. Analogously for the pellet trajectory data, which is
    saved to ``<out_dir>/pellet.npy``.

    :param cfg: Params defining the trained model used to make body part location predictions. The body parts included
        in the model analysis are listed (order is critical) in ``cfg['all_joints_names']``.
    :param markers_side: Detected body part locations for sideCam: ``np.array((N*M, 5), dtype=np.float32)``, where ``N``
        is the frame count and ``M`` is the number of unique body parts included in the output. Each row contains
        ``[frame_num, RxBodyPart.value, x_pix, y_pix, likelihood]``.
    :param markers_front: Analogously for frontCam.
    :param frame_rate: Nominal video frame rate in Hz.
    :param out_dir: Output directory. The hand/pellet trajectory arrays are saved to ``hand.npy`` and ``pellet.npy``
        within this directory.
    :param task: Background task object for message logging, progress updates, and premature cancellation.
    """
    # STEP 1: extract hand and pellet trajectory data from the model outputs
    task.message_logged.emit("Extracting hand and pellet trajectory data from predicted body part location data...")
    body_parts = [RxBodyPart.from_nickname(nickname) for nickname in cfg["all_joints_names"]]
    n_bp = len(body_parts)
    hand_indices = [i for i, part in enumerate(body_parts) if part.is_hand()]
    pellet_idx = next((i for i, part in enumerate(body_parts) if part == RxBodyPart.PELLET), -1)
    n_hand_parts = len(hand_indices)
    assert (n_hand_parts > 0) and (pellet_idx > -1)
    assert (len(markers_front) == len(markers_side)) and (len(markers_side) % n_bp == 0)
    n_frames = int(len(markers_side) / n_bp)

    # pellet: y, z, yz likelihood from sideCam; x, o, xlikelihood from frontCam
    pellet_y = markers_side[pellet_idx::n_bp, 2]  # sideCam X --> 'y'
    pellet_z = markers_side[pellet_idx::n_bp, 3]  # sideCam Y --> 'z'
    pellet_yz_lhood = markers_side[pellet_idx::n_bp, 4]  # sideCam likelihood --> 'yz likelihood'
    pellet_x = markers_front[pellet_idx::n_bp, 2]  # frontCam X --> 'x'
    pellet_x_lhood = markers_front[pellet_idx::n_bp, 4]  # frontCam likelihood --> 'x likelihood'

    # hand: Similarly, except that for each frame, keep only the hand part location with highest likelihood
    likelihood = np.empty((n_frames, n_hand_parts), dtype=np.float32)
    x = np.empty((n_frames, n_hand_parts), dtype=np.float32)
    y = np.empty((n_frames, n_hand_parts), dtype=np.float32)

    # TRICKY: hand_indices contains the zero-based index of each hand part in the original model output, and we're
    # collecting x, y, likelihood for ONLY the hand parts, NOT pellet and any other parts
    for idx, idx_orig in enumerate(hand_indices):
        x[:, idx] = markers_side[idx_orig::n_bp, 2]
        y[:, idx] = markers_side[idx_orig::n_bp, 3]
        likelihood[:, idx] = markers_side[idx_orig::n_bp, 4]
    best_hand_indices = np.argmax(likelihood, axis=1)

    frame_num_vec = np.arange(n_frames)
    hand_y = x[frame_num_vec, best_hand_indices]  # sideCam X --> 'y'
    hand_z = y[frame_num_vec, best_hand_indices]  # sideCam Y --> 'z'
    hand_yz_lhood = likelihood[frame_num_vec, best_hand_indices]  # sideCam likelihood --> 'yz likelihood'

    for idx, idx_orig in enumerate(hand_indices):
        x[:, idx] = markers_front[idx_orig::n_bp, 2]
        y[:, idx] = markers_front[idx_orig::n_bp, 3]
        likelihood[:, idx] = markers_front[idx_orig::n_bp, 4]
    best_hand_indices = np.argmax(likelihood, axis=1)

    hand_x = x[frame_num_vec, best_hand_indices]  # frontCam X --> 'x'
    hand_x_lhood = likelihood[frame_num_vec, best_hand_indices]  # frontCam likelihood --> 'x likelihood'

    if task.was_canceled():
        return

    # STEP 2: Convert to mm, filter trajectory data, compute distance (from image origin) and speed
    task.message_logged.emit("Filtering hand and pellet trajectories...")
    hand_y *= _PIX2MM_SIDECAM
    hand_z *= _PIX2MM_SIDECAM
    pellet_y *= _PIX2MM_SIDECAM
    pellet_z *= _PIX2MM_SIDECAM
    hand_x *= _PIX2MM_FRONTCAM
    pellet_x *= _PIX2MM_FRONTCAM

    # Butterworth filter and Savitz-Golay smoothing filter coefficients
    normalized_cutoff_freq = _BW_CUTOFF_FREQ / (0.5 * frame_rate)
    # noinspection All
    b, a = butter(_BW_FILTER_ORDER, normalized_cutoff_freq, btype='low', analog=False, output='ba')
    sg_coeffs = savgol_coeffs(_SG_WINDOW_LEN, _SG_POLYORDER)

    hand_traj = np.empty((n_frames, len(TrajCol)), dtype=np.float32)
    pellet_traj = np.empty((n_frames, len(TrajCol)), dtype=np.float32)
    for which, traj in enumerate([hand_traj, pellet_traj]):
        x = hand_x if which == 0 else pellet_x
        x_lhood = hand_x_lhood if which == 0 else pellet_x_lhood
        y = hand_y if which == 0 else pellet_y
        z = hand_z if which == 0 else pellet_z
        yz_lhood = hand_yz_lhood if which == 0 else pellet_yz_lhood

        # find indices of low-confidence values
        x_lhood[0] = 1
        x_lhood[-1] = 1
        yz_lhood[0] = 1
        yz_lhood[-1] = 1

        low_confidence_indices_x = x_lhood[:] < CONF_THRESHOLD
        low_confidence_indices_yz = yz_lhood[:] < CONF_THRESHOLD
        interp_counts_x = sum(low_confidence_indices_x)
        interp_counts_yz = sum(low_confidence_indices_yz)

        # linearly interpolate low-confidence values if enough high-confidence values are identified
        # NOTE: This implies you do interpolation if there is just ONE high-confidence value on both cams!
        if (interp_counts_x >= n_frames - 2) or (interp_counts_yz >= n_frames - 2):
            x[:] = 0   # broadcast to entire column
            y[:] = 0
            z[:] = 0
        else:
            # Interpolation
            x[:] = np.interp(frame_num_vec, frame_num_vec[~low_confidence_indices_x], x[~low_confidence_indices_x])
            y[:] = np.interp(frame_num_vec, frame_num_vec[~low_confidence_indices_yz], y[~low_confidence_indices_yz])
            z[:] = np.interp(frame_num_vec, frame_num_vec[~low_confidence_indices_yz], z[~low_confidence_indices_yz])

        # in case interpolation results in any NaNs (?)
        x[:][np.where(np.isnan(x[:]))] = 0
        y[:][np.where(np.isnan(y[:]))] = 0
        z[:][np.where(np.isnan(z[:]))] = 0

        if task.was_canceled():
            return

        # low-pass filtered versions of x, y, z trajectories
        x_filt = filtfilt(b, a, x)
        y_filt = filtfilt(b, a, y)
        z_filt = filtfilt(b, a, z)

        if task.was_canceled():
            return

        # calculate per-frame position change and speed in XYZ space, then apply SG smoothing filter to speed.
        pos_change = np.sqrt(np.diff(x_filt) ** 2 + np.diff(y_filt) ** 2 + np.diff(z_filt) ** 2)
        pos_change = np.concatenate(([0], pos_change))  # cuz output of np.diff is one less than original!
        speed = pos_change * (frame_rate / 1000.0)   # speed in mm/ms
        speed_filt = filtfilt(sg_coeffs, [1], speed)

        if task.was_canceled():
            return

        # store trajectory data in columns of Numpy array
        traj[:, TrajCol.Y.value] = y
        traj[:, TrajCol.Y_FILT.value] = y_filt
        traj[:, TrajCol.Z.value] = z
        traj[:, TrajCol.Z_FILT.value] = z_filt
        traj[:, TrajCol.YZ_LHOOD.value] = yz_lhood
        traj[:, TrajCol.X.value] = x
        traj[:, TrajCol.X_FILT.value] = x_filt
        traj[:, TrajCol.X_LHOOD.value] = x_lhood
        traj[:, TrajCol.DIST.value] = pos_change
        traj[:, TrajCol.SPEED.value] = speed
        traj[:, TrajCol.SPEED_FILT.value] = speed_filt

        if task.was_canceled():
            return
        task.progress_updated.emit(50 if which == 0 else 98)

    # save results
    out_file = Path(out_dir, "hand.npy")
    task.message_logged.emit(f"  Saving filtered hand trajectory data to: {str(out_file)}")
    np.save(out_file, hand_traj)
    out_file = Path(out_dir, "pellet.npy")
    task.message_logged.emit(f"  Saving filtered pellet trajectory data to: {str(out_file)}")
    np.save(out_file, pellet_traj)


def _process_fixed_cam_object_trajectories(
        cfg: Dict[str, Any], markers: Dict[RxCam, np.ndarray], frame_rate: int, flir_calib: FlirCalibration,
        out_dir: Path, task: BackgroundTask) -> None:
    """
    Process the 2D trajectories of each body part on the left- and right-side cameras for a **fixed-cam session**,
    computing 3D trajectory data for each body part included in the analysis.

    This method takes the inference output from a body part detection model and performs the following computations:
     - Extract individual body part trajectories for both left and right cams. A composite left-hand trajectory for
       ``RxBodyPart.L_HAND`` is formed by using the location of whichever left-hand pose (``RxBodyPart.LH_*``) has the
       highest confidence score in each frame. Analogously for the right-hand trajectory, assigned to the composite
       body part ``RxBodyPart.R_HAND``.
     - Interpolate and filter the X-vector and Y-vector for each 2D body part trajectory on each camera.
     - For each body part: triangulate the 2D trajectories on the left- and right-cam to form a ``Nx4`` 3D object
       trajectory ``(x, y, z, p)``, where ``N`` is the #frames and ``p==1`` for high-confidence locations and ``p=0``
       otherwise. Supplied Flir camera calibration information is required for this step.
     - Select ``RxBodyPart.DIAMOND`` to define the origin in 3D space, if present; otherwise use ``RxBodyPart.PELLET``.
       It is an error if neither body part is included in the model inference output. Find the median displacement of
       the selected BP wrt the original origin at (0, 0, 0). If the pellet is the reference point, omit pellet
       locations where it is moving >= 0.004 mm/ms.
     - Transform each individual body part trajectory to a new 3D coordinate space with origin defined in the previous
       step. See ``FlirCalibration.transform_3d_trajectory()``.
     - For each body part trajectory, compute vectors representing distance from origin (in the new coord space),
       speed, and filtered speed. The final trajectory array for each body part is a ``(N, 7) np.float32`` Numpy array
       with columns defined in ``FixedCamTrajCol``.
     - Save trajectory data in a single file ``<out_dir>/trajectories.npz``, where the name of each individual body part
       trajectory array matches that body part's nickname.

   :param cfg: Params defining the trained model used to make body part location predictions. The body parts included
        in the model analysis are listed (order is critical) in ``cfg['all_joints_names']``.
    :param markers: Detected body part locations on each camera, keyed by ``RxCam`` enumerant: ``np.array((N*M, 5),
        dtype=np.float32)``, where ``N`` is the frame count and ``M`` is the number of unique body parts included in the
        output. Each row contain ``[frame_num, RxBodyPart.value, x_pix, y_pix, likelihood]``.
    :param frame_rate: Nominal video frame rate in Hz.
    :param flir_calib: The FLIR camera calibration information needed to transform the left/right 2D object
        trajectories to 3D space.
    :param out_dir: Output directory where ``trajectories.npz`` will be written.
    :param task: Background task object for message logging, progress updates, and premature cancellation.
    """
    body_parts = [RxBodyPart.from_nickname(nickname) for nickname in cfg["all_joints_names"]]
    n_bp = len(body_parts)

    task.message_logged.emit(f"Processing trajectory data for {n_bp} body parts from predicted location data "
                             "on left and right cams...")

    r_hand_indices = [i for i, part in enumerate(body_parts) if part in RxBodyPart.fixed_cam_right_hand_poses()]
    l_hand_indices = [i for i, part in enumerate(body_parts) if part in RxBodyPart.fixed_cam_left_hand_poses()]
    pellet_idx = next((i for i, part in enumerate(body_parts) if part == RxBodyPart.PELLET), -1)
    assert (len(r_hand_indices) > 0) and (len(l_hand_indices) > 0) and (pellet_idx > -1)
    assert (RxCam.LEFT in markers) and (RxCam.RIGHT in markers)
    assert (len(markers[RxCam.LEFT]) == len(markers[RxCam.RIGHT])) and (len(markers[RxCam.LEFT]) % n_bp == 0)
    n_frames = int(len(markers[RxCam.LEFT]) / n_bp)
    frame_num_vec = np.arange(n_frames)

    # 3D trajectories are always centered about one of two body parts: preferably RxBodyPart.DIAMOND, which by design
    # is stationary, or the pellet itself. It is a fatal error if neither is available
    if not any([bp in body_parts for bp in [RxBodyPart.DIAMOND, RxBodyPart.PELLET]]):
        raise Exception("Missing diamond and pellet in trajectory data. Cannot continue.")

    trajectories: Dict[RxCam, Dict[RxBodyPart, np.ndarray]] = {RxCam.LEFT: dict(), RxCam.RIGHT: dict()}
    """ 
    Individual body part trajectories on the two cameras, keyed by body part. Each Numpy array is #frames x 3,
    with each row holding (x, y, likelihood).
    """

    # STEP 1: extract individual body part trajectories (x, y, likelihood) on each cam from the model output, with
    # special treatment for L,R hands
    for cam in [RxCam.LEFT, RxCam.RIGHT]:
        m = markers[cam]

        # prepare "composite" L- and R-hand trajectories, always using the hand part location with highest confidence
        for hand, indices in {RxBodyPart.R_HAND: r_hand_indices, RxBodyPart.L_HAND: l_hand_indices}.items():
            x = np.empty((n_frames, len(indices)), dtype=np.float32)
            y = np.empty((n_frames, len(indices)), dtype=np.float32)
            l_hood = np.empty((n_frames, len(indices)), dtype=np.float32)
            for idx, idx_orig in enumerate(indices):
                x[:, idx] = m[idx_orig::n_bp, 2]
                y[:, idx] = m[idx_orig::n_bp, 3]
                l_hood[:, idx] = m[idx_orig::n_bp, 4]
            best_hand_indices = np.argmax(l_hood, axis=1)

            trajectories[cam][hand] = np.column_stack(
                (x[frame_num_vec, best_hand_indices], y[frame_num_vec, best_hand_indices],
                 l_hood[frame_num_vec, best_hand_indices]))

        # handle all other body parts
        for idx, bp in enumerate(body_parts):
            if (idx in l_hand_indices) or (idx in r_hand_indices):
                continue
            trajectories[cam][bp] = m[idx::n_bp, 2:5]   # each row = (x, y, likelihood) for given frame

        if task.was_canceled():
            return

    # STEP 2: Interpolate and filter every X and Y trajectory for each body part on each cam
    task.message_logged.emit("Interpolating and filtering all object trajectories")

    # Butterworth filter
    normalized_cutoff_freq = _BW_CUTOFF_FREQ / (0.5 * frame_rate)
    # noinspection All
    b, a = butter(_BW_FILTER_ORDER, normalized_cutoff_freq, btype='low', analog=False, output='ba')

    task.progress_updated.emit(0)
    n_chunks = 2 * len(trajectories[RxCam.LEFT])
    i_chunk = 0
    for cam in [RxCam.LEFT, RxCam.RIGHT]:
        for bp_name, bp_traj in trajectories[cam].items():
            x = bp_traj[:, 0]
            y = bp_traj[:, 1]
            lhood = bp_traj[:, 2]

            # find indices where likelihood is below confidence threshold, but always accept 1st and last
            lhood[0] = 1
            lhood[-1] = 1
            below_threshold = lhood[:] < CONF_THRESHOLD
            if sum(below_threshold) >= n_frames - 2:
                x[:] = 0
                y[:] = 0
            else:
                x[:] = np.interp(frame_num_vec, frame_num_vec[~below_threshold], x[~below_threshold])
                y[:] = np.interp(frame_num_vec, frame_num_vec[~below_threshold], y[~below_threshold])
                # in case interpolation introduces any NaNs
                x[:][np.where(np.isnan(x[:]))] = 0
                y[:][np.where(np.isnan(y[:]))] = 0

            bp_traj[:, 0] = filtfilt(b, a, x)
            bp_traj[:, 1] = filtfilt(b, a, y)

            if task.was_canceled():
                return
            task.progress_updated.emit(int(100 * (i_chunk + 1) / n_chunks))

    task.message_logged.emit("Performing triangulation to compute 3D object trajectories...")
    task.progress_updated.emit(0)
    trajectories_3d: Dict[RxBodyPart, np.ndarray] = dict()
    n_traj = len(trajectories[RxCam.LEFT])
    for i_bp, bp in enumerate(trajectories[RxCam.LEFT].keys()):
        left_2d = trajectories[RxCam.LEFT][bp]
        right_2d = trajectories[RxCam.RIGHT][bp]
        trajectories_3d[bp] = _compute_3d_fixed_cam_trajectory(left_2d, right_2d, flir_calib)
        if task.was_canceled():
            return
        task.progress_updated.emit(int(100 * (i_bp + 1) / n_traj))

    # compute translation offset of centering BP from current origin in X,Y,Z. Since the current origin is (0,0,0),
    # we simply take the median value of the body part's X, Y, and Z trajectories -- only including high-confidence
    # (p==1) locations in the calculation. If the pellet will serve as the new origin, then we must also exclude
    # frames when it is in motion. The diamond is at a fixed location in the rig.
    center_bp = RxBodyPart.DIAMOND if RxBodyPart.DIAMOND in body_parts else RxBodyPart.PELLET
    task.message_logged.emit(f"Reorienting and centering 3D trajectories about origin at {str(center_bp)}...")
    task.progress_updated.emit(0)
    x_off, y_off, z_off = 0.0, 0.0, 0.0
    if len(trajectories_3d[center_bp]) < 10:
        task.message_logged.emit(f"WARNING - Cannot recenter on {str(center_bp)} -- trajectories too short!")
    else:
        trj = trajectories_3d[center_bp]
        x, y, z, p = trj[:, 0], trj[:, 1], trj[:, 2], trj[:, 3]
        if p.size == 0:
            task.message_logged.emit(f"WARNING: Cannot recenter on {str(center_bp)} - no high-confidence locations!")
        else:
            x = x[np.where(p == 1)]
            y = y[np.where(p == 1)]
            z = z[np.where(p == 1)]
            if center_bp == RxBodyPart.PELLET:
                # since the pellet moves sometimes, we must additionally exclude frames when it is in motion
                dist_vec = np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2)
                dist_vec = np.concatenate(([dist_vec[0]], dist_vec))  # cuz output of np.diff is one less than original!
                speed_vec = dist_vec * (frame_rate / 1000)  # convert to speed in mm/ms
                not_moving = np.abs(speed_vec) < 0.004
                if len(not_moving) > 0:
                    x = x[not_moving]
                    y = y[not_moving]
                    z = z[not_moving]
                else:
                    task.message_logged.emit(f"WARNING: Cannot recenter on pellet - found no staionary locations!")
            x_off = float(np.median(x))
            y_off = float(np.median(y))
            z_off = float(np.median(z))

    for i_bp, bp in enumerate(trajectories_3d.keys()):
        traj_3d = trajectories_3d[bp]
        # this updates the first 3 columns of traj_3d while leaving the 4th column ("p=0 or 1") untouched
        flir_calib.transform_3d_trajectory(traj_3d[:, :3], x_off, y_off, z_off)
        if task.was_canceled():
            return
        task.progress_updated.emit(int(100 * (i_bp + 1) / n_traj))

    # compute 3D distance and speed for each body part, appending columns to already computed X,Y,Z,P. For storage in
    # a Numpy NPZ file, each (N, M) body part trajectory array is assigned to a name matching the body part nickname.
    task.message_logged.emit("Computing 3D distance from origin and speed vectors for each body part...")
    task.progress_updated.emit(0)
    sg_coeffs = savgol_coeffs(_SG_WINDOW_LEN, _SG_POLYORDER)
    traj_3d_final: Dict[str, np.ndarray] = dict()   # body part nickname -> (N, len(FixedCamTrajCol)) traj data
    for i_bp, bp in enumerate(trajectories_3d.keys()):
        traj_3d = trajectories_3d[bp]
        x, y, z = traj_3d[:, 0], traj_3d[:, 1], traj_3d[:, 2]
        dist_from_ori = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)  # calculate dist from origin
        dist_from_ori = np.concatenate(([dist_from_ori[0]], dist_from_ori))  # cuz output of np.diff is one less!
        speed = dist_from_ori * (frame_rate / 1000)  # convert to speed in mm/ms
        speed_filt = filtfilt(sg_coeffs, [1], speed)  # filtered version

        # we have to convert each of the (N,) 1D vectors computed above to (N, 1) column vectors
        traj_3d_final[str(bp)] = np.hstack((traj_3d, dist_from_ori[:, np.newaxis], speed[:, np.newaxis],
                                            speed_filt[:, np.newaxis])).astype(dtype=np.float32)
        if task.was_canceled():
            return
        task.progress_updated.emit(int(100 * (i_bp + 1) / n_traj))

    # save computed trajectory data to NPZ file
    out_file = Path(out_dir, "trajectories.npz")
    task.message_logged.emit(f"  Saving 3D body part trajectory data to: {str(out_file)}")
    np.savez(out_file, **traj_3d_final)


MIN_CLUSTER: int = 10
""" Minimum length of a sequence of low-confidence 3D points to mark entire sequence as low-confidence. """


def _compute_3d_fixed_cam_trajectory(left_2d: np.ndarray, right_2d: np.ndarray, flir: FlirCalibration) -> np.ndarray:
    """
    Compute the 3D trajectory of an object given its 2D trajectories on the left and right FLIR camera views of a
    calibrated fixed-cam rig.

    Each 2D trajectory is undistorted IAW the FLIR calibration information, then the 2D trajectories are triangulated
    to form a unified 3D trajectory. A fourth column in the output array marks each 3D point as "accepted" (1) or
    not (0). This column is derived from an analysis of a "low confidence mask", set to True for each 3D point where
    either or both corresponding L/R 2D points have a confidence score < 0.9. Clusters of sequential low-confidence
    points of length 10 or more are marked as "not accepted"; all other points are "accepted". This effectively
    "passes" more of the 3D points so long as they're not in a low-confidence cluster.

    :param left_2d: A 2D body part trajectory on left cam. An ``(N, 3)`` array holding ``(x_pix, y_pix, likelihood)``
        for each of ``N`` video frames.
    :param right_2d: Analogouslly for that body part's 2D trajectory on right cam.
    :param flir: FLIR camera calibration information for the rig on which session was recorded.
    :return: The body part's computed 3D trajectory: ``(N, 4)`` ``np.float32`` array holding ``(x, y, z, p)`` for each
        of N frames. Note ``p=1`` for "accepted" points and ``p=0`` for all points in low-confidence clusters, as
        described above.
    """
    # undistort (X,Y) trajectories on both cams. The likelihood column is left unchanged.
    flir.undistort_2d_trajectory(left_2d, is_left=True)
    flir.undistort_2d_trajectory(right_2d, is_left=False)

    # triangulate XY left and XY right -> XYZ. Note that likelihood columns excluded.
    traj_3d = flir.triangulate_2d_trajectories(left_2d[:, :2], right_2d[:, :2])

    # boolean mask marking all 3D points as low- (True) or high-confidence (False). Note that both left and right 2D
    # locations must have high confidence for the corres 3D point to have high confidence.
    conf_left = left_2d[:, 2] >= CONF_THRESHOLD
    conf_right = right_2d[:, 2] >= CONF_THRESHOLD
    low_confidence = ~(conf_left & conf_right)

    # mark all clusters of low-confidence of sufficient length, but treat all others as high-confidence. This
    # effectively "passes" more of the 3D points if they're not in a low-confidence cluster!
    cluster_mask = np.zeros_like(low_confidence, dtype=bool)  # all high-confidence (False) initially
    start = None
    for i, value in enumerate(low_confidence):
        if value:  # Found a low-confidence value
            if start is None:
                start = i  # Start of a new cluster
        else:
            if start is not None:
                # End of a cluster, check its length
                if i - start >= MIN_CLUSTER:
                    cluster_mask[start:i] = True  # Mark the entire cluster as True
                start = None

    # handle the case where the cluster goes until the last element
    if start is not None and len(low_confidence) - start >= MIN_CLUSTER:
        cluster_mask[start:] = True

    high_confidence = ~cluster_mask
    return np.column_stack((traj_3d, high_confidence.astype(np.float32)))


def test_process_fixed_cam_traj(n_frames: int, frame_rate: int, scorer_dir: Path, calib_path: Path) -> None:
    """
    Temporary test fixture for testing ``_process_fixed_cam_object_trajectories()``.

    Calls the method directly (not on a background thread), and supplies a dummy ``BackgroundTask`` object which just
    prints messages to the console. The method call is wrapped in a try-except block to catch any exceptions and
    print a stack trace for debugging purposes. Time elapsed is reported each time a 100% progress update is
    received, which will give a sense of how long each step takes in ``_process_fixed_cam_object_trajectories()``.

    The results of model inference analysis on the left- and right-cam videos are stored as ``(N*P, 5)`` NumPy arrays in
    the scorer folder in ``left.npy`` and ``right.npy``, respectively. Here ``N`` is the # of video frames, and ``P`` is
    the number of body parts analyzed. Each row represents a single body part location prediction: ``[frame_num,
    bp_value, x, y, likelihood]``, where ``int(bp_value)`` is an ``RxBodyPart`` enumerated value identifying the body
    part. The first ``P`` rows contain the body part location predictions for the first frame, and so on -- always in
    the same order. Thus, we can identify the body parts included in the analysis and their order by looking at the
    values in the second column of the first P rows.

    :param n_frames: Video duration in frames. Method uses this to identify body parts accounted for in left.npy and
        right.npy, as described.
    :param frame_rate: The video frame rate in Hz.
    :param scorer_dir: The scorer folder. The ``left.npy`` and `right.npy`` files must be present. If the call to
        ``_process_fixed_cam_object_trajectories()`` is successful, the body part trajectories are stored as individual
        named arrays in ``trajectories.npy``, where each array name is the corresponding body part's nickname.
    :param calib_path: The calibration folder for the session analyzed.
    """
    try:
        print(f"Loading predicted body part location arrays for left/right cams...")
        left_markers_file = Path(scorer_dir, "left.npy")
        right_markers_file = Path(scorer_dir, "right.npy")
        if not all([f.is_file() for f in [left_markers_file, right_markers_file]]):
            raise Exception("Bad scorer folder, or missing left.npy and/or right.npy.")
        markers: Dict[RxCam, np.ndarray] = dict()
        markers[RxCam.LEFT] = np.load(left_markers_file)
        markers[RxCam.RIGHT] = np.load(right_markers_file)

        # validate shape of left/right outputs and get body part nicknames analyzed, in order
        shape = markers[RxCam.LEFT].shape
        ok = ((len(shape) == 2) and (shape[1] == 5) and (shape[0] % n_frames == 0) and
              (shape == markers[RxCam.RIGHT].shape))
        if not ok:
            raise Exception("Something is wrong with contents of left/right.npy!")
        n_bp = int(shape[0] / n_frames)
        body_part_vals = markers[RxCam.LEFT][0:n_bp, 1]
        body_part_order: List[str] = [str(RxBodyPart(val)) for val in body_part_vals]
        print(f"Body parts: {body_part_order}")

        # prepare pseudo model config dictionary. _process_fixed_cam_object_trajectories() only accesses one field.
        cfg: Dict[str, Any] = dict()
        cfg["all_joints_names"] = body_part_order

        # load calibration information
        emsg, flir_calib = FlirCalibration.create_flir_calibration(calib_path)
        if flir_calib is None:
            raise Exception(f"Bad calibration folder for fixed-cam session: {emsg}")
        else:
            print(f"Loaded calibration info from {calib_path.parent.name}/{calib_path.name}")

        t0 = time.time()

        def print_elapsed_time(pct: int):
            if pct == 100:
                print(f"Elapsed time: {time.time() - t0:0.3f} seconds.")

        dummy_task = BackgroundTask()
        dummy_task.message_logged.connect(lambda msg: print(msg))
        dummy_task.progress_updated.connect(print_elapsed_time)
        dummy_task.finished.connect(lambda: print("Finished!"))

        print(f"Calling process_fixed_cam_object_trajectories() at elapsed time: {time.time() - t0:0.3f} seconds.")
        _process_fixed_cam_object_trajectories(cfg, markers, frame_rate, flir_calib, scorer_dir, dummy_task)
        print_elapsed_time(100)
    except Exception as e:
        print(f"ERROR: {str(e)}")
        traceback.print_exc()


_BATCH_FRM = 10
""" TODO: Describe (constant used in reach segmentation). """
_BATCH_DIST = 5
""" TODO: Describe (constant used in reach segmentation). """
_BATCH_DROP = 25
""" TODO: Describe (constant used in reach segmentation). """
_BATCH_SPEED = 5
""" TODO: Describe (constant used in reach segmentation). """
_BATCH_STALL = 25
""" TODO: Describe (constant used in reach segmentation). """


def find_reach_segments(sesh_id: RxSessionID, rx_scorer: str, seg_parms: RxReachSegControls, task: BackgroundTask,
                        verbose: bool = False) -> None:
    """
    Examine the hand and pellet trajectory data for the specified experiment session -- derived from analysis using the
    specified body part detection model -- and find and categorize all reach segments during that session.

    The segmentation algorithm for a fixed-cam session is significantly different from that for a legacy-cam session.
    For details, see ``_find_reaches_legacy_cam()`` and ``_find_reaches_fixed_cam()``. While both algorithms use some
    of the  same user-specified control parameters, others are unique to one or the other. The ``RxReachSegControls``
    class encapsulates all control parameters for either type of session.

    :param sesh_id: The session identifier.
    :param rx_scorer: Identifies the "scorer", ie, the particular model instance that generated the output. Analysis
        results needed for reach segmentation are located in the directory: ``<session_folder/<rx_scorer>``, and the
        reach segmentation results for that scorer are placed in ``.../<rx_scorer>/detected_reaches.txt``.
    :param seg_parms: The user-specified control parameters for the reach segmentation algorithm.
    :param task: Background task object for message logging, progress updates, and premature cancellation.
    :param verbose: If True, detailed messages are posted via the task object as the algorithm detects and classifies
        individual reach segment. Intended for debugging purposes. Default = False.
    """
    if sesh_id.is_fixed_cam:
        _find_reaches_fixed_cam(sesh_id, rx_scorer, seg_parms, task, verbose)
    else:
        _find_reaches_legacy_cam(sesh_id, rx_scorer, seg_parms, task, verbose)


def _find_reaches_legacy_cam(sesh_id: RxSessionID, rx_scorer: str, seg_parms: RxReachSegControls, task: BackgroundTask,
                             verbose: bool = False) -> None:
    """
    Helper method performs automated reach segmentation for a legacy-cam session.

    The method expects the hand and pellet trajectory data to be found in files ``hand.npy`` and ``pellet.npy``,
    respectively, in the directory ``<session_folder>/<rx_scorer>/``. It also reads in the session's event timestamps
    from ``<session_folder>/<session_file_prefix>events.txt``.

    The algorithm scans each "pellet delivery epoch", ie, all frames between consecutive ``RxEvent.DELIVERY`` events,
    looking sequentially for a reach segment. The segmentation control parameters are used to perform a series of tests
    based on the hand and pellet trajectories. These tests detect when a reach segment starts, when the "max reach"
    point occurs, and when the reach segment ends. Other tests determine whether the pellet was dropped,
    missed, or grabbed, or whether the reach segment stalled out. When a reach segment completes on a stall, pellet
    drop, or successful pellet grab, the algorithm appends the completed reach and moves on to the next delivery epoch.
    Reach segmments ending in a "miss" are also kept, but the algorithm continues looking in the current delivery epoch
    for another reach attempt.

    Once finished, the list of reach segments found are saved at ``<session_folder>/<rx_scorer>/detected_reaches.txt``.
    Thus, researchers can perform body part detection analysis and reach segmentation on a session using more than one
    "scorer". Furthermore, the model-driven reach segments are kept distinct from the session's user-curated reaches,
    which are stored in ``<session_folder>/<session_file_prefix>reaches.txt``.

    For details on the structure of the Numpy arrays holding the hand and pellet trajectory data, see the function
    ``_save_filtered_hand_and_pellet_trajectories()``.

    :param sesh_id: The session identifier.
    :param rx_scorer: Identifies the "scorer", ie, the particular model instance that generated the output.
    :param seg_parms: The user-specified control parameters for the reach segmentation algorithm.
    :param task: Background task object for message logging, progress updates, and premature cancellation.
    :param verbose: If True, detailed messages are posted via the task object as the algorithm detects and classifies
        individual reach segment. Intended for debugging purposes. Default = False.
    """
    task.progress_updated.emit(0)
    task.message_logged.emit(f"Starting reach segmentation on session '{str(sesh_id)}' using scorer: {rx_scorer}...")
    try:
        assert not sesh_id.is_fixed_cam
        task.message_logged.emit("Loading trajectory data and session events..")

        # load hand and pellet trajectory data and session events
        out_dir = Path(sesh_id.location, rx_scorer)
        if not out_dir.is_dir():
            raise Exception(f"Analysis results directory not found: {str(out_dir)}")

        hand = np.load(Path(out_dir, "hand.npy"))
        pellet = np.load(Path(out_dir, "pellet.npy"))
        pellet_x_lhood = pellet[:, TrajCol.X_LHOOD.value]
        pellet_yz_lhood = pellet[:, TrajCol.YZ_LHOOD.value]
        hand_x_lhood = hand[:, TrajCol.X_LHOOD.value]
        hand_yz_lhood = hand[:, TrajCol.YZ_LHOOD.value]
        pellet_x_filt = pellet[:, TrajCol.X_FILT.value]
        pellet_y_filt = pellet[:, TrajCol.Y_FILT.value]
        pellet_z_filt = pellet[:, TrajCol.Z_FILT.value]
        hand_x_filt = hand[:, TrajCol.X_FILT.value]
        hand_y_filt = hand[:, TrajCol.Y_FILT.value]
        hand_z_filt = hand[:, TrajCol.Z_FILT.value]
        pellet_speed_filt = pellet[:, TrajCol.SPEED_FILT.value]
        hand_speed_filt = hand[:, TrajCol.SPEED_FILT.value]

        sesh_mgr = SessionManager(suppress_log=True, auto_save=False)
        emsg = sesh_mgr.load(sesh_id)
        if isinstance(emsg, str):
            raise Exception(f"Error loading session: {emsg}")
        frame_count = sesh_mgr.total_frames
        frame_rate = sesh_mgr.master_frame_rate
        # we scan for reach segments between successive pellet delivery events
        key_frames = sesh_mgr.events.get_all_occurrences_of(RxEvent.DELIVERY)
        sesh_mgr.unload()

        if task.was_canceled():
            return

        batch_frm = _BATCH_FRM
        batch_dist = _BATCH_DIST
        batch_drop = _BATCH_DROP
        batch_speed = _BATCH_SPEED
        batch_stall = _BATCH_STALL
        sg_coeffs = savgol_coeffs(_SG_WINDOW_LEN, _SG_POLYORDER)
        task.message_logged.emit(f"Finding reach segments in {len(key_frames)} pellet delivery epochs...")
        reaches: List[RxReachSegment] = list()
        for kf_idx, start_frame in enumerate(key_frames):
            # define dist and velo for each reach sequence (changes according to start_frame)
            search_end = key_frames[kf_idx + 1] - batch_frm - 1 if kf_idx < len(key_frames) - 1 else frame_count
            search_list = np.arange(start_frame + 20, search_end)

            # precompute pellet likelihoods for the search range
            pellet_detected_indices = np.where(
                (np.convolve(pellet_x_lhood[search_list] > seg_parms.confidence, np.ones(batch_frm),
                             mode='valid') / batch_frm > 0.75) &
                (np.convolve(pellet_yz_lhood[search_list] > seg_parms.confidence, np.ones(batch_frm),
                             mode='valid') / batch_frm > 0.75)
            )[0]

            if len(pellet_detected_indices) == 0:
                task.message_logged.emit(f"Skipping reach epoch #{kf_idx} starting at frame {start_frame}: "
                                         f"pellet origin not found.")
                continue

            sfp = search_list[pellet_detected_indices[0]]
            distance_p = np.sqrt((pellet_x_filt - pellet_x_filt[sfp]) ** 2 +
                                 (pellet_y_filt - pellet_y_filt[sfp]) ** 2 +
                                 (pellet_z_filt - pellet_z_filt[sfp]) ** 2)
            distance_hvpp = np.sqrt((hand_x_filt - pellet_x_filt[sfp]) ** 2 +
                                    (hand_y_filt - pellet_y_filt[sfp]) ** 2 +
                                    (hand_z_filt - pellet_z_filt[sfp]) ** 2)
            z_dist_h = hand_z_filt - pellet_z_filt[sfp]
            z_dist_p = pellet_z_filt[sfp] - pellet_z_filt
            y_dist_p = pellet_y_filt[sfp] - pellet_y_filt

            velocity_h = np.diff(distance_hvpp) * (frame_rate / 1000)
            velocity_h_filt = filtfilt(sg_coeffs, [1], velocity_h)

            food_was_dropped = False
            frame = sfp - 20
            pellet_detected = False
            curr_reach_init = -1   # starting frame number for reach segment  under construction
            curr_reach_max = -1    # max hand velocity frame number for reach segment under construction
            while frame < frame_count - batch_frm:
                # stop when we reach the next pellet delivery epoch
                if kf_idx < len(key_frames) - 1 and frame >= key_frames[kf_idx + 1]:
                    if (curr_reach_init == -1) and verbose:
                        if not pellet_detected:
                            task.message_logged.emit(f"No pellet was detected during delivery epoch #{kf_idx}")
                        else:
                            task.message_logged.emit(f"Hit end of epoch #{kf_idx} without finding additional reach")
                    break

                # Precompute likelihoods for the current frame range
                hand_detected = (
                        (np.sum(hand_x_lhood[frame:frame + batch_frm] > seg_parms.confidence) / batch_frm > 0.75) and
                        (np.sum(hand_yz_lhood[frame:frame + batch_frm] > seg_parms.confidence) / batch_frm > 0.75)
                )
                pellet_detected = (
                        (np.sum(pellet_x_lhood[frame:frame + batch_frm] > seg_parms.confidence) / batch_frm > 0.75) and
                        (np.sum(pellet_yz_lhood[frame:frame + batch_frm] > seg_parms.confidence) / batch_frm > 0.75)
                )

                if curr_reach_init == -1:  # Search for a reach initiation
                    if pellet_detected and hand_detected:
                        test_a = z_dist_h[frame] > seg_parms.distZ_hand_start
                        test_b = np.mean(velocity_h_filt[frame:frame + batch_speed]) < seg_parms.reach_init_speed
                        if test_a and test_b:
                            if verbose:
                                task.message_logged.emit(f"Starting new reach segment at frame {frame}.")
                            curr_reach_init = frame
                            food_was_dropped = False
                            speed_hvh_init = np.diff(distance_hvpp - distance_hvpp[frame]) * (frame_rate / 1000)
                            speed_hvh_init = filtfilt(sg_coeffs, [1], speed_hvh_init)
                elif curr_reach_max == -1:  # Search for reach max
                    if np.any(pellet_speed_filt[frame:frame + batch_drop] > seg_parms.pellet_drop_speed):
                        food_was_dropped = True
                    if np.any(y_dist_p[frame:frame + batch_drop] < seg_parms.pellet_drop_distY):
                        food_was_dropped = True
                    if np.any(z_dist_p[frame:frame + batch_drop] < seg_parms.pellet_drop_distZ):
                        z_dist_indices = np.where(z_dist_p[frame:frame + batch_drop] < seg_parms.pellet_drop_distZ)
                        if np.any(pellet_yz_lhood[frame:frame + batch_drop][z_dist_indices] > seg_parms.confidence):
                            food_was_dropped = True
                    # noinspection PyUnboundLocalVariable
                    if np.mean(speed_hvh_init[frame:frame + batch_speed]) > seg_parms.reach_dirchange_speed:
                        if verbose:
                            task.message_logged.emit(f"Detected reach max at frame {frame + 3}.")
                        curr_reach_max = frame + 3
                        frame += 3
                else:  # Search for reach end
                    keep_looking = True
                    test_a = np.mean(distance_hvpp[frame:frame + batch_dist]) > seg_parms.dist_thresh_end
                    test_b = np.mean(speed_hvh_init[frame:frame + batch_speed]) < seg_parms.reach_dirchange_speed
                    test_c = np.mean(velocity_h_filt[frame:frame + batch_speed]) < seg_parms.reach_init_speed
                    test_d = np.allclose(np.mean(distance_hvpp[frame:frame + batch_stall]),
                                         distance_hvpp[frame], atol=2)
                    test_e = np.isclose(np.mean(hand_speed_filt[frame:frame + batch_stall]), 0, atol=0.025)
                    test_f = np.all(distance_hvpp[frame:frame + batch_stall] < 6)

                    if np.any(pellet_speed_filt[frame:frame + batch_drop] > seg_parms.pellet_drop_speed):
                        food_was_dropped = True
                    if np.any(y_dist_p[frame:frame + batch_drop] < seg_parms.pellet_drop_distY):
                        food_was_dropped = True
                    if np.any(z_dist_p[frame:frame + batch_drop] < seg_parms.pellet_drop_distZ):
                        z_dist_indices = np.where(z_dist_p[frame:frame + batch_drop] < seg_parms.pellet_drop_distZ)
                        if np.any(pellet_yz_lhood[frame:frame + batch_drop][z_dist_indices] > seg_parms.confidence):
                            food_was_dropped = True

                    reach_ended = False
                    if test_a and test_c and not food_was_dropped:
                        if verbose:
                            task.message_logged.emit(f"Reach segment ended at frame {frame - 1}: MISSED.")
                        reaches.append(
                            RxReachSegment(curr_reach_init, curr_reach_max - curr_reach_init,
                                           frame - 1 - curr_reach_init, RxReach.MISSED, RxHandPos.UNSPECIFIED))
                        reach_ended = True
                    elif test_d and test_e and test_f:
                        if verbose:
                            task.message_logged.emit(f"Reach segment ended at frame {frame + 10}: STALLED.")
                        reaches.append(
                            RxReachSegment(curr_reach_init, curr_reach_max - curr_reach_init,
                                           frame + 10 - curr_reach_init, RxReach.STALLED, RxHandPos.UNSPECIFIED))
                        reach_ended = True
                        keep_looking = False
                    elif test_a and test_b:
                        p_test = np.mean(distance_p[frame:frame + batch_frm]) < seg_parms.pellet_dist_to_origin
                        if food_was_dropped:
                            if verbose:
                                task.message_logged.emit(f"Reach segment ended at frame {frame + 2}: DROPPED.")
                            reaches.append(
                                RxReachSegment(curr_reach_init, curr_reach_max - curr_reach_init,
                                               frame + 2 - curr_reach_init, RxReach.DROPPED, RxHandPos.UNSPECIFIED))
                            reach_ended = True
                            keep_looking = False
                        elif pellet_detected and p_test:
                            if verbose:
                                task.message_logged.emit(f"Reach segment ended at frame {frame + 2}: MISSED.")
                            reaches.append(
                                RxReachSegment(curr_reach_init, curr_reach_max - curr_reach_init,
                                               frame + 2 - curr_reach_init, RxReach.MISSED, RxHandPos.UNSPECIFIED))
                            reach_ended = True
                            frame += 2
                        else:
                            if verbose:
                                task.message_logged.emit(f"Reach segment ended at frame {frame + 2}: GRABBED.")
                            reaches.append(
                                RxReachSegment(curr_reach_init, curr_reach_max - curr_reach_init,
                                               frame + 2 - curr_reach_init, RxReach.GRABBED, RxHandPos.UNSPECIFIED))
                            reach_ended = True
                            keep_looking = False

                    # if we found a reach segment, discard it if its duration is too long or too short, or the hand
                    # is too far from pellet at the max velocity frame (?).
                    if reach_ended:
                        curr_reach_init = curr_reach_max = -1
                        test_x = reaches[-1].dur < seg_parms.min_frame
                        test_y = reaches[-1].dur > seg_parms.max_frame
                        test_z = distance_hvpp[reaches[-1].max_frame] > 15
                        if test_x or test_y or test_z:
                            if verbose:
                                task.message_logged.emit(f"Reach segment at frame {reaches[-1].frame} DISCARDED!")
                            reaches.pop()

                    # we only keep looking in the current epoch when the pellet was missed
                    if not keep_looking:
                        break
                frame += 1

            if task.was_canceled():
                return
            pct = int(100 * (kf_idx + 1) / len(key_frames))
            task.progress_updated.emit(pct)

        # save the detected reach segments to dedicated file within session subfolder containing scorer's results
        if len(reaches) > 0:
            out_file = Path(sesh_id.location, rx_scorer, "detected_reaches.txt")
            task.message_logged.emit(f"Saving {len(reaches)} reach segments found: {str(out_file)}")
            emsg = RxReachSegment.save_reach_segments(reaches, out_file)
            if len(emsg) > 0:
                task.message_logged.emit(emsg)

    except Exception as e:
        task.message_logged.emit(f"Reach segmenation FAILED: {e}")
        traceback.print_exc()


_TIME_TO_PLACE: float = 0.05
""" Duration (sec) that pellet must be near enough to 'pellet home' to be considered 'placed'. """
_MAX_DIST_FROM_HOME: float = 2.0
""" Maximum distance (mm) that pellet can be from 'pellet home' and still be 'placed'. """
_TIME_TO_LOST: float = 0.1
""" Duration (sec) that pellet must be undetected or far enough from 'pellet home' to be considered 'lost'. """
_MIN_INTER_PELLET_INTV: float = 5.0
""" Minimum duration (sec) after pellet 'lost' before the next pellet 'placed' can occur. """
_MIN_DIST_FOR_GRAB: float = 15.0
""" 
Minimum distance (mm) of hand from pellet home at 'pellet lost' frame to consider the pellet 'grabbed'. In 
addition, a candidate reach segment is rejected if distance of hand from pellet home exceeds this value at the
"reach max" frame.
"""
_POS_WINDOW_NUM_FRAMES: int = 100
""" Size of a position window (# video frames) for certain tests."""
_SPEED_WINDOW_NUM_FRAMES: int = 100
""" Size of a speed window (# video frames) for certain tests. """


def _find_reaches_fixed_cam(sesh_id: RxSessionID, rx_scorer: str, seg_parms: RxReachSegControls, task: BackgroundTask,
                            verbose: bool = False) -> None:
    """
    Helper method performs automated reach segmentation for a fixed-cam session.

    The method expects all body part trajectories to be stored in named ``(N, 7) np.float32`` Numpy arrays in the file
    ``trajectories.npz`` in the directory ``<session_folder>/<rx_scorer>/``. Each array's name matches the corresponding
    body part's "nickname". Trajectories for the left- and right-hand are denoted by the special body part enumerants
    ``RxBodyPart.L_HAND`` and ``RxBodyPart.R_HAND``, respectively. Each row in the array is ``(X, Y, Z, P, D, S, SF)``
    for each video frame during the session. Here ``P=1`` if body part was detected in that frame; else 0. ``D`` is
    distance from origin, ``S`` is raw speed (frame-by-frame diffs in ``D``), and ``SF`` is a filtered version of ``S``.

    Algorithm details:
     - **Phase 1**: Determine frames in timeline where pellet is placed and ready for "grabbing". These conditions
       must be met for ``0.05*frame_rate`` consecutive frames: pellet is detected, pellet is within 2mm of "pellet
       home" (where it is placed for grabbing), and the pellet cover is far enough away from pellet arm (or cover not
       installed. The method then "waits" for the pellet to "disappear" (eaten, lost, or moved away from animal), then
       "waits" another 5 seconds (minimimum inter-pellet interval) before searching for the next pellet placement frame.
     - **Phase 2**: Searches each pellet epoch for one or more reach segments. A pellet epoch starts 20 frames prior to
       when the pellet is placed, as determined in phase 1. This analysis is in many ways similar to the legacy-cam
       algorithm and uses many of the same segmentation control parameters. These parameters are used to perform tests
       based on the **right hand** and pellet trajectories. These tests detect when a reach segment starts, when
       the "max reach" point occurs, and when the reach segment ends. Other tests determine whether the pellet was
       dropped, missed, or grabbed, or whether the reach segment stalled out. When a reach segment completes on a stall,
       pellet drop, or successful pellet grab, the algorithm appends the completed reach and moves on to the next
       pellet placemeent frame. Reach segmments ending in a "miss" are also kept, but the algorithm continues looking in
       the current pellet epoch for another reach attempt.

    Once finished, the list of reach segments found are saved at ``<session_folder>/<rx_scorer>/detected_reaches.txt``.
    Thus, researchers can perform body part detection analysis and reach segmentation on a session using more than one
    "scorer". Furthermore, the model-driven reach segments are kept distinct from the session's user-curated reaches,
    which are stored in ``<session_folder>/<session_file_prefix>reaches.txt``.

    NOTE 1: Phase 1 generates a list of "pellet events", each of which is a dictionary containing the fields 'placed',
    'lost', 'method', and 'outcome'. The "outcome" is either "eaten" or "dropped", and the "method" field is the
    "reason" the pellet was "lost": "left_hand", "right_hand", "tongue", or "other". A final dict {"x": x_home,
    "y": y_home, "z": z_home} is appended to the list which gives the coordinates of "pellet home". This list is saved
    to ``<rx_scorer>/pelletHistory.pickle``.

    NOTE 2: Only right-handed reach segments are generated, even though it is possible the animal could use its left
    hand. So the contents of pelletHistory.pickle could be useful.

    :param sesh_id: The session identifier.
    :param rx_scorer: Identifies the "scorer", ie, the particular model instance that generated the output.
    :param seg_parms: The user-specified control parameters for the reach segmentation algorithm.
    :param task: Background task object for message logging, progress updates, and premature cancellation.
    :param verbose: If True, detailed messages are posted via the task object as the algorithm detects and classifies
        individual reach segment. Intended for debugging purposes. Default = False.
    """
    task.progress_updated.emit(0)
    task.message_logged.emit(f"Starting reach segmentation on session '{str(sesh_id)}' using scorer: {rx_scorer}...")
    try:
        assert sesh_id.is_fixed_cam

        # need frame rate because some time-valued control params are in seconds, not frames
        videos = _load_required_videos_for_analysis(sesh_id)
        frame_rate = next(iter(videos.values())).frame_rate

        task.message_logged.emit("Loading body part trajectory data...")
        out_dir = Path(sesh_id.location, rx_scorer)
        if not out_dir.is_dir():
            raise Exception(f"Analysis results directory not found: {str(out_dir)}")

        r_hand: np.ndarray
        l_hand: np.ndarray
        pellet: np.ndarray
        star: Optional[np.ndarray]
        triangle: Optional[np.ndarray]
        mid_tongue: Optional[np.ndarray]
        diamond_is_origin: bool
        with np.load(Path(out_dir, "trajectories.npz")) as data:
            # these absolutely must exist
            r_hand = data[str(RxBodyPart.R_HAND)]
            l_hand = data[str(RxBodyPart.L_HAND)]
            pellet = data[str(RxBodyPart.PELLET)]

            # these we can live without
            star = data[str(RxBodyPart.STAR)] if str(RxBodyPart.STAR) in data else None
            triangle = data[str(RxBodyPart.TRIANGLE)] if str(RxBodyPart.TRIANGLE) in data else None
            mid_tongue = data[str(RxBodyPart.TONGUE_MID)] if str(RxBodyPart.TONGUE_MID) in data else None

            # Diamond is origin if present, else pellet home is origin
            diamond_is_origin = str(RxBodyPart.DIAMOND) in data

        n_total_frames = r_hand.shape[0]

        if any([traj is None for traj in [star, triangle]]):
            task.message_logged.emit("WARNING: Missing Star and/or Triangle; cannot determine pellet cover state.")

        # calculate "pellet home" -- its location when not moving, ready to be grabbed. If the Diamond is present, the
        # 3D space's origin is at its fixed location. Otherwise, pellet home itself is the origin!
        if not diamond_is_origin:
            pellet_home = [0.0, 0.0, 0.0]
        else:
            # since pellet spends a lot of frames at "pellet home", the median values of X,Y,Z work here
            x, y = pellet[:, FixedCamTrajCol.X.value], pellet[:, FixedCamTrajCol.Y.value]
            z, p = pellet[:, FixedCamTrajCol.Z.value], pellet[:, FixedCamTrajCol.P.value]
            x, y, z = x[np.where(p == 1)], y[np.where(p == 1)], z[np.where(p == 1)]
            pellet_home = [float(np.median(c)) for c in [x, y, z]]

        # per-frame distance from pellet home for the body parts we care about
        distance_p = np.sqrt((pellet[:, FixedCamTrajCol.X.value] - pellet_home[0]) ** 2 +
                             (pellet[:, FixedCamTrajCol.Y.value] - pellet_home[1]) ** 2 +
                             (pellet[:, FixedCamTrajCol.Z.value] - pellet_home[2]) ** 2)
        distance_rh = np.sqrt((r_hand[:, FixedCamTrajCol.X.value] - pellet_home[0]) ** 2 +
                              (r_hand[:, FixedCamTrajCol.Y.value] - pellet_home[1]) ** 2 +
                              (r_hand[:, FixedCamTrajCol.Z.value] - pellet_home[2]) ** 2)
        distance_lh = np.sqrt((l_hand[:, FixedCamTrajCol.X.value] - pellet_home[0]) ** 2 +
                              (l_hand[:, FixedCamTrajCol.Y.value] - pellet_home[1]) ** 2 +
                              (l_hand[:, FixedCamTrajCol.Z.value] - pellet_home[2]) ** 2)

        # whether (1 or 0) pellet, R_Hand, and L_Hand were detected, per frame
        pellet_p = pellet[:, FixedCamTrajCol.P.value]
        r_hand_p = r_hand[:, FixedCamTrajCol.P.value]
        l_hand_p = l_hand[:, FixedCamTrajCol.P.value]

        # if star or triangle are missing, their per-frame separation distance is set to all NaNs
        dist_star_tri: np.ndarray
        if (star is not None) and (triangle is not None):
            dist_star_tri = np.sqrt(
                (star[:, FixedCamTrajCol.X.value] - triangle[:, FixedCamTrajCol.X.value]) ** 2 +
                (star[:, FixedCamTrajCol.Y.value] - triangle[:, FixedCamTrajCol.Y.value]) ** 2 +
                (star[:, FixedCamTrajCol.Z.value] - triangle[:, FixedCamTrajCol.Z.value]) ** 2
            )
        else:
            dist_star_tri = np.full(distance_p.shape, np.nan)

        # if tongue_mid is missing, these two column vectors are set to ensure tests below fail
        dist_midtongue: np.ndarray
        mid_tongue_p: np.ndarray
        if mid_tongue is not None:
            dist_midtongue = np.sqrt(
                (mid_tongue[:, FixedCamTrajCol.X.value] - pellet_home[0]) ** 2 +
                (mid_tongue[:, FixedCamTrajCol.Y.value] - pellet_home[1]) ** 2 +
                (mid_tongue[:, FixedCamTrajCol.Z.value] - pellet_home[2]) ** 2
            )
            mid_tongue_p = mid_tongue[:, FixedCamTrajCol.P.value]
        else:
            dist_midtongue = np.full(distance_p.shape, 10000)
            mid_tongue_p = np.full(distance_p.shape, 0)

        # Z-coordinate of pellet WRT to pellet home, with NaN where pellet not detected
        coord_pz = pellet[:, FixedCamTrajCol.Z.value] - pellet_home[2]
        coord_pz[pellet_p == 0] = np.nan

        # Z-coordinate of right hand wrt to pellet home
        coord_rhz = r_hand[:, FixedCamTrajCol.Z.value] - pellet_home[2]

        # filtered speed of pellet and right hand in original coord system, and right-hand speed rel to pellet home
        pellet_speed = pellet[:, FixedCamTrajCol.SPEED_FILT.value]
        rh_speed = r_hand[:, FixedCamTrajCol.SPEED_FILT.value]
        rh_rel_pos = np.diff(distance_rh) * (frame_rate / 1000)
        sg_coeffs = savgol_coeffs(_SG_WINDOW_LEN, _SG_POLYORDER)
        rh_rel_pos = np.concatenate(([rh_rel_pos[0]], rh_rel_pos))  # cuz output of np.diff is one less!
        rh_rel_speed = filtfilt(sg_coeffs, [1], rh_rel_pos)

        if task.was_canceled():
            return
        task.progress_updated.emit(5)

        # control parameters for PHASE 1 - for now these are all hard-coded
        n_consec_frames_to_place = _TIME_TO_PLACE * frame_rate
        n_consec_frames_to_lost = _TIME_TO_LOST * frame_rate
        max_dist_p_from_home = _MAX_DIST_FROM_HOME
        n_frames_inter_pellet_interval = _MIN_INTER_PELLET_INTV * frame_rate
        min_dist_for_grab = _MIN_DIST_FOR_GRAB

        # control parameters for phase 2 - some hard-coded, some specified by user
        batch_frm = _BATCH_FRM
        batch_dist = _BATCH_DIST
        batch_speed = _BATCH_SPEED
        batch_stall = _BATCH_STALL
        pos_window_nframes = _POS_WINDOW_NUM_FRAMES       # also used in phase 1
        speed_window_nframes = _SPEED_WINDOW_NUM_FRAMES

        confidence = seg_parms.confidence
        reach_init_speed = seg_parms.reach_init_speed
        reach_dirchange_speed = seg_parms.reach_dirchange_speed
        pellet_drop_speed = seg_parms.pellet_drop_speed
        pellet_drop_dist = seg_parms.pellet_drop_distZ    # also used in phase 1
        pellet_dist_to_origin = seg_parms.pellet_dist_to_origin
        max_dur_frames = seg_parms.max_frame
        min_dur_frames = seg_parms.min_frame
        dist_thresh_end = seg_parms.dist_thresh_end
        min_dist_from_pellet = _MIN_DIST_FOR_GRAB

        # PHASE 1: find all pellet events: pellet placed ready for grabbing, followed by pellet lost...
        task.message_logged.emit("Starting phase 1: Finding all pellet placement epochs...")
        frame_number = -1
        start_frame = 0
        count = 0
        pellet_state = 0  # = find next 'placed'; 1 = find 'lost'; 2 = wait for min inter-pellet interval
        pellet_events: List[Dict[str, Any]] = list()
        last_pct_update = 5
        for dp, st, pp in zip(distance_p, dist_star_tri, pellet_p):
            frame_number += 1

            # progress update
            pct = int((frame_number + 1) * 100 / n_total_frames)
            if (pct > last_pct_update) and (pct % 5 == 0):
                task.progress_updated.emit(pct)
                last_pct_update = pct
                if task.was_canceled():
                    return

            if pellet_state == 0:
                # search for next pellet placed frame
                test_a = dp <= max_dist_p_from_home  # pellet is close enough to pellet home
                test_b = pp == 1                     # pellet was detected in current frame
                test_c = np.isnan(st) or st > 12     # pellet cover open (or status not available)
                if not (test_a and test_b and test_c):
                    count = 0
                    start_frame = frame_number
                    continue

                count += 1
                if count >= n_consec_frames_to_place:  # pellet placed -- add new pellet event
                    pellet_events.append(
                        {'placed': start_frame, 'lost': -1, 'method': 'none', 'outcome': 'none'})
                    pellet_state = 1
                    count = 0
                    if verbose:
                        task.message_logged.emit(f"Pellet placed at {start_frame}.")

            elif pellet_state == 1:
                # search for pellete lost
                if dp > max_dist_p_from_home or pp == 0:
                    count += 1
                else:
                    count = 0
                    start_frame = frame_number

                if count >= n_consec_frames_to_lost:  # pellet lost -- fill out the current pellet event accordingly
                    pellet_dict = pellet_events[-1]
                    pellet_dict['lost'] = start_frame

                    r_grabbed = (distance_rh[start_frame] < min_dist_for_grab) and (r_hand_p[start_frame] == 1)
                    l_grabbed = (distance_lh[start_frame] < min_dist_for_grab) and (l_hand_p[start_frame] == 1)
                    tongue_present = mid_tongue_p[start_frame] == 1
                    r_closer_than_l = distance_rh[start_frame] < distance_lh[start_frame]
                    t_closer_than_r = dist_midtongue[start_frame] < distance_rh[start_frame]
                    t_closer_than_l = dist_midtongue[start_frame] < distance_lh[start_frame]
                    pellet_dict['outcome'] = 'eaten'
                    if tongue_present and t_closer_than_r and t_closer_than_l:
                        pellet_dict['method'] = 'tongue'
                    elif r_grabbed and r_closer_than_l:
                        pellet_dict['method'] = 'right_hand'
                    elif l_grabbed and not r_closer_than_l:
                        pellet_dict['method'] = 'left_hand'
                    else:
                        pellet_dict['method'] = 'other'
                        pellet_dict['outcome'] = 'dropped'

                    if verbose:
                        task.message_logged.emit(f"Pellet event completed: {pellet_dict}")

                    # start waiting for the minimum inter-pellet intv from when pellet was lost -- so count not reset!
                    pellet_state = 2

            elif pellet_state == 2:
                count += 1
                if count >= n_frames_inter_pellet_interval:
                    if verbose:
                        task.message_logged.emit(f"Finished inter-pellet interval wait at frame {frame_number}.")
                    pellet_state = 0
                    count = 0

        # if a pellet was placed close to the end of the session, it's possible the last pellet event is incomplete.
        if (len(pellet_events) > 0) and (pellet_events[-1]['lost'] < 0):
            pellet_events.pop()

        # correction: Go back through all pellet valid pellet events. If min Z-coord of pellet within a window after
        # the 'pellet placed' frame is less than a specified threshold, set the pellet event's outcome to 'dropped'
        for p in range(len(pellet_events)):
            if pellet_events[p]['lost'] >= 0:
                idx_start = pellet_events[p]['placed']
                idx_end = pellet_events[p]['lost'] + pos_window_nframes
                if idx_end >= n_total_frames:
                    idx_end = n_total_frames - 1
                if np.nanmin(coord_pz[idx_start:idx_end]) <= pellet_drop_dist:
                    pellet_events[p]['outcome'] = 'dropped'

        # before next step, accumulate the list of frame numbers at which the pellet is "placed". These are in chrono
        # order already. They are used in Phase 2.
        key_frames: List[int] = [evt['placed'] for evt in pellet_events]

        # append the pellet home location to the pellet events and save to pickle file in scorer folder
        pellet_events.append({'x': pellet_home[0], 'y': pellet_home[1], 'z': pellet_home[2]})
        with open(Path(out_dir, "pelletHistory.pickle"), "wb") as f:
            pickle.dump(pellet_events, f)

        if task.was_canceled():
            return

        # PHASE 2: find reach segment(s) in each pellet event epoch found in phase 1...
        task.message_logged.emit(f"Phase 2: Finding reach segments in {len(key_frames)} pellet placement epochs...")
        task.progress_updated.emit(0)
        reaches: List[RxReachSegment] = list()
        for kf_idx, start_frame in enumerate(key_frames):

            food_was_dropped = False
            frame = start_frame - 20   # in case a reach started before pellet was "placed"
            pellet_detected = False
            curr_reach_init = -1   # starting frame number for reach segment  under construction
            curr_reach_max = -1    # max hand velocity frame number for reach segment under construction
            while frame < n_total_frames - batch_frm:
                # stop if we've reached the next epoch
                if (kf_idx < len(key_frames) - 1) and (frame >= key_frames[kf_idx + 1]):
                    if (curr_reach_init == -1) and verbose:
                        if not pellet_detected:
                            task.message_logged.emit(f"No pellet was detected during delivery epoch #{kf_idx}")
                        else:
                            task.message_logged.emit(f"Hit end of epoch #{kf_idx} without finding additional reach")
                    break

                pellet_detected = bool(np.sum(pellet_p[frame:frame+batch_frm] > confidence)/batch_frm > 0.75)
                rhand_detected = bool(np.sum(r_hand_p[frame:frame+batch_frm] > confidence)/batch_frm > 0.75)

                if curr_reach_init == -1:  # search for reach initiation
                    if pellet_detected and rhand_detected:
                        test_a = coord_rhz[frame] > -4
                        test_b = np.mean(rh_rel_speed[frame:frame+batch_speed]) < reach_init_speed
                        if test_a and test_b:
                            if verbose:
                                task.message_logged.emit(f"Starting new reach segment at frame {frame}.")
                            curr_reach_init = frame
                            food_was_dropped = False
                            speed_hvh_init = np.diff(distance_rh - distance_rh[frame]) * (frame_rate / 1000)
                            speed_hvh_init = filtfilt(sg_coeffs, [1], speed_hvh_init)

                elif curr_reach_max == -1:  # search for reach max frame
                    speed_seg = np.asarray(pellet_speed[frame:frame + speed_window_nframes])
                    if np.sum(speed_seg > pellet_drop_speed) > 1:
                        food_was_dropped = True
                    if np.any(coord_pz[frame:frame + pos_window_nframes] < pellet_drop_dist):
                        # food dropped if pellet is too low
                        z_dist_indices = np.where(coord_pz[frame:frame + pos_window_nframes] < pellet_drop_dist)
                        if np.any(pellet_p[frame:frame + pos_window_nframes][z_dist_indices] > confidence):
                            food_was_dropped = True
                    # noinspection PyUnboundLocalVariable
                    if np.mean(speed_hvh_init[frame:frame + batch_speed]) > reach_dirchange_speed:
                        if verbose:
                            task.message_logged.emit(f"Detected reach max at frame {frame + 3}.")
                        curr_reach_max = frame + 3
                        frame += 3

                else:   # search for reach end and determine outcome
                    keep_looking = True  # set to false if pellet dropped, grabbed, stalled
                    test_a = np.mean(distance_rh[frame:frame + batch_dist]) > dist_thresh_end
                    test_b = np.mean(speed_hvh_init[frame:frame + batch_speed]) < reach_dirchange_speed
                    test_c = np.mean(rh_rel_speed[frame:frame + batch_speed]) < reach_init_speed
                    test_d = np.allclose(np.mean(distance_rh[frame:frame + batch_stall]), distance_rh[frame], atol=2)
                    test_e = np.isclose(np.mean(rh_speed[frame:frame + batch_stall]), 0, atol=0.025)
                    test_f = np.all(distance_rh[frame:frame + batch_stall] < 6)
                    speed_seg = np.asarray(pellet_speed[frame:frame + speed_window_nframes])
                    if np.sum(speed_seg > pellet_drop_speed) > 1:
                        food_was_dropped = True
                    if np.any(coord_pz[frame:frame + pos_window_nframes] < pellet_drop_dist):
                        z_dist_indices = np.where(coord_pz[frame:frame + pos_window_nframes] < pellet_drop_dist)
                        if np.any(pellet_p[frame:frame + pos_window_nframes][z_dist_indices] > confidence):
                            food_was_dropped = True

                    reach_ended = False
                    if test_a and test_c and not food_was_dropped:
                        if verbose:
                            task.message_logged.emit(f"Reach segment ended at frame {frame - 1}: MISSED.")
                        reaches.append(
                            RxReachSegment(curr_reach_init, curr_reach_max - curr_reach_init,
                                           frame - 1 - curr_reach_init, RxReach.MISSED, RxHandPos.UNSPECIFIED))
                        reach_ended = True
                    elif test_d and test_e and test_f:
                        if verbose:
                            task.message_logged.emit(f"Reach segment ended at frame {frame + 10}: STALLED.")
                        reaches.append(
                            RxReachSegment(curr_reach_init, curr_reach_max - curr_reach_init,
                                           frame + 10 - curr_reach_init, RxReach.STALLED, RxHandPos.UNSPECIFIED))
                        reach_ended = True
                        keep_looking = False
                    elif test_a and test_b:
                        p_test = np.mean(distance_p[frame:frame + batch_frm]) < pellet_dist_to_origin
                        if food_was_dropped:
                            if verbose:
                                task.message_logged.emit(f"Reach segment ended at frame {frame + 2}: DROPPED.")
                            reaches.append(
                                RxReachSegment(curr_reach_init, curr_reach_max - curr_reach_init,
                                               frame + 2 - curr_reach_init, RxReach.DROPPED, RxHandPos.UNSPECIFIED))
                            reach_ended = True
                            keep_looking = False
                        elif pellet_detected and p_test:
                            if verbose:
                                task.message_logged.emit(f"Reach segment ended at frame {frame + 2}: MISSED.")
                            reaches.append(
                                RxReachSegment(curr_reach_init, curr_reach_max - curr_reach_init,
                                               frame + 2 - curr_reach_init, RxReach.MISSED, RxHandPos.UNSPECIFIED))
                            reach_ended = True
                            frame += 2
                        else:
                            if verbose:
                                task.message_logged.emit(f"Reach segment ended at frame {frame + 2}: GRABBED.")
                            reaches.append(
                                RxReachSegment(curr_reach_init, curr_reach_max - curr_reach_init,
                                               frame + 2 - curr_reach_init, RxReach.GRABBED, RxHandPos.UNSPECIFIED))
                            reach_ended = True
                            keep_looking = False

                    # if we found a reach segment, discard it if its duration is too long or too short, or the hand
                    # is too far from pellet at the max velocity frame (?).
                    if reach_ended:
                        # unless keep_looking is false, start looking for another reach within current pellet epoch
                        curr_reach_init = curr_reach_max = -1
                        test_x = reaches[-1].dur < min_dur_frames
                        test_y = reaches[-1].dur > max_dur_frames
                        test_z = distance_rh[reaches[-1].max_frame] > min_dist_from_pellet
                        if test_x or test_y or test_z:
                            if verbose:
                                task.message_logged.emit(f"Reach segment at frame {reaches[-1].frame} DISCARDED!")
                            reaches.pop()

                    # we only keep looking in the current pellet epoch if the pellet was "missed"
                    if not keep_looking:
                        break

                frame += 1

            if task.was_canceled():
                return
            pct = int(100 * (kf_idx + 1) / len(key_frames))
            task.progress_updated.emit(pct)

        # save the detected reach segments to dedicated file within session subfolder containing scorer's results
        if len(reaches) > 0:
            out_file = Path(out_dir, "detected_reaches.txt")
            task.message_logged.emit(f"Saving {len(reaches)} reach segments found: {str(out_file)}")
            emsg = RxReachSegment.save_reach_segments(reaches, out_file)
            if len(emsg) > 0:
                task.message_logged.emit(emsg)

    except Exception as e:
        task.message_logged.emit(f"Reach segmenation FAILED: {e}")
        traceback.print_exc()


def test_find_fixed_cam_reaches(p: Path, rx_scorer: str) -> None:
    """
    Temporary test fixture for testing ``_find_reaches_fixed_cam()``.

    Calls the method directly (not on a background thread), and supplies a dummy ``BackgroundTask`` object which just
    prints messages to the console. Time elapsed is reported each time a 100% progress update is received and upon
    successful completion.

    :param p: Path to session folder. Must be a fixed-cam session.
    :param rx_scorer: The scorer folder name. The ``trajectories.npz`` file must be present there. If the call to
        ``_find_reaches_fixed_cam() is successful``, the body part trajectories are stored as individual
        named arrays in ``trajectories.npy``, where the array name is the corresponding body part's nickname.
    """
    sesh_id = RxSessionID.from_session_folder(p)
    ok = isinstance(sesh_id, RxSessionID) and sesh_id.is_fixed_cam
    if not ok:
        print(f"ERROR: Bad or nonexistent session folder, or NOT a fixed-cam session!")
        return

    # use default seg control params
    seg_parms = RxReachSegControls()

    t0 = time.time()

    def print_elapsed_time(pct: int):
        if pct == 100:
            print(f"Elapsed time: {time.time() - t0:0.3f} seconds.")

    dummy_task = BackgroundTask()
    dummy_task.message_logged.connect(lambda msg: print(msg))
    dummy_task.progress_updated.connect(print_elapsed_time)
    dummy_task.finished.connect(lambda: print("Finished!"))

    print("Calling _find_reaches_fixed_cam at elapsed time...")
    _find_reaches_fixed_cam(sesh_id, rx_scorer, seg_parms, dummy_task, verbose=True)
    print_elapsed_time(100)
