"""
train.py - Training a neural network model to perform object detection in an image.

This module contains functions based on similar code in the `pose_estimation_tensorflow` package within the DeepLabCut
repository. DLC provides a lot of flexibility for defining how the network is trained, tested, and evaluated, plus
support for a relatively large number of pre-trained neural nets. ReachX, on the other hand, does not use most of the
DLC-provideed options for training, only supports the `resnet_50` and `resnet_152` pre-trained architectures, and
always uses the `img_aug` dataset format for the training data.

Given the large size and complexity of DLC and the tensorflow library which does the real work, the goal here is to
remove all dependencies on DLC and just use Tensorflow directly.

--------------------------------------
CREDIT: The code in this module is derived from the DeepLabCut toolbox repository:

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
import threading
from pathlib import Path
from typing import Dict, Any

import tensorflow as tf
import tf_slim as slim

from reachx.uicommon import BackgroundTask
from reachx.modeling.imgaug import Batch, ImgaugPoseDataset
from reachx.modeling.resnet import (PoseResnet, get_pretrained_resnet_checkpoint_file,
                                    download_pretrained_resnet_checkpoint_file)


class LearningRate(object):
    def __init__(self, cfg):
        self.steps = cfg["multi_step"]
        self.current_step = 0

    def get_lr(self, iteration):
        lr = self.steps[self.current_step][0]
        if iteration == self.steps[self.current_step][1]:
            self.current_step += 1

        return lr


def get_batch_spec(cfg):
    num_joints = cfg["num_joints"]
    batch_size = cfg["batch_size"]
    return {
        Batch.inputs: [batch_size, None, None, 3],
        Batch.part_score_targets: [batch_size, None, None, num_joints],
        Batch.part_score_weights: [batch_size, None, None, num_joints],
        Batch.locref_targets: [batch_size, None, None, num_joints * 2],
        Batch.locref_mask: [batch_size, None, None, num_joints * 2],
    }


_QUEUE_SIZE: int = 20


def setup_preloading(batch_spec):
    placeholders = {
        name: tf.compat.v1.placeholder(tf.float32, shape=spec)
        for (name, spec) in batch_spec.items()
    }
    names = placeholders.keys()
    placeholders_list = list(placeholders.values())

    q = tf.queue.FIFOQueue(_QUEUE_SIZE, [tf.float32] * len(batch_spec))
    enqueue_op = q.enqueue(placeholders_list)
    batch_list = q.dequeue()

    batch = {}
    for idx, name in enumerate(names):
        batch[name] = batch_list[idx]
        batch[name].set_shape(batch_spec[name])
    return batch, enqueue_op, placeholders


def load_and_enqueue(sess, enqueue_op, coord, dataset, placeholders):
    while not coord.should_stop():
        batch_np = dataset.next_batch()
        food = {pl: batch_np[name] for (name, pl) in placeholders.items()}
        sess.run(enqueue_op, feed_dict=food)


def start_preloading(sess, enqueue_op, dataset, placeholders):
    coord = tf.compat.v1.train.Coordinator()
    t = threading.Thread(
        target=load_and_enqueue,
        args=(sess, enqueue_op, coord, dataset, placeholders),
    )
    t.start()
    return coord, t


def get_optimizer(loss_op, cfg):
    tstep = tf.compat.v1.placeholder(tf.int32, shape=[], name="tstep")
    learning_rate = tf.compat.v1.placeholder(tf.float32, shape=[])

    if cfg["optimizer"] == "sgd":
        optimizer = tf.compat.v1.train.MomentumOptimizer(
            learning_rate=learning_rate, momentum=0.9
        )
    elif cfg["optimizer"] == "adam":
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
    else:
        raise ValueError("unknown optimizer {}".format(cfg["optimizer"]))
    train_op = slim.learning.create_train_op(loss_op, optimizer)

    return learning_rate, train_op, tstep


def train(cfg: Dict[str, Any], task: BackgroundTask, max_keep: int = 5, allow_growth: bool = True) -> None:
    """
    Train a ReachX objection detection model IAW the supplied training parameters.

    This method is adapted from the Tensorflow-specific "single animal" version of the training function in the
    DeepLabCut toolkit in `pose_estimation_tensorflow.core.train.py` -- adapted and simplified for ReachX's purposes.

    :param cfg: Dict of training parameters. See ``reachx.modeling.training.get_default_training_params()`` for details.
    :param task: The task object on the background thread from which this method is invoked. Provides facilities for
        reporting progress, posting info/error/progress messages (for GUI display), as well as to check for task
        cancellation.
    :param max_keep: Max number of model snapshots to keep. Default = 5.
    :param allow_growth: If ``True``, the memory allocator does not pre-allocate the entire specified GPU memory region,
        instead starting small and growing as needed. No effect if GPU support not available.
    """
    # switch to the directory where model snapshots are to be written. Remember current working directory.
    model_dir = Path(cfg["model_dir"])
    if not model_dir.is_dir():
        raise Exception(f"Configuration error - Model output folder not found: {str(model_dir)}")
    start_path = os.getcwd()
    os.chdir(str(model_dir))

    # some basic checks
    train_dir = Path(cfg["train_dir"])
    if not (train_dir.is_dir() and Path(train_dir, "images").is_dir()):
        raise Exception(f"Configuration error - Invalid/missing training data folder: {str(train_dir)}")

    # we have to download the pretrained resnet checkpoint file on first use (too big to include in repo)
    net_type = cfg["net_type"]
    if net_type not in ["resnet_50", "resnet_152"]:
        raise Exception(f"Configuration error - Pretrained neural net not supported: {net_type}")
    ckpt_file_path = get_pretrained_resnet_checkpoint_file(net_type)
    if not ckpt_file_path.is_file():
        task.message_logged.emit(f"Downloading pretrained checkpoint file for {net_type}...")
        emsg = download_pretrained_resnet_checkpoint_file(net_type)
        if emsg is not None:
            raise Exception(emsg)
        assert ckpt_file_path.is_file()

    task.message_logged.emit("Setting up Tensorflow infrastructure for training session...")

    dataset = ImgaugPoseDataset(cfg)
    batch_spec = get_batch_spec(cfg)
    batch, enqueue_op, placeholders = setup_preloading(batch_spec)

    task.message_logged.emit(f"Loading ImageNet-pretrained: {net_type}")
    losses = PoseResnet(cfg).train(batch)
    total_loss = losses["total_loss"]

    for k, t in losses.items():
        tf.compat.v1.summary.scalar(k, t)
    merged_summaries = tf.compat.v1.summary.merge_all()

    variables_to_restore = slim.get_variables_to_restore(include=["resnet_v1"])

    restorer = tf.compat.v1.train.Saver(variables_to_restore)
    saver = tf.compat.v1.train.Saver(max_to_keep=max_keep)  # last N model snapshots are saved in model folder

    if allow_growth:
        # Pycharm cannot find this, though it is in tensorflow._api.v2.compat.v1.__init__.py
        # noinspection PyUnresolvedReferences
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)
    else:
        sess = tf.compat.v1.Session()

    coord, thread = start_preloading(sess, enqueue_op, dataset, placeholders)
    train_writer = tf.compat.v1.summary.FileWriter(cfg["log_dir"], sess.graph)

    # TODO: This code block does not actually verify that the system is running on M1/M2. And "sgd" can run on an
    #   Intel-based Mac. And 'adam' did not! (losses blew up). Investigate further. For now, defaulting to "sgd".
    # Auto-switch to Adam on Apple M1/M2 chips, as the momentum optimizer crashes
    # from tensorflow.python.platform import build_info
    # info = build_info.build_info
    # if not info["is_cuda_build"]:
    #     task.message_logged.emit("-- Warning: Switching to Adam optimizer, as SGD crashes on Apple Silicon.")
    #     cfg["optimizer"] = "adam"

    # NOTE: ReachX doesn't support training param 'freezeencoder'
    task.message_logged.emit(f"Loading optimizer: {cfg['optimizer']}")
    learning_rate, train_op, tstep = get_optimizer(total_loss, cfg)

    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(tf.compat.v1.local_variables_initializer())

    # Restore variables from pretrained neural net checkpoint file
    restorer.restore(sess, str(ckpt_file_path.absolute()))

    max_iter = int(cfg["multi_step"][-1][1])
    display_iters = max(1, int(cfg["display_iters"]))
    save_iters = max(1, int(cfg["save_iters"]))

    cum_loss = 0.0
    lr_gen = LearningRate(cfg)

    stats_path = Path(model_dir, "learning_stats.csv")
    lrf = open(str(stats_path), "w")

    snapshot_pfx = str(Path(model_dir, "snapshot"))  # full path + prefix for model snapshots
    task.message_logged.emit(f"Starting training. Num iterations = {max_iter}.\n"
                             f"Model snapshots saved to {snapshot_pfx}")
    for it in range(0, max_iter + 1):
        current_lr = lr_gen.get_lr(it)
        lr_dict = {learning_rate: current_lr}

        [_, loss_val, summary] = sess.run([train_op, total_loss, merged_summaries], feed_dict=lr_dict)
        cum_loss += loss_val
        train_writer.add_summary(summary, it)

        # check for task cancellation and update progress regularly
        if task.was_canceled():
            task.message_logged.emit("Operation cancelled...please wait.")
            break
        pct = int(100.0 * (it + 1) / max_iter + 0.5)
        task.progress_updated.emit(pct)

        if it % display_iters == 0 and it > 0:
            average_loss = cum_loss / display_iters
            cum_loss = 0.0
            task.message_logged.emit(f"iteration {it}: avg loss = {average_loss:.4f}, learn rate = {current_lr}")
            lrf.write("{}, {:.5f}, {}\n".format(it, average_loss, current_lr))
            lrf.flush()

        # save model snapshot
        if (it % save_iters == 0 and it > 0) or it == max_iter:
            saver.save(sess, snapshot_pfx, global_step=it)

    task.message_logged.emit("Cleaning up...")
    lrf.close()
    sess.close()
    coord.request_stop()
    coord.join([thread])
    # restore original working directory
    os.chdir(str(start_path))
