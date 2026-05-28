"""
resnet.py - Implementation of the "resnet_50" and "resnet_152" pre-trained Residual Neural Networks


--------------------------------------
CREDIT: The code in this module is excerpted from the DeepLabCut toolbox repository:

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

import abc
import re
from pathlib import Path
from typing import Optional

import tensorflow as tf
import tf_slim as slim
from tf_slim.nets import resnet_v1

from reachx.common import RX
from reachx.modeling.imgaug import Batch

_RESNET50_URL: str = "http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz"
""" Download URL for the pretrained ``resnet_v1_50`` neural net. """
_RESNET152_URL: str = "http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz"
""" Download URL for the pretrained ``resnet_v1_152`` neural net. """


def get_pretrained_resnet_checkpoint_file(net_type: str) -> Path:
    """
    Get file path locating the checkpoint file (``.ckpt``) for the resnet_v1_50 or _152 pretrained neural network.

    The checkpoint files are quite large (98MB, 231MB), so they are not included in the ReachX Git repo nor in the
    installation wheel. Instead, they are downloaded on first use and stored in the ReachX application home directory.

    :param net_type: Must be "resnet_50" or "resnet_152"
    :return: File system path to requested checkpoint file. The file may not yet exist.
    :raises ValueError: If ``net_type`` is invalid.
    """
    if net_type not in ["resnet_50", "resnet_152"]:
        raise ValueError("Only 'resnet_50' or 'resnet_152' supported!")
    return Path(RX().HOME, f"resnet_v1_{net_type[7:]}.ckpt")


def download_pretrained_resnet_checkpoint_file(net_type: str) -> Optional[str]:
    """
    Download the pretrained checkpoint file for the specified ResNet architecture and store it the ReachX application
    home directory.

    The checkpoint files are quite large (98MB, 231MB), so they are not included in the ReachX Git repo nor in the
    installation wheel. Instead, they are downloaded on first use.

    :param net_type: Must be "resnet_50" or "resnet_152"
    :return: None if successful, else an error description.
    :raises ValueError: If ``net_type`` is invalid.
    """
    if net_type not in ["resnet_50", "resnet_152"]:
        raise ValueError("Only 'resnet_50' or 'resnet_152' supported!")

    url = _RESNET50_URL if net_type == "resnet_50" else _RESNET152_URL

    import urllib.request
    import tarfile
    from io import BytesIO

    try:
        response = urllib.request.urlopen(url)
        with tarfile.open(fileobj=BytesIO(response.read()), mode="r:gz") as tar:
            tar.extractall(path=RX().HOME)
        return None
    except Exception as e:
        return f"Download and extraction of checkpoint file failed: {e}"


def prediction_layer(cfg, tensor, name, num_outputs):
    with slim.arg_scope(
        [slim.conv2d, slim.conv2d_transpose],
        padding="SAME",
        activation_fn=None,
        normalizer_fn=None,
        weights_regularizer=slim.l2_regularizer(cfg["weight_decay"]),
    ):
        with tf.compat.v1.variable_scope(name):
            pred = slim.conv2d_transpose(
                tensor, num_outputs, kernel_size=[3, 3], stride=2, scope="block4"
            )
            return pred


# noinspection PyTypeChecker
def make_2d_gaussian_kernel(sigma, size):
    sigma = tf.convert_to_tensor(sigma, dtype=tf.float32)
    k = tf.range(-size // 2 + 1, size // 2 + 1)
    k = tf.cast(k**2, sigma.dtype)
    k = tf.nn.softmax(-k / (2 * (sigma**2)))
    return tf.einsum("i,j->ij", k, k)


class BasePoseNet(metaclass=abc.ABCMeta):
    def __init__(self, cfg):
        self.cfg = cfg

    @abc.abstractmethod
    def extract_features(self, inputs):
        ...

    @abc.abstractmethod
    def get_net(self, inputs):
        ...

    def train(self, batch):
        heads = self.get_net(batch[Batch.inputs])
        if self.cfg["weigh_part_predictions"]:
            part_score_weights = batch[Batch.part_score_weights]
        else:
            part_score_weights = 1.0

        def add_part_loss(pred_layer):
            return tf.compat.v1.losses.sigmoid_cross_entropy(
                batch[Batch.part_score_targets], heads[pred_layer], part_score_weights
            )

        loss = {"part_loss": add_part_loss("part_pred")}
        total_loss = loss["part_loss"]

        if (
            self.cfg["intermediate_supervision"]
            and "efficientnet" not in self.cfg["net_type"]
        ):
            loss["part_loss_interm"] = add_part_loss("part_pred_interm")
            total_loss += loss["part_loss_interm"]

        if self.cfg["location_refinement"]:
            locref_pred = heads["locref"]
            locref_targets = batch[Batch.locref_targets]
            locref_weights = batch[Batch.locref_mask]
            loss_func = (
                tf.compat.v1.losses.huber_loss
                if self.cfg["locref_huber_loss"]
                else tf.compat.v1.losses.mean_squared_error
            )
            loss["locref_loss"] = self.cfg["locref_loss_weight"] * loss_func(
                locref_targets, locref_pred, locref_weights
            )
            total_loss += loss["locref_loss"]

        if self.cfg["pairwise_predict"] or self.cfg["partaffinityfield_predict"]:
            pairwise_pred = heads["pairwise_pred"]
            pairwise_targets = batch[Batch.pairwise_targets]
            pairwise_weights = batch[Batch.pairwise_mask]
            loss_func = (
                tf.compat.v1.losses.huber_loss
                if self.cfg["pairwise_huber_loss"]
                else tf.compat.v1.losses.mean_squared_error
            )
            loss["pairwise_loss"] = self.cfg["pairwise_loss_weight"] * loss_func(
                pairwise_targets, pairwise_pred, pairwise_weights
            )
            total_loss += loss["pairwise_loss"]

        loss["total_loss"] = total_loss
        return loss

    def test(self, inputs):
        heads = self.get_net(inputs)
        return self.add_inference_layers(heads)

    # TODO: Take out any multi-animal stuff?
    def prediction_layers(
        self,
        features,
        scope="pose",
        reuse=None,
    ):
        out = {}
        n_joints = self.cfg["num_joints"]
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            out["part_pred"] = prediction_layer(
                self.cfg,
                features,
                "part_pred",
                n_joints + self.cfg.get("num_idchannel", 0),
            )
            if self.cfg["location_refinement"]:
                out["locref"] = prediction_layer(
                    self.cfg,
                    features,
                    "locref_pred",
                    n_joints * 2,
                )
            if (
                self.cfg["pairwise_predict"]
                and "multi-animal" not in self.cfg["dataset_type"]
            ):
                out["pairwise_pred"] = prediction_layer(
                    self.cfg,
                    features,
                    "pairwise_pred",
                    n_joints * (n_joints - 1) * 2,
                )
            if (
                self.cfg["partaffinityfield_predict"]
                and "multi-animal" in self.cfg["dataset_type"]
            ):
                out["pairwise_pred"] = prediction_layer(
                    self.cfg,
                    features,
                    "pairwise_pred",
                    self.cfg["num_limbs"] * 2,
                )
        out["features"] = features
        return out

    def inference(self, inputs):
        """Direct TF inference on GPU.
        Added with: https://arxiv.org/abs/1909.11229
        """
        heads = self.get_net(inputs)
        locref = heads["locref"]
        probs = tf.sigmoid(heads["part_pred"])

        if self.cfg["batch_size"] == 1:
            probs = tf.squeeze(probs, axis=0)
            locref = tf.squeeze(locref, axis=0)
            l_shape = tf.shape(input=probs)
            locref = tf.reshape(locref, (l_shape[0] * l_shape[1], -1, 2))
            probs = tf.reshape(probs, (l_shape[0] * l_shape[1], -1))
            maxloc = tf.argmax(input=probs, axis=0)
            loc = tf.unravel_index(
                maxloc, (tf.cast(l_shape[0], tf.int64), tf.cast(l_shape[1], tf.int64))
            )
            maxloc = tf.reshape(maxloc, (1, -1))

            joints = tf.reshape(
                tf.range(0, tf.cast(l_shape[2], dtype=tf.int64)), (1, -1)
            )
        else:
            l_shape = tf.shape(
                input=probs
            )  # batchsize times x times y times body parts
            locref = tf.reshape(
                locref, (l_shape[0], l_shape[1], l_shape[2], l_shape[3], 2)
            )
            # turn into x times y time bs * bpts
            locref = tf.transpose(a=locref, perm=[1, 2, 0, 3, 4])
            probs = tf.transpose(a=probs, perm=[1, 2, 0, 3])

            l_shape = tf.shape(input=probs)  # x times y times batch times body parts

            locref = tf.reshape(locref, (l_shape[0] * l_shape[1], -1, 2))
            probs = tf.reshape(probs, (l_shape[0] * l_shape[1], -1))
            maxloc = tf.argmax(input=probs, axis=0)
            loc = tf.unravel_index(
                maxloc, (tf.cast(l_shape[0], tf.int64), tf.cast(l_shape[1], tf.int64))
            )  # tuple of max indices
            maxloc = tf.reshape(maxloc, (1, -1))
            joints = tf.reshape(
                tf.range(0, tf.cast(l_shape[2] * l_shape[3], dtype=tf.int64)), (1, -1)
            )

        # extract corresponding locref x and y as well as probability
        indices = tf.transpose(a=tf.concat([maxloc, joints], axis=0))
        offset = tf.gather_nd(locref, indices)
        offset = tf.gather(offset, [1, 0], axis=1)
        likelihood = tf.reshape(tf.gather_nd(probs, indices), (-1, 1))

        pose = (
            self.cfg["stride"] * tf.cast(tf.transpose(a=loc), dtype=tf.float32)
            + self.cfg["stride"] * 0.5
            + offset * self.cfg["locref_stdev"]
        )
        pose = tf.concat([pose, likelihood], axis=1)
        return {"pose": pose}

    def add_inference_layers(self, heads):
        """initialized during inference"""
        prob = tf.sigmoid(heads["part_pred"])
        nms_radius = int(self.cfg.get("nmsradius", 5))

        # Filter predicted heatmaps with a 2D Gaussian kernel as in:
        # https://openaccess.thecvf.com/content_CVPR_2020/papers/
        # Huang_The_Devil_Is_in_the_Details_Delving_Into_Unbiased_Data_CVPR_2020_paper.pdf
        scmaps = tf.gather(prob, tf.range(self.cfg["num_joints"]), axis=3)
        kernel = make_2d_gaussian_kernel(
            sigma=self.cfg.get("sigma", 1),
            size=nms_radius * 2 + 1,
        )
        kernel = kernel[:, :, tf.newaxis, tf.newaxis]

        kernel_sc = tf.tile(kernel, [1, 1, tf.shape(scmaps)[3], 1])
        scmaps = tf.nn.depthwise_conv2d(
            scmaps,
            kernel_sc,
            strides=[1, 1, 1, 1],
            padding="SAME",
        )

        # copied from deeplabcut.pose_estimation_tensorflow.core.predict_multianimal.py
        threshold = self.cfg.get("minconfidence", 0.01)
        pooled = tf.nn.max_pool2d(scmaps, [nms_radius, nms_radius], strides=1, padding="SAME")
        maxima = scmaps * tf.cast(tf.equal(scmaps, pooled), tf.float32)
        peak_inds = tf.cast(tf.where(maxima >= threshold), tf.int32)

        outputs = {"part_prob": prob, "peak_inds": peak_inds}
        if self.cfg["location_refinement"]:
            locref = heads["locref"]
            if self.cfg.get("locref_smooth", False):
                kernel_loc = tf.tile(kernel, [1, 1, tf.shape(locref)[3], 1])
                locref = tf.nn.depthwise_conv2d(
                    locref,
                    kernel_loc,
                    strides=[1, 1, 1, 1],
                    padding="SAME",
                )
            outputs["locref"] = locref

        if self.cfg["pairwise_predict"] or self.cfg["partaffinityfield_predict"]:
            outputs["pairwise_pred"] = heads["pairwise_pred"]

        if "features" in heads:
            outputs["features"] = heads["features"]

        return outputs

    def center_inputs(self, inputs):
        mean = tf.constant(
            self.cfg["mean_pixel"],
            dtype=tf.float32,
            shape=[1, 1, 1, 3],
            name="img_mean",
        )
        return inputs - mean


class PoseResnet(BasePoseNet):
    _NET_FUNCS = {
        "resnet_50": resnet_v1.resnet_v1_50,
        "resnet_101": resnet_v1.resnet_v1_101,
        "resnet_152": resnet_v1.resnet_v1_152,
    }
    """ Maps ResNet type to corresponding function in tf.slim package. """

    def __init__(self, cfg):
        super(PoseResnet, self).__init__(cfg)

    def extract_features(self, inputs):
        net_fun = self._NET_FUNCS[self.cfg["net_type"]]
        im_centered = self.center_inputs(inputs)
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            net, end_points = net_fun(
                im_centered,
                global_pool=False,
                output_stride=16,
                is_training=False,
            )
        return net, end_points

    def get_prediction_layers(
        self,
        features,
        end_points,
        scope="pose",
        reuse=None,
    ):
        out = super(PoseResnet, self).prediction_layers(
            features,
            scope,
            reuse,
        )
        out["features"] = features
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            if self.cfg["intermediate_supervision"]:
                layer_name = "resnet_v1_{}/block{}/unit_{}/bottleneck_v1"
                num_layers = re.findall("resnet_([0-9]*)", self.cfg["net_type"])[0]
                interm_name = layer_name.format(
                    num_layers, 3, self.cfg["intermediate_supervision_layer"]
                )
                block_interm_out = end_points[interm_name]
                out["part_pred_interm"] = prediction_layer(
                    self.cfg,
                    block_interm_out,
                    "intermediate_supervision",
                    self.cfg["num_joints"] + self.cfg.get("num_idchannel", 0),
                )
        return out

    def get_net(self, inputs):
        net, end_points = self.extract_features(inputs)
        return self.get_prediction_layers(net, end_points)
