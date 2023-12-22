# Copyright © 2023 Huawei Technologies Co, Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""trainer"""

import os
import numpy as np
import mindspore as ms

from mindspore import Parameter, Tensor, context
from mindspore import dtype as mstype
from mindspore import nn, ops, Tensor
from mindspore.communication.management import get_group_size
from mindspore.context import ParallelMode
from mindspore.parallel._auto_parallel_context import auto_parallel_context

from models.facerender.modules.keypoint_detector import HEEstimator
from models.facerender.modules.utils import make_coordinate_grid_2d
from models.facerender.modules.make_animation import headpose_pred_to_degree
from models.face3d.facexlib.resnet import Bottleneck
from models.facerender.networks import Hopenet
from utils.preprocess import CropAndExtract


def apply_image_normalization(input):
    r"""Normalize using ImageNet mean and std.

    Args:
        input (4D tensor NxCxHxW): The input images, assuming to be [-1, 1].

    Returns:
        Normalized inputs using the ImageNet normalization.
    """
    datatype = input.dtype
    # normalize the input back to [0, 1]
    # normalized_input = (input + 1.0) / 2.0

    # normalize the input using the ImageNet mean and std
    mean = Tensor([0.485, 0.456, 0.406], dtype=datatype).view(1, 3, 1, 1)
    std = Tensor([0.229, 0.224, 0.225], dtype=datatype).view(1, 3, 1, 1)
    output = (input - mean) / std
    output = ops.cast(output, datatype)
    return output


class Transform:
    """
    Random tps transformation for equivariance constraints.
    """

    def __init__(self, bs, **kwargs):
        noise = ops.normal(shape=(bs, 2, 3), mean=0, stddev=kwargs["sigma_affine"])
        self.theta = noise + ops.eye(2, 3).view(1, 2, 3)
        self.bs = bs

        if ("sigma_tps" in kwargs) and ("points_tps" in kwargs):
            self.tps = True
            self.control_points = make_coordinate_grid_2d(
                (kwargs["points_tps"], kwargs["points_tps"]), type=noise.dtype
            )
            self.control_points = self.control_points.unsqueeze(0)
            self.control_params = ops.normal(
                shape=(bs, 1, kwargs["points_tps"] ** 2),
                mean=0,
                stddev=kwargs["sigma_tps"],
            )
        else:
            self.tps = False

    def transform_frame(self, frame):
        grid = make_coordinate_grid_2d(frame.shape[2:], type=frame.dtype).unsqueeze(0)
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        grid = self.warp_coordinates(grid).view(
            self.bs, frame.shape[2], frame.shape[3], 2
        )
        return ops.grid_sample(frame, grid, padding_mode="reflection")

    def warp_coordinates(self, coordinates):
        theta = self.theta.astype(coordinates.dtype)
        theta = theta.unsqueeze(1)
        transformed = (
            ops.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1))
            + theta[:, :, :, 2:]
        )
        transformed = transformed.squeeze(-1)

        if self.tps:
            control_points = self.control_points.astype(coordinates.dtype)
            control_params = self.control_params.astype(coordinates.dtype)
            distances = coordinates.view(
                coordinates.shape[0], -1, 1, 2
            ) - control_points.view(1, 1, -1, 2)
            distances = ops.abs(distances).sum(-1)

            result = distances**2
            result = result * ops.log(distances + 1e-6)
            result = result * control_params
            result = result.sum(axis=2).view(self.bs, coordinates.shape[1], 1)
            transformed = transformed + result

        return transformed

    def jacobian(self, coordinates):
        new_coordinates = self.warp_coordinates(coordinates)
        grad_x = ops.grad(new_coordinates[..., 0].sum(), coordinates, create_graph=True)
        grad_y = ops.grad(new_coordinates[..., 1].sum(), coordinates, create_graph=True)
        jacobian = ops.cat([grad_x[0].unsqueeze(-2), grad_y[0].unsqueeze(-2)], axis=-2)
        return jacobian


class TrainOneStepCell(nn.Cell):
    """TrainOneStepCell"""

    def __init__(self, network, optimizer, initial_scale_sense=1.0):
        super(TrainOneStepCell, self).__init__()
        self.network = network
        self.network.set_grad()
        self.optimizer = optimizer
        self.weights = self.optimizer.parameters
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.scale_sense = Tensor(initial_scale_sense, dtype=mstype.float32)

        self.reducer_flag = False
        self.grad_reducer = None
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [
            ParallelMode.DATA_PARALLEL,
            ParallelMode.HYBRID_PARALLEL,
        ]:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = context.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(
                optimizer.parameters, mean, degree
            )


class GTrainOneStepCell(TrainOneStepCell):
    """Generator TrainOneStepCell"""

    def __init__(self, network, optimizer, initial_scale_sense=1.0):
        super(GTrainOneStepCell, self).__init__(network, optimizer, initial_scale_sense)
        self.network.vgg_feat_extractor.set_grad(False)
        self.network.discriminator.set_grad(False)

    def set_train(self, mode=True):
        super().set_train(mode)
        self.network.vgg_feat_extractor.set_train(False)
        self.network.discriminator.set_train(False)
        return self

    def construct(self, *inputs):
        network_fwd_bwd = ops.value_and_grad(
            self.network, grad_position=None, weights=self.weights, has_aux=True
        )

        (loss, pred), grads = network_fwd_bwd(*inputs)

        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        opt_res = self.optimizer(grads)
        return ops.depend((loss, pred), opt_res)


class DTrainOneStepCell(TrainOneStepCell):
    """Discriminator TrainOneStepCell"""

    def construct(self, *inputs):
        loss = self.network(*inputs)
        grads = self.grad(self.network, self.weights)(*inputs, self.scale_sense * 1.0)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        loss = ops.depend(loss, self.optimizer(grads))
        return loss


class GWithLossCell(nn.Cell):
    """Generator with loss cell"""

    def __init__(self, generator, discriminator, vgg_feat_extractor, cfg):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.vgg_feat_extractor = vgg_feat_extractor
        self.cfg = cfg

        # Equivariance loss
        self.semantic_radius = 13
        self.extractor = CropAndExtract(self.cfg.preprocess)

        # Keypoint prior loss
        self.dt = 0.1  # distance threshold
        self.zt = 0.33  # target value

        # Head pose loss
        self.resize_mode = "bilinear"
        hopenet = Hopenet(Bottleneck, [3, 4, 6, 3], 66)

        hopenet_path = self.cfg.facerender.train.pretrained_hopenet
        if hopenet_path:
            hopenet_params = ms.load_checkpoint(hopenet_path)
            ms.load_param_into_net(hopenet, hopenet_params)
            print(
                f"Finished loading the pretrained checkpoint {hopenet_path} into Hopenet."
            )

        self.hopenet = hopenet
        self.hopenet.set_train(False)

        # Deformation prior loss

        # Perceptual loss
        self.l1 = nn.L1Loss()
        self.pct_weights = [0.03125, 0.0625, 0.125, 0.25, 1.0]

        # GAN loss
        self.criterion = nn.BCEWithLogitsLoss(reduction="mean")
        self.real_target = Tensor(1.0, mstype.float32)

        # Keypoint unsupervised loss
        he_estimator = HEEstimator(
            **cfg.facerender.model_params.he_estimator_params,
            **cfg.facerender.model_params.common_params,
        )
        he_estimator_path = os.path.join(
            cfg.facerender.path.get("checkpoint_dir", ""),
            cfg.facerender.path.get("he_estimator_checkpoint", None),
        )
        estimator_params = ms.load_checkpoint(he_estimator_path)
        ms.load_param_into_net(he_estimator, estimator_params)
        print(
            f"Finished loading the pretrained checkpoint {he_estimator_path} into HEEstimator."
        )

        self.he_estimator = he_estimator
        self.he_estimator.set_train(False)

    def construct(
        self,
        source_image,
        source_semantics,
        source_image_binary,
        target_semantics,
        target_image_ts,
    ):
        (
            pred_semantics,
            kp_canonical,
            kp_source,
            kp_driving,
            he_source,
            he_driving,
        ) = self.generator(source_image, source_semantics, target_semantics)

        logits = pred_semantics
        labels = target_image_ts

        # perceptual loss
        vgg_output = self.vgg_feat_extractor(logits)
        vgg_ground_truth = self.vgg_feat_extractor(labels)
        loss_perceptual = 0.0
        for i, w in enumerate(self.pct_weights):
            loss_perceptual += w * self.l1(vgg_output[i], vgg_ground_truth[i])

        # GAN loss (+ feature matching)
        output_pred = self.discriminator(logits)

        real_target = self.real_target.expand_as(output_pred)

        output_pred = output_pred.astype(mstype.float32)

        loss_adversarial = self.criterion(output_pred, real_target)

        # Equivariance loss
        # transform = Transform(
        #     source_image.shape[0], **self.cfg.facerender.train.transform_params
        # )

        # transformed_frame = transform.transform_frame(
        #     source_image_binary.astype(mstype.float32)
        # )  # (bs, 256, 256, 3)
        # coeff_dict = self.extractor.extract_3dmm(list(ops.unbind(transformed_frame)))
        # transformed_semantics = coeff_dict["coeff_3dmm"][:, : source_semantics.shape[1]]
        # transformed_semantics = Tensor(transformed_semantics, mstype.float32).unsqueeze(
        #     -1
        # )
        # transformed_semantics = transformed_semantics.repeat(
        #     self.semantic_radius * 2 + 1, axis=-1
        # )

        # transformed_he_source = self.generator.mapping(transformed_semantics)

        # transformed_kp = self.generator.keypoint_transformation_train(
        #     kp_canonical, transformed_he_source
        # )

        # ## Value loss part
        # # project 3d -> 2d
        # kp_source_2d = kp_source[:, :, :2]
        # transformed_kp_2d = transformed_kp[:, :, :2]
        # loss_eqv = ops.abs(
        #     kp_source_2d - transform.warp_coordinates(transformed_kp_2d)
        # ).mean()
        loss_eqv = 0.0

        # Keypoint prior loss
        dist = ops.cdist(kp_driving, kp_driving, p=2.0).pow(2)
        dist = self.dt - dist  # set Dt = 0.1
        dd = ops.gt(dist, 0)
        loss_kp = (dist * dd).mean(axis=0).sum()

        kp_mean_depth = kp_driving[:, :, -1].mean(-1)
        value_depth = ops.abs(kp_mean_depth - self.zt).mean()  # set Zt = 0.33

        loss_kp += value_depth

        # Headpose loss
        source_224 = ops.interpolate(
            source_image, mode=self.resize_mode, size=(224, 224), align_corners=False
        )
        source_224 = apply_image_normalization(source_224)

        yaw_gt, pitch_gt, roll_gt = self.hopenet(source_224)

        yaw_gt = headpose_pred_to_degree(yaw_gt)
        pitch_gt = headpose_pred_to_degree(pitch_gt)
        roll_gt = headpose_pred_to_degree(roll_gt)

        yaw, pitch, roll = he_source[0], he_source[1], he_source[2]

        yaw = headpose_pred_to_degree(yaw)
        pitch = headpose_pred_to_degree(pitch)
        roll = headpose_pred_to_degree(roll)

        loss_headpose = (
            ops.abs(yaw - yaw_gt).mean()
            + ops.abs(pitch - pitch_gt).mean()
            + ops.abs(roll - roll_gt).mean()
        )
        # loss_headpose = 0.0

        # Deformation prior loss
        exp_driving = he_driving[-1]
        loss_deform = ops.norm(exp_driving, ord=1, dim=-1).mean()

        # Keypoint unsupervised loss
        he_source_vid2vid = self.he_estimator(source_image)
        kp_vid2vid = self.generator.keypoint_transformation_train(
            kp_canonical, he_source_vid2vid
        )
        loss_vid2vid = ops.abs(kp_vid2vid - kp_source).sum(axis=-1).mean()

        loss_g = (
            1.0 * loss_adversarial
            + 10.0 * loss_perceptual
            + 20.0 * loss_eqv  # TODO!!
            + 10.0 * loss_kp
            + 20.0 * loss_headpose
            + 5.0 * loss_deform
            + 20.0 * loss_vid2vid
        )

        result = ops.stop_gradient(logits)
        return loss_g, result


class DWithLossCell(nn.Cell):
    """Discriminator with loss cell"""

    def __init__(self, discriminator):
        super().__init__()
        self.discriminator = discriminator
        self.criterion = nn.BCEWithLogitsLoss(reduction="mean")
        self.real_target = Tensor(1.0, mstype.float32)
        self.fake_target = Tensor(0.0, mstype.float32)

    def construct(self, ground_truth, generated_samples):
        real_pred = self.discriminator(ground_truth).astype(mstype.float32)
        fake_pred = self.discriminator(generated_samples).astype(mstype.float32)

        real_target = self.real_target.expand_as(real_pred)
        fake_target = self.fake_target.expand_as(fake_pred)

        loss_adversarial = self.criterion(real_pred, real_target) + self.criterion(
            fake_pred, fake_target
        )
        return loss_adversarial


class FaceRenderTrainer:
    """FaceRenderTrainer"""

    def __init__(self, train_one_step_g, train_one_step_d, cfg, finetune=False):
        super(FaceRenderTrainer, self).__init__()
        self.train_one_step_g = train_one_step_g
        self.train_one_step_d = train_one_step_d
        self.finetune = finetune
        self.cfg = cfg

    def run(self, data_batch):
        source_image = data_batch["source_image"]
        source_semantics = data_batch["source_semantics"]
        source_image_binary = data_batch["source_image_binary"]
        target_semantics = data_batch["target_semantics"]
        target_image_ts = data_batch["target_image_ts"]

        print("running train_one_step_g ...")
        self.train_one_step_g.set_train(not self.finetune)
        loss_g, output = self.train_one_step_g(
            source_image,
            source_semantics,
            source_image_binary,
            target_semantics,
            target_image_ts,
        )

        print("running train_one_step_d ...")
        self.train_one_step_d.set_train()
        loss_d = self.train_one_step_d(target_image_ts, output)
        return loss_g, loss_d

    def train(self, total_steps, dataset, callbacks, save_ckpt_logs=True, **kwargs):
        # callbacks = get_callbacks(
        #     self.cfg,
        #     self.train_one_step_g.network.generator,
        #     self.train_one_step_g.network.discriminator,
        #     self.finetune,
        # )
        print(f"Start training for {total_steps} iterations")

        dataset_size = dataset.get_dataset_size()
        repeats_num = (total_steps + dataset_size - 1) // dataset_size
        dataset = dataset.repeat(repeats_num)
        dataloader = dataset.create_dict_iterator()
        for num_batch, databatch in enumerate(dataloader):
            if num_batch >= total_steps:
                print("Reached the target number of iterations")
                break

            # import pdb; pdb.set_trace()

            # sample -> ['source_image', 'source_semantics', 'target_semantics', 'frame_num']

            loss_g, loss_d = self.run(databatch)

            print(loss_g)
            print(loss_d)

            # if save_ckpt_logs:
            #     callbacks([loss_g.asnumpy().mean(), loss_d.asnumpy().mean()])

        print("Training completed")
