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

from mindspore import Parameter, Tensor, context
from mindspore import dtype as mstype
from mindspore import nn, ops
from mindspore.communication.management import get_group_size
from mindspore.context import ParallelMode
from mindspore.parallel._auto_parallel_context import auto_parallel_context


class TrainOneStepCell(nn.Cell):
    """TrainOneStepCell"""

    def __init__(self, network, optimizer, initial_scale_sense=1.0):
        super(TrainOneStepCell, self).__init__()
        self.network = network
        self.network.set_grad()
        self.optimizer = optimizer
        self.weights = self.optimizer.parameters
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.scale_sense = Parameter(Tensor(initial_scale_sense, dtype=mstype.float32), name="scale_sense")
        self.reducer_flag = False
        self.grad_reducer = None
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = context.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)


class GTrainOneStepCell(TrainOneStepCell):
    """Generator TrainOneStepCell"""

    def __init__(self, network, optimizer, initial_scale_sense=1.0):
        super(GTrainOneStepCell, self).__init__(network, optimizer, initial_scale_sense)
        # self.network.vgg_feat_extractor.set_grad(False)
        self.network.discriminator.set_grad(False)

    def set_train(self, mode=True):
        super().set_train(mode)
        # self.network.vgg_feat_extractor.set_train(False)
        self.network.discriminator.set_train(False)
        return self

    def construct(self, *inputs):
        network_fwd_bwd = ops.value_and_grad(self.network, grad_position=None, weights=self.weights, has_aux=True)
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


class VAEGTrainer:
    """CTSDGTrainer"""

    def __init__(self, train_one_step_g, train_one_step_d, cfg, finetune=False):
        super(VAEGTrainer, self).__init__()
        self.train_one_step_g = train_one_step_g
        self.train_one_step_d = train_one_step_d
        self.finetune = finetune
        self.cfg = cfg

    def run(self, *inputs):
        x_gt, x_class, x_indiv_mels = inputs

        self.train_one_step_g.set_train(not self.finetune)
        loss_g, output = self.train_one_step_g(*inputs)
        self.train_one_step_d.set_train()
        loss_d = self.train_one_step_d(x_gt, output)
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
        for num_batch, sample in enumerate(dataloader):
            if num_batch >= total_steps:
                print("Reached the target number of iterations")
                break

            x_gt = sample['gt']
            x_class = sample['class']
            x_indiv_mels = sample['indiv_mels']

            loss_g, loss_d = self.run(x_gt, x_class, x_indiv_mels)
            # if save_ckpt_logs:
            #     callbacks([loss_g.asnumpy().mean(), loss_d.asnumpy().mean()])

        print("Training completed")


class GWithLossCell(nn.Cell):
    """Generator with loss cell"""

    def __init__(self, generator, discriminator, cfg):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

        self.l1 = nn.L1Loss()
        self.criterion = nn.BCELoss(reduction="mean")
        self.real_target = Tensor(1.0, mstype.float32)

        self.mse = nn.MSELoss()
        self.kl = nn.KLDivLoss()

    def construct(self, data_batch):

        data_batch = self.generator(data_batch)

        logits = data_batch['pose_motion_pred']
        labels = data_batch['pose_motion_gt']

        mu = data_batch['mu']
        logvar = data_batch['logvar']
        std = ops.Exp()(0.5 * logvar)

        loss_mse = self.mse(logits, labels)
        loss_kl = ops.sum(mu ** 2 + std ** 2 - ops.log(std) - 0.5)

        output_pred = self.discriminator(logits, is_real=False)

        real_target = self.real_target.expand_as(output_pred)
        loss_adversarial = self.criterion(output_pred, real_target)


        loss_g = loss_mse + loss_kl + 0.7 * loss_adversarial

        result = ops.stop_gradient(logits)
        return loss_g, result


class DWithLossCell(nn.Cell):
    """Discriminator with loss cell"""

    def __init__(self, discriminator):
        super().__init__()
        self.discriminator = discriminator
        self.criterion = nn.BCELoss(reduction="mean")
        self.real_target = Tensor(1.0, mstype.float32)
        self.fake_target = Tensor(0.0, mstype.float32)

    def construct(self, ground_truth, generated_samples):
        real_pred = self.discriminator(ground_truth, is_real=True)
        fake_pred = self.discriminator(generated_samples, is_real=False)

        real_target = self.real_target.expand_as(real_pred)
        fake_target = self.fake_target.expand_as(fake_pred)

        loss_adversarial = (
            self.criterion(real_pred, real_target)
            + self.criterion(fake_pred, fake_target)
        )
        return loss_adversarial