import sys

from absl import logging
from absl import flags

import jax
from jax import numpy as jnp

from flax import traverse_util

import optax

import numpy as np

from jaxline import experiment
from jaxline import platform
from jaxline import utils as jl_utils

from clu.parameter_overview import count_parameters

from einops import rearrange

import models
import input_pipeline
import utils

FLAGS = flags.FLAGS


class Experiment(experiment.AbstractExperiment):
    CHECKPOINT_ATTRS = {'_params': 'params', '_opt_state': 'opt_state'}
    NON_BROADCAST_CHECKPOINT_ATTRS = {'_step': 'step'}

    def __init__(self, mode, config, init_rng):
        super().__init__(mode=mode)
        self.mode = mode
        self.config = config
        self.init_rng = init_rng

        # Checkpointed experiment state.
        self._step = None
        self._params = None
        self._opt_state = None

        # Optimizer.
        self._opt = None

        # Input pipelines.
        self._train_input = None
        self._eval_input = None

        # Get model
        model = getattr(models, self.config.model.name)
        config = getattr(models, self.config.model.name + 'Config')
        self.net = model(config=config(num_classes=self.config.num_classes,
                                       **self.config.model.config_kwargs))

        lr_sched_fn = getattr(optax, self.config.lr_schedule.name)
        self.lr_sched_fn = lr_sched_fn(**self.config.lr_schedule.kwargs)

        self.train_imsize = 224
        self.test_imsize = 224

        donate_argnums = (0, 1, 2)
        self.train_fn = jax.pmap(self._train_fn,
                                 axis_name='i',
                                 donate_argnums=donate_argnums)
        self.eval_fn = jax.pmap(self._eval_fn, axis_name='i')

    def _initialize_train(self):
        self._train_input = self._build_train_input()
        if self._params is None:
            input_shape = (1, self.config.image_size, self.config.image_size, 3)
            inputs = jnp.ones(input_shape, jnp.float32)
            init_net = jax.pmap(lambda *a: self.net.init(*a, is_training=True),
                                axis_name='i')
            init_rng = jl_utils.bcast_local_devices(self.init_rng)
            self._params = init_net(init_rng, inputs)
            num_params = count_parameters(self._params)
            logging.info(f'Net params: {num_params / jax.local_device_count()}')
            self._make_opt()
            self._opt_state = self._opt.init(self._params)

    def _make_opt(self):

        def flattened_traversal(fn):

            def mask(data):
                flat = traverse_util.flatten_dict(data)
                return traverse_util.unflatten_dict(
                    {k: fn(k, v) for k, v in flat.items()})

            return mask

        weights_opt = getattr(optax, self.config.optimizer_weights.name)
        biases_opt = getattr(optax, self.config.optimizer_biases.name)
        self._opt = optax.chain(
            optax.masked(
                weights_opt(**self.config.optimizer_weights.kwargs),
                mask=flattened_traversal(lambda path, _: path[-1] != 'bias')),
            optax.masked(
                biases_opt(**self.config.optimizer_biases.kwargs),
                mask=flattened_traversal(lambda path, _: path[-1] == 'bias')),
        )

    def _one_hot(self, labels):
        return jax.nn.one_hot(labels, self.config.num_classes)

    def _loss_fn(self, params, batch):
        if self.config.get('transpose', False):
            images = rearrange(batch['images'], 'H W C N -> N H W C')
        else:
            images = batch['images']
        if self.config.dtype is jnp.bfloat16:
            images = utils.to_bf16(images)

        logits = self.net.apply(params, images, is_training=True)
        y = self._one_hot(batch['labels'])
        if 'mix_labels' in batch:  # Handle cutmix/mixup label mixing
            logging.info('Using mixup or cutmix!')
            y1 = self._one_hot(batch['mix_labels'])
            y = batch['ratio'][:,
                               None] * y + (1. - batch['ratio'][:, None]) * y1
        if self.config.label_smoothing > 0:
            spositives = 1. - self.config.label_smoothing
            snegatives = self.config.label_smoothing / self.config.num_classes
            y = spositives * y + snegatives
        if self.config.dtype is jnp.bfloat16:
            logits = logits.astype(jnp.float32)
        which_loss = getattr(utils, self.config.which_loss)
        loss = which_loss(logits, y, reduction='mean')
        metrics = utils.topk_correct(logits, batch['labels'], prefix='train_')
        metrics = jax.tree_map(jnp.mean, metrics)
        metrics['train_loss'] = loss
        scaled_loss = loss / jax.device_count()
        return scaled_loss, metrics

    def _train_fn(self, params, opt_state, batch, global_step):
        grad_fn = jax.grad(self._loss_fn, argnums=0, has_aux=True)
        if self.config.dtype is jnp.bfloat16:
            params = jax.tree_map(utils.to_bf16, params)
        grads, metrics = grad_fn(params, batch)
        if self.config.dtype is jnp.bfloat16:
            metrics, grads = jax.tree_map(utils.from_bf16, (metrics, grads))

        grads = jax.lax.psum(grads, 'i')
        metrics = jax.lax.pmean(metrics, 'i')
        lr = self.lr_sched_fn(global_step)
        metrics['learning_rate'] = lr
        updates, opt_state = self._opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return {'params': params, 'opt_state': opt_state, 'metrics': metrics}

    def step(self, global_step, *unused_args, **unused_kwargs):
        if self._train_input is None:
            self._initialize_train()
        batch = next(self._train_input)
        out = self.train_fn(params=self._params,
                            opt_state=self._opt_state,
                            batch=batch,
                            global_step=global_step)
        self._params = out['params']
        self._opt_state = out['opt_state']
        self._step = global_step
        return jl_utils.get_first(out['metrics'])

    def _build_train_input(self):
        num_devices = jax.device_count()
        global_batch_size = self.config.train_batch_size
        bs_per_device, ragged = divmod(global_batch_size, num_devices)
        if ragged:
            raise ValueError(
                f'Global batch size {global_batch_size} must be divisible by '
                f'num devices {num_devices}')
        return input_pipeline.load(
            input_pipeline.Split.TRAIN_AND_VALID,
            is_training=True,
            batch_dims=[jax.local_device_count(), bs_per_device],
            transpose=self.config.get('transpose', False),
            image_size=(self.train_imsize, self.train_imsize),
            augment_name=self.config.augment_name,
            augment_before_mix=self.config.get('augment_before_mix', True),
            name=self.config.which_dataset,
            fake_data=False)

    def evaluate(self, global_step, **unused_args):
        metrics = self._eval_epoch(self._params, self._state)
        logging.info(f'[Step {global_step}] Eval scalars: {metrics}')
        return metrics

    def _eval_epoch(self, params):
        num_samples = 0.
        summed_metrics = None

        for batch in self._build_eval_input():
            num_samples += np.prod(batch['labels'].shape[:2])
            metrics = self._eval_fn(params, batch)
            metrics = jax.tree_map(lambda x: jnp.sum(x[0], axis=0), metrics)
            if summed_metrics is None:
                summed_metrics = metrics
            else:
                summed_metrics = jax.tree_multimap(jnp.add, summed_metrics,
                                                   metrics)
        mean_metrics = jax.tree_map(lambda x: x / num_samples, summed_metrics)
        return jax.device_get(mean_metrics)

    def _eval_fn(self, params, batch):
        if self.config.get('transpose', False):
            images = rearrange(batch['images'], 'H W C N -> N H W C')
        else:
            images = batch['images']

        logits = self.net.apply(params, images, is_training=False)
        y = self._one_hot(batch['labels'])
        which_loss = getattr(utils, self.config.which_loss)
        loss = which_loss(logits, y, reduction=None)
        metrics = utils.topk_correct(logits, batch['labels'], prefix='eval_')
        metrics['eval_loss'] = loss
        return jax.lax.psum(metrics, 'i')

    def _build_eval_input(self):
        bs_per_device = (self.config.eval_batch_size //
                         jax.local_device_count())
        split = input_pipeline.Split.from_string(self.config.eval_subset)
        eval_preproc = self.config.get('eval_preproc', 'crop_resize')
        return input_pipeline.load(
            split,
            is_training=False,
            batch_dims=[jax.local_device_count(), bs_per_device],
            transpose=self.config.get('transpose', False),
            image_size=(self.test_imsize, self.test_imsize),
            name=self.config.which_dataset,
            eval_preproc=eval_preproc,
            fake_data=False)


if __name__ == '__main__':
    flags.mark_flag_as_required('config')
    platform.main(Experiment, sys.argv[1:])
