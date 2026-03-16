# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
# Copyright (c) Meta Platforms, Inc. and affiliates.


import os
import gc
import json
import math
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from utils import get_args_parser
from dataset import FlowDataModule
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities import rank_zero_info


import PIL.Image
import lightning.pytorch as pl
import torch
from flow_matching.path import CondOTProbPath, MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.solver.ode_solver import ODESolver
from flow_matching.utils import ModelWrapper
from models.discrete_unet import DiscreteUNetModel
from models.ema import EMA
from models.model_configs import instantiate_model
from torch.nn import Module
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import save_image

from utils import get_time_discretization

MASK_TOKEN = 256
PRINT_FREQUENCY = 50


class CFGScaledModel(ModelWrapper):
    def __init__(self, model: Module):
        super().__init__(model)
        self.nfe_counter = 0

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, cfg_scale: float, label: torch.Tensor
    ):
        module = self.model
        is_discrete = isinstance(module, DiscreteUNetModel) or (
            isinstance(module, EMA) and isinstance(module.model, DiscreteUNetModel)
        )
        assert (
            cfg_scale == 0.0 or not is_discrete
        ), f"Cfg scaling does not work for the logit outputs of discrete models. Got cfg weight={cfg_scale} and model {type(self.model)}."
        t = torch.zeros(x.shape[0], device=x.device) + t

        if cfg_scale != 0.0:
            with torch.no_grad():
                conditional = self.model(x, t, extra={"label": label})
                condition_free = self.model(x, t, extra={})
            result = (1.0 + cfg_scale) * conditional - cfg_scale * condition_free
        else:
            with torch.no_grad():
                result = self.model(x, t, extra={"label": label})

        self.nfe_counter += 1
        if is_discrete:
            return torch.softmax(result.to(dtype=torch.float32), dim=-1)
        return result.to(dtype=torch.float32)

    def reset_nfe_counter(self) -> None:
        self.nfe_counter = 0

    def get_nfe(self) -> int:
        return self.nfe_counter


def skewed_timestep_sample(num_samples: int, device: torch.device) -> torch.Tensor:
    p_mean = -1.2
    p_std = 1.2
    rnd_normal = torch.randn((num_samples,), device=device)
    sigma = (rnd_normal * p_std + p_mean).exp()
    time = 1 / (1 + sigma)
    time = torch.clip(time, min=0.0001, max=1.0)
    return time


class FlowLightningModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(vars(args))

        self.model = instantiate_model(
            architechture=args.dataset,
            is_discrete=args.discrete_flow_matching,
            use_ema=args.use_ema,
        )

        if args.discrete_flow_matching:
            scheduler = PolynomialConvexScheduler(n=3.0)
            self.train_path = MixtureDiscreteProbPath(scheduler=scheduler)
        else:
            self.train_path = CondOTProbPath()

        self.cfg_scaled_model = None
        self.val_fid_metric = None
        self.val_target_synthetic = 0
        self.val_num_synthetic = 0
        self.snapshots_saved = False

    def forward(self, x_t, t, extra):
        return self.model(x_t, t, extra=extra)

    def _get_conditioning(self, labels: torch.Tensor):
        if self.training and torch.rand(1, device=labels.device) < self.args.class_drop_prob:
            return {}
        return {"label": labels}

    def training_step(self, batch, _batch_idx):
        samples, labels = batch
        conditioning = self._get_conditioning(labels)

        if self.args.discrete_flow_matching:
            samples = (samples * 255.0).to(torch.long)
            t = torch.rand(samples.shape[0], device=samples.device)
            x_0 = torch.zeros(samples.shape, dtype=torch.long, device=samples.device) + MASK_TOKEN
            path_sample = self.train_path.sample(t=t, x_0=x_0, x_1=samples)
            logits = self.model(path_sample.x_t, t=t, extra=conditioning)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape([-1, 257]), samples.reshape([-1])
            ).mean()
        else:
            samples = samples * 2.0 - 1.0
            noise = torch.randn_like(samples)
            if self.args.skewed_timesteps:
                t = skewed_timestep_sample(samples.shape[0], device=samples.device)
            else:
                t = torch.rand(samples.shape[0], device=samples.device)

            path_sample = self.train_path.sample(t=t, x_0=noise, x_1=samples)
            x_t = path_sample.x_t
            u_t = path_sample.dx_t
            loss = torch.pow(self.model(x_t, t, extra=conditioning) - u_t, 2).mean()

        if not math.isfinite(loss.item()):
            raise ValueError(f"Loss is {loss.item()}, stopping training")

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=labels.shape[0],
        )
        return loss

    def on_before_zero_grad(self, _optimizer):
        if isinstance(self.model, EMA):
            self.model.update_ema()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            betas=tuple(self.args.optimizer_betas),
        )

        if self.args.decay_lr:
            lr_schedule = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                total_iters=self.args.epochs,
                start_factor=1.0,
                end_factor=1e-8 / self.args.lr,
            )
        else:
            lr_schedule = torch.optim.lr_scheduler.ConstantLR(
                optimizer,
                total_iters=self.args.epochs,
                factor=1.0,
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_schedule,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_validation_epoch_start(self):
        gc.collect()
        self.cfg_scaled_model = CFGScaledModel(model=self.model)
        self.cfg_scaled_model.train(False)

        world_size = self.trainer.world_size
        per_rank = self.args.fid_samples // world_size
        if self.trainer.is_global_zero:
            self.val_target_synthetic = self.args.fid_samples - (world_size - 1) * per_rank
        else:
            self.val_target_synthetic = per_rank

        self.val_num_synthetic = 0
        self.snapshots_saved = False

        if self.args.compute_fid:
            self.val_fid_metric = FrechetInceptionDistance(normalize=True).to(self.device)
        else:
            self.val_fid_metric = None

        if self.args.output_dir and self.trainer.is_global_zero:
            (Path(self.args.output_dir) / "snapshots").mkdir(parents=True, exist_ok=True)

    def validation_step(self, batch, batch_idx):
        samples, labels = batch

        if self.val_fid_metric is not None:
            self.val_fid_metric.update(samples, real=True)

        if self.val_num_synthetic < self.val_target_synthetic:
            synthetic_samples = self._sample_synthetic_batch(samples, labels)
            if self.val_num_synthetic + synthetic_samples.shape[0] > self.val_target_synthetic:
                synthetic_samples = synthetic_samples[: self.val_target_synthetic - self.val_num_synthetic]

            if self.val_fid_metric is not None:
                self.val_fid_metric.update(synthetic_samples, real=False)

            self.val_num_synthetic += synthetic_samples.shape[0]

            if not self.snapshots_saved and self.args.output_dir and self.trainer.is_global_zero:
                save_image(
                    synthetic_samples,
                    fp=Path(self.args.output_dir)
                    / "snapshots"
                    / f"{self.current_epoch}_{batch_idx}.png",
                )
                self.snapshots_saved = True

            if self.args.save_fid_samples and self.args.output_dir:
                images_np = (
                    (synthetic_samples * 255.0)
                    .clip(0, 255)
                    .to(torch.uint8)
                    .permute(0, 2, 3, 1)
                    .cpu()
                    .numpy()
                )
                for image_index, image_np in enumerate(images_np):
                    image_dir = Path(self.args.output_dir) / "fid_samples"
                    os.makedirs(image_dir, exist_ok=True)
                    image_path = image_dir / f"{self.global_rank}_{batch_idx}_{image_index}.png"
                    PIL.Image.fromarray(image_np, "RGB").save(image_path)

        if self.val_fid_metric is not None and batch_idx % PRINT_FREQUENCY == 0:
            running_fid = self.val_fid_metric.compute()
            rank_zero_info(
                "Evaluating [%s/%s] samples generated [%s/%s] running fid %s",
                batch_idx,
                self.trainer.num_val_batches[0],
                self.val_num_synthetic,
                self.val_target_synthetic,
                running_fid,
            )

    def on_validation_epoch_end(self):
        if self.val_fid_metric is None:
            return

        # FrechetInceptionDistance synchronizes metric state during compute in DDP.
        # Keep the result on the current device and avoid an extra reduction on CPU.
        fid = self.val_fid_metric.compute().detach().float()
        self.log("eval_fid", fid, prog_bar=True)

    def _sample_synthetic_batch(self, samples: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        self.cfg_scaled_model.reset_nfe_counter()

        if self.args.discrete_flow_matching:
            scheduler = PolynomialConvexScheduler(n=3.0)
            path = MixtureDiscreteProbPath(scheduler=scheduler)
            p = torch.zeros(size=[257], dtype=torch.float32, device=self.device)
            p[256] = 1.0
            solver = MixtureDiscreteEulerSolver(
                model=self.cfg_scaled_model,
                path=path,
                vocabulary_size=257,
                source_distribution_p=p,
            )
            x_0 = torch.zeros(samples.shape, dtype=torch.long, device=self.device) + MASK_TOKEN
            if self.args.sym_func:
                sym = lambda t: 12.0 * torch.pow(t, 2.0) * torch.pow(1.0 - t, 0.25)
            else:
                sym = self.args.sym
            dtype = torch.float32 if self.args.sampling_dtype == "float32" else torch.float64
            synthetic_samples = solver.sample(
                x_init=x_0,
                step_size=1.0 / self.args.discrete_fm_steps,
                verbose=False,
                div_free=sym,
                dtype_categorical=dtype,
                label=labels,
                cfg_scale=self.args.cfg_scale,
            )
        else:
            solver = ODESolver(velocity_model=self.cfg_scaled_model)
            x_0 = torch.randn(samples.shape, dtype=torch.float32, device=self.device)

            if self.args.edm_schedule:
                time_grid = get_time_discretization(nfes=self.args.ode_options["nfe"])  # type: ignore[index]
            else:
                time_grid = torch.tensor([0.0, 1.0], device=self.device)

            synthetic_samples = solver.sample(
                time_grid=time_grid,
                x_init=x_0,
                method=self.args.ode_method,
                return_intermediates=False,
                atol=self.args.ode_options.get("atol", 1e-5),
                rtol=self.args.ode_options.get("rtol", 1e-5),
                step_size=self.args.ode_options.get("step_size", None),
                label=labels,
                cfg_scale=self.args.cfg_scale,
            )
            synthetic_samples = torch.clamp(synthetic_samples * 0.5 + 0.5, min=0.0, max=1.0)
            synthetic_samples = torch.floor(synthetic_samples * 255)

        synthetic_samples = synthetic_samples.to(torch.float32) / 255.0
        rank_zero_info(
            "%s samples generated in %s evaluations.",
            samples.shape[0],
            self.cfg_scaled_model.get_nfe(),
        )
        return synthetic_samples

def _get_resume_ckpt_path(args) -> str | None:
    if args.no_resume:
        return None

    last_checkpoint = Path(args.output_dir) / "last.ckpt"
    if last_checkpoint.is_file():
        rank_zero_info("Found %s, resuming with Lightning automatic state restoration.", last_checkpoint)
        return "last"
    return None


def main(args):
    rank_zero_info("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    rank_zero_info("{}".format(args).replace(", ", ",\n"))

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    args_filepath = Path(args.output_dir) / "args.json"
    rank_zero_info(f"Saving args to {args_filepath}")
    with open(args_filepath, "w") as f:
        json.dump(vars(args), f)

    pl.seed_everything(args.seed, workers=True)

    cudnn.benchmark = True

    model = FlowLightningModule(args=args)
    datamodule = FlowDataModule(args=args)

    callbacks = [LearningRateMonitor(logging_interval="epoch")]
    if not args.eval_only:
        callbacks.append(
            ModelCheckpoint(
                dirpath=args.output_dir,
                filename="checkpoint-{epoch}",
                save_last=True,
                save_top_k=-1,
                every_n_epochs=args.eval_frequency if args.eval_frequency > 0 else 1,
                save_on_train_epoch_end=True,
            )
        )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    num_nodes = max(world_size // max(local_world_size, 1), 1)

    if world_size > 1:
        # When launched via torchrun, align Trainer topology with distributed env.
        devices = local_world_size
        strategy = "ddp"
    else:
        devices = "auto"
        strategy = "auto"
    eval_enabled = args.eval_only or args.test_run or args.eval_frequency > 0
    val_batches = 1 if args.test_run else (1.0 if eval_enabled else 0.0)

    trainer = pl.Trainer(
        default_root_dir=args.output_dir,
        accelerator="gpu" if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu",
        devices=devices,
        num_nodes=num_nodes,
        strategy=strategy,
        max_epochs=args.epochs,
        logger=TensorBoardLogger(save_dir=args.output_dir, name="", version=""),
        callbacks=callbacks,
        benchmark=True,
        accumulate_grad_batches=args.accum_iter,
        log_every_n_steps=args.log_every_step,
        check_val_every_n_epoch=args.eval_frequency if args.eval_frequency > 0 else 1,
        num_sanity_val_steps=0,
        limit_train_batches=1 if args.test_run else 1.0,
        limit_val_batches=val_batches,
    )

    ckpt_path = _get_resume_ckpt_path(args)

    if args.eval_only:
        trainer.validate(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        return

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
