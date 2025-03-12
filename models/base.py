import lightning.pytorch as pl
from lightning.pytorch.utilities.model_summary.model_summary import ModelSummary
import logging
import torch
from torch import Tensor
import numpy as np
import os
import warnings
import abc

from utils.loss_functions import RegressionLoss, EnergyLoss
from utils.types import BatchPattern, ReturnPattern
from utils.torch_modules import DataTansform, EncodingModule, PadTau


# Ignore anticipated PL warnings.
warnings.filterwarnings('ignore', '.*infer the indices fetched for your dataloader.*')
warnings.filterwarnings('ignore', '.*You requested to overfit.*')

logger = logging.getLogger('lightning')


class LightningNet(pl.LightningModule):
    """Implements basic training routine.

    Note:
        * This class should take hyperparameters for the training process. Model hyperparameters should be
            handled in the PyTorch module.
        * call 'self.save_hyperparameters()' at the end of subclass `__init__(...)`.
        * The subclass must implement a `forward` method (see PyTorch doc) which takes the arguments
            `x`, the sequencial input, and `s`, the static features.

    Shape:
        The subclass must implement a `forward` method (see PyTorch doc) which takes the arguments `x`, the
        sequencial input, and optional argument `s`, the static input:
        * `x`: (B, C, S)
        * `s`: (B, D)
        * return: (B, O, S)
        where B=batch size, S=sequence length, C=number of dynamic channels, and D=number of static channels.
    """
    def __init__(
            self,
            criterion: str = 'L2',
            es_beta: float= 1.0,
            es_length: int = 1,
            num_members: int = 10,
            inference_taus: list[float] = [0, 1, 0.1],
            norm_args_features: dict | None = None,
            norm_args_stat_features: dict | None = None,
            norm_args_targets: dict | None = None
            ) -> None:
        """Initialize lightning module, should be subclassed.

        Args:
            criterion: the criterion to use, defaults to L2.
            beta : power parameter in the energy score, a float (0, 2], default is 1.0, only
                effective for criterion='es'.
            es_length: length of sequence fed jointly into the loss function, an integer > 0, default is 1, only
                effective for criterion='es'.
            inference_taus: The tau values (probabilities) to evaluate in inference. Only applies for
                distribution-aware criterions. [start, stop, step].
            num_members: Number of members to sample in prediction.
        """

        super().__init__()

        if criterion.lower() == 'es':
            self.loss_fn = EnergyLoss(beta=es_beta, es_length=es_length)
        else:
            self.loss_fn = RegressionLoss(criterion=criterion, sqrt_transform=False)

        self.sample_tau = self.loss_fn.has_tau
        self.inference_taus = list(
            np.arange(inference_taus[0], inference_taus[1] + 1e-10, inference_taus[2])) if self.sample_tau else [0.5]

        self.num_members = num_members

        if norm_args_features is not None:
            self.norm_features = DataTansform(**norm_args_features)
        else:
            self.norm_features = torch.nn.Identity()

        if norm_args_stat_features is not None:
            self.norm_stat_features = DataTansform(**norm_args_stat_features)
        else:
            self.norm_stat_features = torch.nn.Identity()

        if norm_args_targets is not None:
            self.norm_targets = DataTansform(**norm_args_targets)
        else:
            self.norm_targets = torch.nn.Identity()

    def normalize_data(self, batch: BatchPattern) -> tuple[Tensor, Tensor, Tensor]:
        x = self.norm_features(batch.dfeatures)
        s = self.norm_stat_features(batch.sfeatures)
        y = self.norm_targets(batch.dtargets)

        return x, s, y

    def denormalize_target(self, target: Tensor) -> Tensor:
        if isinstance(self.norm_targets, torch.nn.Identity):
            return target
        else:
            return self.norm_targets(target, invert=True)

    def shared_step(
            self,
            batch: BatchPattern,
            tau: float | None,
            step_type: str) -> tuple[Tensor, ReturnPattern | None]:
        """A single training step shared across specialized steps that returns the loss and the predictions.

        Args:
            batch: the bach.
            tau: float, the tau value for the loss function.
            step_type: the step type (training mode), one of (`train`, `val`, `test`, `pred`).

        Returns:
            tuple[Tensor, ReturnPattern]: the loss, the predictions.
        """

        if step_type not in ('train', 'val', 'test', 'pred'):
            raise ValueError(f'`step_type` must be one of (`train`, `val`, `test`, `pred`), is {step_type}.')

        x, s, target = self.normalize_data(batch)

        num_cut = batch.coords.warmup_size[0]
        batch_size = target.shape[0]

        target_hat0 = self(x=x, s=s, tau=tau)

        if step_type == 'pred':
            preds = ReturnPattern(
                dtargets=self.denormalize_target(target_hat0.detach()).cpu(),
                coords=batch.coords,
                tau=tau
            )
            return torch.zeros(1), preds

        if self.loss_fn.criterion == 'es':
            target_hat1 = self(x=x, s=s, tau=tau)
            loss, loss_log = self.loss_fn(
                input0=target_hat0[..., num_cut:],
                input1=target_hat1[..., num_cut:],
                target=target[..., num_cut:]
            )
        elif self.sample_tau:
            loss, loss_log = self.loss_fn(
                input=target_hat0[..., num_cut:],
                target=target[..., num_cut:],
                tau=tau
            )
        else:
            loss, loss_log = self.loss_fn(
                input=target_hat0[..., num_cut:],
                target=target[..., num_cut:],
            )

        preds = None

        self.log_dict(
            {f'{step_type}_{k}': v for k, v in loss_log.items()},
            prog_bar=True,
            on_step=True if step_type == 'train' else False,
            on_epoch=True,
            batch_size=batch_size
        )

        return loss, preds

    def training_step(
            self,
            batch: BatchPattern,
            batch_idx: int) -> Tensor:
        """A single training step.

        Args:
            batch (Iterable[Tensor]): the bach, x, m, y, s tuple.
            batch_idx (int): the batch index (required by pl).

        Returns:
            Tensor: The batch loss.
        """

        if self.sample_tau:
            tau = np.random.uniform()
        else:
            tau = None

        loss, _ = self.shared_step(batch, tau=tau, step_type='train')

        return loss

    def validation_step(
            self,
            batch: BatchPattern,
            batch_idx: int) -> dict[str, Tensor]:
        """A single validation step.

        Args:
            batch (Iterable[Tensor]): the bach, x, m, y, s tuple.
            batch_idx (int): the batch index (required by pl).

        """

        loss, _ = self.shared_step(batch, tau=0.5, step_type='val')

        return {'val_loss': loss}

    def test_step(
            self,
            batch: BatchPattern,
            batch_idx: int) -> dict[str, Tensor]:
        """A single test step.

        Args:
            batch (Iterable[Tensor]): the bach, x, m, y, s tuple.
            batch_idx (int): the batch index (required by pl).

        """

        loss, _ = self.shared_step(batch, tau=0.5, step_type='test')

        return {'test_loss': loss}

    def predict_step(
            self,
            batch: BatchPattern,
            batch_idx: int,
            dataloader_idx: int = 0
            ) -> list[ReturnPattern]:
        """A single predict step.

        Args:
            batch (Iterable[Tensor]): the bach, x, m, y, s tuple.
            batch_idx (int): the batch index (required by pl).

        """

        member_preds = []

        if self.sample_tau:
            for tau in self.inference_taus:
                _, preds = self.shared_step(batch, tau=tau, step_type='pred')
                member_preds.append(preds)
        else:
            for _ in range(self.num_members):
                _, preds = self.shared_step(batch, tau=None, step_type='pred')
                member_preds.append(preds)

        return member_preds

    def summarize(self):
        s = f'=== Summary {"=" * 31}\n'
        s += f'{str(ModelSummary(self))}\n\n'
        s += f'=== Model {"=" * 33}\n'
        s += f'{str(self)}'

        return s

    def on_train_start(self) -> None:

        if self.logger is None:
            raise AttributeError(
                'self.logger is not set.'
            )

        if not hasattr(self.logger, 'log_dir') or (self.logger.log_dir is None):
            raise KeyError('logger has no attribute \'log_dir\', or it is None.')

        os.makedirs(self.logger.log_dir, exist_ok=True)
        with open(os.path.join(self.logger.log_dir, 'model_summary.txt'), 'w') as f:
            f.write(self.summarize())

        return super().on_train_start()


class TemporalNet(LightningNet, abc.ABC):

    def __init__(
            self,
            num_static_in: int,
            num_dynamic_in: int,
            num_outputs: int,
            noise_dim: int = 0,
            noise_std: float = 1.,
            model_dim: int = 8,
            enc_dropout: float = 0.2,
            **kwargs) -> None:

        super().__init__(**kwargs)

        self.noise_dim = noise_dim
        self.noise_std = noise_std

        # Pad tau for quantile/expectile loss.
        if self.sample_tau:
            self.pad_tau = PadTau()

        # Static input encoding
        self.static_encoding = EncodingModule(
            num_in=num_static_in,
            num_enc=model_dim,
            num_layers=2,
            dropout=enc_dropout,
            activation=torch.nn.ReLU(),
            activation_last=torch.nn.Tanh()
        )

        if self.sample_tau:
            extra_dim = 1
        else:
            extra_dim = self.noise_dim

        # Dynamic input encoding
        self.dynamic_encoding = EncodingModule(
            num_in=num_dynamic_in + extra_dim,
            num_enc=model_dim,
            num_layers=2,
            dropout=enc_dropout,
            activation=torch.nn.Tanh(),
            activation_last=torch.nn.Sigmoid()
            )

        # Temporal layer
        # -------------------------------------------------

        # >>>> Is defined in subclass.

        # Output
        # -------------------------------------------------

        self.dropout1d = torch.nn.Dropout1d(enc_dropout)

        self.output_layer = torch.nn.Conv1d(
                in_channels=model_dim,
                out_channels=num_outputs,
                kernel_size=1
        )

        self.out_activation = torch.nn.Softplus(beta=3)  # beta=3 for sharper transition.

    """Implements a temporal base lightning network."""
    @abc.abstractmethod
    def temporal_forward(self, x: Tensor) -> Tensor:
        """Temporal layer forward pass, must be overridden in subclass.

        Shapes:
            x: (batch, channels, sequence).

        Returns:
            Tensor with same shape as input (batch, channels, sequence).

        Args:
            x: The input tensor of shape (batch, channels, sequence),

        Returns:
            A tensor of same shape as the input.
        """

    def forward(self, x: Tensor, s: Tensor, tau: float | None) -> Tensor:
        """Temoral layer with encoding pre-fusion.

        Args:
            x: The dynamic inputs, shape (batch, dynamic_channels, sequence)
            s: The static inputs, shape (batch, static_channels)

        Returns:
            A tensor of predicted values (batch, outputs, sequence).

        """

        # Add noise to dynamic input.
        if self.sample_tau:
            x = self.pad_tau(x=x, tau=tau)
        elif self.noise_dim > 0:
            batch_size, channels, sequence = x.shape
            e = torch.randn(
                batch_size,
                self.noise_dim,
                sequence,
                dtype=self.dtype,
                device=self.device) * self.noise_std
            x = torch.cat((x, e), dim=1)

        # Dynamic encoding: (B, D, S) -> (B, E, S)
        enc = self.dynamic_encoding(x)

        if s is not None:
            # Static encoding and unsqueezing: (B, C) ->  (B, E, 1)
            s_enc = self.static_encoding(s.unsqueeze(-1))

            # Multiply static encoding and dynamic encoding: (B, E, S) + (B, E, 1) -> (B, E, S)
            enc = enc + s_enc

        # Temporal layer and dropout: (B, E, S) -> (B, E, S)
        temp_enc = self.temporal_forward(enc)
        temp_enc = self.dropout1d(temp_enc)

        # Output layer, and activation: (B, E + 1, S) -> (B, O, S)
        out = self.output_layer(temp_enc)
        out = self.out_activation(out)

        return out
