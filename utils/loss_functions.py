import math
import logging
import torch
from torch import nn
from torch import Tensor
from torch.linalg import vector_norm

logger = logging.getLogger('pytorch_lightning')


class RegressionLoss(nn.Module):
    """Loss functions that ignore NaN in target.
    The loss funtion allows for having missing / non-finite values in the target.

    !IMPORTANT NOTE!
    ------------------
    With sqrt_transform=True, negative target values are set to 0, and the model predictions are expected to be >= 0.
    This can be achieved by using a activation function such as ReLU or Softplus.

    Example
    -------
    >>> import torch
    >>> mse_loss = RegressionLoss(sample_wise=True)
    >>> input = torch.ones(2, 2, requires_grad=True)
    >>> target = torch.ones(2, 2, requires_grad=False) + 1.
    >>> target[0, 0] = float('NaN')
    >>> loss = mse_loss(input, target)
    >>> loss.backward()
    >>> print('input:', input)
    input:
     tensor([[1., 1.],
             [1., 1.]], requires_grad=True)
    >>> print('target:', target)
    target:
     tensor([[nan, 2.],
             [2., 2.]])
    >>> print('mse:', loss)
    mse:
     tensor(1., grad_fn=<MeanBackward0>)
    >>> print('gradients:', input.grad)
    gradients:
     tensor([[ 0.0000, -1.0000],
             [-0.5000, -0.5000]])

    Shape
    -----
    * input: (N, *), where * means, any number of additional dimensions
    * target: (N, *), same shape as the input
    """

    LOSS_FUNCTIONS = ('l1', 'l2', 'huber')
    LOSS_FUNCTIONS_WITH_TAU = ('quantile', 'expectile')

    def __init__(
            self,
            criterion: str,
            sample_wise: bool = True,
            sqrt_transform: bool = False) -> None:
        """Initialize RegressionLoss.

        Args:
            criterion : str (``'l1'`` | ``'l2'`` | ``'huber'``)
                ``'l1'`` for Mean Absolute Error (MAE),
                ``'l2'`` for Mean Squared Error (MSE),
                ``'huber'`` for Huber loss (mix of l1 and l2),
            sample_wise : bool
                Whether to calculate sample-wise loss first and average then (`True`, default) or to
                calculate the loss across all elements. The former weights each batch element equally,
                the latter weights each observation equally. This is relevant especially with many NaN
                in the target tensor, while there is no difference without NaN.

        """

        super(RegressionLoss, self).__init__()

        criterion = criterion.lower()

        if criterion not in self.LOSS_FUNCTIONS:
            loss_functions_str = '(\'' + '\' | \''.join(self.LOSS_FUNCTIONS) + '\')'
            raise ValueError(
                f'argument `criterion` must be one of {loss_functions_str}, is \'{criterion}\'.'
            )

        self.criterion = criterion
        self.sample_wise = sample_wise

        loss_fn_args: dict = dict(reduction='none')
        if self.criterion == 'huber':
            loss_fn_args.update(dict(delta=0.3))

        self.sqrt_transform = sqrt_transform

        self.loss_fn = {
            'l1': nn.L1Loss,
            'l2': nn.MSELoss,
            'huber': nn.HuberLoss,
        }[self.criterion](**loss_fn_args)

    def forward(
            self,
            input: Tensor,
            target: Tensor,
            basin_std: Tensor | None = None) -> Tensor:
        """Forward call, calculate loss from input and target, must have same shape.

        Args:
            input: Predicted mean of shape (B x ...)
            target: Target of shape (B x ...)

        Returns:
            The loss, a scalar.
        """

        mask = target.isfinite()
        # By setting target to input for NaNs, we set gradients to zero.
        target = target.where(mask, input)

        element_error = self.loss_fn(input, target)

        if self.sample_wise:
            red_dims = tuple(range(1, element_error.ndim))
            batch_error = element_error.sum(red_dims) / mask.sum(red_dims)
            err = batch_error.mean()

        else:
            err = element_error.sum() / mask.sum()

        return err

    def __repr__(self) -> str:
        criterion = self.criterion
        sample_wise = self.sample_wise
        sqrt_transform = self.sqrt_transform
        s = f'RegressionLoss({criterion=}, {sample_wise=}, {sqrt_transform=})'
        return s


class EnergyLoss(nn.Module):
    """Energy loss that ignore NaN in target.
    https://arxiv.org/pdf/2307.00835
    The loss funtion allows for having missing / non-finite values in the target.

    Shape
    -----
    * input: (N, *), where * means, any number of additional dimensions
    * target: (N, *), same shape as the input
    """

    def __init__(
            self,
            beta: float = 1.0,
            es_length: int = 1) -> None:
        """Initialize RegressionLoss.

        Args:
            beta : float
                Power parameter in the energy score, a float (0, 2). Default is 1.0.
            es_length: int
                Length of sequence fed jointly into the loss function, an integer > 0. Default is 1.

        """

        super(EnergyLoss, self).__init__()

        self.beta = beta
        self.es_length = es_length
        self.criterion = 'es'

    def forward(
            self,
            input0: Tensor,
            input1: Tensor,
            target: Tensor) -> Tensor:
        """Forward call, calculate loss from input0, input1,  and target, must have same shape.

        Args:
            input0 (torch.Tensor): (batch, sequence) iid samples from the estimated distribution.
            input1 (torch.Tensor): (batch, sequence) iid samples from the estimated distribution.
            target (torch.Tensor): (batch, sequence) iid samples from the true distribution.

        Returns:
            The loss, a scalar.
        """

        EPS = 0 if float(self.beta).is_integer() else 1e-5

        # (batch, sequence) -> (batch, L = sequence // es_length, es_length)
        target = self.reshape_to_es_length(target, self.es_length)
        input0 = self.reshape_to_es_length(input0, self.es_length)
        input1 = self.reshape_to_es_length(input1, self.es_length)

        # (batch, L, es_length)
        mask = target.isfinite()
        mask_num_finite = mask.sum((-2, -1))

        # (batch, L, es_length)
        diff0 = input0 - target.where(mask, input0)
        diff1 = input1 - target.where(mask, input1)

        # (batch, L, es_length)
        input_diff = (input0 - input1).where(mask, 0.0)

        # (batch, L)
        s1 = (
            0.5 * self.mean_masked((vector_norm(diff0, 2, dim=-1) + EPS).pow(self.beta), mask_num_finite, dim=-1) +
            0.5 * self.mean_masked((vector_norm(diff1, 2, dim=-1) + EPS).pow(self.beta), mask_num_finite, dim=-1)
        )

        # (batch, L)
        s2 = self.mean_masked((vector_norm(input_diff, 2, dim=-1) + EPS).pow(self.beta), mask_num_finite, dim=-1)

        # (batch, L)
        element_loss = s1 - s2 / 2

        # (1,)
        loss = element_loss.mean()

        return loss

    def reshape_to_es_length(self, x: Tensor, es_length: int) -> Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(-1)

        if es_length == 1:
            return x.unsqueeze(-1)

        if x.ndim == 3:
            if x.shape[1] != 1:
                raise ValueError(
                    f'for a 3D tensor, the second dimension must have size 1 (is {x.shape[1]}) because '
                    'more than one target variable not supported.'
                )

            x = x.squeeze(1)

        batch_size, seq_length = x.shape
        n_drop = seq_length % es_length

        x = x[:, n_drop:].view(batch_size, -1, es_length)

        return x

    def mean_masked(self, x: Tensor, mask_num_finite: Tensor, dim: int = -1) -> Tensor:
        return x.sum(dim) / mask_num_finite

    def __repr__(self) -> str:
        beta = self.beta
        es_length = self.es_length
        s = f'EnergyLoss({beta=}, {es_length=})'
        return s
