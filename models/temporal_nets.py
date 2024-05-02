
import torch
from torch import Tensor

from models.base import TemporalNet
from utils.torch_modules import Transform, SeqZeroPad, TemporalConvNet, Transform

class TCN(TemporalNet):
    """TCN based rainfall-runoff module."""
    def __init__(
            self,
            model_dim: int = 32,
            tcn_kernel_size: int = 4,
            tcn_dropout:float = 0.0,
            tcn_layers: int = 1,
            **kwargs) -> None:

        super().__init__(model_dim=model_dim, **kwargs)

        self.save_hyperparameters()

        self.model_type = 'TCN'

        self.tcn = TemporalConvNet(
            num_inputs=model_dim,
            num_outputs=-1,
            num_hidden=model_dim,
            kernel_size=tcn_kernel_size,
            num_layers=tcn_layers,
            dropout=tcn_dropout
        )

    def temporal_forward(self, x: Tensor) -> Tensor:
        out = self.tcn(x)
        return out


class LSTM(TemporalNet):
    """LSTM based rainfall-runoff module."""
    def __init__(
            self,
            model_dim: int = 32,
            lstm_layers: int = 1,
            **kwargs) -> None:

        super().__init__(model_dim=model_dim, **kwargs)

        self.model_type = 'LSTM'

        self.save_hyperparameters()

        self.to_channel_last = Transform(transform_fun=lambda x: x.transpose(1, 2), name='Transpose(1, 2)')

        self.lstm = torch.nn.LSTM(
            input_size=model_dim,
            hidden_size=model_dim,
            num_layers=lstm_layers,
            batch_first=True,
        )

        self.to_sequence_last = Transform(transform_fun=lambda x: x.transpose(1, 2), name='Transpose(1, 2)')

    def temporal_forward(self, x: Tensor) -> Tensor:
        out = self.to_channel_last(x)
        out, _ = self.lstm(out)
        out = self.to_sequence_last(out)
        return out
