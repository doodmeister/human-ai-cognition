import torch
import torch.nn as nn


def make_mlp(input_dim, output_dim, hidden_dims=[64, 64], activation=nn.ReLU()):
    layers = []
    dims = [input_dim] + hidden_dims
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        layers.append(activation)
    layers.append(nn.Linear(dims[-1], output_dim))
    return nn.Sequential(*layers)


class DPADRNN(nn.Module):
    def __init__(
        self,
        input_size=64,
        hidden_size=128,
        output_size=1,
        nonlinear_input=True,
        nonlinear_recurrence=True,
        nonlinear_behavior_readout=True,
        nonlinear_reconstruction=True
    ):
        super(DPADRNN, self).__init__()

        # Neural input transformation (optional nonlinearity)
        if nonlinear_input:
            self.input_map = make_mlp(input_size, hidden_size)
        else:
            self.input_map = nn.Linear(input_size, hidden_size)

        # RNN for behaviorally relevant states
        if nonlinear_recurrence:
            self.rnn_behavior = nn.GRU(hidden_size, hidden_size, batch_first=True)
        else:
            self.rnn_behavior = nn.RNN(hidden_size, hidden_size, batch_first=True)

        # Behavior readout
        if nonlinear_behavior_readout:
            self.behavior_readout = make_mlp(hidden_size, output_size)
        else:
            self.behavior_readout = nn.Linear(hidden_size, output_size)

        # RNN for residual dynamics (reconstruction)
        self.rnn_recon = nn.GRU(hidden_size, hidden_size, batch_first=True)

        # Reconstruction readout
        if nonlinear_reconstruction:
            self.reconstruction_head = make_mlp(hidden_size, input_size)
        else:
            self.reconstruction_head = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_size) or (seq_len, input_size)
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)

        # Input transformation
        x_mapped = self.input_map(x)

        # Behaviorally relevant RNN
        out_behavior, _ = self.rnn_behavior(x_mapped)
        behavior_pred = self.behavior_readout(out_behavior)

        # Residual dynamics RNN (same input, separate modeling)
        out_recon, _ = self.rnn_recon(x_mapped)
        recon_seq = self.reconstruction_head(out_recon)

        return behavior_pred.squeeze(-1), recon_seq
