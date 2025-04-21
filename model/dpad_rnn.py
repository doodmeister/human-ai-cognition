import torch
import torch.nn as nn


def make_mlp(input_dim, output_dim, hidden_dims=[64, 64], activation=nn.ReLU(), dropout=0.1):
    layers = []
    dims = [input_dim] + hidden_dims
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        layers.append(activation)
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
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
        nonlinear_reconstruction=True,
        use_layernorm=True,
        dropout=0.1
    ):
        super(DPADRNN, self).__init__()

        # Input transformation
        self.input_map = make_mlp(input_size, hidden_size, [hidden_size], dropout=dropout) if nonlinear_input else nn.Linear(input_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size) if use_layernorm else nn.Identity()

        # RNN for behaviorally relevant latent dynamics
        self.rnn_behavior = nn.GRU(hidden_size, hidden_size, batch_first=True) if nonlinear_recurrence else nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.behavior_readout = make_mlp(hidden_size, output_size, dropout=dropout) if nonlinear_behavior_readout else nn.Linear(hidden_size, output_size)

        # RNN for residual modeling
        self.rnn_recon = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.reconstruction_head = make_mlp(hidden_size, input_size, dropout=dropout) if nonlinear_reconstruction else nn.Linear(hidden_size, input_size)

    def forward(self, x, debug=False):
        """
        Args:
            x: (batch_size, seq_len, input_size) or (seq_len, input_size)
        Returns:
            dict with 'behavior', 'reconstruction', 'latents'
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)

        # Input mapping + normalization
        x_mapped = self.norm(self.input_map(x))
        if debug:
            print("[DPAD] x_mapped:", x_mapped.shape)

        # Behavior-predictive RNN
        behavior_out, _ = self.rnn_behavior(x_mapped)
        behavior_pred = self.behavior_readout(behavior_out).squeeze(-1)
        if debug:
            print("[DPAD] behavior_pred:", behavior_pred.shape)

        # Residual dynamics RNN
        recon_out, _ = self.rnn_recon(x_mapped)
        recon_pred = self.reconstruction_head(recon_out)
        if debug:
            print("[DPAD] recon_pred:", recon_pred.shape)

        return {
            "behavior": behavior_pred,
            "reconstruction": recon_pred,
            "latents": behavior_out
        }
