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

        # Section 1: Behaviorally-relevant transformation
        self.input_map = make_mlp(input_size, hidden_size, [hidden_size], dropout=dropout) if nonlinear_input else nn.Linear(input_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size) if use_layernorm else nn.Identity()

        self.rnn_behavior = nn.GRU(hidden_size, hidden_size, batch_first=True) if nonlinear_recurrence else nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.behavior_readout = make_mlp(hidden_size, output_size, dropout=dropout) if nonlinear_behavior_readout else nn.Linear(hidden_size, output_size)

        # Section 2: Residual dynamics
        self.rnn_residual = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.reconstruction_head = make_mlp(hidden_size, input_size, dropout=dropout) if nonlinear_reconstruction else nn.Linear(hidden_size, input_size)

    def forward(self, x, debug=False):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, input_size)
        Returns:
            dict with:
                - 'behavior': shape (batch_size, seq_len)
                - 'reconstruction': shape (batch_size, seq_len, input_size)
                - 'latents': intermediate hidden states from RNN 1
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (1, seq_len, input_size)

        x_mapped = self.norm(self.input_map(x))
        if debug:
            print("[DPAD] Mapped input shape:", x_mapped.shape)

        # Section 1: Behaviorally relevant dynamics
        h_behavior, _ = self.rnn_behavior(x_mapped)
        behavior_pred = self.behavior_readout(h_behavior).squeeze(-1)
        if debug:
            print("[DPAD] Behavior output shape:", behavior_pred.shape)

        # Section 2: Residual dynamics for input reconstruction
        h_residual, _ = self.rnn_residual(x_mapped)
        recon_pred = self.reconstruction_head(h_residual)
        if debug:
            print("[DPAD] Recon output shape:", recon_pred.shape)

        return {
            "behavior": behavior_pred,
            "reconstruction": recon_pred,
            "latents": h_behavior
        }
