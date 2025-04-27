import torch
import torch.nn as nn


def make_mlp(input_dim, output_dim, hidden_dims=None, activation=nn.ReLU(), dropout=0.1):
    """
    Utility function to create a Multi-Layer Perceptron (MLP).

    Args:
        input_dim (int): Input dimensionality.
        output_dim (int): Output dimensionality.
        hidden_dims (list, optional): List of hidden layer dimensions. Defaults to [64, 64].
        activation (nn.Module): Activation function to use. Defaults to nn.ReLU().
        dropout (float): Dropout rate. Defaults to 0.1.

    Returns:
        nn.Sequential: A sequential MLP model.
    """
    if hidden_dims is None:
        hidden_dims = [64, 64]

    layers = []
    dims = [input_dim] + hidden_dims
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(activation)
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(dims[-1], output_dim))
    return nn.Sequential(*layers)


class DPADRNN(nn.Module):
    """
    Dynamic Predictive Attention RNN (DPADRNN).

    This model combines behaviorally relevant dynamics and residual dynamics
    for tasks such as behavior prediction and input reconstruction.

    Attributes:
        input_map (nn.Module): Input transformation module.
        norm (nn.Module): Normalization layer.
        rnn_behavior (nn.Module): RNN for behaviorally relevant dynamics.
        behavior_readout (nn.Module): Readout layer for behavior prediction.
        rnn_residual (nn.Module): RNN for residual dynamics.
        reconstruction_head (nn.Module): Readout layer for input reconstruction.
    """

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
        dropout=0.1,
    ):
        """
        Initialize the DPADRNN model.

        Args:
            input_size (int): Dimensionality of the input features.
            hidden_size (int): Dimensionality of the hidden state.
            output_size (int): Dimensionality of the output predictions.
            nonlinear_input (bool): Whether to use a nonlinear input transformation.
            nonlinear_recurrence (bool): Whether to use a nonlinear RNN (GRU).
            nonlinear_behavior_readout (bool): Whether to use a nonlinear behavior readout.
            nonlinear_reconstruction (bool): Whether to use a nonlinear reconstruction head.
            use_layernorm (bool): Whether to apply layer normalization.
            dropout (float): Dropout rate for regularization.
        """
        super(DPADRNN, self).__init__()

        # Input transformation
        self.input_map = (
            make_mlp(input_size, hidden_size, [hidden_size], dropout=dropout)
            if nonlinear_input
            else nn.Linear(input_size, hidden_size)
        )
        self.norm = nn.LayerNorm(hidden_size) if use_layernorm else nn.Identity()

        # Behaviorally relevant dynamics
        self.rnn_behavior = (
            nn.GRU(hidden_size, hidden_size, batch_first=True)
            if nonlinear_recurrence
            else nn.RNN(hidden_size, hidden_size, batch_first=True)
        )
        self.behavior_readout = (
            make_mlp(hidden_size, output_size, dropout=dropout)
            if nonlinear_behavior_readout
            else nn.Linear(hidden_size, output_size)
        )

        # Residual dynamics
        self.rnn_residual = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.reconstruction_head = (
            make_mlp(hidden_size, input_size, dropout=dropout)
            if nonlinear_reconstruction
            else nn.Linear(hidden_size, input_size)
        )

    def forward(self, x, debug=False):
        """
        Forward pass of the DPADRNN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).
            debug (bool): If True, prints intermediate shapes for debugging.

        Returns:
            dict: A dictionary containing:
                - 'behavior': Behavior predictions of shape (batch_size, seq_len).
                - 'reconstruction': Reconstructed inputs of shape (batch_size, seq_len, input_size).
                - 'latents': Latent states from the behavior RNN.
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension if missing

        # Input transformation
        x_mapped = self.norm(self.input_map(x))
        if debug:
            print("[DPAD] Mapped input shape:", x_mapped.shape)

        # Behaviorally relevant dynamics
        h_behavior, _ = self.rnn_behavior(x_mapped)
        behavior_pred = self.behavior_readout(h_behavior).squeeze(-1)
        if debug:
            print("[DPAD] Behavior output shape:", behavior_pred.shape)

        # Residual dynamics for input reconstruction
        h_residual, _ = self.rnn_residual(x_mapped)
        recon_pred = self.reconstruction_head(h_residual)
        if debug:
            print("[DPAD] Reconstruction output shape:", recon_pred.shape)

        return {
            "behavior": behavior_pred,
            "reconstruction": recon_pred,
            "latents": h_behavior,
        }
