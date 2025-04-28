import torch
import torch.nn as nn
import torch.optim as optim


def make_mlp(input_dim, output_dim, hidden_dims=None, activation=nn.ReLU(), dropout=0.1):
    """
    Utility function to create a Multi-Layer Perceptron (MLP).
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
        residual_rnn_type="GRU",  # New parameter
    ):
        super(DPADRNN, self).__init__()

        self.input_map = (
            make_mlp(input_size, hidden_size, [hidden_size], dropout=dropout)
            if nonlinear_input else nn.Linear(input_size, hidden_size)
        )
        self.norm = nn.LayerNorm(hidden_size) if use_layernorm else nn.Identity()

        self.rnn_behavior = (
            nn.GRU(hidden_size, hidden_size, batch_first=True)
            if nonlinear_recurrence else nn.RNN(hidden_size, hidden_size, batch_first=True)
        )
        self.behavior_readout = (
            make_mlp(hidden_size, output_size, dropout=dropout)
            if nonlinear_behavior_readout else nn.Linear(hidden_size, output_size)
        )

        # Configurable residual dynamics
        if residual_rnn_type == "GRU":
            self.rnn_residual = nn.GRU(hidden_size, hidden_size, batch_first=True)
        elif residual_rnn_type == "RNN":
            self.rnn_residual = nn.RNN(hidden_size, hidden_size, batch_first=True)
        elif residual_rnn_type == "LSTM":
            self.rnn_residual = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        else:
            raise ValueError("Unsupported residual_rnn_type: choose from 'GRU', 'RNN', 'LSTM'")

        self.reconstruction_head = (
            make_mlp(hidden_size, input_size, dropout=dropout)
            if nonlinear_reconstruction else nn.Linear(hidden_size, input_size)
        )

    def forward(self, x, debug=False):
        """
        Forward pass through the DPADRNN model.
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)

        x_mapped = self.norm(self.input_map(x))
        if debug: print("[DPAD] Mapped input shape:", x_mapped.shape)

        h_behavior, _ = self.rnn_behavior(x_mapped)
        behavior_pred = self.behavior_readout(h_behavior).squeeze(-1)
        if debug: print("[DPAD] Behavior prediction shape:", behavior_pred.shape)

        h_residual, _ = self.rnn_residual(x_mapped)
        recon_pred = self.reconstruction_head(h_residual)
        if debug: print("[DPAD] Reconstruction output shape:", recon_pred.shape)

        return {
            "behavior": behavior_pred,
            "reconstruction": recon_pred,
            "latents": h_behavior,
        }

    def save(self, path, optimizer=None):
        """
        Save model and optimizer states to the specified path.
        """
        checkpoint = {"model_state": self.state_dict()}
        if optimizer:
            checkpoint["optimizer_state"] = optimizer.state_dict()
        torch.save(checkpoint, path)

    def load(self, path, optimizer=None, device='cpu'):
        """
        Load model and optimizer states from the specified path.
        """
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint["model_state"])
        if optimizer and "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.eval()


class DPADTrainer:
    """
    Trainer class for phased training of DPADRNN.
    """

    def __init__(self, model, device="cpu"):
        self.model = model.to(device)
        self.device = device

    def train_behavior_phase(self, data_loader, optimizer, criterion, epochs=10):
        """
        Train only the behavior prediction phase.
        """
        for param in self.model.rnn_residual.parameters():
            param.requires_grad = False
        for param in self.model.reconstruction_head.parameters():
            param.requires_grad = False

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in data_loader:
                x, y_behavior = batch
                x, y_behavior = x.to(self.device), y_behavior.to(self.device)

                output = self.model(x)
                loss = criterion(output['behavior'], y_behavior)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            print(f"[Behavior Phase] Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(data_loader):.4f}")

    def train_residual_phase(self, data_loader, optimizer, criterion, epochs=10):
        """
        Train only the residual reconstruction phase.
        """
        for param in self.model.rnn_behavior.parameters():
            param.requires_grad = False
        for param in self.model.behavior_readout.parameters():
            param.requires_grad = False

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in data_loader:
                x, y_input = batch
                x, y_input = x.to(self.device), y_input.to(self.device)

                output = self.model(x)
                loss = criterion(output['reconstruction'], y_input)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            print(f"[Residual Phase] Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(data_loader):.4f}")