import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from dpad_transformer import DPADRNN, DPADTrainer

def create_synthetic_dataset(seq_len=30, num_samples=500, input_size=64, output_size=1):
    # Create synthetic data for initial testing
    X = torch.randn(num_samples, seq_len, input_size)
    Y_behavior = torch.sum(X, dim=2)  # Simple behavior proxy
    return TensorDataset(X, Y_behavior, X)  # (input, behavior label, input reconstruction)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    input_size = 64
    hidden_size = 128
    output_size = 1
    batch_size = 32
    lr = 1e-3
    behavior_epochs = 10
    residual_epochs = 10

    # Initialize model
    model = DPADRNN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        nonlinear_input=True,
        nonlinear_recurrence=True,
        nonlinear_behavior_readout=True,
        nonlinear_reconstruction=True,
        use_layernorm=True,
        dropout=0.1,
    )

    # Load Data
    dataset = create_synthetic_dataset(seq_len=30, num_samples=1000, input_size=input_size, output_size=output_size)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    trainer = DPADTrainer(model, device=device)

    # Optimizer and Loss Functions
    optimizer_behavior = optim.Adam(
        list(model.input_map.parameters()) +
        list(model.rnn_behavior.parameters()) +
        list(model.behavior_readout.parameters()), lr=lr)

    optimizer_residual = optim.Adam(
        list(model.rnn_residual.parameters()) +
        list(model.reconstruction_head.parameters()), lr=lr)

    behavior_criterion = nn.MSELoss()
    residual_criterion = nn.MSELoss()

    # Behavior Phase
    trainer.train_behavior_phase(data_loader, optimizer_behavior, behavior_criterion, epochs=behavior_epochs)

    # Residual Phase
    trainer.train_residual_phase(data_loader, optimizer_residual, residual_criterion, epochs=residual_epochs)

    # Save the model
    model.save("dpad_model.pth")
    print("✅ Model training completed and saved as dpad_model.pth")

if __name__ == "__main__":
    main()
