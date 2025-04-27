import torch
from torch.utils.data import DataLoader, TensorDataset
from model.dpad_rnn import DPADRNN
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def train_dpad(training_data, input_size=64, output_size=1, epochs=20, batch_size=16, device=None, save_path="model/dpad_trained.pt"):
    """
    Train the DPAD RNN model.

    Args:
        training_data (list): List of (seq_tensor, flags_tensor) tuples.
        input_size (int): Dimensionality of input sequences.
        output_size (int): Dimensionality of output predictions.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        device (torch.device): Device to use for training (CPU/GPU).
        save_path (str): Path to save the trained model.

    Returns:
        None
    """
    if not isinstance(training_data, list) or not all(isinstance(pair, tuple) and len(pair) == 2 for pair in training_data):
        raise ValueError("training_data must be a list of (seq_tensor, flags_tensor) tuples.")

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"[DPAD Trainer] Using device: {device}")
    dpad = DPADRNN(input_size=input_size, output_size=output_size).to(device)
    if torch.cuda.device_count() > 1:
        print(f"[DPAD Trainer] Using {torch.cuda.device_count()} GPUs")
        dpad = torch.nn.DataParallel(dpad)
    optimizer = torch.optim.Adam(dpad.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion_recon = torch.nn.MSELoss()
    criterion_behavior = torch.nn.BCEWithLogitsLoss()

    # Expect training_data to be a list of (seq_tensor, flags_tensor)
    seqs = torch.stack([torch.tensor(pair[0]) for pair in training_data]).float()
    flags = torch.stack([torch.tensor(pair[1]) for pair in training_data]).float()

    dataset = TensorDataset(seqs, flags)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    writer = SummaryWriter(log_dir="logs/dpad_training")

    best_loss = float("inf")
    patience = 5
    no_improve_epochs = 0

    for epoch in range(epochs):
        dpad.train()
        total_loss, recon_loss_total, behavior_loss_total = 0, 0, 0

        for seq_batch, flags_batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            seq_batch, flags_batch = seq_batch.to(device), flags_batch.to(device)
            optimizer.zero_grad()
            outputs = dpad(seq_batch)

            loss_behavior = criterion_behavior(outputs["behavior"], flags_batch)
            loss_recon = criterion_recon(outputs["reconstruction"], seq_batch)
            loss = loss_behavior + loss_recon

            loss.backward()
            torch.nn.utils.clip_grad_norm_(dpad.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            recon_loss_total += loss_recon.item()
            behavior_loss_total += loss_behavior.item()

            tqdm.write(f"Batch Loss: {loss.item():.4f}")

        scheduler.step()
        print(f"[Epoch {epoch+1}] Learning rate: {scheduler.get_last_lr()[0]:.6f}")
        writer.add_scalar("Loss/Total", total_loss, epoch)
        writer.add_scalar("Loss/Behavior", behavior_loss_total, epoch)
        writer.add_scalar("Loss/Reconstruction", recon_loss_total, epoch)

        print(f"[Epoch {epoch+1}] Total: {total_loss:.4f} | Behavior: {behavior_loss_total:.4f} | Recon: {recon_loss_total:.4f}")

        if total_loss < best_loss:
            best_loss = total_loss
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print("[✓] Early stopping triggered.")
                break

        if epoch % 5 == 0:  # Save every 5 epochs
            checkpoint_path = f"model/dpad_checkpoint_epoch_{epoch}.pt"
            torch.save(dpad.state_dict(), checkpoint_path)
            print(f"[✓] Checkpoint saved to {checkpoint_path}")

    torch.save(dpad.state_dict(), save_path)
    print(f"[✓] Model saved to {save_path}")
    writer.close()


def validate(model, val_loader, criterion_behavior, criterion_recon, device):
    model.eval()
    val_loss, val_behavior_loss, val_recon_loss = 0, 0, 0
    with torch.no_grad():
        for seq_batch, flags_batch in val_loader:
            seq_batch, flags_batch = seq_batch.to(device), flags_batch.to(device)
            outputs = model(seq_batch)
            val_behavior_loss += criterion_behavior(outputs["behavior"], flags_batch).item()
            val_recon_loss += criterion_recon(outputs["reconstruction"], seq_batch).item()
    val_loss = val_behavior_loss + val_recon_loss
    return val_loss, val_behavior_loss, val_recon_loss
