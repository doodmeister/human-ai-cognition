import torch
from torch.utils.data import DataLoader, TensorDataset
from model.dpad_rnn import DPADRNN
from tqdm import tqdm


def train_dpad(training_data, input_size=64, output_size=1, epochs=20, batch_size=16, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"[DPAD Trainer] Using device: {device}")
    dpad = DPADRNN(input_size=input_size, output_size=output_size).to(device)

    optimizer = torch.optim.Adam(dpad.parameters(), lr=1e-3)
    criterion_recon = torch.nn.MSELoss()
    criterion_behavior = torch.nn.BCEWithLogitsLoss()

    # Expect training_data to be a list of (seq_tensor, flags_tensor)
    seqs = torch.stack([torch.tensor(pair[0]) for pair in training_data]).float()
    flags = torch.stack([torch.tensor(pair[1]) for pair in training_data]).float()

    dataset = TensorDataset(seqs, flags)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        dpad.train()
        total_loss, recon_loss_total, behavior_loss_total = 0, 0, 0

        for seq_batch, flags_batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            seq_batch = seq_batch.to(device)       # (B, T, D)
            flags_batch = flags_batch.to(device)   # (B, T) or (B,)

            optimizer.zero_grad()
            outputs = dpad(seq_batch)

            behavior_pred = outputs["behavior"]
            recon_pred = outputs["reconstruction"]

            loss_behavior = criterion_behavior(behavior_pred, flags_batch)
            loss_recon = criterion_recon(recon_pred, seq_batch)

            loss = loss_behavior + loss_recon
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            recon_loss_total += loss_recon.item()
            behavior_loss_total += loss_behavior.item()

        print(f"[Epoch {epoch+1}] Total: {total_loss:.4f} | Behavior: {behavior_loss_total:.4f} | Recon: {recon_loss_total:.4f}")

    torch.save(dpad.state_dict(), "model/dpad_trained.pt")
    print("[✓] Model saved to model/dpad_trained.pt")
