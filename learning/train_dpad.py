import torch
from torch.utils.data import DataLoader
from model.dpad_rnn import DPADRNN
from tqdm import tqdm

def train_dpad(training_data, epochs=20, batch_size=16, device='cuda' if torch.cuda.is_available() else 'cpu'):
    dpad = DPADRNN().to(device)
    optimizer = torch.optim.Adam(dpad.parameters(), lr=1e-3)
    criterion_recon = torch.nn.MSELoss()
    criterion_behavior = torch.nn.BCEWithLogitsLoss()

    loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        recon_loss_total = 0
        behavior_loss_total = 0

        dpad.train()
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            seq_batch, flags_batch = batch
            seq_tensor = seq_batch.to(device).float()
            flags_tensor = flags_batch.to(device).float()

            optimizer.zero_grad()
            behavior_pred, recon_seq = dpad(seq_tensor)

            loss_recon = criterion_recon(recon_seq, seq_tensor)
            loss_behavior = criterion_behavior(behavior_pred, flags_tensor)
            loss = loss_recon + loss_behavior

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            recon_loss_total += loss_recon.item()
            behavior_loss_total += loss_behavior.item()

        print(f"Epoch {epoch+1} | Total Loss: {total_loss:.4f} | Recon Loss: {recon_loss_total:.4f} | Behavior Loss: {behavior_loss_total:.4f}")

    torch.save(dpad.state_dict(), "model/dpad_trained.pt")
    print("Model saved to model/dpad_trained.pt")