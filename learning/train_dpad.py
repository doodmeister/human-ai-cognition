import torch
from model.dpad_rnn import DPADRNN

def train_dpad(training_data, epochs=20):
    dpad = DPADRNN()
    optimizer = torch.optim.Adam(dpad.parameters(), lr=1e-3)
    criterion_recon = torch.nn.MSELoss()
    criterion_behavior = torch.nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        total_loss = 0
        for seq, flags in training_data:
            seq_tensor = torch.tensor(seq)
            flags_tensor = torch.tensor(flags).float()
            optimizer.zero_grad()
            behavior_pred, recon_seq = dpad(seq_tensor)
            loss_recon = criterion_recon(recon_seq, seq_tensor)
            loss_behavior = criterion_behavior(behavior_pred, flags_tensor)
            loss = loss_recon + loss_behavior
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}, Loss: {total_loss}")
    torch.save(dpad.state_dict(), "model/dpad_trained.pt")