import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import psutil
import os
import argparse

class AlzheimerModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.layer2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.layer3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.output = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.bn1(self.layer1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.layer2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.layer3(x)))
        x = self.dropout(x)
        return torch.sigmoid(self.output(x))

def load_data(client_id):
    data = pd.read_csv(f"{client_id}_data.csv")
    target_column = "Diagnosis"
    X = data.drop(columns=[target_column]).select_dtypes(include=[np.number])
    y = data[target_column].astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=np.random.randint(1000)
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train.values, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test.values, dtype=torch.float32),
    )

class AlzheimerClient(fl.client.NumPyClient):
    def __init__(self, model, train_data, test_data, client_id):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.client_id = client_id

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = dict(zip(self.model.state_dict().keys(), [torch.tensor(p) for p in parameters]))
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config=None):
        self.set_parameters(parameters)
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        train_loader = DataLoader(TensorDataset(*self.train_data), batch_size=32, shuffle=True)

        start_time = time.time()
        process = psutil.Process()
        mem_before = process.memory_info().rss
        losses = []

        for epoch in range(10):
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = self.model(X_batch).squeeze()
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

        mem_after = process.memory_info().rss
        duration = time.time() - start_time
        print(f"[{self.client_id}] Training Time: {duration:.2f}s, Memory Usage: {(mem_after - mem_before)/1e6:.2f} MB")

        np.save(f"{self.client_id}_losses.npy", np.array(losses))
        return self.get_parameters(), len(self.train_data[0]), {}

    def evaluate(self, parameters, config=None):
        self.set_parameters(parameters)
        self.model.eval()
        test_loader = DataLoader(TensorDataset(*self.test_data), batch_size=32)
        criterion = nn.BCELoss()

        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                output = self.model(X_batch).squeeze()
                loss = criterion(output, y_batch)
                total_loss += loss.item() * len(y_batch)
                pred = (output > 0.5).float()
                correct += (pred == y_batch).sum().item()
                total += len(y_batch)

        accuracy = correct / total
        avg_loss = total_loss / total

        print(f"[{self.client_id}] Accuracy: {accuracy:.4f}, Loss: {avg_loss:.4f}")

        with open(f"{self.client_id}_metrics.csv", "a") as f:
            f.write(f"{accuracy},{avg_loss}\n")

        return avg_loss, total, {"accuracy": accuracy}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Federated Learning Client')
    parser.add_argument('--client-id', type=str, required=True, help='Client ID (e.g., client1, client2, etc.)')
    args = parser.parse_args()
    
    client_id = args.client_id
    X_train, y_train, X_test, y_test = load_data(client_id)
    model = AlzheimerModel(input_size=X_train.shape[1])
    client = AlzheimerClient(model, (X_train, y_train), (X_test, y_test), client_id)
    
    fl.client.start_client(server_address="localhost:8081", client=client.to_client())
