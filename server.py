import flwr as fl
from flwr.server.strategy import FedAvg
import numpy as np

def aggregate_accuracy(metrics):
    accuracies = []
    for _, m in metrics:
        if "accuracy" in m:
            accuracies.append(m["accuracy"])
    return {"accuracy": float(np.mean(accuracies)) if accuracies else 0.0}

strategy = FedAvg(
    fraction_fit=0.5,
    min_available_clients=5,
    evaluate_metrics_aggregation_fn=aggregate_accuracy  
)

if __name__ == "__main__":
    config = fl.server.ServerConfig(num_rounds=5)
    fl.server.start_server(
        server_address="localhost:8081",
        config=config,
        strategy=strategy
    )
