import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def load_client_metrics():
    metrics = {}
    for i in range(1, 6):
        filename = f"client{i}_metrics.csv"
        if os.path.exists(filename):
            data = pd.read_csv(filename, header=None, names=['accuracy', 'loss'])
            metrics[f"Client {i}"] = data
    return metrics

def plot_metrics():
    metrics = load_client_metrics()
    
    if not metrics:
        print("No metrics files found. Please run the simulation first.")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for client, data in metrics.items():
        rounds = range(1, len(data) + 1)
        ax1.plot(rounds, data['accuracy'], marker='o', label=client)
        ax2.plot(rounds, data['loss'], marker='s', label=client)
    
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy per Round for Each Client')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss per Round for Each Client')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('federated_learning_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nFinal Round Metrics:")
    print("-" * 50)
    for client, data in metrics.items():
        final_accuracy = data['accuracy'].iloc[-1]
        final_loss = data['loss'].iloc[-1]
        print(f"{client}: Accuracy = {final_accuracy:.4f}, Loss = {final_loss:.4f}")
    
    all_accuracies = [data['accuracy'].iloc[-1] for data in metrics.values()]
    print(f"\nAverage Final Accuracy: {np.mean(all_accuracies):.4f}")
    print(f"Standard Deviation: {np.std(all_accuracies):.4f}")

def plot_loss_distribution():
    plt.figure(figsize=(10, 6))
    
    for i in range(1, 6):
        filename = f"client{i}_losses.npy"
        if os.path.exists(filename):
            losses = np.load(filename)
            plt.plot(losses, alpha=0.7, label=f'Client {i}')
    
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time for Each Client')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('training_losses.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Visualizing Federated Learning Results...")
    plot_metrics()
    plot_loss_distribution()
