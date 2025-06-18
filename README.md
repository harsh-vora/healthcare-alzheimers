# Federated Learning for Collaborative Healthcare System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Federated Learning](https://img.shields.io/badge/Federated-Learning-purple.svg)](https://flower.dev/)

## Overview

A **privacy-preserving machine learning system** that enables multiple healthcare institutions to collaboratively train models for Alzheimer's disease prediction without sharing sensitive patient data, ensuring HIPAA and GDPR compliance.

![Federated Learning Architecture](https://img.shields.io/badge/Architecture-Federated_Learning-blue)

## Key Features

- **Privacy-Preserving**: Patient data never leaves local institutions
- **Collaborative Learning**: Multiple clients contribute to a global model
- **Healthcare-Focused**: Designed for Alzheimer's disease prediction
- **Performance Monitoring**: Real-time tracking of accuracy, loss, and resource usage
- **Visualization**: Automatic generation of performance plots

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/healthcare-alzheimers.git
cd healthcare-alzheimers
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Quick Start

### Option 1: Automated Simulation (Recommended)

Run the complete federated learning simulation with one command:

```bash
python run_simulation.py
```

This will:
- Generate synthetic Alzheimer's disease data for 5 clients
- Start the federated learning server
- Launch 5 client instances
- Run 5 rounds of federated training
- Save metrics and model updates

### Option 2: Manual Execution

1. **Generate Data** (if not already present):
```bash
python generate_data.py
```

2. **Start the Server** (in one terminal):
```bash
python server.py
```

3. **Start Clients** (in separate terminals):
```bash
# Terminal 2
python client.py --client-id client1

# Terminal 3
python client.py --client-id client2

# Terminal 4
python client.py --client-id client3

# Terminal 5
python client.py --client-id client4

# Terminal 6
python client.py --client-id client5
```

## Visualizing Results

After the simulation completes, visualize the results:

```bash
python visualize_results.py
```

This generates:
- `federated_learning_results.png`: Accuracy and loss trends per round
- `training_losses.png`: Detailed training loss progression
- Console output with final metrics summary

## Project Structure

```
healthcare-alzheimers/
│
├── server.py              # Federated learning server
├── client.py              # Federated learning client
├── generate_data.py       # Synthetic data generator
├── run_simulation.py      # Automated simulation runner
├── visualize_results.py   # Results visualization
├── requirements.txt       # Python dependencies
├── README.md             # This file
│
├── client1_data.csv      # Generated client datasets
├── client2_data.csv
├── client3_data.csv
├── client4_data.csv
├── client5_data.csv
│
├── client1_metrics.csv   # Training metrics (generated)
├── client2_metrics.csv
├── client3_metrics.csv
├── client4_metrics.csv
├── client5_metrics.csv
│
└── *.png                 # Visualization outputs (generated)
```

## Configuration

### Server Configuration
Edit `server.py` to modify:
- Number of training rounds (default: 5)
- Client selection fraction (default: 0.5)
- Minimum clients required (default: 5)

### Client Configuration
Edit `client.py` to modify:
- Learning rate (default: 0.001)
- Batch size (default: 32)
- Local epochs (default: 10)
- Model architecture

## Model Architecture

The system uses a neural network with:
- Input layer (33 features from Alzheimer's dataset)
- Hidden layers: 128 → 64 → 32 neurons
- Batch normalization and dropout (0.3) for regularization
- Sigmoid output for binary classification

## Performance Metrics

The system tracks:
- **Accuracy**: Classification accuracy on test data
- **Loss**: Binary cross-entropy loss
- **Training Time**: Time per federated round
- **Memory Usage**: RAM consumption during training

## Troubleshooting

### Port Already in Use
If you see "Address already in use" error:
```bash
# Find process using port 8081
lsof -i :8081  # macOS/Linux
netstat -ano | findstr :8081  # Windows

# Kill the process or use a different port in server.py
```

### Missing Data Files
If clients can't find data files:
```bash
python generate_data.py
```

### Memory Issues
For systems with limited RAM:
- Reduce batch size in `client.py`
- Decrease model size (hidden layer dimensions)
- Run fewer clients simultaneously

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Flower (flwr) framework for federated learning infrastructure
- PyTorch for deep learning capabilities
- Healthcare institutions participating in collaborative research

## Contact

For questions or support, please open an issue in the GitHub repository.
