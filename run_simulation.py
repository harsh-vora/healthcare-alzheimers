import subprocess
import time
import os
import signal
import sys

processes = []

def signal_handler(sig, frame):
    print('\nStopping all processes...')
    for p in processes:
        p.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def start_server():
    print("Starting FL Server...")
    server_process = subprocess.Popen(
        [sys.executable, "server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    processes.append(server_process)
    return server_process

def start_client(client_id):
    print(f"Starting {client_id}...")
    client_process = subprocess.Popen(
        [sys.executable, "client.py", "--client-id", client_id],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    processes.append(client_process)
    return client_process

def main():
    if not all(os.path.exists(f"client{i}_data.csv") for i in range(1, 6)):
        print("Data files not found. Generating data first...")
        subprocess.run([sys.executable, "generate_data.py"])
        print()
    
    server = start_server()
    time.sleep(3)
    
    clients = []
    for i in range(1, 6):
        client = start_client(f"client{i}")
        clients.append(client)
        time.sleep(1)
    
    print("\nFederated Learning simulation is running...")
    print("Press Ctrl+C to stop\n")
    
    try:
        server.wait()
    except KeyboardInterrupt:
        pass
    
    print("\nSimulation completed!")
    
    for p in processes:
        p.terminate()

if __name__ == "__main__":
    main()
