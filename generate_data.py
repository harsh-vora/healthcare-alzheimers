import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def generate_alzheimer_data(n_samples=2150):
    np.random.seed(42)
    
    data = {
        'PatientID': range(1, n_samples + 1),
        'Age': np.random.randint(50, 90, n_samples),
        'Gender': np.random.randint(0, 2, n_samples),
        'Ethnicity': np.random.randint(0, 5, n_samples),
        'EducationLevel': np.random.randint(0, 5, n_samples),
        'BMI': np.random.uniform(18.5, 35.0, n_samples),
        'Smoking': np.random.randint(0, 2, n_samples),
        'AlcoholConsumption': np.random.uniform(0, 10, n_samples),
        'PhysicalActivity': np.random.uniform(0, 10, n_samples),
        'DietQuality': np.random.uniform(0, 10, n_samples),
        'SleepQuality': np.random.uniform(0, 10, n_samples),
        'FamilyHistoryAlzheimers': np.random.randint(0, 2, n_samples),
        'CardiovascularDisease': np.random.randint(0, 2, n_samples),
        'Diabetes': np.random.randint(0, 2, n_samples),
        'Depression': np.random.randint(0, 2, n_samples),
        'HeadInjury': np.random.randint(0, 2, n_samples),
        'Hypertension': np.random.randint(0, 2, n_samples),
        'SystolicBP': np.random.randint(90, 180, n_samples),
        'DiastolicBP': np.random.randint(60, 110, n_samples),
        'CholesterolTotal': np.random.uniform(150, 300, n_samples),
        'CholesterolLDL': np.random.uniform(50, 200, n_samples),
        'CholesterolHDL': np.random.uniform(30, 80, n_samples),
        'CholesterolTriglycerides': np.random.uniform(50, 400, n_samples),
        'MMSE': np.random.uniform(0, 30, n_samples),
        'FunctionalAssessment': np.random.uniform(0, 10, n_samples),
        'MemoryComplaints': np.random.randint(0, 2, n_samples),
        'BehavioralProblems': np.random.randint(0, 2, n_samples),
        'ADL': np.random.uniform(0, 10, n_samples),
        'Confusion': np.random.randint(0, 2, n_samples),
        'Disorientation': np.random.randint(0, 2, n_samples),
        'PersonalityChanges': np.random.randint(0, 2, n_samples),
        'DifficultyCompletingTasks': np.random.randint(0, 2, n_samples),
        'Forgetfulness': np.random.randint(0, 2, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    risk_score = (
        (df['Age'] - 50) / 40 * 0.3 +
        df['FamilyHistoryAlzheimers'] * 0.25 +
        df['CardiovascularDisease'] * 0.1 +
        df['Diabetes'] * 0.1 +
        df['Depression'] * 0.05 +
        df['HeadInjury'] * 0.05 +
        df['Hypertension'] * 0.05 +
        (30 - df['MMSE']) / 30 * 0.3 +
        df['MemoryComplaints'] * 0.1 +
        df['Confusion'] * 0.1 +
        df['Disorientation'] * 0.1 +
        df['PersonalityChanges'] * 0.05 +
        df['DifficultyCompletingTasks'] * 0.05 +
        df['Forgetfulness'] * 0.05 +
        np.random.normal(0, 0.1, n_samples)
    )
    
    df['Diagnosis'] = (risk_score > 0.5).astype(int)
    
    doctors = ['Dr. Smith', 'Dr. Johnson', 'Dr. Williams', 'Dr. Brown', 'Dr. Davis']
    df['DoctorInCharge'] = np.random.choice(doctors, n_samples)
    
    return df

def split_data_for_clients(df, n_clients=5):
    client_data = []
    samples_per_client = len(df) // n_clients
    
    for i in range(n_clients):
        if i == n_clients - 1:
            client_df = df.iloc[i * samples_per_client:]
        else:
            client_df = df.iloc[i * samples_per_client:(i + 1) * samples_per_client]
        
        client_df = client_df.reset_index(drop=True)
        client_df['PatientID'] = range(1, len(client_df) + 1)
        
        client_data.append(client_df)
    
    return client_data

if __name__ == "__main__":
    print("Generating Alzheimer's Disease dataset...")
    df = generate_alzheimer_data()
    
    print(f"Total samples generated: {len(df)}")
    print(f"Positive cases (Alzheimer's): {df['Diagnosis'].sum()}")
    print(f"Negative cases (Healthy): {len(df) - df['Diagnosis'].sum()}")
    
    print("\nSplitting data for 5 clients...")
    client_datasets = split_data_for_clients(df, n_clients=5)
    
    for i, client_df in enumerate(client_datasets):
        filename = f"client{i+1}_data.csv"
        client_df.to_csv(filename, index=False)
        print(f"Created {filename} with {len(client_df)} samples")
    
    print("\nData generation complete!")
