import pandas as pd
import numpy as np
import os

def generate_synthetic_data(num_machines=100, days=30):
    # Using 100 machines for faster generation but can be scaled to 500
    print(f"Generating data for {num_machines} machines over {days} days...")
    np.random.seed(42)
    
    dates = pd.date_range(start='2023-01-01', periods=days*24, freq='h')
    all_data = []
    
    for machine_id in range(num_machines):
        # Base normal operation parameters
        base_temp = np.random.uniform(60, 75)
        base_vib = np.random.uniform(1.0, 5.0)
        base_pres = np.random.uniform(100, 120)
        
        temp = np.random.normal(base_temp, 2.0, len(dates))
        vib = np.random.normal(base_vib, 0.5, len(dates))
        pres = np.random.normal(base_pres, 5.0, len(dates))
        
        # Inject failures
        # Rare failure event (1% of machines have 1 failure)
        failure_labels = np.zeros(len(dates))
        if np.random.random() < 0.2: # 20% of machines will have a failure
            failure_time = np.random.randint(24*5, len(dates) - 48) # failure happens at least 5 days in and 2 days before end
            
            # Pattern before failure (e.g., 48 hours before failure, values start to deviate)
            temp[failure_time-48:failure_time+1] += np.linspace(0, 15, 49) # Temp rises
            vib[failure_time-48:failure_time+1] += np.linspace(0, 8, 49)   # Vib rises
            temp[failure_time-48:failure_time+1] += np.random.normal(0, 3, 49) # Higher variance
            vib[failure_time-48:failure_time+1] += np.random.normal(0, 2, 49)
            
            # Actual failure timestamp
            failure_labels[failure_time] = 1
            
        machine_df = pd.DataFrame({
            'timestamp': dates,
            'machine_id': f'M_{machine_id:03d}',
            'temperature': temp,
            'vibration': vib,
            'pressure': pres,
            'failure': failure_labels
        })
        
        # Create target: failure in next 24 hours
        # We look ahead 24 hours to see if there's a failure
        # Shift the failure column backwards by up to 24 hours to create the 24h predictive window
        machine_df['failure_in_24h'] = machine_df['failure'].rolling(window=24, min_periods=1).max().shift(-24)
        machine_df['failure_in_24h'] = machine_df['failure_in_24h'].fillna(0)
        
        all_data.append(machine_df)
        
    final_df = pd.concat(all_data, ignore_index=True)
    
    # Save the data
    os.makedirs('data', exist_ok=True)
    final_df.to_csv('data/sensor_data.csv', index=False)
    print(f"Data generated and saved to data/sensor_data.csv with shape {final_df.shape}")
    print(f"Number of positive cases (failure within 24h): {final_df['failure_in_24h'].sum()}")

if __name__ == "__main__":
    # Generate 500 machines as per use case
    generate_synthetic_data(num_machines=500, days=60)
