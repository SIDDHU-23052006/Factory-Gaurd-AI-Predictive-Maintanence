import requests
import time
import json

def test_api():
    url = "http://localhost:5000/predict"
    
    # We simulate sending data for a specific machine
    machine_id = "M_001"
    
    # Simulate a sequence of readings (normal)
    print("Simulating normal readings...")
    for i in range(5):
        payload = {
            "machine_id": machine_id,
            "readings": {
                "temperature": 65.5 + i*0.1,
                "vibration": 2.1 + i*0.05,
                "pressure": 110.0 + i*0.5
            }
        }
        
        response = requests.post(url, json=payload)
        result = response.json()
        risk = result.get('failure_probability', 0) * 100
        
        if result.get('alert'):
            print(f"Reading {i+1} | [DANGER] Machine {machine_id} is predicted to FAIL within 24 hours! (Risk: {risk:.1f}%)")
        else:
            print(f"Reading {i+1} | [OK] Machine {machine_id} is healthy. (Risk: {risk:.1f}%)")
            
        time.sleep(0.1)
        
    print("\nSimulating failure pattern (spiking values)...")
    for i in range(5):
        payload = {
            "machine_id": machine_id,
            "readings": {
                "temperature": 80.0 + i*2.0, # High temperature
                "vibration": 8.0 + i*1.0,    # High vibration
                "pressure": 130.0 + i*2.0
            }
        }
        
        response = requests.post(url, json=payload)
        result = response.json()
        risk = result.get('failure_probability', 0) * 100
        
        if result.get('alert'):
            print(f"Spike Reading {i+1} | [DANGER] Machine {machine_id} is predicted to FAIL within 24 hours! (Risk: {risk:.1f}%)")
        else:
            print(f"Spike Reading {i+1} | [OK] Machine {machine_id} is healthy. (Risk: {risk:.1f}%)")
            
        time.sleep(0.1)

if __name__ == "__main__":
    test_api()
