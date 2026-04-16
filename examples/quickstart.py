"""
Quickstart example for Genesys API.

This script demonstrates the complete workflow:
1. Register user
2. Create project
3. Start training
4. Monitor training
5. Activate model
6. Run inference
"""

import time
import requests

# Configuration
BASE_URL = "http://localhost:8000/v1"


def register_user():
    """Register a new user."""
    response = requests.post(
        f"{BASE_URL}/auth/register",
        json={
            "name": "Test User",
            "email": "test@example.com",
            "password": "testpassword123",
        },
    )
    
    if response.status_code == 201:
        data = response.json()
        print(f"✅ User registered: {data['email']}")
        print(f"🔑 API Key: {data['api_key']}")
        return data["api_key"]
    else:
        print(f"❌ Registration failed: {response.json()}")
        # Try to get existing user
        response = requests.post(
            f"{BASE_URL}/auth/login",
            json={
                "email": "test@example.com",
                "password": "testpassword123",
            },
        )
        if response.status_code == 200:
            return response.json()["user"]["api_key"]
        return None


def create_project(api_key: str):
    """Create a new project."""
    response = requests.post(
        f"{BASE_URL}/projects",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "name": "GridWorld Example",
            "description": "Example GridWorld navigation project",
            "template_default": "grid_world",
        },
    )
    
    if response.status_code == 201:
        data = response.json()
        print(f"✅ Project created: {data['name']} (ID: {data['id']})")
        return data["id"]
    else:
        print(f"❌ Project creation failed: {response.json()}")
        return None


def start_training(api_key: str, project_id: str):
    """Start a training job."""
    response = requests.post(
        f"{BASE_URL}/jobs/projects/{project_id}/train",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "template": "grid_world",
            "config": {
                "state_size": 8,
                "action_space": [0, 1, 2, 3],
                "episodes": 100,  # Small number for quick demo
                "max_steps": 50,
                "gamma": 0.99,
                "learning_rate": 0.001,
                "epsilon_start": 1.0,
                "epsilon_end": 0.01,
                "epsilon_decay": 0.995,
                "batch_size": 32,
                "memory_size": 5000,
                "target_update_freq": 50,
            },
        },
    )
    
    if response.status_code == 201:
        data = response.json()
        print(f"✅ Training job created: {data['id']}")
        print(f"📊 Status: {data['status']}")
        return data["id"]
    else:
        print(f"❌ Training job creation failed: {response.json()}")
        return None


def wait_for_training(api_key: str, job_id: str, timeout: int = 300):
    """Wait for training to complete."""
    print("⏳ Waiting for training to complete...")
    
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        response = requests.get(
            f"{BASE_URL}/jobs/{job_id}",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        
        if response.status_code == 200:
            data = response.json()
            status = data["status"]
            
            if status == "completed":
                print(f"✅ Training completed!")
                print(f"📈 Episodes: {data.get('total_episodes')}")
                print(f"🎯 Avg Reward: {data.get('avg_reward', 'N/A')}")
                return True
            elif status == "failed":
                print(f"❌ Training failed: {data.get('error_message')}")
                return False
            elif status == "cancelled":
                print(f"⚠️ Training cancelled")
                return False
            else:
                print(f"⏳ Status: {status}...")
        
        time.sleep(5)
    
    print("⏱️ Timeout waiting for training")
    return False


def get_models(api_key: str, project_id: str):
    """Get list of models for project."""
    response = requests.get(
        f"{BASE_URL}/models/projects/{project_id}/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    
    if response.status_code == 200:
        data = response.json()
        models = data.get("items", [])
        if models:
            print(f"✅ Found {len(models)} model(s)")
            return models[0]["id"]
    
    print("❌ No models found")
    return None


def activate_model(api_key: str, model_id: str):
    """Activate a model for inference."""
    response = requests.post(
        f"{BASE_URL}/models/{model_id}/activate",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    
    if response.status_code == 200:
        print(f"✅ Model activated")
        return True
    else:
        print(f"❌ Model activation failed: {response.json()}")
        return False


def run_inference(api_key: str, project_id: str):
    """Run inference with the trained model."""
    # Example state for GridWorld (normalized positions + sensors)
    state = [0.0, 0.0, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0]
    
    response = requests.post(
        f"{BASE_URL}/inference/projects/{project_id}/predict",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"state": state},
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Inference result:")
        print(f"   Action: {data['action']}")
        print(f"   Confidence: {data['confidence']:.2%}")
        print(f"   Time: {data['inference_time_ms']:.2f}ms")
        return data
    else:
        print(f"❌ Inference failed: {response.json()}")
        return None


def main():
    """Run the complete workflow."""
    print("🚀 Genesys Quickstart Example\n")
    
    # Step 1: Register user
    api_key = register_user()
    if not api_key:
        print("Failed to get API key")
        return
    
    # Step 2: Create project
    project_id = create_project(api_key)
    if not project_id:
        print("Failed to create project")
        return
    
    # Step 3: Start training
    job_id = start_training(api_key, project_id)
    if not job_id:
        print("Failed to start training")
        return
    
    # Step 4: Wait for training
    if not wait_for_training(api_key, job_id):
        print("Training did not complete successfully")
        return
    
    # Step 5: Get and activate model
    model_id = get_models(api_key, project_id)
    if not model_id:
        print("No models available")
        return
    
    if not activate_model(api_key, model_id):
        print("Failed to activate model")
        return
    
    # Step 6: Run inference
    result = run_inference(api_key, project_id)
    
    if result:
        print("\n✨ Workflow completed successfully!")
    else:
        print("\n❌ Workflow failed")


if __name__ == "__main__":
    main()
