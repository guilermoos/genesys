#!/bin/bash

# Genesys API Examples using curl
# ================================

BASE_URL="http://localhost:8000/v1"
API_KEY=""  # Fill in after registration
PROJECT_ID=""  # Fill in after project creation
JOB_ID=""  # Fill in after training starts
MODEL_ID=""  # Fill in after training completes

echo "=== Genesys API Examples ==="
echo ""

# 1. Health Check
echo "1. Health Check"
curl -s $BASE_URL/health | jq .
echo ""

# 2. Register User
echo "2. Register User"
curl -s -X POST "$BASE_URL/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Demo User",
    "email": "demo@example.com",
    "password": "demopassword123"
  }' | jq .
echo ""

# 3. Login
echo "3. Login"
curl -s -X POST "$BASE_URL/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "demo@example.com",
    "password": "demopassword123"
  }' | jq .
echo ""

# 4. List Templates
echo "4. List Templates"
curl -s "$BASE_URL/templates" \
  -H "Authorization: Bearer $API_KEY" | jq .
echo ""

# 5. Create Project
echo "5. Create Project"
curl -s -X POST "$BASE_URL/projects" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "GridWorld Demo",
    "description": "Demo project for GridWorld navigation",
    "template_default": "grid_world"
  }' | jq .
echo ""

# 6. List Projects
echo "6. List Projects"
curl -s "$BASE_URL/projects" \
  -H "Authorization: Bearer $API_KEY" | jq .
echo ""

# 7. Start Training (GridWorld)
echo "7. Start Training"
curl -s -X POST "$BASE_URL/jobs/projects/$PROJECT_ID/train" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "template": "grid_world",
    "config": {
      "state_size": 8,
      "action_space": [0, 1, 2, 3],
      "episodes": 200,
      "max_steps": 100,
      "gamma": 0.99,
      "learning_rate": 0.001,
      "epsilon_start": 1.0,
      "epsilon_end": 0.01,
      "epsilon_decay": 0.995,
      "batch_size": 64,
      "memory_size": 10000,
      "target_update_freq": 100
    }
  }' | jq .
echo ""

# 8. Check Training Status
echo "8. Check Training Status"
curl -s "$BASE_URL/jobs/$JOB_ID" \
  -H "Authorization: Bearer $API_KEY" | jq .
echo ""

# 9. Get Training Logs
echo "9. Get Training Logs"
curl -s "$BASE_URL/jobs/$JOB_ID/logs" \
  -H "Authorization: Bearer $API_KEY" | jq .
echo ""

# 10. List Models
echo "10. List Models"
curl -s "$BASE_URL/models/projects/$PROJECT_ID/models" \
  -H "Authorization: Bearer $API_KEY" | jq .
echo ""

# 11. Activate Model
echo "11. Activate Model"
curl -s -X POST "$BASE_URL/models/$MODEL_ID/activate" \
  -H "Authorization: Bearer $API_KEY" | jq .
echo ""

# 12. Run Inference
echo "12. Run Inference"
curl -s -X POST "$BASE_URL/inference/projects/$PROJECT_ID/predict" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "state": [0.0, 0.0, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0]
  }' | jq .
echo ""

# 13. Batch Inference
echo "13. Batch Inference"
curl -s -X POST "$BASE_URL/inference/projects/$PROJECT_ID/predict/batch" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "states": [
      [0.0, 0.0, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0],
      [0.2, 0.2, 0.8, 0.8, 0.5, 1.0, 1.0, 0.5],
      [0.5, 0.5, 0.8, 0.8, 0.0, 0.5, 1.0, 0.0]
    ]
  }' | jq .
echo ""

# 14. Get Project Stats
echo "14. Get Project Stats"
curl -s "$BASE_URL/projects/$PROJECT_ID/stats" \
  -H "Authorization: Bearer $API_KEY" | jq .
echo ""

# 15. Cancel Training (if needed)
# echo "15. Cancel Training"
# curl -s -X POST "$BASE_URL/jobs/$JOB_ID/cancel" \
#   -H "Authorization: Bearer $API_KEY" | jq .
# echo ""

echo "=== Examples Complete ==="
