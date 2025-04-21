# Recruitment Candidate Selection - SVM Project

This project predicts whether a candidate should be hired based on their experience and technical test score using the SVM algorithm.

---

## Project Overview

The system consists of:
- Machine Learning Model: Trained using SVM on synthetic candidate data.
- API Backend: FastAPI service to make predictions based on input.
- Web UI: A user-friendly Streamlit interface for interaction.
- Docker Support: Containerization for deployment.

---

## Components

### 1. Core Python Scripts
- `data_generator.py`: Generates labeled synthetic data using Faker.
- `model_training.py`: Trains a linear SVM model and saves the model and scaler.
- `visualize_boundary.py`: (Optional) Plots the SVM decision boundary.
- `predict_user_input.py`: Command-line interface for manual predictions.

### 2. FastAPI Backend
- `main.py`: REST API to serve predictions using the trained model.
- Endpoint: `POST /predict`
- Input: Experience (years) and technical score (0-100)
- Rule-based logic is applied for early rejection.

### 3. Streamlit Web UI
- `streamlit_app.py`: Lightweight web interface to input candidate data and view prediction results.
- Communicates with FastAPI backend via HTTP request.

---

## How to Run the Project Locally

### Step 1. Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2. Generate data
```bash
python data_generator.py
```

### Step 3. Train and save model
```bash
python model_training.py
```

### Step 4. Start FastAPI server
```bash
uvicorn main:app --reload
```
Visit: http://localhost:8000/docs to test the API.

---

## Run the Web UI (Streamlit)

### Step 1. Ensure FastAPI is running (see above)
### Step 2. Run the frontend
```bash
streamlit run streamlit_app.py
```
Visit: http://localhost:8501 to use the web app.

---

## Docker Setup (Manual)

### Step 1. Build the Docker image
```bash
docker build -t recruitment-api .
```

### Step 2. Run the container
```bash
docker run -d -p 8000:8000 recruitment-api
```

### Step 3. Access the API
Open http://localhost:8000/docs

Note: This container only runs the FastAPI backend. Streamlit runs separately on the host or another container.

---

## Docker Setup with Docker Compose

### Step 1. Ensure Docker is installed and running

### Step 2. Run the entire system
```bash
docker-compose up --build
```

### Step 3. Access the interfaces
- API: http://localhost:8000/docs
- Web UI: http://localhost:8501