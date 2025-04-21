# Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-%3E%3D13-blue?logo=postgresql)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-red)
![ML](https://img.shields.io/badge/ML-GradientBoosting-yellow)

A full-stack machine learning application for predicting customer churn in a telecommunications company, featuring interactive data exploration and real-time predictions.

## Demo

![App Demo](media/demo.gif)




## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Machine Learning Details](#machine-learning-details)
- [Setup and Installation](#setup-and-installation)
- [Application Screenshots](#application-screenshots)
- [Data Analysis Insights](#data-analysis-insights)
- [API Documentation](#api-documentation)
- [Future Enhancements](#future-enhancements)
- [License](#license)
- [Contact](#contact)

## 🔭 Overview

This project provides a comprehensive solution for telecommunications customer churn analysis and prediction. It demonstrates an end-to-end data science workflow from data loading and preprocessing to model training, evaluation, deployment, and visualization through an interactive dashboard.

The solution helps businesses answer critical questions like:
- Which customers are likely to churn?
- What are the primary drivers of customer churn?
- How can we develop targeted retention strategies based on customer segments?

## 🌟 Key Features

- **Interactive Dashboard**: Comprehensive Streamlit-based visualization dashboard with filtering capabilities
- **Real-time Predictions**: FastAPI backend for quick churn predictions via REST API
- **Machine Learning Pipeline**: End-to-end ML pipeline with feature engineering, preprocessing, and hyperparameter tuning
- **Data Persistence**: PostgreSQL database integration for storing customer data
- **Containerization**: Docker and Docker Compose for easy deployment and scaling
- **Dynamic Visualizations**: Multiple interactive plots and charts using Plotly
- **Statistical Insights**: Advanced statistical analysis of churn factors
- **ML Model Optimization**: Threshold optimization for classification metrics
- **Model Reporting**: Comprehensive reports including ROC curves, confusion matrices, and classification metrics

## 🏗️ Architecture

The project follows a modern microservices architecture with three main components:

1. **Backend Service (FastAPI)**: 
   - Serves ML model predictions via RESTful API
   - Handles model training and evaluation
   - Generates model performance reports

2. **Frontend Application (Streamlit)**:
   - Provides an interactive dashboard for data exploration
   - Visualizes churn patterns and insights
   - Offers a user interface for making predictions

3. **Database (PostgreSQL)**:
   - Stores customer data and metadata
   - Enables SQL-based data querying

All services are containerized using Docker and orchestrated with Docker Compose for seamless deployment.

## 💻 Technology Stack

- **Backend**: 
  - FastAPI for API development
  - scikit-learn for machine learning
  - SQLAlchemy for ORM

- **Frontend**:
  - Streamlit for the interactive dashboard
  - Plotly for advanced data visualizations

- **Database**:
  - PostgreSQL for data storage

- **DevOps**:
  - Docker for containerization
  - Docker Compose for multi-container orchestration

- **Data Analysis & ML**:
  - pandas for data manipulation
  - scikit-learn for machine learning algorithms
  - matplotlib/Plotly for visualization

## 🧠 Machine Learning Details

### Model Selection
The project uses a **Gradient Boosting Classifier** with hyperparameter tuning via RandomizedSearchCV and cross-validation.

### Feature Engineering
- Created derived features such as `avg_charge_per_month`
- Service usage count metrics
- Tenure categorization into meaningful buckets

### Preprocessing Pipeline
- Separate numerical and categorical preprocessing pipelines
- Missing value imputation
- Feature scaling
- One-hot encoding for categorical variables

### Hyperparameter Tuning
- RandomizedSearchCV for efficient hyperparameter space exploration
- 5-fold cross-validation to prevent overfitting
- ROC AUC as the optimization metric

### Threshold Optimization
- Custom threshold selection to balance precision and recall
- F1 score optimization for classification threshold

### Evaluation Metrics
- ROC AUC score
- F1 score
- Classification report
- Confusion matrix



## 🚀 Setup and Installation

### Prerequisites
- Docker and Docker Compose
- Git

### Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/MahtabRanjbar/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. Start the application:
   ```bash
   docker-compose up -d
   ```

3. Access the components:
   - Streamlit Dashboard: http://localhost:8501
   - FastAPI Documentation: http://localhost:8000/docs

### Manual Setup (Without Docker)

1. Setup PostgreSQL database:
   ```bash
   psql -U postgres -c "CREATE DATABASE churn_db;"
   psql -U postgres -d churn_db -f postgres/create_schema.sql
   ```

2. Install backend dependencies:
   ```bash
   cd backend
   pip install -r api/requirements.txt
   python api/train_model.py
   uvicorn api.main:app --reload
   ```

3. Install frontend dependencies:
   ```bash
   cd frontend
   pip install -r requirements.txt
   streamlit run app.py
   ```

## 📚 API Documentation

The FastAPI backend exposes the following endpoints:

### Prediction Endpoint
```
POST /predict
```

Sample request:
```json
{
  "tenure": 12,
  "monthly_charges": 70.0,
  "contract": "Month-to-month",
  "gender": "Female",
  "senior_citizen": 0,
  "partner": "No",
  "dependents": "No"
}
```

Sample response:
```json
{
  "prediction": "Yes",
  "probability": 0.72
}
```


## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

For questions, feedback, or collaboration opportunities, please reach out:

- Email: mahtabranjbar93@gmail.com
- LinkedIn: [Your LinkedIn Profile](https://linkedin.com/in/MahtabRanjbar)
- GitHub: [Your GitHub Profile](https://github.com/MahtabRanjbar)

---

⭐ If you find this project useful, please consider giving it a star on GitHub!
