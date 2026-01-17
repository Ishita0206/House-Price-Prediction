# 🏠 House Price Prediction using Machine Learning

This project predicts **median house values** using machine learning techniques.  
It demonstrates an **end-to-end ML workflow** including data preprocessing, stratified sampling, pipeline building, model training, and inference.

---

## 📌 Project Overview

The goal of this project is to:
- Train a regression model on housing data
- Use a robust preprocessing pipeline
- Save the trained model and pipeline
- Perform inference on unseen data and generate predictions

---

## 🧠 Machine Learning Approach

- **Model Used:** Random Forest Regressor
- **Sampling Technique:** Stratified Shuffle Split (based on income category)
- **Evaluation Metric:** Root Mean Squared Error (RMSE)

---

## 🧹 Data Preprocessing

The preprocessing is handled using **Scikit-learn Pipelines**:

### Numerical Features
- Missing values handled using **Median Imputation**
- Feature scaling using **StandardScaler**

### Categorical Features
- Encoding using **One-Hot Encoder**

All preprocessing steps are combined using a **ColumnTransformer** to ensure consistency.

---

## ⚙️ Project Workflow

### 1️⃣ Training Phase
- Load housing dataset
- Create income categories for stratified sampling
- Split data into training and test sets
- Build preprocessing pipeline
- Train Random Forest model
- Save trained model and pipeline using `joblib`

### 2️⃣ Inference Phase
- Load saved model and pipeline
- Read test input data
- Apply preprocessing pipeline
- Generate predictions
- Save results to `predictions.csv`

---

## 🎯 Key Learnings

- Proper use of **Pipelines** to avoid data leakage  
- Importance of **Stratified Sampling**  
- Clean separation of **training and inference logic**  
- Best practices for **ML project structure and Git usage**

---

## 👩‍💻 Author

**Ishita Sharma**  
CSE & AIML Student


## 📂 Project Structure

