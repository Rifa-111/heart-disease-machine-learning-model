# heart-disease-machine-learning-model

# ❤️ Machine Learning-Based Heart Disease Risk Analyser Web App

This project is a machine learning-powered web app that predicts the risk of heart disease based on sleep quality and lifestyle habits using a dataset from Kaggle.

Built with:
- ✅ Python & Streamlit
- ✅ XGBoost Classifier
- ✅ SHAP-ready architecture
- ✅ Sleep & stress-based features

---

## 📊 Features

- Interactive web form to enter lifestyle data
- Predicts **heart disease risk** with confidence %
- Uses custom rules to label heart disease based on:
  - Obesity/Overweight
  - High Stress
  - Sleep Disorders (Insomnia, Sleep Apnea)
  - High Resting Heart Rate

---

## 📁 Dataset

Dataset used:  
**Sleep Health and Lifestyle Dataset**  
📎 [Kaggle Link](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset)

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/Rifa-111/heart-disease-machine-learning-model.git
cd heart-disease-machine-learning-model
pip install -r requirements.txt
streamlit run app.py
