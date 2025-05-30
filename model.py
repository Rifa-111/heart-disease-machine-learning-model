import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("data/sleep_health_and_lifestyle_dataset.csv")
df.dropna(inplace=True)

# Create new feature: Sleep Efficiency
df["Sleep Efficiency"] = (df["Sleep Duration"] / 8) * 100

# ✅ Create a more balanced Heart Disease label using multiple risk factors
df["Heart Disease"] = (
    (df["BMI Category"].isin(["Obese", "Overweight"])) &
    (df["Stress Level"] >= 7) &
    (df["Sleep Disorder"].isin(["Sleep Apnea", "Insomnia"])) &
    (df["Heart Rate"] >= 80)
).astype(int)

# ✅ Debug: check class balance
print("Class distribution in 'Heart Disease':")
print(df["Heart Disease"].value_counts())

# Define numeric and categorical features
numeric = ["Sleep Duration", "Quality of Sleep", "Stress Level", "Physical Activity Level", "Heart Rate"]
categorical = ["Gender", "Occupation", "Sleep Disorder", "BMI Category"]

# Define input (X) and target (y)
X = df[numeric + categorical]
y = df["Heart Disease"]

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ("num", MinMaxScaler(), numeric),
    ("cat", OneHotEncoder(), categorical)
])

# Full model pipeline
model = Pipeline([
    ("prep", preprocessor),
    ("clf", XGBClassifier(use_label_encoder=False, eval_metric="logloss"))
])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Train the model
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\nModel Evaluation:")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
















# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import classification_report, roc_auc_score
# from xgboost import XGBClassifier

# df = pd.read_csv("data/sleep_health_and_lifestyle_dataset.csv")
# df.dropna(inplace=True)
# df["Sleep Efficiency"] = (df["Sleep Duration"] / 8) * 100

# X = df.drop(columns=["Heart Disease"])
# y = df["Heart Disease"]

# numeric = ["Sleep Duration", "Quality of Sleep", "Stress Level", "Physical Activity Level", "BMI", "Heart Rate"]
# categorical = ["Gender", "Occupation", "Sleep Disorder"]

# preprocessor = ColumnTransformer([
#     ("num", MinMaxScaler(), numeric),
#     ("cat", OneHotEncoder(), categorical)
# ])

# model = Pipeline([
#     ("prep", preprocessor),
#     ("clf", XGBClassifier(use_label_encoder=False, eval_metric="logloss"))
# ])

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)
# y_proba = model.predict_proba(X_test)[:, 1]

# print(classification_report(y_test, y_pred))
# print("ROC-AUC:", roc_auc_score(y_test, y_proba))

