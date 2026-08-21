import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier # <-- ใช้ Neural Network จาก scikit-learn
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest

print("🚀 เริ่มกระบวนการเทรนและบันทึกโมเดล...")

# 1. โหลดข้อมูล
#url = "https://raw.githubusercontent.com/selva86/datasets/master/Banknote_Authentication.csv"
#df = pd.read_csv(url)
df = pd.read_csv('BankNote_Authentication.csv')

X = df.drop('class', axis=1)
y = df['class']

# 2. แบ่ง Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. SMOTE
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# 5. เทรนโมเดล
models = {}

print("🤖 เทรน Logistic Regression...")
models['Logistic Regression'] = LogisticRegression(random_state=42, max_iter=1000).fit(X_train_resampled, y_train_resampled)

print("🌲 เทรน Random Forest...")
models['Random Forest'] = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced').fit(X_train_resampled, y_train_resampled)

print("🔥 เทรน XGBoost...")
models['XGBoost'] = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, eval_metric='logloss').fit(X_train_resampled, y_train_resampled)

print("🌳 เทรน Isolation Forest...")
models['Isolation Forest'] = IsolationForest(n_estimators=100, contamination=0.1, random_state=42).fit(X_train_resampled)

print("🧠 เทรน Neural Network (MLPClassifier)...")
# MLPClassifier คือ Artificial Neural Network (ANN) แบบ Multi-Layer Perceptron
models['Neural Network (MLP)'] = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=42).fit(X_train_resampled, y_train_resampled)

# 6. บันทึกไฟล์
os.makedirs('models', exist_ok=True)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('models/models.pkl', 'wb') as f:
    pickle.dump(models, f)

print("\n✅ บันทึกโมเดลลงโฟลเดอร์ 'models/' เรียบร้อย! พร้อมใช้งาน Streamlit App")