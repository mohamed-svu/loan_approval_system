import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os

# إنشاء مجلدات إذا لم تكن موجودة
os.makedirs('models', exist_ok=True)
os.makedirs('static/images', exist_ok=True)

# تحميل البيانات
data = pd.read_csv('/home/mohamedsvu/loan_approval_system/data/loan_prediction.csv')

# معالجة البيانات المفقودة
data['Dependents'] = data['Dependents'].replace('3+', '3')
data['Dependents'] = data['Dependents'].fillna('0')
data['LoanAmount'] = data['LoanAmount'].fillna(data['LoanAmount'].median())
data['Loan_Amount_Term'] = data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].median())
data['Credit_History'] = data['Credit_History'].fillna(data['Credit_History'].median())
data['Gender'] = data['Gender'].fillna(data['Gender'].mode()[0])
data['Married'] = data['Married'].fillna(data['Married'].mode()[0])
data['Self_Employed'] = data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])

# تحويل المتغيرات الفئوية
cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
le = LabelEncoder()
for col in cat_cols:
    data[col] = le.fit_transform(data[col])

# تحديد الميزات والهدف
X = data.drop(['Loan_ID', 'Loan_Status'], axis=1)
y = data['Loan_Status']

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تدريب النموذج
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# تقييم النموذج
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# حفظ النموذج
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# حفظ المقاييس
metrics = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1
}

import json
with open('metrics.json', 'w') as f:
    json.dump(metrics, f)

# إنشاء بعض الرسوم البيانية للتحليل
plt.figure(figsize=(10, 6))
sns.countplot(x='Education', hue='Loan_Status', data=data)
plt.title('توزيع حالة القرض حسب التعليم')
plt.savefig('/home/mohamedsvu/loan_approval_system/static/images/education_loan_status.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Loan_Status', y='ApplicantIncome', data=data)
plt.title('دخل المتقدم حسب حالة القرض')
plt.savefig('/home/mohamedsvu/loan_approval_system/static/images/income_loan_status.png')
plt.close()