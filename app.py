from flask import Flask, render_template, request, url_for, jsonify
import pandas as pd
import pickle
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import logging

# تهيئة التطبيق
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# إعداد نظام التسجيل
logging.basicConfig(filename='loan_app.log', level=logging.INFO)

# تحميل البيانات
def load_data():
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'loan_prediction.csv')
    return pd.read_csv(data_path)

# تحميل النموذج والمقاييس من المجلد الرئيسي
def load_model_and_metrics():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, 'model.pkl')
    metrics_path = os.path.join(base_dir, 'metrics.json')

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    return model, metrics

# تحميل البيانات عند بدء التشغيل
try:
    data = load_data()
    model, metrics = load_model_and_metrics()
    le = LabelEncoder()
    app.logger.info("تم تحميل البيانات والنموذج بنجاح")
except Exception as e:
    app.logger.error(f"خطأ في تحميل البيانات: {str(e)}")
    raise e

@app.route('/keep_alive')
def keep_alive():
    return "OK", 200

# الصفحة الرئيسية
@app.route('/')
def index():
    return render_template('index.html')

# صفحة إضافة قرض
@app.route('/add_loan', methods=['GET', 'POST'])
def add_loan():
    if request.method == 'POST':
        try:
            new_loan = {
                'Gender': request.form.get('gender'),
                'Married': request.form.get('married'),
                'Dependents': request.form.get('dependents', '0'),
                'Education': request.form.get('education'),
                'Self_Employed': request.form.get('self_employed'),
                'ApplicantIncome': float(request.form.get('applicant_income', 0)),
                'CoapplicantIncome': float(request.form.get('coapplicant_income', 0)),
                'LoanAmount': float(request.form.get('loan_amount', 0)),
                'Loan_Amount_Term': float(request.form.get('loan_term', 360)),
                'Credit_History': float(request.form.get('credit_history', 0)),
                'Property_Area': request.form.get('property_area')
            }

            # تحويل البيانات إلى DataFrame
            new_df = pd.DataFrame([new_loan])

            # تحويل المتغيرات الفئوية
            cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
            for col in cat_cols:
                new_df[col] = le.fit_transform(new_df[col])

            # التنبؤ
            prediction = model.predict(new_df)[0]
            result = 'مقبول' if prediction == 1 else 'مرفوض'

            return render_template('add_loan.html', prediction=result, show_result=True)

        except Exception as e:
            app.logger.error(f"خطأ في معالجة طلب القرض: {str(e)}")
            return render_template('add_loan.html', error_message="حدث خطأ أثناء معالجة الطلب", show_result=False)

    return render_template('add_loan.html', show_result=False)

# صفحة التحليلات
@app.route('/analysis')
def analysis():
    try:
        # إنشاء مجلد الصور إذا لم يكن موجوداً
        images_dir = os.path.join(os.path.dirname(__file__), 'static', 'images')
        os.makedirs(images_dir, exist_ok=True)

        # مسارات الصور
        img1_path = os.path.join(images_dir, 'education_loan_status.png')
        img2_path = os.path.join(images_dir, 'income_loan_status.png')

        # إنشاء الرسوم البيانية
        plt.figure(figsize=(10, 6))
        sns.countplot(x='Education', hue='Loan_Status', data=data)
        plt.title('توزيع حالة القرض حسب التعليم')
        plt.savefig(img1_path)
        plt.close()

        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Loan_Status', y='ApplicantIncome', data=data)
        plt.title('دخل المتقدم حسب حالة القرض')
        plt.savefig(img2_path)
        plt.close()

        stats = {
            'total_loans': len(data),
            'approved_loans': len(data[data['Loan_Status'] == 'Y']),
            'rejected_loans': len(data[data['Loan_Status'] == 'N']),
            'avg_income': round(data['ApplicantIncome'].mean(), 2),
            'avg_loan_amount': round(data['LoanAmount'].mean(), 2),
            'img1': url_for('static', filename='images/education_loan_status.png'),
            'img2': url_for('static', filename='images/income_loan_status.png')
        }

        return render_template('analysis.html', stats=stats)

    except Exception as e:
        app.logger.error(f"خطأ في تحليل البيانات: {str(e)}")
        return render_template('error.html', error_message="حدث خطأ أثناء تحليل البيانات")

# صفحة المقاييس
@app.route('/metrics')
def show_metrics():
    try:
        return render_template('metrics.html',
                             accuracy=round(metrics['accuracy']*100, 2),
                             precision=round(metrics['precision']*100, 2),
                             recall=round(metrics['recall']*100, 2),
                             f1=round(metrics['f1']*100, 2))

    except Exception as e:
        app.logger.error(f"خطأ في تحميل المقاييس: {str(e)}")
        return render_template('error.html', error_message="حدث خطأ أثناء تحميل المقاييس")

# صفحة التحقق من صحة النظام
@app.route('/health')
def health_check():
    base_dir = os.path.dirname(__file__)
    files = {
        'data_file': os.path.exists(os.path.join(base_dir, 'loan_prediction.csv')),
        'model_file': os.path.exists(os.path.join(base_dir, 'model.pkl')),
        'metrics_file': os.path.exists(os.path.join(base_dir, 'metrics.json')),
        'images_dir': os.path.exists(os.path.join(base_dir, 'static', 'images'))
    }
    return jsonify(files)

# صفحة للتعامل مع الأخطاء
@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', error_message="الصفحة غير موجودة"), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('error.html', error_message="حدث خطأ داخلي في الخادم"), 500

if __name__ == '__main__':
    app.run(debug=True)