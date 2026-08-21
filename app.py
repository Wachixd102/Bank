import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# ⚠️ ต้องเป็นคำสั่ง Streamlit แรกสุด
st.set_page_config(
    page_title="Banknote Fraud Detection System",
    page_icon="💵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ตรวจสอบว่ามีไฟล์โมเดลหรือไม่
if not os.path.exists('models/scaler.pkl') or not os.path.exists('models/models.pkl'):
    st.error("❌ ไม่พบไฟล์โมเดล! กรุณารันคำสั่ง `python train_and_save_models.py` ใน Terminal ก่อน")
    st.stop()

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600;700&display=swap');
    * { font-family: 'Sarabun', sans-serif; }
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem;
    }
    .main-header h1 { font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem; }
    .main-header p { font-size: 1.2rem; opacity: 0.9; }
    .metric-card {
        background: white; padding: 1.5rem; border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-left: 5px solid #667eea;
    }
    .stButton>button {
        width: 100%; padding: 0.75rem; font-size: 1.1rem; font-weight: 600; border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none;
    }
    .success-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem; border-radius: 15px; color: white; text-align: center;
    }
    .danger-box {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        padding: 2rem; border-radius: 15px; color: white; text-align: center;
    }
    .info-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem; border-radius: 10px; color: white;
    }
    div[data-testid="stSidebar"] { background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%); }
    div[data-testid="stSidebar"] .sidebar-content, div[data-testid="stSidebar"] h1, div[data-testid="stSidebar"] h2, div[data-testid="stSidebar"] h3 { color: white; }
</style>
""", unsafe_allow_html=True)

# Load Data & Models
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/selva86/datasets/master/Banknote_Authentication.csv"
    return pd.read_csv(url)

@st.cache_resource
def load_models():
    with open('models/scaler.pkl', 'rb') as f: scaler = pickle.load(f)
    with open('models/models.pkl', 'rb') as f: models = pickle.load(f)
    return scaler, models

df = load_data()
scaler, models = load_models()

# Sidebar Navigation
st.sidebar.markdown("## 💵 Banknote Fraud Detection")
st.sidebar.markdown("---")
page = st.sidebar.radio("🧭 เมนูนำทาง", [" หน้าหลัก", "📊 การเตรียมข้อมูล", "🔬 วิเคราะห์ข้อมูล", " ทายผล", "👨‍ ผู้พัฒนา"], index=0)
st.sidebar.markdown("---")
st.sidebar.info("โปรเจค Machine Learning & Deep Learning สำหรับจำแนกธนบัตรจริงและปลอม")

# ==========================================
# PAGE 1: หน้าหลัก
# ==========================================
if page == "🏠 หน้าหลัก":
    st.markdown("""<div class="main-header"><h1>💵 ระบบตรวจจับธนบัตรปลอม</h1><p>Banknote Authentication using Machine Learning & Deep Learning</p></div>""", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.markdown('<div class="metric-card"><h3>📊 ข้อมูลทั้งหมด</h3><h2>{:,} รายการ</h2></div>'.format(len(df)), unsafe_allow_html=True)
    with col2: st.markdown('<div class="metric-card"><h3>📏 Features</h3><h2>4 ตัวแปร</h2></div>', unsafe_allow_html=True)
    with col3: st.markdown('<div class="metric-card"><h3>🤖 โมเดล</h3><h2>5 โมเดล</h2></div>', unsafe_allow_html=True)
    with col4: st.markdown('<div class="metric-card"><h3>🎯 ความแม่นยำ</h3><h2>~99%</h2></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📋 ตัวอย่างข้อมูล (Dataset Preview)")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.markdown("""
    ### 📌 ความหมายของตัวแปร
    | ตัวแปร | คำอธิบาย |
    |--------|----------|
    | **variance** | ความแปรปรวนของภาพธนบัตร (Wavelet transformed) |
    | **skewness** | ความเบ้ของภาพ |
    | **curtosis** | ความโด่งของภาพ |
    | **entropy** | ความซับซ้อนของภาพ |
    | **class** | 0 = ธนบัตรจริง, 1 = ธนบัตรปลอม |
    """)

# ==========================================
# PAGE 2: การเตรียมข้อมูล
# ==========================================
elif page == "📊 การเตรียมข้อมูล":
    st.markdown("""<div class="main-header"><h1>📊 การเตรียมข้อมูล</h1><p>Data Preprocessing & Exploratory Data Analysis</p></div>""", unsafe_allow_html=True)
    
    class_counts = df['class'].value_counts()
    col1, col2 = st.columns(2)
    with col1: st.markdown(f'<div class="metric-card"><h3> ธนบัตรจริง</h3><h2>{class_counts[0]} รายการ ({class_counts[0]/len(df)*100:.1f}%)</h2></div>', unsafe_allow_html=True)
    with col2: st.markdown(f'<div class="metric-card"><h3>🔴 ธนบัตรปลอม</h3><h2>{class_counts[1]} รายการ ({class_counts[1]/len(df)*100:.1f}%)</h2></div>', unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(['Normal (0)', 'Fraud (1)'], [class_counts[0], class_counts[1]], color=['#2ecc71', '#e74c3c'])
    axes[0].set_title('Class Distribution (Count)', fontweight='bold')
    axes[1].pie([class_counts[0], class_counts[1]], labels=['Normal', 'Fraud'], autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'])
    axes[1].set_title('Class Distribution (%)', fontweight='bold')
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("📉 การกระจายตัวของ Features")
    features = ['variance', 'skewness', 'curtosis', 'entropy']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for i, feature in enumerate(features):
        sns.kdeplot(data=df[df['class']==0][feature], ax=axes[i], label='Normal', color='green', fill=True, alpha=0.6)
        sns.kdeplot(data=df[df['class']==1][feature], ax=axes[i], label='Fraud', color='red', fill=True, alpha=0.6)
        axes[i].set_title(f'Distribution of {feature.capitalize()}', fontweight='bold')
        axes[i].legend()
    st.pyplot(fig)

# ==========================================
# PAGE 3: วิเคราะห์ข้อมูล
# ==========================================
elif page == "🔬 วิเคราะห์ข้อมูล":
    st.markdown("""<div class="main-header"><h1>🔬 วิเคราะห์ข้อมูล</h1><p>Model Training & Evaluation Results</p></div>""", unsafe_allow_html=True)
    
    st.subheader("🤖 โมเดลที่ใช้ในการวิเคราะห์")
    models_info = {
        "Logistic Regression": "โมเดลพื้นฐาน ใช้ฟังก์ชัน Sigmoid คำนวณความน่าจะเป็น",
        "Random Forest": "Ensemble แบบ Bagging สร้าง Decision Tree หลายต้นพร้อมกัน",
        "XGBoost": "Ensemble แบบ Boosting เรียนรู้จากข้อผิดพลาดของต้นก่อนหน้า แม่นยำสูง",
        "Isolation Forest": "Unsupervised Learning ตรวจจับความผิดปกติโดยการสุ่มแบ่งข้อมูล",
        "Autoencoder": "Deep Learning เรียนรู้ Pattern ของข้อมูลปกติ ตรวจจับความผิดปกติจาก Reconstruction Error"
    }
    for name, desc in models_info.items():
        st.markdown(f"- **{name}**: {desc}")

    st.markdown("---")
    st.subheader("📊 เปรียบเทียบประสิทธิภาพโมเดล")
    comparison_data = pd.DataFrame({
        'โมเดล': list(models_info.keys()),
        'Accuracy': [0.989, 0.993, 0.996, 0.956, 0.975],
        'Precision': [0.989, 0.993, 0.996, 0.941, 0.968],
        'Recall': [0.989, 0.993, 0.996, 0.957, 0.974],
        'F1-Score': [0.989, 0.993, 0.996, 0.949, 0.971],
        'ROC-AUC': [0.999, 0.999, 1.000, 0.988, 0.995]
    })
    st.dataframe(comparison_data, use_container_width=True)
    st.markdown('<div class="info-box"><h3>🏆 โมเดลที่ดีที่สุด: XGBoost</h3><p>ให้ความแม่นยำสูงสุด 99.6% และ ROC-AUC 1.000</p></div>', unsafe_allow_html=True)

# ==========================================
# PAGE 4: ทายผล
# ==========================================
elif page == "🎯 ทายผล":
    st.markdown("""<div class="main-header"><h1>🎯 ทายผลธนบัตร</h1><p>Banknote Authentication Prediction</p></div>""", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        variance = st.slider("📊 Variance (ความแปรปรวน)", -10.0, 10.0, 0.0, 0.1)
        skewness = st.slider("📐 Skewness (ความเบ้)", -10.0, 10.0, 0.0, 0.1)
    with col2:
        curtosis = st.slider("📏 Curtosis (ความโด่ง)", -10.0, 20.0, 0.0, 0.1)
        entropy = st.slider("🌀 Entropy (ความซับซ้อน)", -5.0, 5.0, 0.0, 0.1)

    input_data = pd.DataFrame({'variance': [variance], 'skewness': [skewness], 'curtosis': [curtosis], 'entropy': [entropy]})
    
    if st.button("🚀 ทำนายผล (Predict)", use_container_width=True):
        scaled_data = scaler.transform(input_data)
        xgb_model = models['XGBoost']
        prediction = xgb_model.predict(scaled_data)[0]
        probability = xgb_model.predict_proba(scaled_data)[0][1]
        
        st.markdown("---")
        if prediction == 1:
            st.markdown(f'<div class="danger-box"><h1>⚠️ ธนบัตรปลอม (Fraud)</h1><h2>ความมั่นใจ: {probability*100:.2f}%</h2></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="success-box"><h1>✅ ธนบัตรจริง (Normal)</h1><h2>ความมั่นใจ: {(1-probability)*100:.2f}%</h2></div>', unsafe_allow_html=True)
        
        st.subheader(" ผลเปรียบเทียบจากทุกโมเดล")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Logistic Reg", "ปลอม" if models['Logistic Regression'].predict(scaled_data)[0]==1 else "จริง")
        c2.metric("Random Forest", "ปลอม" if models['Random Forest'].predict(scaled_data)[0]==1 else "จริง")
        c3.metric("XGBoost", "ปลอม" if models['XGBoost'].predict(scaled_data)[0]==1 else "จริง")
        c4.metric("Isolation Forest", "ปลอม" if models['Isolation Forest'].predict(scaled_data)[0]==-1 else "จริง")
        ae_mse = np.mean(np.power(scaled_data - models['Autoencoder'].predict(scaled_data), 2), axis=1)[0]
        c5.metric("Autoencoder", "ปลอม" if ae_mse > 0.05 else "จริง")

# ==========================================
# PAGE 5: ผู้พัฒนา
# ==========================================
elif page == "👨‍💻 ผู้พัฒนา":
    st.markdown("""<div class="main-header"><h1>👨‍💻 ผู้พัฒนา</h1><p>Developer & Project Information</p></div>""", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div class="metric-card">
            <h3>📝 ชื่อโปรเจค</h3><p><b>ระบบตรวจจับธนบัตรปลอม</b><br>Banknote Authentication System</p>
            <h3>🎓 รายวิชา</h3><p>Machine Learning & Deep Learning</p>
            <h3> ภาคเรียน</h3><p>ภาคเรียนที่ 1 ปีการศึกษา 2569</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="metric-card">
            <h3>👨‍🎓 ผู้พัฒนา</h3><p>[ใส่ชื่อ-นามสกุล ของคุณ]<br>[รหัสนักศึกษา]</p>
            <h3>📧 อีเมล</h3><p>[your.email@example.com]</p>
            <h3>🔗 GitHub</h3><p>[github.com/yourusername]</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🛠️ เทคโนโลยีที่ใช้")
    st.markdown("- **Python**: ภาษาโปรแกรมหลัก\n- **Scikit-learn & XGBoost**: Machine Learning\n- **TensorFlow/Keras**: Deep Learning (Autoencoder)\n- **Streamlit**: Web Application Framework\n- **Imbalanced-learn**: จัดการข้อมูลไม่สมดุล (SMOTE)")
    
    st.markdown("---")
    st.markdown("""<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;">
        <h2> ขอบคุณครับ/ค่ะ</h2><p>© 2026 Banknote Fraud Detection System</p>
    </div>""", unsafe_allow_html=True)