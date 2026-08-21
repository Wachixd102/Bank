# 💵 Banknote Fraud Detection System

ระบบตรวจจับธนบัตรปลอม (Banknote Authentication) ที่พัฒนาโดยใช้เทคนิค Machine Learning และ Deep Learning เพื่อจำแนกธนบัตรจริงและธนบัตรปลอม โดยวิเคราะห์จากลักษณะทางสถิติของภาพ

##  วัตถุประสงค์
- ศึกษาและเปรียบเทียบประสิทธิภาพของโมเดล Machine Learning 5 ชนิด
- พัฒนาโมเดลที่สามารถจำแนกธนบัตรจริงและปลอมได้อย่างแม่นยำ
- สร้าง Web Application สำหรับใช้งานจริงผ่าน Streamlit

## 🛠️ เทคโนโลยีที่ใช้
- Python
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn, XGBoost, Imbalanced-learn
- TensorFlow / Keras
- Streamlit

## 📊 Dataset
ใช้ **Banknote Authentication Dataset** จาก UCI Machine Learning Repository
- จำนวนข้อมูล: 1,372 รายการ
- Features: variance, skewness, curtosis, entropy
- Target: class (0 = ของจริง, 1 = ของปลอม)

## 🚀 วิธีรันโปรเจค

### 1. ติดตั้ง Dependencies
```bash
pip install -r requirements.txt