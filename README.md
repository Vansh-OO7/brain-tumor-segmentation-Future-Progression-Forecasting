# 🧠 Automated Brain Tumor Segmentation and Future Progression Forecasting

AI-powered healthcare project that performs **brain tumor segmentation** from multi-modal MRI scans and predicts **future tumor progression** using machine learning.

---

## 🚀 Features

- Upload `.h5` MRI scan
- Automatic tumor segmentation
- Tumor area estimation
- Short / Mid / Long-term progression forecasting
- Growth percentage analysis
- Progression status detection
- Clinical-style visual output
- Interactive Streamlit web app

---

## 🧩 Workflow

1. Upload multi-modal MRI file  
2. Preprocess input scan  
3. Segment tumor using U-Net  
4. Calculate current tumor area  
5. Predict future progression  
6. Display results in dashboard  

---

## 🧠 Models Used

### Tumor Segmentation
- **U-Net** (TensorFlow / Keras)

### Progression Forecasting
- **Random Forest Regressor** (Multi-output)

---

## 📊 Sample Results

| Metric | Value |
|------|------|
| MAE | 3.5277 |
| RMSE | 6.0007 |
| R² Score | 0.8669 |

### Forecast Accuracy

| Horizon | R² |
|--------|------|
| Short Term | 0.9653 |
| Mid Term | 0.8765 |
| Long Term | 0.7589 |

---

## 🛠 Tech Stack

- Python
- TensorFlow / Keras
- Scikit-learn
- NumPy / Pandas
- Matplotlib
- Streamlit
- h5py / joblib

---

## 📁 Project Structure

```bash
brain-tumor-segmentation-future-growth-prediction/
│── streamlit_app.py
│── requirements.txt
│── README.md
├── models/
├── src/
├── uploads/
└── outputs/
