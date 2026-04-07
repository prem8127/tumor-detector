# 🧠 Lung & Pancreatic Tumor Detection System

A **production-grade AI-powered web application** for detecting tumors from CT scan images using both **Machine Learning (SVM)** and **Deep Learning (ResNet-50 CNN)** models.

---

## 🚀 Key Features

* 🧠 **Dual Model System**

  * SVM (HOG + LBP features)
  * ResNet-50 (Deep Learning)

* 🫁 **Multi-Organ Detection**

  * Lung Tumor Detection
  * Pancreatic Tumor Detection

* 📤 **Image Upload & Analysis**

  * Supports PNG, JPG, BMP, TIFF

* 📊 **Detailed Predictions**

  * Tumor / No Tumor classification
  * Confidence score
  * Probability distribution

* ⚡ **Fast & Interactive UI**

  * Single-page Flask interface
  * Real-time prediction results

---

## 🛠️ Tech Stack

* **Frontend:** HTML, CSS, JavaScript
* **Backend:** Flask (Python)
* **ML Models:** Scikit-learn (SVM)
* **DL Models:** TensorFlow / Keras (ResNet-50)
* **Image Processing:** OpenCV, PIL

---

## 📁 Project Structure

```
tumor_detection/
├── app.py
├── requirements.txt
├── README.md
├── models/
│   ├── lung_model.pkl
│   ├── pancreatic_model.pkl
│   ├── lung_cancer_resnet50.h5
│   └── pancreas_cancer_resnet50.h5
├── templates/
│   └── index.html
├── static/
│   ├── css/style.css
│   └── js/app.js
└── uploads/
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/tumor-detection-app.git
cd tumor-detection-app
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📥 Model Files Setup (IMPORTANT)

Due to large size, model files are not included in the repository.

Place the following files inside the `models/` folder:

```
lung_model.pkl
pancreatic_model.pkl
lung_cancer_resnet50.h5
pancreas_cancer_resnet50.h5
```

👉 Add your Google Drive link here:

```
Download models: [YOUR_LINK_HERE]
```

---

## ▶️ Run the Application

```bash
python app.py
```

Open browser:

```
http://localhost:5000
```

---

## 🧪 How It Works

### 🔹 SVM Pipeline

* Grayscale conversion → Resize (128×128)
* Feature Extraction:

  * HOG (8100 features)
  * LBP (64 features)
* Combined → 8164 features
* StandardScaler → SVM classifier

### 🔹 CNN Pipeline (ResNet-50)

* Resize to 224×224 RGB
* Normalize pixel values
* Deep feature extraction using ResNet-50
* Final classification layer

---

## 🔗 API Endpoints

| Method | Endpoint   | Description    |
| ------ | ---------- | -------------- |
| GET    | `/`        | Web interface  |
| POST   | `/predict` | Run prediction |
| GET    | `/health`  | Health check   |

---

## 📊 Example Response

```json
{
  "label": "Lung Cancer",
  "detected": true,
  "confidence": 91.34,
  "organ": "lung",
  "model_type": "SVM"
}
```

---

## ⚠️ Troubleshooting

| Issue               | Solution                        |
| ------------------- | ------------------------------- |
| TensorFlow error    | `pip install tensorflow`        |
| Model not found     | Ensure files exist in `/models` |
| Port already in use | Change port in `app.py`         |

---

## 🚀 Deployment

Use Gunicorn for production:

```bash
pip install gunicorn
gunicorn -w 2 -b 0.0.0.0:5000 app:app
```

---

## 📌 Future Enhancements

* Grad-CAM visualization for CNN 🔥
* User authentication system
* Cloud deployment (AWS / Render)
* Medical report generation

---

## 👨‍💻 Author

**Prem Sagar**
B.Tech CSE (AI & ML)
VBIT

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**.
It is **not a substitute for professional medical diagnosis**.
