# 🔫 AI-Powered Weapons Classification with Custom CNN & VGG19  
[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Demo-brightgreen)](https://ai-powered-weapons-classification-with-custom-cnn-vgg19-hmfjcf.streamlit.app/)

## 🚀 About the Project  
This project classifies different types of weapons using **Deep Learning models**:  
✅ **VGG19 Pretrained Model**  
✅ **Custom CNN Model**  

The application is built using **Streamlit** and allows users to **upload images** and classify them using both models.

---

## 🛠️ Installation & Setup  
### **1️⃣ Clone the Repository**  
```sh
git clone https://github.com/ysaikumar21/Weapons-Classification-using-custom-and-VGG19.git
cd Weapons-Classification-using-custom-and-VGG19
2️⃣ Install Dependencies
sh
Copy
Edit
pip install -r requirements.txt
3️⃣ Run the Streamlit App Locally
sh
Copy
Edit
streamlit run main.py
📂 Project Structure
bash
Copy
Edit
📂 Weapons-Classification
│── 📄 main.py              # Streamlit app with model downloading & classification
│── 📄 requirements.txt     # Required Python packages
│── 📂 models/              # (Models downloaded from Google Drive)
│── 📂 data/                # (Optional: Sample images)
🔗 Live Demo
🌍 Try the Live App Here:
🔗 AI-Powered Weapons Classification

📌 How the Models Work
VGG19: A deep learning model pretrained on ImageNet, used for feature extraction.
Custom CNN: A lightweight model trained from scratch for classification.
Prediction Comparison: The app displays classifications from both models side by side.
📥 Downloading Models from Google Drive
If models are missing, they will be automatically downloaded from Google Drive.

python
Copy
Edit
models = {
    "vgg19_multilabel_model_sample.h5": "1D4fx9bD9t4b-CE2sJDMm0wLLk9g1hyIQ",
    "binary_cnn_model.h5": "15X2VokJ3l-JiQtzi0OjoG7_1YMrRIfBi"
}
🛠️ Built With
Python
TensorFlow & Keras
Streamlit
NumPy, Pandas, PIL
gdown (for model downloads)
📧 Contact
📩 Saikumar – LinkedIn

🚀 Star the repo ⭐ and try out the live demo!

yaml
Copy
Edit

---

### **✅ Next Steps**
1️⃣ **Save this file as `README.md` in your project folder**  
2️⃣ **Push it to GitHub**  
```sh
git add README.md
git commit -m "Added README file with project details"
git push origin main
