
# 🎓 Learning-Style-Predictor

A Streamlit-based web app that predicts a student's **learning style** — Visual, Auditory, Kinesthetic, or Reading/Writing — using a machine learning model (Random Forest Classifier) trained on study behavior data.

https://learning-style-predictor-9iawobcgmd4sduxl5exrru.streamlit.app/

---

## 📌 Features

- 🔍 Predicts a student's learning style based on their inputs
- 📊 Displays prediction confidence visually
- 💡 Provides personalized study tips
- 🧠 Uses Random Forest for classification
- 💾 Saves model using `.pkl` for quick deployment
- 🎨 Built with Streamlit for a clean user interface

---

## 🚀 Live Demo

https://learning-style-predictor-9iawobcgmd4sduxl5exrru.streamlit.app/

---

## 🧠 Learning Styles Predicted

| Learning Style        | Description |
|-----------------------|-------------|
| 🗾 Visual              | Learns best with images, diagrams, videos |
| 🎧 Auditory           | Prefers listening to lectures and discussions |
| 🏃 Kinesthetic        | Learns by doing, touching, or movement |
| 📖 Reading/Writing    | Prefers reading and writing as learning tools |

---

## 🗂️ Project Structure
├── streamlit_app.py  
├── learning_style_data.csv # Cleaned dataset  
├── train_model.py # Trained model  
├── encoders.pkl  
├── model.pkl  
├── requirement.txt # Dependencies  
├── .gitignore  
└── README.md

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ShrutiTate/Learning-Style-Predictor.git
cd Learning-Style-Predictor
```
2. Install Dependencies
```bash
pip install -r requirement.txt

```
3. Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

📘 Dataset
The dataset contains categorized study behaviors for students. It has been manually reviewed and slightly shuffled to maintain realistic variability while avoiding overfitting.

🤝 Contribution
Feel free to fork this repo and raise a pull request with improvements, or create an issue for suggestions and bugs.

👩‍💻 Author
Shruti Tate
https://github.com/ShrutiTate

"# Learning-Style-Prediction-app" 
