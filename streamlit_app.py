import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# 🌟 Page Configuration
st.set_page_config(page_title="📚 Learning Styles Predictor", page_icon="🎓", layout="wide")

# 🔄 Load Dataset
@st.cache_data
def load_data():
    try:
        return pd.read_csv("learning_style_data.csv")
    except FileNotFoundError:
        st.error("⚠️ Error: Dataset file not found!")
        return None

df = load_data()

# 🧠 Load Trained Model and Encoders
try:
    model = joblib.load("model.pkl")
    encoders = joblib.load("encoders.pkl")
except FileNotFoundError:
    st.error("⚠️ Error: model.pkl or encoders.pkl not found!")
    model = None
    encoders = None

# 🧩 Learning Style Mapping
style_mapping = {
    0: "Auditory Learner 🎧",
    1: "Kinesthetic Learner 🏃",
    2: "Reading/Writing Learner 📖",
    3: "Visual Learner 🗾️"
}

# 💡 Study Tips
study_tips = {
    "Visual Learner 🗾️": [
        "📌 Use mind maps and flowcharts to structure notes.",
        "🎨 Try apps like Canva, Miro, or Sketchbook for visual note-taking.",
        "📺 Watch explainer videos or tutorials for concept clarity."
    ],
    "Auditory Learner 🎧": [
        "🎥 Record and listen to lectures or podcasts.",
        "🗣️ Use text-to-speech tools for reading notes aloud.",
        "💬 Engage in discussions or debates to reinforce learning."
    ],
    "Kinesthetic Learner 🏃": [
        "🔬 Perform hands-on experiments or role-playing exercises.",
        "⚡ Use real-world applications for better retention.",
        "🏃 Take short breaks and move while studying."
    ],
    "Reading/Writing Learner 📖": [
        "📝 Summarize concepts using structured notes.",
        "🕊️ Write explanations in your own words for better understanding.",
        "📖 Use bullet points, lists, and outlines for retention."
    ]
}

# 🎭 Sidebar Navigation
with st.sidebar:
    st.subheader("📌 Navigation")
    choice = st.radio("Choose a section:", ["Home", "Dataset", "Predictor"])
    st.write("🔍 Machine Learning Model")

# 🏠 Home Page
if choice == "Home":
    st.title("🎓 Welcome to the Learning Styles Predictor!")
    st.markdown("""
    <div style="font-size:22px; line-height:1.5;"><br><br>
    🌟 <b>Discover how you learn best</b><br>
    📊 <b>Analyze study habits effectively</b><br>
    </div>
    """, unsafe_allow_html=True)

# 📂 Dataset Overview
elif choice == "Dataset":
    st.title("📊 Dataset Overview")
    if df is not None:
        st.dataframe(df, use_container_width=True)
        st.success("✅ Dataset loaded successfully!")

# 🔮 Predictor Page
elif choice == "Predictor":
    st.title("🔎 Predict Your Learning Style")

    if df is None or model is None or encoders is None:
        st.error("⚠️ Required files missing. Please check dataset, model and encoders.")
    else:
        with st.expander("📘 Fill in your study behavior", expanded=True):
            col1, col2 = st.columns(2)
            user_input = {}
            input_cols = df.columns[:-1]  # exclude target

            for i, col in enumerate(input_cols):
                options = sorted(df[col].dropna().unique().tolist())
                with (col1 if i % 2 == 0 else col2):
                    user_input[col] = st.selectbox(f"{col}", options)

        # Encode user input
        input_df = pd.DataFrame([user_input])
        for col in input_df.columns:
            if col in encoders:
                input_df[col] = encoders[col].transform(input_df[col])

        if st.button("✨ Analyze My Learning Style"):
            with st.spinner("🧠 Processing your study patterns..."):
                prediction_proba = model.predict_proba(input_df)[0]
                prediction = model.predict(input_df)[0]
                prediction_label = style_mapping.get(prediction, "Unknown")

            confidence = prediction_proba[prediction] * 100

            st.success(f"🎓 **Your Learning Style:** {prediction_label}")
            st.info(f"🧮 Confidence Level: **{confidence:.2f}%**")

            # 📊 Horizontal Bar Chart
            st.markdown("### 📈 Prediction Confidence")
            chart_df = pd.DataFrame({
                "Learning Style": [style_mapping.get(cls, str(cls)) for cls in model.classes_],
                "Probability": prediction_proba
            })

            fig = px.bar(
                chart_df,
                x="Probability",
                y="Learning Style",
                orientation="h",
                color="Learning Style",
                color_discrete_sequence=px.colors.qualitative.Set2,
                height=320
            )
            fig.update_layout(
                xaxis_title="Confidence",
                yaxis_title="",
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=14)
            )
            st.plotly_chart(fig, use_container_width=True)

            # 💡 Tips
            st.markdown("### ✨ Personalized Study Tips")
            for tip in study_tips.get(prediction_label, ["No tips available."]):
                st.markdown(f"""
                <div style='background-color:#f7f7f7; padding:10px; border-radius:8px; margin-bottom:10px; color:#333; font-weight:600;'>{tip}</div>
                """, unsafe_allow_html=True)
