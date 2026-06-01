import sys
import os


disease_info = {
    "Normal ECG":
        "The ECG appears normal with no major abnormalities detected.",

    "Abnormal Heartbeat":
        "An irregular heartbeat pattern was detected. Medical evaluation is recommended.",

    "Myocardial Infarction":
        "Myocardial Infarction (heart attack) occurs when blood flow to the heart muscle is blocked, reducing oxygen supply to the heart muscle.",

    "Post MI History":
        "The ECG shows patterns that may be associated with a previous heart attack."
}

recommendations = {
    "Normal ECG":
        "Maintain a healthy lifestyle, regular exercise, balanced diet, and routine health checkups.",

    "Abnormal Heartbeat":
        "Consult a cardiologist for further evaluation and follow-up testing.",

    "Myocardial Infarction":
        "Seek immediate medical attention and consult a healthcare professional as soon as possible.",

    "Post MI History":
        "Regular follow-up with a cardiologist and adherence to prescribed treatment is recommended."
}


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(PROJECT_ROOT)



import streamlit as st
import os
from PIL import Image

from backend.inference import predict_image

st.set_page_config(
    page_title="Heart Disease AI",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Heart Disease Detection System")

st.write(
    "Upload an ECG image and the AI model will analyze it."
)

uploaded_file = st.file_uploader(
    "Upload ECG Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    os.makedirs("uploads", exist_ok=True)

    image_path = os.path.join(
        "uploads",
        uploaded_file.name
    )

    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    image = Image.open(image_path)

    st.image(
        image,
        caption="Uploaded ECG Image",
        use_container_width=True
    )

    if st.button("Analyze ECG"):

        prediction, confidence = predict_image(image_path)

        st.success(f"Prediction: {prediction}")

        st.info(f"Confidence: {confidence:.2f}%")

        if confidence < 80:
            st.warning(
                "Low confidence prediction. Please consult a healthcare professional."
            )

        st.subheader("Disease Explanation")
        st.write(disease_info.get(
            prediction,
            "No information available."
        ))

        st.subheader("Healthcare Recommendations")
        st.info(recommendations.get(
            prediction,
            "No recommendation available."
        ))

        st.warning(
            "This AI system is developed for educational and research purposes only. "
            "It should not be used as a substitute for professional medical diagnosis."
        )

    st.subheader("Medical Chatbot")

    question = st.text_input(
        "Ask a question about the ECG result:"
    )

    if question:

        q = question.lower()

        if "heart attack" in q:
            st.write(
                "A heart attack (Myocardial Infarction) occurs when blood flow to part of the heart is blocked."
            )

        elif "normal ecg" in q:
            st.write(
                "A normal ECG generally indicates normal electrical activity of the heart."
            )

        elif "abnormal heartbeat" in q:
            st.write(
                "An abnormal heartbeat may indicate an arrhythmia."
            )

        elif "recommendation" in q:
            st.write(
                "Please follow the healthcare recommendations shown above."
            )

        else:
            st.write(
                "Please consult a healthcare professional for detailed medical advice."
            )