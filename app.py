import streamlit as st
import numpy as np
import joblib
from PIL import Image
from disease_model import predict_disease


model = joblib.load("crop_model.pkl")

st.set_page_config(page_title="Smart Agriculture AI", layout="wide")

# ---------------- THEME TOGGLE ----------------
theme = st.sidebar.toggle("🌙 Dark Mode")

if theme:
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1c1c1c, #2c2c2c);
    }
    h1,h2,h3,p,label {
        color: white;
    }
    .stNumberInput input {
        background-color: #2c2c2c;
        color: white;
    }
    .stButton>button {
        background: linear-gradient(45deg, #00c6ff, #0072ff);
        color: white;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-color:white;
        background-size: 400% 400%;
        animation: gradientMove 10s ease infinite;
    }
    @keyframes gradientMove {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    [data-testid="stSidebar"] {
        background-color: #dff5e1;
    }
    h1 {
        color: #2e7d32;
        text-align:center;
    }
    .stButton>button {
        background: linear-gradient(45deg, #43a047, #66bb6a);
        color: white;
        border-radius: 10px;
        transition:0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.1);
    }
    .card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        transition: 0.3s;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .card:hover {
        transform: scale(1.1);
        background: #e8f5e9;
    }
    </style>
    """, unsafe_allow_html=True)


# ---------------- SIDEBAR ----------------
feature = st.sidebar.radio("Select Feature", [
    "Home",
    "Crop Recommendation",
    "Fertilizer Suggestion",
    "Profit Prediction",
    "Disease Prediction",
    "AI Chatbot"
])

# ---------------- HOME ----------------
if feature == "Home":

    st.markdown("""
    <div style="
    position:relative;
    height:90vh;
    border-radius:20px;
    overflow:hidden;
    background-image:url('https://images.unsplash.com/photo-1500382017468-9049fed747ef');
    background-size:cover;
    background-position:center;
    ">

    <div style="
    position:absolute;
    top:0;
    left:0;
    width:100%;
    height:100%;
    background:rgba(0,0,0,0.4);
    "></div>

    <div style="
    position:absolute;
    top:50%;
    left:50%;
    transform:translate(-50%,-50%);
    text-align:center;
    color:white;
    width:90%;
    ">

    <h1 style="
    font-size:75px;
    font-weight:900;
    font-color:black;
    text-shadow:4px 4px 20px black;
    ">

    🌾 WELCOME TO SMART AI AGRICULTURE

    </h1>

    <p style="
    font-size:50px;
    font-weight:bold;
    text-shadow:2px 2px 10px black;
    ">

    This Application can Provide an AI-Powered Crop Recommendation,
    Disease Detection and Smart Farming Solutions with an AI Chatbots

    </p>

    </div>

    </div>
    """, unsafe_allow_html=True)
# ---------------- CROP ----------------
elif feature == "Crop Recommendation":

    st.subheader("Enter Soil Data")

    col1, col2 = st.columns(2)

    with col1:
        N = st.number_input("Nitrogen", 0.0)
        P = st.number_input("Phosphorus", 0.0)
        K = st.number_input("Potassium", 0.0)

    with col2:
        temp = st.number_input("Temperature", 0.0)
        humidity = st.number_input("Humidity", 0.0)
        ph = st.number_input("pH", 0.0)
        rainfall = st.number_input("Rainfall", 0.0)

    if st.button("Predict Crop"):

        data = np.array([[N,P,K,temp,humidity,ph,rainfall]])
        probs = model.predict_proba(data)[0]
        top3 = np.argsort(probs)[-3:][::-1]
        crops = model.classes_

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"<div class='card'>🥇<br>{crops[top3[0]]}<br>{round(probs[top3[0]]*100,2)}%</div>", unsafe_allow_html=True)

        with col2:
            st.markdown(f"<div class='card'>🥈<br>{crops[top3[1]]}<br>{round(probs[top3[1]]*100,2)}%</div>", unsafe_allow_html=True)

        with col3:
            st.markdown(f"<div class='card'>🥉<br>{crops[top3[2]]}<br>{round(probs[top3[2]]*100,2)}%</div>", unsafe_allow_html=True)

# ---------------- FERTILIZER ----------------
elif feature == "Fertilizer Suggestion":

    N = st.number_input("Nitrogen", 0.0)
    P = st.number_input("Phosphorus", 0.0)
    K = st.number_input("Potassium", 0.0)

    if st.button("Suggest"):

        if N < 50:
            st.success("Use Nitrogen fertilizer")
        elif P < 30:
            st.success("Use Phosphorus fertilizer")
        elif K < 30:
            st.success("Use Potassium fertilizer")
        else:
            st.success("Soil balanced")

# ---------------- PROFIT ----------------
elif feature == "Profit Prediction":

    price = st.number_input("Price", 0.0)
    cost = st.number_input("Cost", 0.0)
    yield_amt = st.number_input("Yield", 0.0)

    if st.button("Calculate"):
        st.success(f"Profit: ₹{(price*yield_amt)-cost}")

# ---------------- DISEASE ----------------
elif feature == "Disease Prediction":

    uploaded = st.file_uploader("Upload Leaf Image")

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, width=300)

        result = predict_disease(img)
        st.success(f"Disease: {result}")

