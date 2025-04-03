import streamlit as st
import requests

# Backend API URL
BACKEND_URL = "http://localhost:8000/analyze"  # Ensure FastAPI is running

# Streamlit UI
st.title("🐍 Code Bug Detector & Fixing")
st.write("🔍 Paste your Python code below, and I will detect & fix bugs for you!")

# Code input box
code = st.text_area("Enter your Python code here:", height=200)

if st.button("Analyze Code"):
    if code.strip():
        # Send request to backend
        response = requests.post(BACKEND_URL, json={"code_snippet": code})

        if response.status_code == 200:
            result = response.json()["analysis"]
            st.subheader("Code Analysis & Fix:")
            st.text_area("Output:", result, height=300)
        else:
            st.error("Error: Unable to connect to the backend!")
    else:
        st.warning("Please enter some code to analyze.")

