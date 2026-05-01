import streamlit as st
from model import predict_review

st.title("🕵️ Fake Review Detector")

st.write("Enter a product review below:")

user_input = st.text_area("Review:")

if st.button("Check"):
    if user_input.strip() != "":
        result = predict_review(user_input)
        st.subheader("Result:")
        st.write(result)
    else:
        st.warning("Please enter a review")