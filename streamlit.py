import streamlit as st
import requests

st.title('Dynamic Learning Pathway with RAG')

course_id = st.text_input("Enter the course ID:")
actual_time = st.number_input(
    "Enter the actual time taken for the course (hours):")
exam_score = st.number_input("Enter the exam score (percentage):")

if st.button('Evaluate'):
    if not course_id:
        st.error("Course ID is required.")
    elif actual_time <= 0:
        st.error("Actual time must be greater than 0.")
    elif exam_score < 0 or exam_score > 100:
        st.error("Exam score must be between 0 and 100.")
    else:
        data = {
            "course_id": course_id,
            "actual_time": actual_time,
            "exam_score": exam_score
        }
        response = requests.post('http://127.0.0.1:5000/evaluate', json=data)
        if response.status_code == 200:
            result = response.json()
            st.write(f"Evaluation Result: {result['result']}")
            st.write(f"Recommended Next Course: {result['next_course']}")
        else:
            st.error(
                "Failed to evaluate the progress. Please check the input data and try again.")
