import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_extras.add_vertical_space import add_vertical_space

def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text + "\n"
    return text

def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    return cosine_similarities

# Streamlit UI
st.set_page_config(page_title="AI Resume Ranking", layout="wide")
st.markdown(
    """
    <style>
        .main {
            background-color: #f4f4f4;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
            padding: 10px 20px;
        }
        .stTextInput>div>div>input {
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title & Instructions
st.title("🚀 AI Resume Ranking App")
st.write("### Rank resumes based on job description using AI-powered matching!")
add_vertical_space(2)

# Job description input
st.subheader("📝 Job Description")
job_description = st.text_area("Enter the job description", height=150)

# File uploader
st.subheader("📂 Upload Resumes (PDF Only)")
uploaded_files = st.file_uploader("Upload multiple PDF resumes", type=["pdf"], accept_multiple_files=True)

if uploaded_files and job_description:
    st.subheader("📊 Ranking Results")
    resumes = []
    
    with st.spinner("Processing resumes..."):
        for file in uploaded_files:
            text = extract_text_from_pdf(file)
            resumes.append(text)
    
    scores = rank_resumes(job_description, resumes)
    results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Score": scores})
    results = results.sort_values(by="Score", ascending=False)
    
    # Display results in a better format
    st.dataframe(results.style.format({"Score": "{:.2f}"}).set_properties(**{"background-color": "#f9f9f9", "color": "black"}))
    
    st.success("✅ Ranking Complete!")
