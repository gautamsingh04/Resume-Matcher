import streamlit as st

# Set config first (must be first Streamlit call)
st.set_page_config(page_title="📄 Resume Matcher", layout="centered")

import pickle
import numpy as np
import re
import PyPDF2  # ✅ safer alternative to fitz
from sklearn.metrics.pairwise import cosine_similarity

# --- Load model and dataset ---
@st.cache_resource
def load_model_data():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("dataset.pkl", "rb") as f:
        dataset = pickle.load(f)
    return model, dataset

model, dataset = load_model_data()

# --- Extract text from PDF ---
def extract_resume_text(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        return ""

# --- Recommendation logic ---
def get_recommendation(resume_text):
    resume = resume_text.lower()
    resume = re.sub(r"[^a-z0-9\s]", "", resume)
    words = resume.split()
    word_vec = [model.wv[word] for word in words if word in model.wv]
    final_res = np.mean(word_vec, axis=0) if word_vec else np.zeros(model.vector_size)

    recommendations = []
    for idx, emb in enumerate(dataset["embeddings"]):
        score = cosine_similarity([final_res], [emb])[0][0]
        recommendations.append((score, int(idx)))

    recommendations.sort(reverse=True, key=lambda x: x[0])
    seen_titles = set()
    results = []

    for score, idx in recommendations[:100]:
        title = dataset.iloc[idx]["title"]
        if title not in seen_titles and score >= 0.80:
            seen_titles.add(title)
            results.append((title, score))
            if len(results) >= 5:
                break

    return results if results else [("Nothing can be done, sorry.", 0.0)]

# --- Streamlit UI ---
st.title("📄 Resume Matcher")
st.write("Upload your resume in PDF format and get top job title suggestions based on your skills.")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if uploaded_file:
    with st.spinner("🔍 Processing resume..."):
        resume_text = extract_resume_text(uploaded_file)
        if not resume_text:
            st.error("❌ Could not extract text from the uploaded PDF. Please upload a valid resume.")
        else:
            recommendations = get_recommendation(resume_text)
            st.success("✅ Matching complete!")
            st.markdown("### 🎯 Top Job Recommendations:")
            for title, score in recommendations:
                st.markdown(f"- **{title}** — Similarity Score: `{score:.2f}`")
