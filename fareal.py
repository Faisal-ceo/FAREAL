import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sounddevice as sd
import numpy as np
import wave
import io
import speech_recognition as sr
import pyttsx3
from datetime import datetime

genai.configure(api_key="HHHHHHHHHHHHHHHHHHHHHHHHH")

st.set_page_config(
    page_title="🎤 AI-Powered Voice-to-Content Generator",
    page_icon="🤖",
    layout="wide"
)

st.title("🎤 AI-Powered Voice-to-Content Generator")

def extract_text_from_pdf(pdf_file):
    try:
        reader = PdfReader(pdf_file)
        text = "".join(page.extract_text() for page in reader.pages)
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def classify_and_match(user_input, pdf_text):
    documents = [pdf_text, user_input]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity_score > 0.5, similarity_score

def text_to_speech(text):
    engine = pyttsx3.init()
    engine.save_to_file(text, "output_audio.mp3")
    engine.runAndWait()
    return "output_audio.mp3"

uploaded_file = st.file_uploader("📂 Upload PDF File:", type="pdf")
pdf_content = ""
if uploaded_file:
    pdf_content = extract_text_from_pdf(uploaded_file)
    st.success("📄 PDF content loaded successfully.")

st.markdown("### 🎙️ Record Audio")
duration = st.slider("⏳ Select Recording Duration (seconds):", 1, 10, 5)

if st.button("🔴 Start Recording"):
    st.write("🎤 Recording...")
    fs = 44100
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.int16)
    sd.wait()
    filename = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(recording.tobytes())

    st.success(f"✅ Recording saved as {filename}")
    st.audio(filename, format="audio/wav")

    with open(filename, "rb") as audio_file:
        st.download_button(label="📥 Download Recording", data=audio_file, file_name=filename, mime="audio/wav")

    st.markdown("### 📝 Convert Speech to Text")
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)
        try:
            transcribed_text = recognizer.recognize_google(audio, language="ar")
            st.text_area("🔍 Transcribed Text:", value=transcribed_text, height=200)
        except sr.UnknownValueError:
            st.error("❌ Speech not recognized. Try again.")
            transcribed_text = ""
        except sr.RequestError as e:
            st.error(f"⚠️ Speech recognition service error: {e}")
            transcribed_text = ""

    if st.button("⚡ Generate Content"):
        if not pdf_content:
            st.warning("❗ Please upload a PDF file first.")
        elif not transcribed_text.strip():
            st.warning("❗ Please record and transcribe audio first.")
        else:
            with st.spinner("⏳ Processing..."):
                try:
                    full_prompt = f"📄 PDF Content:\n{pdf_content}\n\n🎤 Transcribed Speech:\n{transcribed_text}"
                    match, score = classify_and_match(transcribed_text, pdf_content)
                    if match:
                        st.success(f"✅ Matched with PDF content ({score:.2f} similarity).")
                    else:
                        st.warning(f"⚠️ No strong match found. Similarity score: {score:.2f}.")

                    model = genai.GenerativeModel("gemini-1.5-flash")
                    response = model.generate_content(full_prompt)
                    audio_path = text_to_speech(response.text)
                    st.audio(audio_path)
                    st.markdown(f'<div class="response-box">{response.text}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"⚠️ Error processing request: {e}")
