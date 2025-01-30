import streamlit as st
import sounddevice as sd
import wave
from datetime import datetime
from PyPDF2 import PdfReader
from openai import OpenAI
import os

api_key = open('secret.txt').read()
client = OpenAI(api_key=api_key)

st.set_page_config(page_title="📂 PDF & Voice Analysis Tool", page_icon="🤖", layout="wide")
st.title("📂 PDF & Voice Analysis Tool")
st.write("Upload a PDF file and ask questions via voice for AI analysis.")

if 'pdf_content' not in st.session_state:
    st.session_state.pdf_content = None

uploaded_file = st.file_uploader("📥 Upload your PDF file:", type="pdf")
if uploaded_file:
    try:
        reader = PdfReader(uploaded_file)
        pdf_content = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        st.session_state.pdf_content = pdf_content
        
        st.success("📄 PDF content loaded successfully!")
        with st.expander("View PDF Content"):
            st.text_area("PDF Content:", value=pdf_content, height=200)
    except Exception as e:
        st.error(f"Error processing PDF: {e}")

if st.session_state.pdf_content:
    st.markdown("### 🎤 Record Your Question")
    duration = st.slider("⏳ Recording Duration (seconds):", 1, 30, 5)

    if st.button("🎙️ Start Recording"):
        st.write("🎤 Recording in progress...")
        fs = 44100  
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()  

        filename = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(fs)
            wf.writeframes(recording.tobytes())
        
        st.success("✅ Recording completed!")
        st.audio(filename, format="audio/wav")

        st.markdown("### 🎯 Your Question")
        with open(filename, "rb") as audio_file:
            try:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
                question = transcription.text
                st.info(f"🔍 Your question: {question}")

                st.markdown("### 🤖 AI Analysis")
                with st.spinner("Processing your question..."):
                    response = client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that analyzes documents and answers questions about them."},
                            {"role": "user", "content": f"""Here is the document content:

{st.session_state.pdf_content}

Question: {question}

Please provide a detailed answer based on the document content."""}
                        ]
                    )
                    st.markdown("### 🧠 Answer:")
                    st.write(response.choices[0].message.content)
            except Exception as e:
                st.error(f"❌ Error: {e}")

        try:
            os.remove(filename)
        except Exception as e:
            st.warning(f"Could not remove temporary audio file: {e}")
else:
    st.info("👆 Please upload a PDF file first to proceed with voice questions.")
