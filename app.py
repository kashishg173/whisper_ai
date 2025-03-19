
import streamlit as st
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import os

MODELS = {
    "Tiny": "openai/whisper-tiny",
    "Base": "openai/whisper-base",
    "Small": "openai/whisper-small",
    "Medium": "openai/whisper-medium",
    "Large-v3": "openai/whisper-large-v3"
}

@st.cache_resource
def load_model(model_name):
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    return processor, model

def transcribe_audio(audio_path, processor, model):
    audio_data, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
    
    inputs = processor(
        audio_data,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features
    
    predicted_ids = model.generate(inputs)
    transcription = processor.decode(predicted_ids[0], skip_special_tokens=True)
    
    detected_language = processor.tokenizer.decode(predicted_ids[0][0])
    return transcription, detected_language

st.title("Audio to Text Converter")
st.write("Upload an audio file (MP3) to transcribe using Whisper AI")

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox(
        "Select Model",
        options=list(MODELS.keys()),
        index=2  
    )

audio_file = st.file_uploader("Choose audio file", type=["mp3"])

if audio_file:
    file_ext = audio_file.name.split(".")[-1]
    temp_path = f"temp_audio.{file_ext}"
    
    with open(temp_path, "wb") as f:
        f.write(audio_file.getbuffer())
    
    st.audio(temp_path)
    
    if st.button("Transcribe Audio"):
        with st.spinner("Processing..."):
            try:
                processor, model = load_model(MODELS[model_name])
                transcription, detected_language = transcribe_audio(temp_path, processor, model)
            
                st.success("Transcription Complete")
                st.write(f"Detected Language: **{detected_language}**")
                st.text_area("Transcript", transcription, height=200)
            except Exception as e:
                st.error(f"Error: {str(e)}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)