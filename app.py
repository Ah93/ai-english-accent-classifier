import os
import sys
import requests
import tempfile
import streamlit as st
from moviepy.editor import VideoFileClip
from speechbrain.inference.classifiers import EncoderClassifier
st.write(f"Python version: {sys.version}")

st.set_page_config(page_title="Accent Classifier", page_icon="🗣️", layout="centered")

# Custom CSS to change background color and style input box
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f0f0;
    }
    input[type="text"] {
        background-color: white;
        color: black;
        border: 1px solid #ccc;
        padding: 0.5rem;
        border-radius: 5px;
    }
    .stTextInput > div > div > input {
        background-color: white !important;
        color: black !important;
        border: 1px solid #ccc !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

MODEL_ID = "Jzuluaga/accent-id-commonaccent_ecapa"

def download_video(url, output_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        if not os.path.exists(output_path) or os.path.getsize(output_path) < 1024:
            raise Exception("❌ Video download failed or file too small.")
    else:
        raise Exception("❌ Failed to download video.")

def extract_audio(video_path, audio_path):
    clip = VideoFileClip(video_path)
    audio = clip.audio
    audio.write_audiofile(audio_path, fps=16000, nbytes=2, codec='pcm_s16le', ffmpeg_params=["-ac", "1"])
    clip.close()
    audio.close()

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    classifier = EncoderClassifier.from_hparams(
        source=MODEL_ID,
        savedir=os.path.join(os.getcwd(), "accent-id-model")
    )
    return classifier

def classify_accent(audio_path, classifier):
    audio_path_clean = os.path.abspath(audio_path).replace('\\', '/')
    if not os.path.exists(audio_path_clean):
        raise FileNotFoundError(f"Audio file not found: {audio_path_clean}")
    out_prob, score, index, label = classifier.classify_file(audio_path_clean)
    return label, round(score.item() * 100, 2)

# ---------------- UI ----------------
st.title("🗣️ Accent Classifier from Video")
st.markdown("Paste a direct **video URL (MP4)** and then press **Enter** or click **Identify the Accent**.")

with st.form("url_form", clear_on_submit=False):
    video_url = st.text_input("🔗 Video URL", placeholder="https://...")
    submitted = st.form_submit_button("🗣️ Identify the Accent")

if submitted:
    if not video_url:
        st.warning("⚠️ Please enter a video URL.")
    else:
        try:
            if "dropbox.com" in video_url and "raw=1" not in video_url:
                video_url = video_url.replace("dl=0", "raw=1").replace("?dl=0", "?raw=1")

            with st.spinner("🔄 Downloading and processing video..."):
                with tempfile.TemporaryDirectory() as tmpdir:
                    video_path = os.path.join(tmpdir, "input_video.mp4")
                    audio_path = os.path.join(tmpdir, "output_audio.wav")

                    download_video(video_url, video_path)
                    extract_audio(video_path, audio_path)
                    classifier = load_model()
                    label, confidence = classify_accent(audio_path, classifier)

            st.success("✅ Accent classified successfully!")
            st.markdown(f"### 🎯 Prediction: **{label}**")
            st.markdown(f"🧠 Confidence: **{confidence}%**")
            st.info(f"The speaker's accent is predicted to be **{label}** with **{confidence}%** confidence.")
        except Exception as e:
            st.error(f"❌ Error: {e}")
