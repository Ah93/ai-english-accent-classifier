import os
import requests
import tempfile
from moviepy.editor import VideoFileClip
from speechbrain.inference.classifiers import EncoderClassifier

MODEL_ID = "Jzuluaga/accent-id-commonaccent_ecapa"

def download_video(url, output_path):
    print("[*] Downloading video...")
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
    print("[*] Extracting audio using MoviePy...")
    clip = VideoFileClip(video_path)
    audio = clip.audio
    # Write audio as 16kHz mono wav
    audio.write_audiofile(audio_path, fps=16000, nbytes=2, codec='pcm_s16le', ffmpeg_params=["-ac", "1"])
    clip.close()
    audio.close()

def load_model():
    print("[*] Loading accent classification model...")
    classifier = EncoderClassifier.from_hparams(
        source=MODEL_ID,
        savedir=os.path.join(os.getcwd(), "accent-id-model")
    )
    return classifier

def classify_accent(audio_path, classifier):
    audio_path_clean = os.path.abspath(audio_path).replace('\\', '/')
    print(f"✅ Using audio file: {audio_path_clean}")

    if not os.path.exists(audio_path_clean):
        raise FileNotFoundError(f"Audio file not found: {audio_path_clean}")

    out_prob, score, index, label = classifier.classify_file(audio_path_clean)
    return label, round(score.item() * 100, 2)

def analyze_accent_from_video_url(video_url):
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "input_video.mp4")
        audio_path = os.path.join(tmpdir, "output_audio.wav")

        download_video(video_url, video_path)
        print(f"📁 Downloaded to: {video_path}")
        print(f"🎧 Will extract audio to: {audio_path}")

        extract_audio(video_path, audio_path)

        classifier = load_model()
        label, confidence = classify_accent(audio_path, classifier)

        return {
            "accent": label,
            "confidence": confidence,
            "explanation": f"The speaker's accent is predicted to be **{label}** with {confidence}% confidence."
        }

if __name__ == "__main__":
    video_url = input("🔗 Enter direct video URL (mp4): ").strip()
    if not video_url:
        print("❌ No URL provided.")
    else:
        try:
            result = analyze_accent_from_video_url(video_url)
            print("\n🎯 Accent Classification Result:")
            for key, value in result.items():
                print(f"{key.capitalize()}: {value}")
        except Exception as e:
            print(f"\n❌ Error: {e}")
