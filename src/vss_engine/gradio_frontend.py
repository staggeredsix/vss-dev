import argparse
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import gradio as gr

# Allow importing pipeline when executed from repository root
sys_path = Path(__file__).resolve().parent
import sys
sys.path.append(str(sys_path))
from pipeline import LocalPipeline


def extract_media(video_path: str | os.PathLike):
    """Extract audio and a representative frame using ffmpeg."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required but not installed")

    video_path = os.fspath(video_path)
    tmpdir = tempfile.mkdtemp()
    audio_path = os.path.join(tmpdir, "audio.wav")
    frame_path = os.path.join(tmpdir, "frame.jpg")

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                video_path,
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                audio_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        subprocess.run(
            [
                "ffmpeg",
                "-i",
                video_path,
                "-vf",
                "select=eq(n\\,0)",
                "-vframes",
                "1",
                frame_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed: {e}") from e

    return audio_path, frame_path, tmpdir


class GradioApp:
    def __init__(self, ollama_url: str):
        self.pipeline = LocalPipeline(ollama_url)
        self.transcript = ""

    def process_upload(self, video_file):
        if video_file is None:
            return "", ""
        audio, frame, tmp = extract_media(video_file)
        self.transcript = self.pipeline.transcribe(audio)
        caption = self.pipeline.caption(frame)
        # cleanup tmpdir later
        return self.transcript, caption

    def answer(self, question, history):
        if not self.transcript:
            return history + [[question, "Upload a video first."],]
        response = self.pipeline.answer(question, self.transcript)
        ts_match = re.search(r"(\d{1,2}:\d{2})", response)
        if ts_match:
            mmss = ts_match.group(1)
            m, s = map(int, mmss.split(":"))
            sec = m * 60 + s
            link = f'<a href="#" onclick="document.getElementById(\'video\').currentTime={sec}; return false;">{mmss}</a>'
            response = response.replace(mmss, link)
        history.append([question, response])
        return history

    def launch(self):
        with gr.Blocks() as demo:
            state = gr.State([])
            video = gr.Video(label="Video", elem_id="video")
            transcript_box = gr.Textbox(label="Transcript")
            caption_box = gr.Textbox(label="Caption")
            chatbot = gr.Chatbot()
            question = gr.Textbox(label="Question")
            send = gr.Button("Ask")

            video.upload(self.process_upload, inputs=video, outputs=[transcript_box, caption_box])
            send.click(self.answer, inputs=[question, chatbot], outputs=chatbot)
        demo.launch()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    args = parser.parse_args()
    app = GradioApp(args.ollama_url)
    app.launch()


if __name__ == "__main__":
    main()
