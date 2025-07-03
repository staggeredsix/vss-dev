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
    """Extract audio and all frames using ffmpeg."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required but not installed")

    video_path = os.fspath(video_path)

    tmpdir = tempfile.mkdtemp()
    audio_path = os.path.join(tmpdir, "audio.wav")

    frames_dir = os.path.join(tmpdir, "frames")
    os.makedirs(frames_dir, exist_ok=True)


    try:
        res = subprocess.run(
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
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        subprocess.run(
            [
                "ffmpeg",
                "-i",
                video_path,
                os.path.join(frames_dir, "frame_%05d.jpg"),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        msg = e.stderr.strip() if e.stderr else str(e)
        raise RuntimeError(f"ffmpeg failed: {msg}") from e

    frame_paths = sorted(
        str(p) for p in Path(frames_dir).glob("frame_*.jpg")
    )
    return audio_path, frame_paths, tmpdir


class GradioApp:
    def __init__(self, ollama_url: str):
        self.pipeline = LocalPipeline(ollama_url)
        self.transcript = ""
        self.frames: list[str] = []
        self.captions: list[str] = []

    def process_upload(self, video_file):
        if video_file is None:
            return "", ""
        audio, self.frames, tmp = extract_media(video_file)
        self.transcript = self.pipeline.transcribe(audio)
        self.captions = [self.pipeline.caption(f) for f in self.frames]
        caption = self.captions[0] if self.captions else ""
        # cleanup tmpdir later
        return self.transcript, caption

    def answer(self, question, history):
        if not self.transcript:

            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": "Upload a video first."})
            return history
        response = self.pipeline.answer(question, self.transcript, self.captions)

        ts_match = re.search(r"(\d{1,2}:\d{2})", response)
        if ts_match:
            mmss = ts_match.group(1)
            m, s = map(int, mmss.split(":"))
            sec = m * 60 + s
            link = f'<a href="#" onclick="document.getElementById(\'video\').currentTime={sec}; return false;">{mmss}</a>'
            response = response.replace(mmss, link)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": response})
        return history

    def launch(self):
        with gr.Blocks() as demo:
            state = gr.State([])
            video = gr.Video(label="Video", elem_id="video")
            transcript_box = gr.Textbox(label="Transcript")
            caption_box = gr.Textbox(label="Caption")
            chatbot = gr.Chatbot(type="messages")
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
