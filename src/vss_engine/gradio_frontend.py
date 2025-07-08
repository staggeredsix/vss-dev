import argparse
import os
import re
import json
import hashlib
import time

import shutil

import subprocess
import tempfile
from pathlib import Path
import sys

import gradio as gr

# Allow importing pipeline when executed from repository root
sys_path = Path(__file__).resolve().parent

sys.path.append(str(sys_path))
from pipeline import LocalPipeline  # noqa: E402


def extract_media(video_path: str | os.PathLike):

    """Extract audio (if present) and all frames using ffmpeg.

    Returns a tuple ``(audio_path, frame_paths, fps, tmpdir)`` where ``audio_path``
    may be ``None`` if no audio track was found.
    """

    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required but not installed")

    video_path = os.fspath(video_path)

    tmpdir = tempfile.mkdtemp()
    audio_path = os.path.join(tmpdir, "audio.wav")

    frames_dir = os.path.join(tmpdir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # Determine frames per second of the input video using ffprobe
    ffprobe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "0",
            "-of",
            "csv=p=0",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            video_path,
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    rate = ffprobe.stdout.strip()
    try:
        num, den = rate.split("/")
        fps = float(num) / float(den)
    except ValueError:
        fps = 1.0

    try:
        audio_proc = subprocess.run(

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

            check=False,

            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        subprocess.run(
            [
                "ffmpeg",
                "-i",
                video_path,

                "-vf",
                "scale=224:224",
                "-vsync",
                "0",

                os.path.join(frames_dir, "frame_%05d.jpg"),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if audio_proc.returncode != 0:
            # remove partial audio and continue without transcription
            if os.path.exists(audio_path):
                os.remove(audio_path)
            audio_path = None
    except subprocess.CalledProcessError as e:
        msg = e.stderr.strip() if e.stderr else str(e)
        raise RuntimeError(f"ffmpeg failed: {msg}") from e

    frame_paths = sorted(
        str(p) for p in Path(frames_dir).glob("frame_*.jpg")
    )
    return audio_path, frame_paths, fps, tmpdir


def file_sha256(path: str | os.PathLike) -> str:
    """Return the sha256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def save_db(db_dir: Path, vid_hash: str, data: dict) -> None:
    with open(db_dir / f"{vid_hash}.json", "w", encoding="utf-8") as f:
        json.dump(data, f)


def load_db(db_dir: Path, vid_hash: str) -> dict | None:
    fp = db_dir / f"{vid_hash}.json"
    if not fp.exists():
        return None
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)


class GradioApp:
    def __init__(self, ollama_url: str):
        self.pipeline = LocalPipeline(ollama_url)
        self.transcript = ""
        self.frames: list[str] = []
        self.captions: list[dict] = []
        self.fps: float = 1.0
        self.video_dir = Path("data/videos")
        self.db_dir = Path("data/db")
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.db_dir.mkdir(parents=True, exist_ok=True)

    def _list_videos(self) -> list[str]:
        vids = []
        for p in self.db_dir.glob("*.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    vids.append(meta.get("file", p.stem))
            except Exception:
                continue
        return sorted(vids)

    def process_stream(self, url: str, progress=gr.Progress()):
        if not url:
            return gr.update(), "", "", gr.update(choices=self._list_videos())

        out_file = self.video_dir / f"stream_{int(time.time())}.mp4"
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    url,
                    "-t",
                    "10",
                    str(out_file),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(e.stderr.strip() if e.stderr else str(e)) from e

        transcript, caption = self.process_upload(str(out_file), progress)
        return gr.update(value=str(out_file)), transcript, caption, gr.update(choices=self._list_videos())

    def load_existing(self, file_name: str):
        vid_hash = None
        for p in self.db_dir.glob("*.json"):
            with open(p, "r", encoding="utf-8") as f:
                meta = json.load(f)
                if meta.get("file") == file_name:
                    vid_hash = meta.get("hash")
                    break
        if not vid_hash:
            return "", ""
        data = load_db(self.db_dir, vid_hash)
        if not data:
            return "", ""
        self.transcript = data.get("transcript", "")
        self.captions = data.get("captions", [])
        self.fps = data.get("fps", 1.0)
        caption = self.captions[0]["caption"] if self.captions else ""
        video_path = self.video_dir / data.get("file")
        return str(video_path), self.transcript, caption

    def process_upload(self, video_file, progress=gr.Progress()):
        if video_file is None:
            return "", ""
        # save the uploaded file to the videos directory
        vid_hash = file_sha256(video_file)
        ext = Path(video_file).suffix
        saved_path = self.video_dir / f"{vid_hash}{ext}"
        if not saved_path.exists():
            shutil.copy(video_file, saved_path)

        audio, self.frames, fps, tmp = extract_media(saved_path)
        # Run captioning on every third frame to speed up processing
        step_frames = self.frames[::3]
        total = len(step_frames)
        progress((0, total), desc="Processing frames")
        self.transcript = self.pipeline.transcribe(audio)
        self.captions = self.pipeline.caption_frames(step_frames, fps=fps, progress=progress)
        caption = self.captions[0]["caption"] if self.captions else ""

        save_db(
            self.db_dir,
            vid_hash,
            {
                "file": saved_path.name,
                "hash": vid_hash,
                "fps": fps,
                "transcript": self.transcript,
                "captions": self.captions,
            },
        )
        return self.transcript, caption

    def answer(self, question, history):

        if not self.frames:

            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": "Upload a video first."})
            return history
        response = self.pipeline.answer(question, self.transcript, self.captions)

        ts_match = re.search(r"(\d{1,2}:\d{2})", response)
        if ts_match:
            mmss = ts_match.group(1)
            m, s = map(int, mmss.split(":"))
            sec = m * 60 + s
            link = (
                f'<a href="#" onclick="document.getElementById(\'video\')'
                f'.currentTime={sec}; return false;">{mmss}</a>'
            )
            response = response.replace(mmss, link)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": response})
        return history

    def launch(self, share: bool = False):
        with gr.Blocks() as demo:
            video = gr.Video(label="Video", elem_id="video")
            url = gr.Textbox(label="Stream URL")
            capture_btn = gr.Button("Capture Stream")
            existing = gr.Dropdown(label="Existing Videos", choices=self._list_videos())
            transcript_box = gr.Textbox(label="Transcript")
            caption_box = gr.Textbox(label="Caption")
            chatbot = gr.Chatbot(type="messages")
            question = gr.Textbox(label="Question")
            send = gr.Button("Ask")

            video.upload(self.process_upload, inputs=video, outputs=[transcript_box, caption_box])
            capture_btn.click(
                self.process_stream,
                inputs=url,
                outputs=[video, transcript_box, caption_box, existing],
            )
            existing.change(
                self.load_existing,
                inputs=existing,
                outputs=[video, transcript_box, caption_box],
            )
            send.click(self.answer, inputs=[question, chatbot], outputs=chatbot)
        demo.queue().launch(server_name="0.0.0.0", share=share)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Share the Gradio interface publicly",
    )
    args = parser.parse_args()
    app = GradioApp(args.ollama_url)
    app.launch(share=args.share)


if __name__ == "__main__":
    main()
