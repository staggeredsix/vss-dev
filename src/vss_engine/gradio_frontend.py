import argparse
import os
import re
import json
import hashlib
import time
import logging

import shutil

try:
    import imageio_ffmpeg  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    imageio_ffmpeg = None

import subprocess
import tempfile
import numpy as np
from pathlib import Path
import sys

# Assets from legacy UI
LEGACY_ASSETS = Path(__file__).resolve().parent / "../vss-engine/src/client/assets"

import gradio as gr

# Allow importing as a package when executed from repository root
sys.path.append(str(Path(__file__).resolve().parent.parent))
from vss_engine.pipeline import LocalPipeline  # noqa: E402
from telemetry import Telemetry  # noqa: E402


def _ffmpeg() -> str:
    """Return path to a working ffmpeg executable with a fallback."""
    path = shutil.which("ffmpeg")
    if path:
        try:
            subprocess.run(
                [path, "-version"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            return path
        except Exception:
            pass
    if imageio_ffmpeg is not None:
        return imageio_ffmpeg.get_ffmpeg_exe()
    raise RuntimeError(
        "ffmpeg is required but was not found. Install it or install the"
        " imageio-ffmpeg Python package."
    )

def extract_audio_bytes(video_path: str | os.PathLike) -> bytes | None:
    """Return raw 16 kHz mono PCM audio from a video without writing to disk."""
    video_path = os.fspath(video_path)
    proc = subprocess.run(
        [
            _ffmpeg(),
            "-i",
            video_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-f",
            "s16le",
            "-",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout


def reencode_video(video_path: Path, size: int = 244) -> None:
    """Reencode ``video_path`` in-place scaled to ``size`` square pixels."""
    tmp = video_path.with_suffix(".tmp" + video_path.suffix)
    try:
        subprocess.run(
            [
                _ffmpeg(),
                "-y",
                "-i",
                str(video_path),
                "-vf",
                f"scale={size}:{size}",
                "-c:a",
                "copy",
                str(tmp),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(e.stderr.strip() if e.stderr else str(e)) from e
    tmp.replace(video_path)


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
    def __init__(self, ollama_url: str, telemetry: Telemetry | None = None):
        self.telemetry = telemetry or Telemetry(os.environ.get("TELEMETRY_URL"))
        asr_url = os.environ.get("ASR_URL")
        reranker_url = os.environ.get("RERANKER_URL")
        spec_url = os.environ.get("SPEC_URL")
        self.pipeline = LocalPipeline(
            ollama_url,
            rag_db_dir="data/db",
            telemetry=self.telemetry,
            asr_url=asr_url,
            reranker_url=reranker_url,
            spec_url=spec_url,
        )
        self.transcript = ""
        self.frames: list[str] = []
        self.captions: list[dict] = []
        self.fps: float = 1.0
        self.current_vid: str | None = None
        self.video_dir = Path("data/videos")
        self.db_dir = Path("data/db")
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.db_dir.mkdir(parents=True, exist_ok=True)

        # Legacy UI assets
        theme_json = LEGACY_ASSETS / "kaizen-theme.json"
        css_file = LEGACY_ASSETS / "kaizen-theme.css"
        app_bar_file = LEGACY_ASSETS / "app_bar.html"
        self.theme = None
        if theme_json.exists():
            self.theme = gr.themes.Default().load(str(theme_json))
        self.css = css_file.read_text() if css_file.exists() else ""
        self.app_bar = app_bar_file.read_text() if app_bar_file.exists() else ""

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
                    _ffmpeg(),
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
            reencode_video(out_file)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(e.stderr.strip() if e.stderr else str(e)) from e

        transcript, caption, dropdown_update = self.process_upload(
            str(out_file), progress
        )
        return (
            gr.update(value=str(out_file)),
            transcript,
            caption,
            dropdown_update,
        )

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
        self.current_vid = vid_hash
        self.transcript = data.get("transcript", "")
        self.captions = data.get("captions", [])
        self.fps = data.get("fps", 1.0)
        caption = self.captions[0]["caption"] if self.captions else ""
        video_path = self.video_dir / data.get("file")
        return gr.update(value=str(video_path)), self.transcript, caption

    def process_upload(self, video_file, progress=gr.Progress()):
        if video_file is None:
            return "", "", gr.update(choices=self._list_videos())
        # save the uploaded file to the videos directory
        vid_hash = file_sha256(video_file)
        ext = Path(video_file).suffix
        saved_path = self.video_dir / f"{vid_hash}{ext}"
        if not saved_path.exists():
            shutil.copy(video_file, saved_path)
        reencode_video(saved_path)

        audio_bytes = extract_audio_bytes(saved_path)
        self.current_vid = vid_hash
        self.transcript = self.pipeline.transcribe(audio_bytes, source_id=vid_hash)
        progress((0, 0), desc="Processing frames")
        # Process video frames with the local pipeline. The realtime caption
        # method now controls frame sampling internally, so only the video path
        # and progress callback are required.
        self.captions = self.pipeline.caption_realtime(
            str(saved_path), progress=progress, source_id=vid_hash
        )
        # ``fps`` is kept for backwards compatibility with previously saved DB
        # entries even though it is no longer used by the pipeline.
        fps = 4.0
        self.fps = fps
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
        return (
            self.transcript,
            caption,
            gr.update(choices=self._list_videos()),
        )

    def answer(self, question, history, use_rerank: bool):

        if not self.current_vid:

            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": "Upload a video first."})
            return history, ""
        response, context_docs = self.pipeline.answer(
            question, source_id=self.current_vid, use_rerank=use_rerank
        )

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
        return history, "\n".join(context_docs)

    def launch(self, share: bool = False):
        with gr.Blocks(theme=self.theme, css=self.css) as demo:
            if self.app_bar:
                gr.HTML(self.app_bar)
            video = gr.Video(label="Video", elem_id="video")
            url = gr.Textbox(label="Stream URL")
            capture_btn = gr.Button("Capture Stream")
            existing = gr.Dropdown(label="Existing Videos", choices=self._list_videos())
            transcript_box = gr.Textbox(label="Transcript")
            caption_box = gr.Textbox(label="Caption")
            chatbot = gr.Chatbot(type="messages")
            question = gr.Textbox(label="Question")
            use_rerank = gr.Checkbox(value=True, label="Use Reranking")
            rag_box = gr.Textbox(label="RAG Context")
            send = gr.Button("Ask")

            video.upload(
                self.process_upload,
                inputs=video,
                outputs=[transcript_box, caption_box, existing],
            )
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
            send.click(
                self.answer,
                inputs=[question, chatbot, use_rerank],
                outputs=[chatbot, rag_box],
            )
        demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=share)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ollama-url",
        default=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
    )
    parser.add_argument(
        "--asr-url",
        default=os.environ.get("ASR_URL"),
        help="URL of the ASR service. If not provided, Whisper runs locally.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Share the Gradio interface publicly",
    )
    args = parser.parse_args()
    logging.basicConfig(level=os.environ.get("VSS_LOG_LEVEL", "INFO").upper())
    telem = Telemetry(os.environ.get("TELEMETRY_URL"))
    if args.asr_url:
        os.environ["ASR_URL"] = args.asr_url
    app = GradioApp(args.ollama_url, telemetry=telem)
    app.launch(share=args.share)


if __name__ == "__main__":
    main()
