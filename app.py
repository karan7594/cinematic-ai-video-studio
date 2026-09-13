import os
import re
import json
import math
import shutil
import subprocess
import asyncio
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
import edge_tts
import imageio_ffmpeg


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Cinematic AI Video Studio",
    page_icon="🎬",
    layout="wide"
)


# ============================================================
# RESPONSIVE / MOBILE / TABLET UI
# ============================================================

st.markdown(
    """
    <style>
    /* Main responsive spacing */
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        padding-left: clamp(0.7rem, 3vw, 3rem);
        padding-right: clamp(0.7rem, 3vw, 3rem);
        max-width: 1500px;
        margin: auto;
    }

    /* Make uploaded media and Streamlit images fluid */
    img, video {
        max-width: 100% !important;
        height: auto !important;
    }

    /* Buttons remain easy to tap on phones/tablets */
    button {
        min-height: 42px !important;
    }

    /* Prevent long filenames/paths from breaking mobile layout */
    code, pre {
        overflow-x: auto !important;
        max-width: 100% !important;
    }

    /* Responsive editor menu */
    .video-editor-menu {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: clamp(8px, 2vw, 18px);
        margin: 12px 0 20px 0;
        flex-wrap: wrap;
    }

    .video-editor-center {
        width: clamp(62px, 13vw, 78px);
        height: clamp(62px, 13vw, 78px);
        min-width: 62px;
        min-height: 62px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        border: 2px solid rgba(128,128,128,.35);
        font-weight: 700;
        font-size: clamp(11px, 2.5vw, 13px);
        text-align: center;
        line-height: 1.15;
        box-sizing: border-box;
    }

    /* Mobile-friendly headings */
    h1 {
        font-size: clamp(1.55rem, 6vw, 2.5rem) !important;
    }

    h2 {
        font-size: clamp(1.25rem, 5vw, 2rem) !important;
    }

    h3 {
        font-size: clamp(1.05rem, 4vw, 1.5rem) !important;
    }

    /* Small-screen adjustments */
    @media (max-width: 900px) {
        .block-container {
            padding-left: 0.65rem;
            padding-right: 0.65rem;
        }

        [data-testid="stHorizontalBlock"] {
            gap: 0.5rem !important;
        }

        [data-testid="stFileUploader"] {
            width: 100% !important;
        }
    }

    @media (max-width: 640px) {
        .block-container {
            padding-left: 0.45rem;
            padding-right: 0.45rem;
        }

        .video-editor-menu {
            gap: 7px;
        }

        .video-editor-center {
            width: 60px;
            height: 60px;
            min-width: 60px;
            min-height: 60px;
            font-size: 10px;
        }

        /* Streamlit columns naturally stack when needed; these
           rules make content readable instead of forcing desktop widths. */
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
        }

        [data-testid="stHorizontalBlock"] > [data-testid="column"] {
            min-width: min(100%, 300px) !important;
            flex: 1 1 100% !important;
        }

        /* Keep compact controls usable */
        .stButton > button,
        .stDownloadButton > button {
            width: 100%;
        }

        textarea, input {
            font-size: 16px !important;
        }
    }

    /* Touch-friendly radio/select controls */
    @media (pointer: coarse) {
        button,
        [role="button"] {
            min-height: 44px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)


APP_DIR = Path(__file__).resolve().parent
PROJECTS_DIR = APP_DIR / "projects"
PROJECTS_DIR.mkdir(exist_ok=True)

VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
FPS = 30
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]
VIDEO_EXTENSIONS = [".mp4", ".mov", ".mkv", ".avi", ".webm"]
AUDIO_EXTENSIONS = [".mp3", ".wav", ".m4a", ".aac", ".ogg"]

VOICE_OPTIONS = {
    "Hindi Female": "hi-IN-SwaraNeural",
    "Hindi Male": "hi-IN-MadhurNeural",
    "English India Female": "en-IN-NeerjaNeural",
    "English India Male": "en-IN-PrabhatNeural",
}


# ============================================================
# SESSION STATE
# ============================================================

if "selected_project" not in st.session_state:
    st.session_state.selected_project = None

if "page" not in st.session_state:
    st.session_state.page = "Projects"

if "delete_project" not in st.session_state:
    st.session_state.delete_project = None

if "delete_scene" not in st.session_state:
    st.session_state.delete_scene = None


# ============================================================
# HELPERS
# ============================================================

def safe_project_name(name):
    name = name.strip()
    name = re.sub(r'[<>:"/\\|?*]', "", name)
    name = re.sub(r"\s+", "_", name)
    return name[:80] or "New_Project"


def now_string():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def project_path(project_name):
    return PROJECTS_DIR / project_name


def project_json_path(project_name):
    return project_path(project_name) / "project.json"


def load_project(project_name):
    path = project_json_path(project_name)
    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_project(project):
    project["updated_at"] = now_string()
    path = project_json_path(project["name"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(project, f, indent=4, ensure_ascii=False)


def list_projects():
    projects = []
    for folder in PROJECTS_DIR.iterdir():
        if not folder.is_dir():
            continue
        data = load_project(folder.name)
        if data:
            projects.append(data)

    projects.sort(
        key=lambda x: x.get("updated_at", ""),
        reverse=True
    )
    return projects


def run_ffmpeg(args):
    command = [FFMPEG] + args

    creationflags = (
        subprocess.CREATE_NO_WINDOW
        if os.name == "nt"
        else 0
    )

    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=creationflags
    )

    if process.returncode != 0:
        raise RuntimeError(
            "FFmpeg Error:\n\n" + process.stderr[-8000:]
        )

    return process


def get_media_duration(file_path):
    result = subprocess.run(
        [FFMPEG, "-i", str(file_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=(
            subprocess.CREATE_NO_WINDOW
            if os.name == "nt"
            else 0
        )
    )

    match = re.search(
        r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)",
        result.stderr
    )

    if not match:
        raise RuntimeError(
            f"Media duration read नहीं हो सकी:\n{file_path}"
        )

    return (
        int(match.group(1)) * 3600
        + int(match.group(2)) * 60
        + float(match.group(3))
    )


def resize_cover(image, target_width, target_height):
    h, w = image.shape[:2]

    target_ratio = target_width / target_height
    image_ratio = w / h

    if image_ratio > target_ratio:
        new_h = target_height
        new_w = int(new_h * image_ratio)
    else:
        new_w = target_width
        new_h = int(new_w / image_ratio)

    image = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    x = max(0, (new_w - target_width) // 2)
    y = max(0, (new_h - target_height) // 2)

    return image[
        y:y + target_height,
        x:x + target_width
    ]


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def cinematic_frame(image, progress):
    progress = np.clip(progress, 0, 1)
    smooth = smoothstep(progress)

    h, w = image.shape[:2]

    # Subtle zoom out
    start_zoom = 1.12
    end_zoom = 1.03
    zoom = start_zoom + (end_zoom - start_zoom) * smooth

    crop_w = min(int(w / zoom), w)
    crop_h = min(int(h / zoom), h)

    # Right -> left
    max_x = max(0, w - crop_w)
    start_x = max_x * 0.85
    end_x = max_x * 0.15

    x = int(start_x + (end_x - start_x) * smooth)

    # Small vertical movement
    max_y = max(0, h - crop_h)
    start_y = max_y * 0.45
    end_y = max_y * 0.55

    y = int(start_y + (end_y - start_y) * smooth)

    x = max(0, min(x, w - crop_w))
    y = max(0, min(y, h - crop_h))

    frame = image[
        y:y + crop_h,
        x:x + crop_w
    ]

    return cv2.resize(
        frame,
        (VIDEO_WIDTH, VIDEO_HEIGHT),
        interpolation=cv2.INTER_AREA
    )


def save_uploaded_file(uploaded_file, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "wb") as f:
        f.write(uploaded_file.getbuffer())


# ============================================================
# VOICE
# ============================================================

async def generate_voice_async(text, output_file, voice):
    communicator = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate="+0%",
        volume="+0%"
    )
    await communicator.save(str(output_file))


def generate_voice(text, output_file, voice):
    try:
        asyncio.run(
            generate_voice_async(
                text,
                output_file,
                voice
            )
        )
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(
                generate_voice_async(
                    text,
                    output_file,
                    voice
                )
            )
        finally:
            loop.close()


# ============================================================
# PROJECT
# ============================================================

def create_project(name):
    clean_name = safe_project_name(name)
    folder = project_path(clean_name)

    if folder.exists():
        raise RuntimeError(
            "इस नाम का project पहले से मौजूद है."
        )

    folder.mkdir(parents=True)
    (folder / "images").mkdir()
    (folder / "audio").mkdir()
    (folder / "videos").mkdir()

    project = {
        "name": clean_name,
        "display_name": name.strip(),
        "status": "Draft",
        "created_at": now_string(),
        "updated_at": now_string(),
        "voice": "hi-IN-SwaraNeural",
        "scene_script": "",
        "background_music": "",
        "background_music_volume": 0.18,
        "items": [],
        "videos": []
    }

    save_project(project)
    return project


def next_scene_order(project):
    numbers = []
    for item in project.get("items", []):
        try:
            numbers.append(int(item.get("order", 0)))
        except Exception:
            pass
    return max(numbers) + 1 if numbers else 1


def add_image_to_project(project, uploaded_file):
    project_folder = project_path(project["name"])
    images_folder = project_folder / "images"

    original_name = Path(uploaded_file.name).name
    stem = Path(original_name).stem
    ext = Path(original_name).suffix.lower()

    if ext not in IMAGE_EXTENSIONS:
        raise RuntimeError(f"Unsupported image format: {ext}")

    order = next_scene_order(project)
    filename = (
        f"{order:03d}_{safe_project_name(stem)}{ext}"
    )

    destination = images_folder / filename
    save_uploaded_file(uploaded_file, destination)

    item = {
        "id": f"item_{order}_{datetime.now().timestamp()}",
        "order": order,
        "type": "image",
        "media": str(Path("images") / filename),
        "image": str(Path("images") / filename),
        "script": "",
        "voice_file": "",
        "duration": 0
    }

    project["items"].append(item)
    save_project(project)


def add_video_to_project(project, uploaded_file):
    project_folder = project_path(project["name"])
    videos_folder = project_folder / "videos"

    original_name = Path(uploaded_file.name).name
    stem = Path(original_name).stem
    ext = Path(original_name).suffix.lower()

    if ext not in VIDEO_EXTENSIONS:
        raise RuntimeError(f"Unsupported video format: {ext}")

    order = next_scene_order(project)
    filename = (
        f"{order:03d}_{safe_project_name(stem)}{ext}"
    )

    destination = videos_folder / filename
    save_uploaded_file(uploaded_file, destination)

    item = {
        "id": f"item_{order}_{datetime.now().timestamp()}",
        "order": order,
        "type": "video",
        "media": str(Path("videos") / filename),
        "image": "",
        "script": "",
        "voice_file": "",
        "duration": 0
    }

    project["items"].append(item)
    save_project(project)


def remove_item(project, item_id):
    project_folder = project_path(project["name"])
    remaining = []

    for item in project.get("items", []):
        if item["id"] == item_id:
            media_value = item.get("media") or item.get("image", "")
            if media_value:
                media_path = project_folder / media_value
                if media_path.exists():
                    try:
                        media_path.unlink()
                    except Exception:
                        pass

            if item.get("voice_file"):
                voice_path = project_folder / item["voice_file"]
                if voice_path.exists():
                    try:
                        voice_path.unlink()
                    except Exception:
                        pass
        else:
            remaining.append(item)

    project["items"] = remaining
    reorder_items(project)
    save_project(project)


def reorder_items(project):
    project["items"].sort(
        key=lambda x: int(x.get("order", 0))
    )

    for index, item in enumerate(
        project["items"],
        start=1
    ):
        item["order"] = index


def move_item_up(project, index):
    if index <= 0:
        return

    items = project["items"]
    items[index - 1], items[index] = (
        items[index],
        items[index - 1]
    )

    reorder_items(project)
    save_project(project)


def move_item_down(project, index):
    items = project["items"]

    if index >= len(items) - 1:
        return

    items[index], items[index + 1] = (
        items[index + 1],
        items[index]
    )

    reorder_items(project)
    save_project(project)


# ============================================================
# SCENE SCRIPT
# ============================================================

def parse_scene_script(text):
    """
    Scene scripts are separated by a blank line.
    Example:

    Scene 1 script...

    Scene 2 script...

    Scene 3 script...
    """
    if not text or not text.strip():
        return []

    blocks = re.split(r"\n\s*\n+", text.strip())
    return [block.strip() for block in blocks if block.strip()]


def apply_scene_script(project):
    scripts = parse_scene_script(
        project.get("scene_script", "")
    )

    if not scripts:
        return 0

    items = sorted(
        project.get("items", []),
        key=lambda x: int(x.get("order", 0))
    )

    changed = 0

    for index, item in enumerate(items):
        if index < len(scripts):
            # Only fill empty scene scripts.
            if not item.get("script", "").strip():
                item["script"] = scripts[index]
                changed += 1

    if changed:
        save_project(project)

    return changed


# ============================================================
# BACKGROUND MUSIC
# ============================================================

def add_background_music(project, uploaded_file):
    project_folder = project_path(project["name"])
    audio_folder = project_folder / "audio"

    ext = Path(uploaded_file.name).suffix.lower()
    if ext not in AUDIO_EXTENSIONS:
        raise RuntimeError(
            "Unsupported background music format."
        )

    # Remove old background music
    old_music = project.get("background_music", "")
    if old_music:
        old_path = project_folder / old_music
        if old_path.exists():
            try:
                old_path.unlink()
            except Exception:
                pass

    filename = (
        "background_music"
        + ext
    )

    destination = audio_folder / filename
    save_uploaded_file(uploaded_file, destination)

    project["background_music"] = str(
        Path("audio") / filename
    )

    save_project(project)


# ============================================================
# MEDIA RENDERING
# ============================================================

def render_image_scene(image_path, duration, output_path):
    """
    Render a still image with cinematic movement.
    Uses OpenCV only for the frames of this scene.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(
            f"Image load नहीं हुई:\n{image_path}"
        )

    image = resize_cover(
        image,
        VIDEO_WIDTH,
        VIDEO_HEIGHT
    )

    total_frames = max(
        1,
        int(math.ceil(duration * FPS))
    )

    command = [
        FFMPEG,
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{VIDEO_WIDTH}x{VIDEO_HEIGHT}",
        "-r", str(FPS),
        "-i", "-",
        "-an",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path)
    ]

    creationflags = (
        subprocess.CREATE_NO_WINDOW
        if os.name == "nt"
        else 0
    )

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=creationflags
    )

    try:
        for frame_number in range(total_frames):
            progress = frame_number / max(
                1,
                total_frames - 1
            )

            frame = cinematic_frame(
                image,
                progress
            )

            process.stdin.write(frame.tobytes())

        process.stdin.close()
        _, stderr = process.communicate()

        if process.returncode != 0:
            raise RuntimeError(
                stderr.decode(
                    "utf-8",
                    errors="ignore"
                )[-6000:]
            )

    except Exception:
        try:
            process.kill()
        except Exception:
            pass
        raise


def render_video_scene(
    video_path,
    duration,
    output_path
):
    """
    Prepare an uploaded video scene.
    The video is scaled/cropped to 1920x1080 and looped
    if it is shorter than the voice-over duration.
    """
    source_duration = get_media_duration(video_path)

    if source_duration <= 0:
        raise RuntimeError(
            f"Video duration invalid: {video_path}"
        )

    # If the uploaded clip is shorter than required,
    # loop it. Otherwise start from the beginning and trim.
    run_ffmpeg([
        "-y",
        "-stream_loop", "-1",
        "-i", str(video_path),
        "-t", f"{duration:.3f}",
        "-vf",
        (
            "scale=1920:1080:force_original_aspect_ratio=increase,"
            "crop=1920:1080"
        ),
        "-r", str(FPS),
        "-an",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path)
    ])


def create_scene_video(
    project,
    item,
    voice_duration,
    output_path
):
    project_folder = project_path(project["name"])
    media_value = item.get("media") or item.get("image", "")

    if not media_value:
        raise RuntimeError(
            "Scene media missing."
        )

    media_path = project_folder / media_value

    if not media_path.exists():
        raise RuntimeError(
            f"Scene media missing:\n{media_path}"
        )

    if item.get("type", "image") == "video":
        render_video_scene(
            media_path,
            voice_duration,
            output_path
        )
    else:
        render_image_scene(
            media_path,
            voice_duration,
            output_path
        )


def concat_scene_videos(scene_files, output_path):
    list_file = output_path.parent / "scene_list.txt"

    with open(list_file, "w", encoding="utf-8") as f:
        for scene_file in scene_files:
            safe_path = (
                str(Path(scene_file).resolve())
                .replace("\\", "/")
                .replace("'", "'\\''")
            )
            f.write(f"file '{safe_path}'\n")

    run_ffmpeg([
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        "-movflags", "+faststart",
        str(output_path)
    ])

    try:
        list_file.unlink()
    except Exception:
        pass


# ============================================================
# AUDIO
# ============================================================

def merge_voice_files(project, items, output_file):
    project_folder = project_path(project["name"])
    audio_folder = project_folder / "audio"
    audio_folder.mkdir(exist_ok=True)

    list_file = audio_folder / "voice_list.txt"

    with open(list_file, "w", encoding="utf-8") as f:
        for item in items:
            voice_path = (
                project_folder / item["voice_file"]
            )

            if not voice_path.exists():
                raise RuntimeError(
                    f"Voice file missing:\n{voice_path}"
                )

            safe_path = (
                str(voice_path.resolve())
                .replace("\\", "/")
                .replace("'", "'\\''")
            )
            f.write(f"file '{safe_path}'\n")

    run_ffmpeg([
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c:a", "libmp3lame",
        "-b:a", "192k",
        str(output_file)
    ])

    try:
        list_file.unlink()
    except Exception:
        pass


def add_background_music_to_video(
    voice_video,
    music_path,
    volume,
    output_path
):
    """
    Mix voice-over + background music.
    Music is looped and trimmed to the voice video duration.
    """
    run_ffmpeg([
        "-y",
        "-i", str(voice_video),
        "-stream_loop", "-1",
        "-i", str(music_path),
        "-filter_complex",
        (
            f"[1:a]volume={float(volume):.3f},"
            f"aloop=loop=-1:size=2e+09,"
            f"atrim=duration=99999[music];"
            f"[0:a][music]amix=inputs=2:"
            f"duration=first:dropout_transition=2[aout]"
        ),
        "-map", "0:v:0",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        str(output_path)
    ])


# ============================================================
# GENERATE COMPLETE VIDEO
# ============================================================

def generate_project_video(
    project,
    progress_callback=None
):
    project_folder = project_path(project["name"])
    items = sorted(
        project.get("items", []),
        key=lambda x: int(x.get("order", 0))
    )

    if not items:
        raise RuntimeError(
            "Project में कोई image/video नहीं है."
        )

    # --------------------------------------------------------
    # Validate scripts
    # --------------------------------------------------------
    for index, item in enumerate(items, start=1):
        if not item.get("script", "").strip():
            raise RuntimeError(
                f"Scene {index} के लिए Script खाली है."
            )

    # --------------------------------------------------------
    # Generate voice files
    # --------------------------------------------------------
    voice_code = project.get(
        "voice",
        "hi-IN-SwaraNeural"
    )

    total_duration = 0

    for index, item in enumerate(items, start=1):
        audio_folder = project_folder / "audio"
        audio_folder.mkdir(exist_ok=True)

        voice_filename = f"voice_{index:03d}.mp3"
        voice_relative = (
            Path("audio") / voice_filename
        )
        voice_path = project_folder / voice_relative

        # Regenerate when script changes or voice file is missing.
        # A script hash keeps old audio from being reused incorrectly.
        script_hash = str(
            abs(hash(
                voice_code + "|" + item["script"].strip()
            ))
        )

        meta_key = f"voice_hash_{script_hash}"

        if (
            not voice_path.exists()
            or item.get("voice_hash") != meta_key
        ):
            generate_voice(
                item["script"].strip(),
                voice_path,
                voice_code
            )

        duration = get_media_duration(voice_path)

        item["voice_file"] = str(voice_relative)
        item["voice_hash"] = meta_key
        item["duration"] = duration

        total_duration += duration

        if progress_callback:
            progress_callback(
                0.20 * (index / len(items))
            )

    save_project(project)

    # --------------------------------------------------------
    # Render each scene
    # --------------------------------------------------------
    scene_folder = project_folder / "videos" / "scenes"
    scene_folder.mkdir(parents=True, exist_ok=True)

    scene_files = []

    for index, item in enumerate(items, start=1):
        scene_output = (
            scene_folder /
            f"scene_{index:03d}.mp4"
        )

        create_scene_video(
            project,
            item,
            float(item["duration"]),
            scene_output
        )

        scene_files.append(scene_output)

        if progress_callback:
            progress_callback(
                0.20 +
                0.55 * (index / len(items))
            )

    # --------------------------------------------------------
    # Concatenate scenes
    # --------------------------------------------------------
    combined_video = (
        project_folder /
        "videos" /
        "combined_scenes.mp4"
    )

    concat_scene_videos(
        scene_files,
        combined_video
    )

    # --------------------------------------------------------
    # Merge all voice-over files
    # --------------------------------------------------------
    combined_voice = (
        project_folder /
        "audio" /
        "combined_voice.mp3"
    )

    merge_voice_files(
        project,
        items,
        combined_voice
    )

    # --------------------------------------------------------
    # Put voice-over on video
    # --------------------------------------------------------
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    voice_video = (
        project_folder /
        "videos" /
        f"voice_video_{timestamp}.mp4"
    )

    run_ffmpeg([
        "-y",
        "-i", str(combined_video),
        "-i", str(combined_voice),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        str(voice_video)
    ])

    if progress_callback:
        progress_callback(0.90)

    # --------------------------------------------------------
    # Optional background music
    # --------------------------------------------------------
    background_music = project.get(
        "background_music",
        ""
    )

    if background_music:
        music_path = project_folder / background_music

        if music_path.exists():
            final_video = (
                project_folder /
                "videos" /
                f"cinematic_video_{timestamp}.mp4"
            )

            add_background_music_to_video(
                voice_video,
                music_path,
                project.get(
                    "background_music_volume",
                    0.18
                ),
                final_video
            )
        else:
            final_video = voice_video
    else:
        final_video = voice_video

    # --------------------------------------------------------
    # Save video record
    # --------------------------------------------------------
    relative_video = (
        Path("videos") / final_video.name
    )

    project.setdefault("videos", []).append({
        "filename": str(relative_video),
        "created_at": now_string(),
        "duration": total_duration
    })

    project["status"] = "Completed"
    save_project(project)

    if progress_callback:
        progress_callback(1.0)

    return final_video



def check_ffmpeg():
    try:
        result = subprocess.run(
            [FFMPEG, "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=(
                subprocess.CREATE_NO_WINDOW
                if os.name == "nt"
                else 0
            )
        )
        return result.returncode == 0
    except Exception:
        return False


# ============================================================
# DELETE CONFIRMATION
# ============================================================

def confirm_delete_project(project_name):
    st.session_state.delete_project = project_name


def confirm_delete_scene(item_id):
    st.session_state.delete_scene = item_id



st.markdown(
    """
    <style>
    @media (max-width: 640px) {
        div[role="radiogroup"] {
            overflow-x: auto !important;
            flex-wrap: nowrap !important;
            padding-bottom: 6px;
        }
        div[role="radiogroup"] > label {
            white-space: nowrap !important;
            flex-shrink: 0 !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🎬 Cinematic Studio")

page = st.sidebar.radio(
    "Menu",
    ["Projects", "Create Project"]
)

st.session_state.page = page

st.sidebar.divider()

try:
    if check_ffmpeg():
        st.sidebar.success("FFmpeg: Ready")
    else:
        st.sidebar.error("FFmpeg: Not Ready")
except Exception:
    st.sidebar.error("FFmpeg: Not Ready")


# ============================================================
# CREATE PROJECT PAGE
# ============================================================

if page == "Create Project":

    st.title("📁 Create New Project")

    project_name = st.text_input(
        "Project / Folder Name",
        placeholder="Example: Maharana Pratap Story"
    )

    if st.button(
        "📁 CREATE FOLDER",
        type="primary",
        use_container_width=True
    ):
        if not project_name.strip():
            st.error("Project name डालें.")
        else:
            try:
                project = create_project(project_name)

                st.session_state.selected_project = (
                    project["name"]
                )

                st.success(
                    "✅ Folder successfully created."
                )

                st.session_state.page = "Projects"
                st.rerun()

            except Exception as e:
                st.error(str(e))


# ============================================================
# PROJECT LIST
# ============================================================

elif (
    page == "Projects"
    and st.session_state.selected_project is None
):

    st.title("📁 My Video Projects")
    st.write(
        "अपने सभी video projects यहाँ manage करें."
    )

    if st.button(
        "➕ Create New Project",
        use_container_width=True
    ):
        st.session_state.page = "Create Project"
        st.rerun()

    projects = list_projects()

    if not projects:
        st.info(
            "अभी कोई project नहीं है. पहले Create Project करें."
        )

    for project in projects:

        with st.container(border=True):

            col1, col2, col3, col4 = st.columns(
                [3, 1, 1, 1]
            )

            with col1:
                st.subheader(
                    f"📁 {project.get('display_name', project['name'])}"
                )
                st.caption(
                    f"Scenes: {len(project.get('items', []))} | "
                    f"Videos: {len(project.get('videos', []))}"
                )

            with col2:
                status = project.get(
                    "status",
                    "Draft"
                )

                if status == "Completed":
                    st.success("Completed")
                elif status == "Generating":
                    st.warning("Generating")
                else:
                    st.info("Draft")

            with col3:
                if st.button(
                    "📂 Open",
                    key=f"open_{project['name']}"
                ):
                    st.session_state.selected_project = (
                        project["name"]
                    )
                    st.rerun()

            with col4:
                if st.button(
                    "🗑️ Delete",
                    key=f"delete_{project['name']}"
                ):
                    confirm_delete_project(
                        project["name"]
                    )
                    st.rerun()


# ============================================================
# PROJECT EDITOR
# ============================================================

elif st.session_state.selected_project is not None:

    project = load_project(
        st.session_state.selected_project
    )

    if project is None:
        st.session_state.selected_project = None
        st.rerun()

    # Migration for older projects
    project.setdefault("scene_script", "")
    project.setdefault("background_music", "")
    project.setdefault("background_music_volume", 0.18)
    project.setdefault("videos", [])

    for item in project.get("items", []):
        if "type" not in item:
            item["type"] = "image"

        if "media" not in item:
            item["media"] = item.get("image", "")

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    top1, top2, top3 = st.columns([4, 1, 1])

    with top1:
        st.title(
            f"📁 {project.get('display_name', project['name'])}"
        )

    with top2:
        status = project.get(
            "status",
            "Draft"
        )

        if status == "Completed":
            st.success(status)
        elif status == "Generating":
            st.warning(status)
        else:
            st.info(status)

    with top3:
        if st.button(
            "← Projects",
            use_container_width=True
        ):
            st.session_state.selected_project = None
            st.rerun()

    st.divider()

    # --------------------------------------------------------
    # SCENES SCRIPT - ABOVE VOICE SETTINGS
    # --------------------------------------------------------

    st.subheader("📝 Scenes Script")

    st.caption(
        "अगर image/video के साथ अलग script नहीं दी गई है, "
        "तो यहाँ scene-wise script लिखें. हर scene की script "
        "के बीच एक खाली line रखें."
    )

    scene_script = st.text_area(
        "Scene Scripts",
        value=project.get("scene_script", ""),
        height=220,
        key=f"scene_script_{project['name']}",
        placeholder=(
            "Scene 1 की script...\n\n"
            "Scene 2 की script...\n\n"
            "Scene 3 की script..."
        )
    )

    if scene_script != project.get("scene_script", ""):
        project["scene_script"] = scene_script
        project["status"] = "Draft"
        save_project(project)

    sc1, sc2 = st.columns([1, 3])

    with sc1:
        if st.button(
            "⚡ APPLY SCENES SCRIPT",
            use_container_width=True
        ):
            changed = apply_scene_script(project)

            if changed:
                st.success(
                    f"✅ {changed} scene script(s) added."
                )
            else:
                st.info(
                    "कोई empty scene नहीं मिला या Scenes Script खाली है."
                )

            st.rerun()

    with sc2:
        st.caption(
            "यह button केवल उन scenes को fill करेगा "
            "जिनकी individual script खाली है."
        )

    # --------------------------------------------------------
    # VOICE SETTINGS
    # --------------------------------------------------------

    st.divider()
    st.subheader("🎙️ Voice Settings")

    current_voice_code = project.get(
        "voice",
        "hi-IN-SwaraNeural"
    )

    voice_names = list(VOICE_OPTIONS.keys())

    current_voice_name = next(
        (
            name
            for name, code in VOICE_OPTIONS.items()
            if code == current_voice_code
        ),
        "Hindi Female"
    )

    selected_voice_name = st.selectbox(
        "Project Voice",
        voice_names,
        index=voice_names.index(
            current_voice_name
        )
    )

    selected_voice_code = VOICE_OPTIONS[
        selected_voice_name
    ]

    if selected_voice_code != project.get("voice"):
        project["voice"] = selected_voice_code
        project["status"] = "Draft"
        save_project(project)

    # --------------------------------------------------------
    # ADD IMAGES
    # --------------------------------------------------------

    st.divider()
    st.subheader("🖼️ Add Images")

    uploaded_images = st.file_uploader(
        "Images upload करें",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ],
        accept_multiple_files=True,
        key=f"project_image_upload_{project['name']}"
    )

    if st.button(
        "➕ ADD IMAGES TO PROJECT",
        use_container_width=True
    ):
        if not uploaded_images:
            st.warning("पहले images select करें.")
        else:
            for uploaded in uploaded_images:
                add_image_to_project(
                    project,
                    uploaded
                )

            st.success(
                f"✅ {len(uploaded_images)} image(s) added."
            )
            st.rerun()

    # --------------------------------------------------------
    # ADD VIDEOS
    # --------------------------------------------------------

    st.subheader("🎥 Add Videos")

    uploaded_videos = st.file_uploader(
        "Video upload/add करें",
        type=[
            "mp4",
            "mov",
            "mkv",
            "avi",
            "webm"
        ],
        accept_multiple_files=True,
        key=f"project_video_upload_{project['name']}"
    )

    if st.button(
        "➕ ADD VIDEOS TO PROJECT",
        use_container_width=True
    ):
        if not uploaded_videos:
            st.warning("पहले video select करें.")
        else:
            for uploaded in uploaded_videos:
                try:
                    add_video_to_project(
                        project,
                        uploaded
                    )
                except Exception as e:
                    st.error(str(e))

            st.success(
                f"✅ {len(uploaded_videos)} video(s) added."
            )
            st.rerun()

    # --------------------------------------------------------
    # BACKGROUND MUSIC
    # --------------------------------------------------------

    st.divider()
    st.subheader("🎵 Background Music")

    music_file = st.file_uploader(
        "Background Music upload करें",
        type=[
            "mp3",
            "wav",
            "m4a",
            "aac",
            "ogg"
        ],
        key=f"music_upload_{project['name']}"
    )

    if music_file is not None:
        if st.button(
            "➕ ADD BACKGROUND MUSIC",
            use_container_width=True
        ):
            try:
                add_background_music(
                    project,
                    music_file
                )
                st.success(
                    "✅ Background music added."
                )
                st.rerun()
            except Exception as e:
                st.error(str(e))

    current_music = project.get(
        "background_music",
        ""
    )

    if current_music:
        music_path = (
            project_path(project["name"])
            / current_music
        )

        if music_path.exists():
            st.audio(str(music_path))

            music_volume = st.slider(
                "Background Music Volume",
                min_value=0.0,
                max_value=1.0,
                value=float(
                    project.get(
                        "background_music_volume",
                        0.18
                    )
                ),
                step=0.01,
                help="Voice-over clear रखने के लिए 0.10–0.25 recommended."
            )

            if music_volume != project.get(
                "background_music_volume",
                0.18
            ):
                project["background_music_volume"] = music_volume
                save_project(project)

            if st.button(
                "🗑️ REMOVE BACKGROUND MUSIC",
                use_container_width=True
            ):
                old_path = project_path(project["name"]) / current_music

                if old_path.exists():
                    try:
                        old_path.unlink()
                    except Exception:
                        pass

                project["background_music"] = ""
                save_project(project)
                st.rerun()

    # --------------------------------------------------------
    # SCENES
    # --------------------------------------------------------

    st.divider()
    st.subheader("🎞️ Image / Video + Voice-over Scenes")

    items = sorted(
        project.get("items", []),
        key=lambda x: int(x.get("order", 0))
    )

    if not items:
        st.info(
            "इस project में अभी images/videos नहीं हैं."
        )

    for index, item in enumerate(items):

        st.markdown(
            f"### 🎬 Scene {index + 1}"
        )

        col_media, col_data, col_buttons = st.columns(
            [1.2, 2.4, 0.8]
        )

        # ----------------------------------------------------
        # MEDIA
        # ----------------------------------------------------

        with col_media:

            media_value = (
                item.get("media")
                or item.get("image", "")
            )

            media_path = (
                project_path(project["name"])
                / media_value
            )

            media_type = item.get(
                "type",
                "image"
            )

            if media_path.exists():

                if media_type == "video":
                    st.video(str(media_path))
                    st.caption("🎥 Video Scene")
                else:
                    st.image(
                        str(media_path),
                        use_container_width=True
                    )
                    st.caption("🖼️ Image Scene")

            else:
                st.error("Media missing")

        # ----------------------------------------------------
        # SCRIPT
        # ----------------------------------------------------

        with col_data:

            st.caption(
                f"{'🎥 Video' if item.get('type') == 'video' else '🖼️ Image'}: "
                f"{Path(media_value).name}"
            )

            new_script = st.text_area(
                "🎙️ Voice-over Script",
                value=item.get("script", ""),
                height=130,
                key=f"script_{item['id']}",
                placeholder=(
                    "इस scene के लिए voice-over script लिखें..."
                )
            )

            if new_script != item.get("script", ""):
                item["script"] = new_script
                project["status"] = "Draft"
                save_project(project)

        # ----------------------------------------------------
        # ORDER + DELETE
        # ----------------------------------------------------

        with col_buttons:

            st.write(
                f"**Order:** {index + 1}"
            )

            if index > 0:
                if st.button(
                    "⬆️",
                    key=f"up_{item['id']}"
                ):
                    move_item_up(
                        project,
                        index
                    )
                    st.rerun()

            if index < len(items) - 1:
                if st.button(
                    "⬇️",
                    key=f"down_{item['id']}"
                ):
                    move_item_down(
                        project,
                        index
                    )
                    st.rerun()

            if st.button(
                "🗑️ Delete",
                key=f"remove_{item['id']}"
            ):
                confirm_delete_scene(
                    item["id"]
                )
                st.rerun()

        st.divider()

    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    st.subheader("🎬 Generate Video")

    ready_count = sum(
        1
        for item in project.get("items", [])
        if item.get("script", "").strip()
    )

    total_count = len(
        project.get("items", [])
    )

    st.write(
        f"Script Ready: **{ready_count}/{total_count}**"
    )

    if total_count > 0:

        if ready_count == total_count:
            st.success(
                "✅ सभी scenes की scripts ready हैं."
            )
        else:
            st.warning(
                "⚠️ कुछ scenes की scripts अभी खाली हैं."
            )

    generate_button = st.button(
        "🎬 GENERATE COMPLETE VIDEO",
        type="primary",
        use_container_width=True
    )

    if generate_button:

        if not project.get("items"):
            st.error(
                "पहले images/videos add करें."
            )

        elif ready_count != total_count:
            st.error(
                "हर image/video scene के लिए Voice-over Script लिखें."
            )

        else:

            project["status"] = "Generating"
            save_project(project)

            progress_bar = st.progress(0)
            status_text = st.empty()

            try:

                def generation_progress(value):
                    value = max(
                        0,
                        min(1, value)
                    )

                    progress_bar.progress(value)

                    if value < 0.20:
                        status_text.info(
                            "🎙️ Voice-over generate हो रहा है..."
                        )
                    elif value < 0.75:
                        status_text.info(
                            "🎥 Scenes render हो रहे हैं..."
                        )
                    elif value < 0.90:
                        status_text.info(
                            "🎬 Scenes combine हो रहे हैं..."
                        )
                    else:
                        status_text.info(
                            "🎵 Voice + Background Music merge हो रहे हैं..."
                        )

                final_video = generate_project_video(
                    project,
                    generation_progress
                )

                progress_bar.progress(1.0)
                status_text.success(
                    "✅ Video Completed!"
                )

                st.video(str(final_video))

                st.rerun()

            except Exception as e:

                project["status"] = "Draft"
                save_project(project)

                st.error(
                    "❌ Video generation failed."
                )
                st.exception(e)

    # --------------------------------------------------------
    # VIDEO EDITOR
    # --------------------------------------------------------

    st.divider()
    st.subheader("🎬 Video Edit")

    # Circular editor menu — each button opens a real editing control.
    st.markdown(
        """
        <style>
        .video-editor-menu {
            display:flex;
            justify-content:center;
            align-items:center;
            gap:clamp(7px, 2vw, 18px);
            margin:12px 0 20px 0;
            flex-wrap:wrap;
        }
        .video-editor-center {
            width:clamp(60px, 13vw, 78px);
            height:clamp(60px, 13vw, 78px);
            min-width:60px;
            min-height:60px;
            border-radius:50%;
            display:flex;
            align-items:center;
            justify-content:center;
            border:2px solid rgba(128,128,128,.35);
            font-weight:700;
            font-size:clamp(10px, 2.5vw, 13px);
            text-align:center;
            line-height:1.15;
            box-sizing:border-box;
        }
        </style>
        <div class="video-editor-menu">
            <div class="video-editor-center">✂️<br>Trim</div>
            <div class="video-editor-center">T<br>Text</div>
            <div class="video-editor-center">⬜<br>Frame</div>
            <div class="video-editor-center">◫<br>Crop</div>
            <div class="video-editor-center">⏩<br>Speed</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    editor_videos = []
    for editor_video in project.get("videos", []):
        editor_path = (
            project_path(project["name"])
            / editor_video.get("filename", "")
        )
        if editor_path.exists():
            editor_videos.append(
                (editor_video, editor_path)
            )

    if editor_videos:

        editor_names = [
            Path(path).name
            for _, path in editor_videos
        ]

        selected_editor_name = st.selectbox(
            "Video to Edit",
            editor_names,
            key=f"editor_video_select_{project['name']}"
        )

        selected_editor_path = next(
            path for _, path in editor_videos
            if Path(path).name == selected_editor_name
        )

        st.video(str(selected_editor_path))

        editor_action = st.radio(
            "Edit",
            [
                "✂️ Trim",
                "T Text",
                "⬜ Frame",
                "◫ Crop",
                "⏩ Speed"
            ],
            horizontal=True,
            key=f"editor_action_{project['name']}"
        )

        # ----------------------------------------------------
        # TRIM
        # ----------------------------------------------------
        if editor_action == "✂️ Trim":

            duration = get_media_duration(
                selected_editor_path
            )

            st.write(
                f"Original duration: **{duration:.2f} sec**"
            )

            trim_start, trim_end = st.slider(
                "Trim range (seconds)",
                min_value=0.0,
                max_value=float(duration),
                value=(0.0, float(duration)),
                step=0.1,
                key=f"trim_{project['name']}_{selected_editor_name}"
            )

            if st.button(
                "✂️ APPLY TRIM",
                use_container_width=True
            ):
                if trim_end <= trim_start:
                    st.error(
                        "Trim end must be greater than trim start."
                    )
                else:
                    output = (
                        project_path(project["name"])
                        / "videos"
                        / (
                            "edited_trim_"
                            + datetime.now().strftime(
                                "%Y%m%d_%H%M%S"
                            )
                            + ".mp4"
                        )
                    )

                    run_ffmpeg([
                        "-y",
                        "-ss", f"{trim_start:.3f}",
                        "-i", str(selected_editor_path),
                        "-t", f"{trim_end - trim_start:.3f}",
                        "-c:v", "libx264",
                        "-preset", "veryfast",
                        "-crf", "18",
                        "-c:a", "aac",
                        "-b:a", "192k",
                        "-movflags", "+faststart",
                        str(output)
                    ])

                    project.setdefault("videos", []).append({
                        "filename": str(
                            Path("videos") / output.name
                        ),
                        "created_at": now_string(),
                        "duration": trim_end - trim_start
                    })
                    save_project(project)

                    st.success("✅ Trim applied.")
                    st.video(str(output))
                    st.rerun()

        # ----------------------------------------------------
        # TEXT
        # ----------------------------------------------------
        elif editor_action == "T Text":

            overlay_text = st.text_input(
                "Text",
                placeholder="Enter text to display on video",
                key=f"text_value_{project['name']}"
            )

            text_position = st.selectbox(
                "Position",
                [
                    "Top",
                    "Center",
                    "Bottom"
                ],
                key=f"text_position_{project['name']}"
            )

            text_size = st.slider(
                "Text Size",
                20,
                100,
                48,
                key=f"text_size_{project['name']}"
            )

            if st.button(
                "T APPLY TEXT",
                use_container_width=True
            ):
                if not overlay_text.strip():
                    st.warning("पहले text लिखें.")
                else:
                    safe_text = (
                        overlay_text
                        .replace("\\", "\\\\")
                        .replace(":", "\\:")
                        .replace("'", "\\'")
                    )

                    y_value = {
                        "Top": "80",
                        "Center": "(h-text_h)/2",
                        "Bottom": "h-text_h-80"
                    }[text_position]

                    output = (
                        project_path(project["name"])
                        / "videos"
                        / (
                            "edited_text_"
                            + datetime.now().strftime(
                                "%Y%m%d_%H%M%S"
                            )
                            + ".mp4"
                        )
                    )

                    drawtext_filter = (
                        "drawtext="
                        "fontfile=C\\:/Windows/Fonts/arial.ttf:"
                        f"text='{safe_text}':"
                        "fontcolor=white:"
                        f"fontsize={text_size}:"
                        "borderw=3:"
                        "bordercolor=black:"
                        f"x=(w-text_w)/2:"
                        f"y={y_value}"
                    )

                    run_ffmpeg([
                        "-y",
                        "-i", str(selected_editor_path),
                        "-vf", drawtext_filter,
                        "-c:v", "libx264",
                        "-preset", "veryfast",
                        "-crf", "18",
                        "-c:a", "copy",
                        "-movflags", "+faststart",
                        str(output)
                    ])

                    project.setdefault("videos", []).append({
                        "filename": str(
                            Path("videos") / output.name
                        ),
                        "created_at": now_string(),
                        "duration": get_media_duration(output)
                    })
                    save_project(project)

                    st.success("✅ Text added.")
                    st.video(str(output))
                    st.rerun()

        # ----------------------------------------------------
        # FRAME
        # ----------------------------------------------------
        elif editor_action == "⬜ Frame":

            frame_style = st.selectbox(
                "Frame Style",
                [
                    "Classic Border",
                    "Cinematic Border",
                    "Wide Border"
                ],
                key=f"frame_style_{project['name']}"
            )

            frame_width = st.slider(
                "Frame Width",
                4,
                60,
                12,
                key=f"frame_width_{project['name']}"
            )

            # White frame by default; style changes thickness.
            if frame_style == "Cinematic Border":
                frame_width = max(frame_width, 18)
            elif frame_style == "Wide Border":
                frame_width = max(frame_width, 30)

            if st.button(
                "⬜ APPLY FRAME",
                use_container_width=True
            ):
                output = (
                    project_path(project["name"])
                    / "videos"
                    / (
                        "edited_frame_"
                        + datetime.now().strftime(
                            "%Y%m%d_%H%M%S"
                        )
                        + ".mp4"
                    )
                )

                # Pad with a black border around the 1920x1080 video.
                vf = (
                    f"pad=iw+{frame_width*2}:"
                    f"ih+{frame_width*2}:"
                    f"{frame_width}:{frame_width}:color=black"
                )

                # Return to the standard 1920x1080 canvas.
                vf += (
                    ",scale=1920:1080:force_original_aspect_ratio=decrease,"
                    "pad=1920:1080:(ow-iw)/2:(oh-ih)/2"
                )

                run_ffmpeg([
                    "-y",
                    "-i", str(selected_editor_path),
                    "-vf", vf,
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-crf", "18",
                    "-c:a", "copy",
                    "-movflags", "+faststart",
                    str(output)
                ])

                project.setdefault("videos", []).append({
                    "filename": str(
                        Path("videos") / output.name
                    ),
                    "created_at": now_string(),
                    "duration": get_media_duration(output)
                })
                save_project(project)

                st.success("✅ Frame applied.")
                st.video(str(output))
                st.rerun()

        # ----------------------------------------------------
        # CROP
        # ----------------------------------------------------
        elif editor_action == "◫ Crop":

            crop_mode = st.selectbox(
                "Crop",
                [
                    "16:9 Full HD",
                    "1:1 Square",
                    "4:5 Portrait",
                    "9:16 Vertical"
                ],
                key=f"crop_mode_{project['name']}"
            )

            crop_map = {
                "16:9 Full HD": (1920, 1080),
                "1:1 Square": (1080, 1080),
                "4:5 Portrait": (864, 1080),
                "9:16 Vertical": (608, 1080)
            }

            crop_w, crop_h = crop_map[crop_mode]

            if st.button(
                "◫ APPLY CROP",
                use_container_width=True
            ):
                output = (
                    project_path(project["name"])
                    / "videos"
                    / (
                        "edited_crop_"
                        + datetime.now().strftime(
                            "%Y%m%d_%H%M%S"
                        )
                        + ".mp4"
                    )
                )

                vf = (
                    f"scale={crop_w}:{crop_h}:"
                    "force_original_aspect_ratio=increase,"
                    f"crop={crop_w}:{crop_h},"
                    "setsar=1"
                )

                run_ffmpeg([
                    "-y",
                    "-i", str(selected_editor_path),
                    "-vf", vf,
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-crf", "18",
                    "-c:a", "copy",
                    "-movflags", "+faststart",
                    str(output)
                ])

                project.setdefault("videos", []).append({
                    "filename": str(
                        Path("videos") / output.name
                    ),
                    "created_at": now_string(),
                    "duration": get_media_duration(output)
                })
                save_project(project)

                st.success(
                    f"✅ Crop applied: {crop_w}×{crop_h}"
                )
                st.video(str(output))
                st.rerun()

        # ----------------------------------------------------
        # SPEED
        # ----------------------------------------------------
        elif editor_action == "⏩ Speed":

            speed = st.select_slider(
                "Playback Speed",
                options=[
                    0.25,
                    0.5,
                    0.75,
                    1.0,
                    1.25,
                    1.5,
                    2.0,
                    3.0,
                    4.0
                ],
                value=1.0,
                key=f"speed_{project['name']}"
            )

            if st.button(
                "⏩ APPLY SPEED",
                use_container_width=True
            ):
                output = (
                    project_path(project["name"])
                    / "videos"
                    / (
                        "edited_speed_"
                        + datetime.now().strftime(
                            "%Y%m%d_%H%M%S"
                        )
                        + ".mp4"
                    )
                )

                # atempo accepts 0.5–2.0 per filter.
                # Chain filters for values outside that range.
                remaining = float(speed)
                atempo_parts = []

                while remaining < 0.5:
                    atempo_parts.append("atempo=0.5")
                    remaining /= 0.5

                while remaining > 2.0:
                    atempo_parts.append("atempo=2.0")
                    remaining /= 2.0

                atempo_parts.append(
                    f"atempo={remaining:.6f}"
                )

                audio_filter = ",".join(atempo_parts)
                video_filter = f"setpts={1.0/speed:.8f}*PTS"

                run_ffmpeg([
                    "-y",
                    "-i", str(selected_editor_path),
                    "-filter_complex",
                    (
                        f"[0:v]{video_filter}[v];"
                        f"[0:a]{audio_filter}[a]"
                    ),
                    "-map", "[v]",
                    "-map", "[a]",
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-crf", "18",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-movflags", "+faststart",
                    str(output)
                ])

                new_duration = (
                    get_media_duration(
                        selected_editor_path
                    ) / speed
                )

                project.setdefault("videos", []).append({
                    "filename": str(
                        Path("videos") / output.name
                    ),
                    "created_at": now_string(),
                    "duration": new_duration
                })
                save_project(project)

                st.success(
                    f"✅ Speed changed to {speed}×."
                )
                st.video(str(output))
                st.rerun()

    else:
        st.info(
            "पहले कोई generated video बनाएं. उसके बाद यहाँ "
            "Trim, Text, Frame, Crop और Speed tools काम करेंगे."
        )

    # --------------------------------------------------------
    # GENERATED VIDEOS
    # --------------------------------------------------------

    st.divider()
    st.subheader("🎞️ Generated Videos")

    videos = project.get(
        "videos",
        []
    )

    if not videos:
        st.info(
            "अभी कोई generated video नहीं है."
        )

    for video_index, video in enumerate(
        reversed(videos)
    ):

        video_path = (
            project_path(project["name"])
            / video["filename"]
        )

        if not video_path.exists():
            continue

        st.markdown(
            f"### 🎬 Video {len(videos) - video_index}"
        )

        vcol1, vcol2 = st.columns([3, 1])

        with vcol1:
            st.video(str(video_path))

        with vcol2:

            st.write("Created:")
            st.caption(
                video.get(
                    "created_at",
                    ""
                )
            )

            duration = float(
                video.get(
                    "duration",
                    0
                )
            )

            minutes = int(duration // 60)
            seconds = int(duration % 60)

            st.write(
                f"Duration: {minutes:02d}:{seconds:02d}"
            )

            with open(
                video_path,
                "rb"
            ) as f:

                st.download_button(
                    "⬇️ Download",
                    data=f.read(),
                    file_name=video_path.name,
                    mime="video/mp4",
                    key=f"download_{video_index}",
                    use_container_width=True
                )

        st.divider()

    # --------------------------------------------------------
    # FOLDER LOCATION
    # --------------------------------------------------------

    with st.expander(
        "📂 Project Folder Location"
    ):
        st.code(
            str(
                project_path(project["name"])
            )
        )

        st.caption(
            "इस folder में images, videos, audio और generated videos सुरक्षित रहेंगे."
        )


# ============================================================
# DELETE PROJECT CONFIRMATION POPUP
# ============================================================

if st.session_state.delete_project:

    project_to_delete = st.session_state.delete_project

    @st.dialog("⚠️ Delete Project")
    def delete_project_dialog():

        st.warning(
            "Are you sure you want to delete?"
        )

        st.write(
            f"Project: **{project_to_delete}**"
        )

        c1, c2 = st.columns(2)

        with c1:
            if st.button(
                "Yes, Delete",
                type="primary",
                use_container_width=True
            ):
                try:
                    shutil.rmtree(
                        project_path(project_to_delete)
                    )
                    st.session_state.delete_project = None

                    if (
                        st.session_state.selected_project
                        == project_to_delete
                    ):
                        st.session_state.selected_project = None

                    st.rerun()

                except Exception as e:
                    st.error(str(e))

        with c2:
            if st.button(
                "Cancel",
                use_container_width=True
            ):
                st.session_state.delete_project = None
                st.rerun()

    delete_project_dialog()


# ============================================================
# DELETE SCENE CONFIRMATION POPUP
# ============================================================

if st.session_state.delete_scene:

    scene_to_delete = st.session_state.delete_scene

    @st.dialog("⚠️ Delete Scene")
    def delete_scene_dialog():

        st.warning(
            "Are you sure you want to delete?"
        )

        st.write(
            "This scene and its voice-over file will be removed."
        )

        c1, c2 = st.columns(2)

        with c1:
            if st.button(
                "Yes, Delete",
                type="primary",
                use_container_width=True
            ):
                current_project = load_project(
                    st.session_state.selected_project
                )

                if current_project:
                    remove_item(
                        current_project,
                        scene_to_delete
                    )

                st.session_state.delete_scene = None
                st.rerun()

        with c2:
            if st.button(
                "Cancel",
                use_container_width=True
            ):
                st.session_state.delete_scene = None
                st.rerun()

    delete_scene_dialog()
