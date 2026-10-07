#!/usr/bin/env python3
"""Render images as Braille art with xterm-256 color and ncurses dim/normal/bold."""


import argparse
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import curses
from datetime import datetime
import sys
import os

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

_VERSION_ = "1.0"
_AUTHOR_ = "Arun Prakash Jana"
_AUTHOR_EMAIL_ = "engineerarun@gmail.com"
_LICENSE_ = "MIT"
_WEBPAGE_ = "https://github.com/jarun/dotz"

BRAILLE_BASE = 0x2800
BRAILLE_MAP = (
    (0x01, 0x08),  # row 0
    (0x02, 0x10),  # row 1
    (0x04, 0x20),  # row 2
    (0x40, 0x80),  # row 3
)
_BRAILLE_BITS = np.asarray(BRAILLE_MAP, dtype=np.uint16)

ASCII_DENSITY_CHARS = " .:-=+*#%@"
VIDEO_EXTS = frozenset({".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".mpeg", ".mpg"})
IMAGE_EXTS = frozenset({"png", "jpg", "jpeg", "bmp", "gif", "tiff", "webp"})
VIDEO_EXTS_NO_DOT = frozenset(ext.lstrip(".") for ext in VIDEO_EXTS)
VIDEO_SEEK_STEPS = (1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
VIDEO_PLAYBACK_DEFAULT_FPS = 5
VIDEO_PLAYBACK_BUFFER_FRAMES = 4
VIDEO_DECODE_THREADS = max(1, min(4, (os.cpu_count() or 1) // VIDEO_PLAYBACK_BUFFER_FRAMES))
STATUS_LINE_COUNT = 2
IMAGE_BRIGHTNESS = 1.2

# Ordered dither matrix for the 4×2 braille grid.
BAYER_4x2 = np.array([
    [0, 4],
    [2, 6],
    [5, 1],
    [7, 3],
], dtype=np.float32)
ORDERED_THRESHOLDS = (BAYER_4x2 + 0.5) / 8.0


def _precompute_braille_rows(frame, dither_mode):
    """Build immutable Braille output rows for non-error-dither frames."""
    rows, cols = frame.shape[0] // 4, frame.shape[1] // 2
    blocks = frame.reshape(rows, 4, cols, 2)
    if dither_mode == "ordered":
        dots = blocks > ORDERED_THRESHOLDS[None, :, None, :]
    elif dither_mode == "none":
        dots = blocks > 0.5
    else:
        return None  # error diffusion modes (error, atkinson) can't be precomputed
    codes = BRAILLE_BASE + np.sum(dots * _BRAILLE_BITS[None, :, None, :], axis=(1, 3), dtype=np.uint16)
    return tuple("".join(chr(int(code)) for code in row) for row in codes)


# ── xterm-256 color cube helpers ──────────────────────────────────────────────
# Colors 16-231 form a 6×6×6 RGB cube. Values per axis: 0,95,135,175,215,255.
# Colors 232-255 are a 24-step greyscale ramp.
_CUBE_VALS = np.array([0, 0x5f, 0x87, 0xaf, 0xd7, 0xff], dtype=np.float32)
_GREY_VALS = np.array([8 + 10 * i for i in range(24)], dtype=np.float32)
_COLOR_PAIR_ATTRS = None

def _nearest_xterm_indices(rgb_values):
    """Return exact nearest xterm-256 color indices without a full palette tensor."""
    samples = rgb_values.reshape(-1, 3)
    cube_axes = np.abs(samples[:, :, None] - _CUBE_VALS[None, None, :]).argmin(axis=2)
    cube_values = _CUBE_VALS[cube_axes]
    cube_distances = np.sum((samples - cube_values) ** 2, axis=1)

    grey_axes = np.abs(samples.mean(axis=1)[:, None] - _GREY_VALS[None, :]).argmin(axis=1)
    grey_values = _GREY_VALS[grey_axes]
    grey_distances = np.sum((samples - grey_values[:, None]) ** 2, axis=1)

    cube_indices = 16 + 36 * cube_axes[:, 0] + 6 * cube_axes[:, 1] + cube_axes[:, 2]
    return np.where(grey_distances < cube_distances, 232 + grey_axes, cube_indices).astype(np.int32, copy=False)


def _init_color_pairs():
    """Initialise ncurses color pairs 1-240 mapping to xterm colors 16-255."""
    global _COLOR_PAIR_ATTRS
    if _COLOR_PAIR_ATTRS is not None:
        return _COLOR_PAIR_ATTRS
    for i in range(240):
        xterm_idx = i + 16
        curses.init_pair(i + 1, xterm_idx, -1)  # fg=xterm color, bg=default
    _COLOR_PAIR_ATTRS = tuple(curses.color_pair(i + 1) for i in range(240))
    return _COLOR_PAIR_ATTRS


def srgb_to_linear(c):
    """Convert sRGB [0,1] to linear light."""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    """Convert linear light to sRGB [0,1]."""
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(np.clip(c, 0, None), 1.0 / 2.4) - 0.055)


def _transform_image(image, rotation_quadrants=0, flip_horizontal=False):
    """Return an oriented copy of a PIL image without modifying the source."""
    rotation_quadrants = int(rotation_quadrants) % 4
    transpose = getattr(Image, "Transpose", Image)
    rotation_ops = (None, transpose.ROTATE_270, transpose.ROTATE_180, transpose.ROTATE_90)
    if rotation_quadrants:
        image = image.transpose(rotation_ops[rotation_quadrants])
    if flip_horizontal:
        image = image.transpose(transpose.FLIP_LEFT_RIGHT)
    return image


def _load_image(image_path, img_w, img_h, sharpen, color, rotation_quadrants=0, flip_horizontal=False, ascii_mode=False):
    """Load and prepare image data. Accepts a file path or PIL Image. Returns (frames, color_maps, oy, ox, fit_h, fit_w, durations)."""
    img = image_path if isinstance(image_path, Image.Image) else Image.open(image_path)
    is_animated = getattr(img, "is_animated", False)
    n_frames = getattr(img, "n_frames", 1)
    # In ASCII mode, blocks are 2x2; in Braille mode, blocks are 4x2
    block_h = 2 if ascii_mode else 4
    cell_cols, cell_rows = img_w // 2, img_h // block_h
    max_w, max_h = cell_cols * 2, cell_rows * block_h
    # ASCII mode: characters are square (1:1), terminals are ~2:1
    # Need to compensate by stretching image vertically 2x
    aspect_correction = 2.0 if ascii_mode else 1.0
    frames = []
    color_maps = []
    durations = []
    for frame_idx in range(n_frames):
        if is_animated:
            img.seek(frame_idx)
            durations.append(img.info.get("duration", 100))
        frame = _transform_image(img, rotation_quadrants, flip_horizontal)
        img_rgb = ImageEnhance.Brightness(frame.convert("RGB")).enhance(IMAGE_BRIGHTNESS) if color else None
        img_grey = img_rgb.convert("L") if color else ImageEnhance.Brightness(frame.convert("L")).enhance(IMAGE_BRIGHTNESS)
        img_aspect = img_grey.width / img_grey.height
        # Apply aspect correction for ASCII mode
        effective_aspect = img_aspect * aspect_correction
        fit_w, fit_h = (max_w, int(round(max_w / effective_aspect))) if (max_w / effective_aspect) <= max_h else (int(round(max_h * effective_aspect)), max_h)
        fit_w, fit_h = min(fit_w, max_w), min(fit_h, max_h)
        img_grey_r = img_grey.resize((fit_w, fit_h), Image.LANCZOS)
        if sharpen: img_grey_r = img_grey_r.filter(ImageFilter.UnsharpMask(radius=1.2, percent=100, threshold=2))
        oy, ox = (img_h - fit_h) // 2, (img_w - fit_w) // 2
        raw = np.asarray(img_grey_r, dtype=np.float32) / 255.0
        linear = srgb_to_linear(raw)
        canvas = np.zeros((img_h, img_w), dtype=np.float32)
        canvas[oy:oy + fit_h, ox:ox + fit_w] = linear
        perceptual = linear_to_srgb(canvas)
        frames.append(perceptual)
        if color and img_rgb is not None:
            img_rgb_r = img_rgb.resize((fit_w, fit_h), Image.LANCZOS)
            rgb_arr = np.asarray(img_rgb_r, dtype=np.float32)
            canvas_rgb = np.zeros((img_h, img_w, 3), dtype=np.float32)
            canvas_rgb[oy:oy + fit_h, ox:ox + fit_w] = rgb_arr
            blocks = canvas_rgb[:cell_rows*block_h, :cell_cols*2, :].reshape(cell_rows, block_h, cell_cols, 2, 3)
            block_means = blocks.mean(axis=(1, 3), dtype=np.float32)
            color_map = _nearest_xterm_indices(block_means).reshape(cell_rows, cell_cols)
        else:
            color_map = None
        color_maps.append(color_map)
        if not is_animated: break
    return frames, color_maps, oy, ox, fit_h, fit_w, durations


# Helper to extract a video frame using ffmpeg and return a PIL Image
def _is_video_path(path):
    return os.path.splitext(str(path))[1].lower() in VIDEO_EXTS


def _clamp_video_position(position, duration=None):
    """Keep a video position within its known playable range."""
    try:
        position = max(0.0, float(position))
    except (TypeError, ValueError):
        return 0.0
    if duration is None:
        return position
    try:
        duration = float(duration)
    except (TypeError, ValueError):
        return position
    return min(position, max(0.0, duration - 0.001))


def _next_video_seek_step(current_step, direction):
    """Return the next configured video seek step in the requested direction."""
    closest_index = min(
        range(len(VIDEO_SEEK_STEPS)),
        key=lambda index: abs(VIDEO_SEEK_STEPS[index] - float(current_step)),
    )
    next_index = max(0, min(len(VIDEO_SEEK_STEPS) - 1, closest_index + direction))
    return VIDEO_SEEK_STEPS[next_index]


def _format_video_position(seconds):
    total_tenths = max(0, int(round(float(seconds) * 10)))
    minutes, tenths = divmod(total_tenths, 600)
    return f"{minutes}:{tenths // 10:02d}.{tenths % 10}"


def extract_video_frame(path, frametime, extractformat, keyframes_only=False):
    import subprocess, io
    vcodec = 'mjpeg' if extractformat == 'jpeg' else 'png'
    frametime = _clamp_video_position(frametime)
    # Seeking at/after the last frame yields no output, so retry earlier, then from the end.
    for seek_args in (
        ('-ss', f'{frametime:.3f}'),
        ('-ss', f'{max(0.0, frametime - 0.5):.3f}'),
        ('-ss', f'{max(0.0, frametime - 2.0):.3f}'),
        ('-sseof', '-1'),
        ('-sseof', '-0.1'),
        ('-ss', '0'),
    ):
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-nostdin', '-hide_banner', '-loglevel', 'error',
        ]
        if keyframes_only:
            ffmpeg_cmd.extend(('-skip_frame', 'nokey'))
        ffmpeg_cmd.extend((
            *seek_args, '-i', path,
            '-an', '-threads', str(VIDEO_DECODE_THREADS), '-vsync', '0',
            '-vframes', '1',
            '-strict', 'unofficial',
            '-f', 'image2pipe',
            '-vcodec', vcodec,
            '-',
        ))
        result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0 and result.stdout:
            break
    if result.returncode != 0 or not result.stdout:
        raise RuntimeError(f"ffmpeg error: {result.stderr.decode()[:100] or 'no frame at position'}")
    from PIL import Image
    img = Image.open(io.BytesIO(result.stdout))
    img.load()
    return img


def get_image_files(directory, include_videos=False):
    exts = IMAGE_EXTS | VIDEO_EXTS_NO_DOT if include_videos else IMAGE_EXTS
    files = []
    with os.scandir(directory) as entries:
        for entry in entries:
            if not entry.is_file():
                continue
            _, ext = os.path.splitext(entry.name)
            if ext.lower().lstrip(".") in exts:
                files.append(os.path.realpath(entry.path))
    files.sort()
    return files


def _display_name(image_path, display_name):
    return os.path.basename(display_name or image_path or "[video frame]") if display_name or isinstance(image_path, (str, bytes, os.PathLike)) else "[video frame]"


def _format_file_size(size):
    size = max(0, int(size))
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    unit_index = 0
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    precision = 0 if unit_index == 0 else 1
    return f"{size:.{precision}f} {units[unit_index]}"


def _format_duration(seconds):
    try:
        total_seconds = max(0, int(round(float(seconds))))
    except (TypeError, ValueError):
        return "unavailable"
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes}:{seconds:02d}"


def _probe_video_metadata(path):
    """Read video dimensions, container format, and duration with ffprobe JSON output."""
    import json
    import subprocess

    command = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=format_name,duration:stream=width,height",
        "-of", "json",
        os.fspath(path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=5)
        if result.returncode != 0:
            return {}
        probe = json.loads(result.stdout)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return {}

    streams = probe.get("streams", [])
    video_stream = next((stream for stream in streams if stream.get("width") and stream.get("height")), {})
    format_info = probe.get("format", {})
    return {
        "width": video_stream.get("width"),
        "height": video_stream.get("height"),
        "format": format_info.get("format_name"),
        "duration": format_info.get("duration"),
    }


def _metadata_lines(image_item):
    """Build metadata text for an image or video item without altering its rendered state."""
    image_path = image_item
    display_name = None
    if isinstance(image_item, tuple) and len(image_item) == 2:
        image_path, display_name = image_item

    shown_name = display_name or (_display_name(image_path, None) if not isinstance(image_path, Image.Image) else "[in-memory image]")
    file_size = "unavailable"
    modified = "unavailable"
    try:
        file_stat = os.stat(image_path)
        file_size = _format_file_size(file_stat.st_size)
        modified = datetime.fromtimestamp(file_stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    except (OSError, TypeError):
        pass

    if _is_video_path(image_path):
        metadata = _probe_video_metadata(image_path)
        width = metadata.get("width")
        height = metadata.get("height")
        dimensions = f"{width} x {height}" if width and height else "unavailable"
        format_name = (metadata.get("format") or os.path.splitext(str(image_path))[1].lstrip(".") or "unknown").upper()
        return [
            f"File: {shown_name}",
            f"Dimensions: {dimensions}",
            f"Format: {format_name}",
            f"Size: {file_size}",
            f"Modified: {modified}",
            f"Duration: {_format_duration(metadata.get('duration'))}",
        ]

    image = None
    close_image = False
    try:
        if isinstance(image_path, Image.Image):
            image = image_path
        else:
            image = Image.open(image_path)
            close_image = True
        format_name = (image.format or os.path.splitext(str(image_path))[1].lstrip(".") or "unknown").upper()
        lines = [
            f"File: {shown_name}",
            f"Dimensions: {image.width} x {image.height}",
            f"Format: {format_name}",
            f"Size: {file_size}",
            f"Modified: {modified}",
        ]
        if format_name == "GIF":
            lines.append(f"GIF frames: {getattr(image, 'n_frames', 1)}")
        return lines
    except (Image.UnidentifiedImageError, OSError, TypeError, ValueError):
        return [
            f"File: {shown_name}",
            "Dimensions: unavailable",
            "Format: unknown",
            f"Size: {file_size}",
            f"Modified: {modified}",
        ]
    finally:
        if close_image and image is not None:
            image.close()


_HELP_LINES = (
    "NAVIGATION",
    "  Right/n/Space  next",
    "  Left/p         previous",
    "  Up/Down        first / last",
    "VIEW",
    "  + / - / 0      zoom in / out / reset",
    "  h / j / k / l  pan left / down / up / right",
    "  r / f          rotate clockwise / flip",
    "VIDEO",
    "  [ / ]          seek backward / forward",
    "  { / }          smaller / larger seek step",
    "  , / .          previous / next playback frame",
    "  v              toggle video playback",
    "SLIDESHOW",
    "  s / S          forward / reverse",
    "  d / D          delay down / up by 1 sec",
    "THUMBNAILS",
    "  t / T          4 / 9 thumbnails",
    "  Enter          toggle thumbnails / image",
    "INFO & QUIT",
    "  i metadata     ? help     q / Esc quit",
)


def _show_panel(stdscr, title, lines, emphasized_rows=()):
    """Show a centered text panel and wait for one key press."""
    try:
        max_y, max_x = stdscr.getmaxyx()
        if max_y < 5 or max_x < 20:
            return
        panel_width = min(max_x - 2, max(24, len(title) + 6, max(len(line) for line in lines) + 4))
        panel_height = min(max_y - 2, len(lines) + 2)
        top = (max_y - panel_height) // 2
        left = (max_x - panel_width) // 2
        panel = curses.newwin(panel_height, panel_width, top, left)
        panel.box()
        panel.addnstr(0, 2, f" {title} ", panel_width - 4, curses.A_BOLD)
        for row, line in enumerate(lines[:panel_height - 2], start=1):
            attr = curses.A_BOLD if row - 1 in emphasized_rows else curses.A_NORMAL
            panel.addnstr(row, 2, line, panel_width - 4, attr)
        panel.refresh()
        stdscr.nodelay(False)
        panel.getch()
    except curses.error:
        pass
    finally:
        try:
            stdscr.nodelay(True)
        except curses.error:
            pass


def _show_metadata_panel(stdscr, lines):
    _show_panel(stdscr, "Metadata", lines)


def _show_help_panel(stdscr):
    _show_panel(stdscr, "Help", _HELP_LINES, emphasized_rows=(0, 4, 8, 13, 16, 19))


def _prepare_render_item(image_item, img_w, img_h, sharpen, color, seek, extractformat, rotation_quadrants=0, flip_horizontal=False, dither_mode="ordered", keyframes_only=False, ascii_mode=False):
    """Prepare renderable frame buffers for one item. Safe to run in worker threads."""
    image_path = image_item
    display_name = None
    if _is_video_path(image_path):
        display_name, image_path = image_path, extract_video_frame(image_path, seek, extractformat, keyframes_only)
    else:
        if isinstance(image_path, tuple) and len(image_path) == 2:
            image_path, display_name = image_path
        elif isinstance(image_path, Image.Image):
            display_name = '[video frame]'
    frames, color_maps, oy, ox, fit_h, fit_w, durations = _load_image(image_path, img_w, img_h, sharpen, color, rotation_quadrants, flip_horizontal, ascii_mode)
    # Block dimensions depend on mode: Braille uses 4x2, ASCII uses 2x2
    block_h = 2 if ascii_mode else 4
    block_means = [
        frame.reshape(img_h // block_h, block_h, img_w // 2, 2).mean(axis=(1, 3), dtype=np.float32)
        for frame in frames
    ]
    braille_rows = [
        _precompute_braille_rows(frame, dither_mode) for frame in frames
    ] if dither_mode in ("ordered", "none") and not ascii_mode else None
    return {
        "frames": frames,
        "color_maps": color_maps,
        "block_means": block_means,
        "braille_rows": braille_rows,
        "braille_dither_mode": dither_mode,
        "durations": durations,
        "shown_name": _display_name(image_path, display_name),
    }


def _clamp_delay(delay):
    try:
        delay = int(delay)
    except (TypeError, ValueError):
        return 1
    return max(1, min(60, delay))


class _PreparedFrameCache:
    """Bound prepared render items while keeping active cache targets available."""

    def __init__(self, capacity):
        self.capacity = max(1, int(capacity))
        self._items = OrderedDict()

    def get(self, key):
        try:
            value = self._items.pop(key)
        except KeyError:
            return None
        self._items[key] = value
        return value

    def put(self, key, value, protected_keys=()):
        protected_keys = set(protected_keys)
        self._items.pop(key, None)
        self._items[key] = value
        while len(self._items) > self.capacity:
            eviction_key = next((item_key for item_key in self._items if item_key not in protected_keys), None)
            if eviction_key is None:
                eviction_key = next(iter(self._items))
            self._items.pop(eviction_key)

    def discard_except(self, keys):
        keys = set(keys)
        for key in list(self._items):
            if key not in keys:
                self._items.pop(key)

    def keys(self):
        return tuple(self._items)

    def __contains__(self, key):
        return key in self._items


def _neighbor_indices(index, neighbor_count, item_count):
    """Return the current index followed by unique forward/backward neighbors."""
    indices = [index]
    for distance in range(1, neighbor_count + 1):
        for offset in (distance, -distance):
            neighbor = (index + offset) % item_count
            if neighbor not in indices:
                indices.append(neighbor)
    return indices


def _format_status(idx, total, shown_name, zoom_factor, slideshow_active, slideshow_reverse, delay, video_position=None, video_seek_step=None, video_playback=False):
    item_status = f"[{idx + 1}/{total}] {shown_name}"
    detail_status = f"{zoom_factor:.2f}x | slideshow: {'reverse' if slideshow_reverse else 'forward' if slideshow_active else 'off'} | {delay}s"
    if video_position is not None:
        detail_status += f" | video: {_format_video_position(video_position)} | step: {video_seek_step:g}s | {'playback' if video_playback else 'paused'}"
    return item_status, detail_status


def _update_pan_offset(pan_y, pan_x, key):
    pan_step = 0.125
    offsets = {'pan_left': (0.0, -pan_step), 'pan_right': (0.0, pan_step), 'pan_up': (-pan_step, 0.0), 'pan_down': (pan_step, 0.0)}
    delta_y, delta_x = offsets.get(key, (0.0, 0.0))
    return max(-1.0, min(1.0, pan_y + delta_y)), max(-1.0, min(1.0, pan_x + delta_x))


def _zoom_array(array, zoom_factor, target_size, pan_offset=(0.0, 0.0), fill_value=0):
    """Return a centered, optionally panned nearest-neighbor zoom of a two-dimensional array."""
    if array is None:
        return None

    target_h, target_w = target_size
    zoom_factor = float(zoom_factor or 1.0)
    pan_y, pan_x = pan_offset
    pan_y = max(-1.0, min(1.0, float(pan_y)))
    pan_x = max(-1.0, min(1.0, float(pan_x)))
    if zoom_factor <= 0:
        zoom_factor = 1.0
    if zoom_factor == 1.0 and array.shape == (target_h, target_w):
        return array

    source_h, source_w = array.shape
    if zoom_factor > 1.0:
        visible_h = source_h / zoom_factor
        visible_w = source_w / zoom_factor
        top = (source_h - visible_h) * (pan_y + 1.0) / 2.0
        left = (source_w - visible_w) * (pan_x + 1.0) / 2.0
        row_indices = np.minimum(
            source_h - 1,
            (top + (np.arange(target_h) + 0.5) * visible_h / target_h).astype(np.intp),
        )
        col_indices = np.minimum(
            source_w - 1,
            (left + (np.arange(target_w) + 0.5) * visible_w / target_w).astype(np.intp),
        )
        return array[row_indices[:, None], col_indices]

    scaled_h = max(1, int(round(target_h * zoom_factor)))
    scaled_w = max(1, int(round(target_w * zoom_factor)))
    row_indices = np.minimum(
        source_h - 1,
        ((np.arange(scaled_h) + 0.5) * source_h / scaled_h).astype(np.intp),
    )
    col_indices = np.minimum(
        source_w - 1,
        ((np.arange(scaled_w) + 0.5) * source_w / scaled_w).astype(np.intp),
    )
    result = np.full((target_h, target_w), fill_value, dtype=array.dtype)
    top = (target_h - scaled_h) // 2
    left = (target_w - scaled_w) // 2
    result[top:top + scaled_h, left:left + scaled_w] = array[row_indices[:, None], col_indices]
    return result


def _blocks_to_chars(blocks, ascii_mode=False):
    """Convert pixel blocks to characters (ASCII density or Braille)."""
    if ascii_mode:
        coverage = (blocks > 0.5).mean(axis=(1, 3))
        indices = np.rint(coverage * (len(ASCII_DENSITY_CHARS) - 1)).astype(np.intp)
        chars = np.asarray(list(ASCII_DENSITY_CHARS), dtype='<U1')[indices]
        return tuple("".join(row) for row in chars)
    else:
        # Braille mode
        codes = BRAILLE_BASE + np.sum(blocks * _BRAILLE_BITS[None, :, None, :], axis=(1, 3), dtype=np.uint16)
        return tuple("".join(chr(int(code)) for code in row) for row in codes)


def _update_slideshow_state(slideshow_active, slideshow_reverse, key):
    if key == 'toggle_slideshow':
        return (True, False) if not slideshow_active else (False, False) if not slideshow_reverse else (True, False)
    if key == 'toggle_slideshow_reverse':
        return (False, False) if slideshow_active and slideshow_reverse else (True, True)
    return slideshow_active, slideshow_reverse


def _brightness_attr(avg, color_idx=None, color_pair_attrs=None):
    attr = curses.A_DIM if avg < 0.30 else curses.A_NORMAL if avg < 0.62 else curses.A_BOLD
    if color_idx is not None and color_idx >= 16 and color_pair_attrs is not None:
        attr |= color_pair_attrs[color_idx - 16]
    return attr


def _braille_code(block, threshold):
    return BRAILLE_BASE + int(np.sum((block > threshold) * _BRAILLE_BITS, dtype=np.uint16))


def _draw_braille_rows(stdscr, rows, cols, blocks, block_means, thresholds, color_map, color, color_pair_attrs, use_error_dither, use_ordered_dither, char_rows=None):
    for cy in range(rows):
        char_row = char_rows[cy] if char_rows is not None else None
        color_row = color_map[cy] if color and color_map is not None else None
        if char_row is not None:
            current_attr = None
            start_x = 0
            for cx in range(cols):
                attr = _brightness_attr(block_means[cy, cx], color_row[cx] if color_row is not None else None, color_pair_attrs)
                if current_attr is None:
                    start_x, current_attr = cx, attr
                elif current_attr != attr:
                    try:
                        stdscr.addstr(cy, start_x, char_row[start_x:cx], current_attr)
                    except curses.error:
                        pass
                    start_x, current_attr = cx, attr
            if current_attr is not None:
                try:
                    stdscr.addstr(cy, start_x, char_row[start_x:], current_attr)
                except curses.error:
                    pass
            continue

        segments = []
        current_attr = None
        current_chars = []
        start_x = 0
        threshold = thresholds if use_ordered_dither else 0.5
        for cx in range(cols):
            attr = _brightness_attr(block_means[cy, cx], color_row[cx] if color_row is not None else None, color_pair_attrs)
            ch = chr(_braille_code(blocks[cy, :, cx, :], threshold))
            if current_attr is None or current_attr != attr:
                if current_chars:
                    segments.append((start_x, current_attr, "".join(current_chars)))
                start_x, current_attr, current_chars = cx, attr, [ch]
            else:
                current_chars.append(ch)
        if current_chars:
            segments.append((start_x, current_attr, "".join(current_chars)))
        for x, attr, text in segments:
            try:
                stdscr.addstr(cy, x, text, attr)
            except curses.error:
                pass


def _discard_queued_key_repeats(stdscr, repeated_key):
    while True:
        queued_key = stdscr.getch()
        if queued_key == -1:
            return
        if queued_key != repeated_key:
            curses.ungetch(queued_key)
            return


_VIEW_KEY_COMMANDS = {
    ord('+'): 'zoom_in', ord('-'): 'zoom_out', ord('0'): 'reset_zoom',
    ord('h'): 'pan_left', ord('j'): 'pan_down', ord('k'): 'pan_up',
    ord('l'): 'pan_right', ord('r'): 'rotate_clockwise',
    ord('f'): 'flip_horizontal', ord('i'): 'show_metadata', ord('?'): 'show_help',
}

_VIDEO_KEY_COMMANDS = {
    ord('['): 'video_seek_backward', ord(']'): 'video_seek_forward',
    ord('{'): 'video_seek_step_down', ord('}'): 'video_seek_step_up',
    ord(','): 'video_previous_frame', ord('.'): 'video_next_frame',
    ord('v'): 'toggle_video_playback',
}


def _key_command(key, video_position):
    return _VIEW_KEY_COMMANDS.get(key) or (_VIDEO_KEY_COMMANDS.get(key) if video_position is not None else None)


def render(stdscr, image_files, idx, sharpen, dither_mode, color, wait_time=5, slideshow=False, slideshow_reverse=False, prepared=None, zoom_factor=1.0, pan_offset=(0.0, 0.0), rotation_quadrants=0, flip_horizontal=False, video_position=None, video_seek_step=None, video_playback=False, video_playback_fps=VIDEO_PLAYBACK_DEFAULT_FPS, discard_queued_key=None, ascii_mode=False):
    import time
    curses.curs_set(0)
    curses.use_default_colors()
    if color:
        curses.start_color()
        color_pair_attrs = _init_color_pairs()
    else:
        color_pair_attrs = None

    n = len(image_files)
    def floyd_steinberg_dither(img):
        arr = img.copy()
        h, w = arr.shape
        for y in range(h):
            for x in range(w):
                old = arr[y, x]
                new = 1.0 if old > 0.5 else 0.0
                err = old - new
                arr[y, x] = new
                if x + 1 < w:
                    arr[y, x+1] += err * 7/16
                if y + 1 < h:
                    if x > 0:
                        arr[y+1, x-1] += err * 3/16
                    arr[y+1, x] += err * 5/16
                    if x + 1 < w:
                        arr[y+1, x+1] += err * 1/16
        return arr

    def atkinson_dither(img):
        """Atkinson dithering - preserves average brightness better than Floyd-Steinberg."""
        arr = img.copy()
        h, w = arr.shape
        for y in range(h):
            for x in range(w):
                old = arr[y, x]
                new = 1.0 if old > 0.5 else 0.0
                err = old - new
                arr[y, x] = new
                # Distribute error to 6 neighbors (1/8 each)
                if x + 1 < w:
                    arr[y, x+1] += err * 0.125
                if x + 2 < w:
                    arr[y, x+2] += err * 0.125
                if y + 1 < h:
                    if x > 0:
                        arr[y+1, x-1] += err * 0.125
                    arr[y+1, x] += err * 0.125
                    if x + 1 < w:
                        arr[y+1, x+1] += err * 0.125
                if y + 2 < h:
                    arr[y+2, x] += err * 0.125
        return arr

    stdscr.erase()
    max_y, max_x = stdscr.getmaxyx()
    rows = max_y - STATUS_LINE_COUNT
    cols = max_x
    status_y = max_y - STATUS_LINE_COUNT
    # In ASCII mode, each character cell is 2x2 pixels (vs 4x2 for Braille)
    if ascii_mode:
        img_w = cols * 2
        img_h = rows * 2
    else:
        img_w = cols * 2
        img_h = rows * 4
    try:
        image_item = image_files[idx]
        if prepared is None:
            seek = getattr(render, '_seek', 10)
            fmt = getattr(render, '_format', 'jpeg')
            prepared = _prepare_render_item(image_item, img_w, img_h, sharpen, color, seek, fmt, rotation_quadrants, flip_horizontal, dither_mode, ascii_mode=ascii_mode)
        frames = prepared["frames"]
        color_maps = prepared["color_maps"]
        base_block_means = prepared.get("block_means")
        base_braille_rows = prepared.get("braille_rows") if prepared.get("braille_dither_mode") == dither_mode else None
        durations = prepared["durations"]
        shown_name = prepared["shown_name"]
        is_animated = len(frames) > 1
        frame_idx = 0
        key = -1
        stdscr.nodelay(True)
        if discard_queued_key is not None:
            _discard_queued_key_repeats(stdscr, discard_queued_key)
    except Exception as e:
        stdscr.clear()
        stdscr.addstr(0, 0, f"Render error: {e}")
        stdscr.refresh()
        stdscr.getch()
        return -1
    thresholds = ORDERED_THRESHOLDS
    use_error_dither = dither_mode == "error"
    use_atkinson_dither = dither_mode == "atkinson"
    use_ordered_dither = dither_mode == "ordered"
    # In ASCII mode, blocks are 2x2 (1 character cell = 2x2 pixels)
    # In Braille mode, blocks are 4x2 (1 character cell = 4x2 pixels)
    block_h = 2 if ascii_mode else 4
    while True:
        stdscr.erase()
        perceptual = frames[frame_idx]
        color_map = color_maps[frame_idx] if color_maps else None
        if zoom_factor == 1.0:
            frame_view = perceptual
            if use_error_dither:
                dithered = floyd_steinberg_dither(frame_view.copy())
                blocks = dithered.reshape(rows, block_h, cols, 2)
                block_means = blocks.mean(axis=(1, 3), dtype=np.float32)
            elif use_atkinson_dither:
                dithered = atkinson_dither(frame_view.copy())
                blocks = dithered.reshape(rows, block_h, cols, 2)
                block_means = blocks.mean(axis=(1, 3), dtype=np.float32)
            else:
                blocks = frame_view.reshape(rows, block_h, cols, 2)
                block_means = base_block_means[frame_idx] if base_block_means is not None else blocks.mean(axis=(1, 3), dtype=np.float32)
            braille_rows = base_braille_rows[frame_idx] if base_braille_rows is not None else None
        else:
            frame_view = _zoom_array(perceptual, zoom_factor, target_size=(rows * block_h, cols * 2), pan_offset=pan_offset)
            color_map = _zoom_array(color_map, zoom_factor, target_size=(rows, cols), pan_offset=pan_offset)
            if use_error_dither:
                dithered = floyd_steinberg_dither(frame_view.copy())
                blocks = dithered.reshape(rows, block_h, cols, 2)
            elif use_atkinson_dither:
                dithered = atkinson_dither(frame_view.copy())
                blocks = dithered.reshape(rows, block_h, cols, 2)
            else:
                blocks = frame_view.reshape(rows, block_h, cols, 2)
            block_means = blocks.mean(axis=(1, 3), dtype=np.float32)
            braille_rows = None

        # Convert blocks to appropriate characters based on mode
        if ascii_mode:
            char_rows = _blocks_to_chars(blocks, ascii_mode=True)
        else:
            char_rows = braille_rows

        _draw_braille_rows(stdscr, rows, cols, blocks, block_means, thresholds, color_map, color, color_pair_attrs, use_error_dither, use_ordered_dither, char_rows)
        try:
            item_status, detail_status = _format_status(idx, n, shown_name, zoom_factor, slideshow, slideshow_reverse, wait_time, video_position, video_seek_step, video_playback)
            status_width = max(0, max_x - 1)
            if max_y >= STATUS_LINE_COUNT:
                stdscr.addnstr(status_y, 0, item_status, status_width, curses.A_REVERSE)
                stdscr.addnstr(status_y + 1, 0, detail_status, status_width, curses.A_REVERSE)
            else:
                stdscr.addnstr(max_y - 1, 0, f"{item_status} | {detail_status}", status_width, curses.A_REVERSE)
        except curses.error:
            pass
        stdscr.refresh()
        if is_animated:
            duration = durations[frame_idx] / 1000.0 if frame_idx < len(durations) else 0.1
            start_time = time.time()
            while True:
                key = stdscr.getch()
                if key != -1:
                    stdscr.nodelay(False)
                    command = _key_command(key, video_position)
                    if command is not None:
                        return command
                    return key
                if (time.time() - start_time) >= duration:
                    break
                time.sleep(0.01)
            frame_idx = (frame_idx + 1) % len(frames)
        else:
            start_time = time.time() if slideshow or video_playback else None
            while True:
                key = stdscr.getch()
                command = _key_command(key, video_position)
                if command is not None:
                    return command
                if key == ord('s'):
                    return 'toggle_slideshow'
                if key == ord('S'):
                    return 'toggle_slideshow_reverse'
                if key == ord('D'):
                    return 'increase_delay'
                if key == ord('d'):
                    return 'decrease_delay'
                if key != -1:
                    stdscr.nodelay(False)
                    return key
                if slideshow and start_time is not None and (time.time() - start_time) >= wait_time:
                    return 'slideshow_next'
                if video_playback and start_time is not None and (time.time() - start_time) >= 1.0 / video_playback_fps:
                    return 'video_playback_next'
                time.sleep(0.01)
    return key


THUMBNAILS_DEFAULT = 4
THUMBNAILS_CHOICES = (4, 9)
_ENTER_KEYS = (10, 13, curses.KEY_ENTER)


def _thumbnail_layout(count, max_y, max_x):
    """Return (grid_cols, grid_rows, cell_w, cell_h, thumb_w, thumb_h) or None if the terminal is too small."""
    grid_cols = grid_rows = 3 if count == 9 else 2
    cell_w = max_x // grid_cols
    cell_h = (max_y - STATUS_LINE_COUNT) // grid_rows
    thumb_h = cell_h - 1  # last row of a cell holds the caption
    if cell_w < 2 or thumb_h < 1:
        return None
    return grid_cols, grid_rows, cell_w, cell_h, cell_w, thumb_h


class _OffsetWindow:
    """Translate addstr coordinates so a thumbnail can be drawn into a screen region."""

    def __init__(self, win, y, x):
        self.win, self.y, self.x = win, y, x

    def addstr(self, y, x, text, attr=0):
        self.win.addstr(self.y + y, self.x + x, text, attr)


def render_thumbnails(stdscr, page_names, page_prepared, selected, layout, page, page_count, total, idx, color, ascii_mode):
    """Draw one page of thumbnails and block for a key press."""
    curses.curs_set(0)
    curses.use_default_colors()
    color_pair_attrs = None
    if color:
        curses.start_color()
        color_pair_attrs = _init_color_pairs()
    grid_cols, _, cell_w, cell_h, thumb_w, thumb_h = layout
    max_y, max_x = stdscr.getmaxyx()
    stdscr.erase()
    for slot, (name, prepared) in enumerate(zip(page_names, page_prepared)):
        row, col = divmod(slot, grid_cols)
        y0, x0 = row * cell_h, col * cell_w
        if prepared is not None:
            frame = prepared["frames"][0]
            if ascii_mode:
                blocks = frame.reshape(thumb_h, 2, thumb_w, 2)
                char_rows = _blocks_to_chars(blocks, ascii_mode=True)
            else:
                blocks = None
                char_rows = prepared["braille_rows"][0]
            color_map = prepared["color_maps"][0]
            _draw_braille_rows(_OffsetWindow(stdscr, y0, x0), thumb_h, thumb_w, blocks, prepared["block_means"][0], None,
                               color_map, color, color_pair_attrs, False, False, char_rows)
        try:
            attr = curses.A_REVERSE if slot == selected else curses.A_NORMAL
            stdscr.addnstr(y0 + thumb_h, x0, name.ljust(cell_w - 1), cell_w - 1, attr)
        except curses.error:
            pass
    try:
        status_y = max_y - STATUS_LINE_COUNT
        width = max(0, max_x - 1)
        stdscr.addnstr(status_y, 0, f"[{idx + 1}/{total}] {page_names[selected]}", width, curses.A_REVERSE)
        stdscr.addnstr(status_y + 1, 0, f"thumbnails | page {page + 1}/{page_count} | Enter: open image", width, curses.A_REVERSE)
    except curses.error:
        pass
    stdscr.refresh()
    stdscr.nodelay(False)
    return stdscr.getch()


def main():
    class ExtendedArgumentParser(argparse.ArgumentParser):
        @staticmethod
        def print_extended_help(file=None):
            if file is None:
                file = sys.stderr
            file.write(
                "\n"
                f"Version: {_VERSION_}\n"
                f"Author: {_AUTHOR_} <{_AUTHOR_EMAIL_}>\n"
                f"License: {_LICENSE_}\n"
                f"Webpage: {_WEBPAGE_}\n"
            )

        def print_help(self, file=None):
            super().print_help(file)
            self.print_extended_help(file)

    parser = ExtendedArgumentParser(description="Render an image or all images/videos in a directory as Braille and ASCII cells using ncurses with optional xterm-256 color.")
    parser.add_argument("path", nargs="?", help="Path to the image/video file or directory (optional)")
    parser.add_argument("-S", "--no-sharpen", action="store_true", help="Disable edge sharpening")
    parser.add_argument("-C", "--no-color", action="store_true", help="Disable color (greyscale only with dim/normal/bold)")
    parser.add_argument("-d", "--dither", choices=["ordered", "error", "atkinson", "none"], default="ordered",
                        help="Dithering mode: ordered (default, clean), error (Floyd-Steinberg, smooth gradients), atkinson (Atkinson, preserves brightness), none")
    parser.add_argument("-a", "--ascii", action="store_true", help="Use ASCII characters instead of Braille (for terminals without Braille font support)")

    parser.add_argument("-s", "--slideshow", dest="delay", nargs="?", const=5, type=int, help="Enable slideshow mode with optional integer delay in seconds (default: 5).")
    parser.add_argument("-k", "--seek", type=int, default=10, help="Seek position to extract frame from videos in seconds (default: 10)")
    parser.add_argument("-f", "--format", type=str, choices=["jpeg", "png"], default="jpeg", help="Format for extracted video frames: jpeg (default) or png")
    parser.add_argument("-F", "--fps", type=int, choices=range(5, 11), default=VIDEO_PLAYBACK_DEFAULT_FPS,
                        help="Video playback frame rate between 5 and 10 FPS (default: 5)")
    parser.add_argument("-t", "--thumbnails", nargs="?", const=THUMBNAILS_DEFAULT, type=int, choices=THUMBNAILS_CHOICES, metavar="N",
                        help=f"Show N thumbnails per page: 4 (2x2) or 9 (3x3), default: {THUMBNAILS_DEFAULT}; press Enter to open an image.")
    parser.add_argument("-v", "--version", action="version", version=_VERSION_)
    args = parser.parse_args()

    preload_neighbors = 1
    cache_size = max(3, VIDEO_PLAYBACK_BUFFER_FRAMES + 1)

    slideshow = args.delay is not None
    slideshow_delay = _clamp_delay(args.delay if slideshow else 5)

    if args.path == '-':
        # Read image from stdin
        from PIL import Image
        import tempfile
        import shutil
        # Read stdin to a temporary file (since PIL.Image.open(sys.stdin.buffer) may not work for all formats)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
            shutil.copyfileobj(sys.stdin.buffer, tmp)
            tmp_path = tmp.name
        image_files = [tmp_path]
        idx = 0
        # Redirect stdin file descriptor to /dev/tty so curses reads input from the terminal
        # (removed local import os; using global import)
        try:
            tty_fd = os.open('/dev/tty', os.O_RDWR)
            orig_stdin_fd = os.dup(0)
            os.dup2(tty_fd, 0)
            os.close(tty_fd)
            try:
                curses.wrapper(lambda *a, **kw: render(*a, **kw, wait_time=slideshow_delay, slideshow=slideshow, video_playback_fps=args.fps, ascii_mode=args.ascii), image_files, idx, not args.no_sharpen, args.dither, not args.no_color)
            finally:
                os.dup2(orig_stdin_fd, 0)
                os.close(orig_stdin_fd)
        finally:
            os.unlink(tmp_path)
        return
    elif args.path:
        # If a directory is passed, render all images/videos in it
        if os.path.isdir(args.path):
            directory = os.path.abspath(args.path)
            image_files = get_image_files(directory, include_videos=args.seek is not None)
            if not image_files:
                print(f"No images or videos found in directory: {directory}", file=sys.stderr)
                sys.exit(1)
            idx = 0
        else:
            abs_path = os.path.abspath(args.path)
            directory = os.path.dirname(abs_path) or os.getcwd()
            # Always include videos if the selected file is a video
            include_videos = _is_video_path(abs_path) or args.seek is not None
            image_files = get_image_files(directory, include_videos=include_videos)
            if not image_files:
                print(f"No images or videos found in directory: {directory}", file=sys.stderr)
                sys.exit(1)
            try:
                idx = image_files.index(abs_path)
            except ValueError:
                base = os.path.basename(abs_path)
                idx = next((i for i, f in enumerate(image_files) if os.path.basename(f) == base), 0)
    else:
        # No argument: use current directory
        directory = os.getcwd()
        include_videos = args.seek is not None
        image_files = get_image_files(directory, include_videos=include_videos)
        if not image_files:
            print(f"No images or videos found in current directory.", file=sys.stderr)
            sys.exit(1)
        idx = 0


    # Pass seek and format to render via function attributes for video frame extraction
    def render_with_video_support(stdscr, image_files, start_idx, sharpen, dither_mode, color, single_image_mode=False, wait_time=5, slideshow=False):
        idx = start_idx
        n = len(image_files)
        max_workers = min(4, cache_size, max(1, os.cpu_count() or 1))
        prepared_cache = _PreparedFrameCache(cache_size)
        preload_futures = {}
        video_positions = {}
        video_durations = {}
        video_precise_paths = set()
        video_seek_step = VIDEO_SEEK_STEPS[2]
        video_playback_active = False
        video_playback_frame_seconds = 1.0 / args.fps

        def viewport_dims():
            max_y, max_x = stdscr.getmaxyx()
            # Reserve space for status lines
            render_rows = max_y - STATUS_LINE_COUNT
            # In ASCII mode, each character cell is 2x2 pixels (vs 4x2 for Braille)
            if args.ascii:
                return max_x * 2, render_rows * 2
            return max_x * 2, render_rows * 4

        def video_path_for_index(image_idx):
            image_item = image_files[image_idx]
            if isinstance(image_item, tuple) and len(image_item) == 2:
                image_item = image_item[0]
            return image_item if _is_video_path(image_item) else None

        def video_duration(path):
            cache_key = os.fspath(path)
            if cache_key not in video_durations:
                metadata = _probe_video_metadata(path)
                try:
                    duration = float(metadata.get("duration"))
                except (TypeError, ValueError):
                    duration = None
                video_durations[cache_key] = duration if duration is not None and duration >= 0 else None
            return video_durations[cache_key]

        def video_position_for_index(image_idx):
            path = video_path_for_index(image_idx)
            if path is None:
                return None
            cache_key = os.fspath(path)
            if cache_key not in video_positions:
                video_positions[cache_key] = _clamp_video_position(args.seek)
            return video_positions[cache_key]

        def move_video_position(image_idx, offset):
            path = video_path_for_index(image_idx)
            if path is None:
                return False
            cache_key = os.fspath(path)
            current_position = video_position_for_index(image_idx)
            next_position = _clamp_video_position(current_position + offset, video_duration(path))
            video_positions[cache_key] = next_position
            if next_position != current_position:
                video_precise_paths.add(cache_key)
            return next_position != current_position

        def use_fast_initial_extract(image_idx):
            path = video_path_for_index(image_idx)
            return path is not None and os.fspath(path) not in video_precise_paths

        def preload_key(image_idx, img_w, img_h, rotation_quadrants, flip_horizontal, video_position=None):
            if video_position is None:
                video_position = video_position_for_index(image_idx)
            return (
                image_idx,
                img_w,
                img_h,
                sharpen,
                color,
                video_position,
                use_fast_initial_extract(image_idx),
                args.format,
                rotation_quadrants,
                flip_horizontal,
                args.ascii,
            )

        def preload_targets(image_idx, current_rotation, current_flip):
            targets = []
            for target_idx in _neighbor_indices(image_idx, preload_neighbors, n):
                rotation = current_rotation if target_idx == image_idx else 0
                flip = current_flip if target_idx == image_idx else False
                targets.append((target_idx, rotation, flip))
            return targets

        def schedule_preload(image_idx, img_w, img_h, rotation_quadrants, flip_horizontal, executor, video_position=None):
            key = preload_key(image_idx, img_w, img_h, rotation_quadrants, flip_horizontal, video_position)
            if key in prepared_cache or key in preload_futures:
                return
            if video_position is None:
                video_position = video_position_for_index(image_idx)
            keyframes_only = use_fast_initial_extract(image_idx)
            preload_futures[key] = executor.submit(
                _prepare_render_item,
                image_files[image_idx],
                img_w,
                img_h,
                sharpen,
                color,
                args.seek if video_position is None else video_position,
                args.format,
                rotation_quadrants,
                flip_horizontal,
                dither_mode,
                keyframes_only,
                ascii_mode=args.ascii,
            )

        def get_preloaded(image_idx, img_w, img_h, rotation_quadrants, flip_horizontal, protected_keys, executor):
            key = preload_key(image_idx, img_w, img_h, rotation_quadrants, flip_horizontal)
            prepared = prepared_cache.get(key)
            if prepared is not None:
                return prepared
            schedule_preload(image_idx, img_w, img_h, rotation_quadrants, flip_horizontal, executor)
            future = preload_futures.pop(key)
            prepared = future.result()
            prepared_cache.put(key, prepared, protected_keys)
            return prepared

        def collect_completed_preloads(keep_keys):
            for key, future in list(preload_futures.items()):
                if not future.done():
                    continue
                preload_futures.pop(key)
                if future.cancelled():
                    continue
                try:
                    prepared_cache.put(key, future.result(), keep_keys)
                except Exception:
                    pass

        def trim_preload_cache(keep_keys):
            prepared_cache.discard_except(keep_keys)
            stale_keys = [k for k in preload_futures.keys() if k not in keep_keys]
            for key in stale_keys:
                future = preload_futures.pop(key)
                future.cancel()

        # Provide the initial video seek/format for direct render calls.
        render._seek = args.seek
        render._format = args.format
        slideshow_active = slideshow
        slideshow_reverse = False
        current_delay = wait_time
        zoom_factor = 1.0
        pan_y = 0.0
        pan_x = 0.0
        rotation_quadrants = 0
        flip_horizontal = False
        last_rendered_idx = idx
        discard_queued_key = None
        thumb_mode = args.thumbnails is not None
        thumb_count = args.thumbnails or THUMBNAILS_DEFAULT
        thumb_cache = {}
        thumb_futures = {}
        thumb_dither = dither_mode if dither_mode in ("ordered", "none") else "ordered"

        def prepare_thumb_page(page, count, thumb_w, thumb_h):
            block_h = 2 if args.ascii else 4
            items = []
            for i in range(page * count, min(n, (page + 1) * count)):
                try:
                    items.append(_prepare_render_item(image_files[i], thumb_w * 2, thumb_h * block_h, sharpen, color, args.seek,
                                                      args.format, 0, False, thumb_dither, True, ascii_mode=args.ascii))
                except Exception:
                    items.append(None)
            return items

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            while True:
                if idx != last_rendered_idx:
                    stdscr.erase()
                    zoom_factor = 1.0
                    pan_y = 0.0
                    pan_x = 0.0
                    rotation_quadrants = 0
                    flip_horizontal = False
                    video_playback_active = False
                    last_rendered_idx = idx

                if thumb_mode:
                    max_y, max_x = stdscr.getmaxyx()
                    layout = _thumbnail_layout(thumb_count, max_y, max_x)
                    if layout is None:
                        stdscr.erase()
                        try:
                            stdscr.addstr(0, 0, "Terminal too small")
                        except curses.error:
                            pass
                        stdscr.refresh()
                        stdscr.nodelay(False)
                        if stdscr.getch() in (ord('q'), 27):
                            break
                        continue
                    thumb_w, thumb_h = layout[4], layout[5]
                    page = idx // thumb_count
                    thumb_pages = -(-n // thumb_count)
                    wanted = {(p, thumb_count, thumb_w, thumb_h) for p in ((page + d) % thumb_pages for d in (0, 1, -1))}
                    for stale in [k for k in thumb_cache if k not in wanted]:
                        del thumb_cache[stale]
                    for stale in [k for k in thumb_futures if k not in wanted]:
                        thumb_futures.pop(stale).cancel()
                    current_key = (page, thumb_count, thumb_w, thumb_h)
                    for wanted_key in [current_key] + [k for k in wanted if k != current_key]:
                        if wanted_key not in thumb_cache and wanted_key not in thumb_futures:
                            thumb_futures[wanted_key] = executor.submit(prepare_thumb_page, *wanted_key)
                    for done_key, future in list(thumb_futures.items()):
                        if done_key != current_key and future.done():
                            del thumb_futures[done_key]
                            if not future.cancelled() and future.exception() is None:
                                thumb_cache[done_key] = future.result()
                    if current_key not in thumb_cache:
                        thumb_cache[current_key] = thumb_futures.pop(current_key).result()
                    first = page * thumb_count
                    page_prepared = thumb_cache[current_key]
                    page_names = [os.path.basename(str(image_files[i])) for i in range(first, first + len(page_prepared))]
                    key = render_thumbnails(stdscr, page_names, page_prepared, idx - first, layout, page, thumb_pages, n, idx, color, args.ascii)
                    if key in (curses.KEY_RIGHT, ord('n'), ord(' ')):
                        idx = (idx + 1) % n
                    elif key in (curses.KEY_LEFT, ord('p')):
                        idx = (idx - 1) % n
                    elif key == curses.KEY_UP:
                        idx = 0
                    elif key == curses.KEY_DOWN:
                        idx = n - 1
                    elif key in (ord('t'), ord('T')):
                        thumb_count = 4 if key == ord('t') else 9
                    elif key in _ENTER_KEYS:
                        thumb_mode = False
                    elif key == ord('i'):
                        _show_metadata_panel(stdscr, _metadata_lines(image_files[idx]))
                    elif key == ord('?'):
                        _show_help_panel(stdscr)
                    elif key in (ord('q'), 27):
                        break
                    continue

                img_w, img_h = viewport_dims()
                current_video_position = video_position_for_index(idx)
                targets = preload_targets(idx, rotation_quadrants, flip_horizontal)
                target_keys = {
                    preload_key(target_idx, img_w, img_h, target_rotation, target_flip)
                    for target_idx, target_rotation, target_flip in targets
                }
                playback_positions = ()
                if video_playback_active and current_video_position is not None:
                    duration = video_duration(video_path_for_index(idx))
                    playback_positions = tuple(
                        _clamp_video_position(
                            current_video_position + frame_offset * video_playback_frame_seconds,
                            duration,
                        )
                        for frame_offset in range(VIDEO_PLAYBACK_BUFFER_FRAMES + 1)
                    )
                    target_keys.update(
                        preload_key(idx, img_w, img_h, rotation_quadrants, flip_horizontal, position)
                        for position in playback_positions
                    )
                trim_preload_cache(target_keys)
                for position in playback_positions:
                    schedule_preload(
                        idx,
                        img_w,
                        img_h,
                        rotation_quadrants,
                        flip_horizontal,
                        executor,
                        position,
                    )
                for target_idx, target_rotation, target_flip in targets:
                    schedule_preload(target_idx, img_w, img_h, target_rotation, target_flip, executor)

                try:
                    prepared = get_preloaded(idx, img_w, img_h, rotation_quadrants, flip_horizontal, target_keys, executor)
                    collect_completed_preloads(target_keys)
                    key = render(
                        stdscr,
                        image_files,
                        idx,
                        sharpen,
                        dither_mode,
                        color,
                        wait_time=current_delay,
                        slideshow=slideshow_active,
                        slideshow_reverse=slideshow_reverse,
                        prepared=prepared,
                        zoom_factor=zoom_factor,
                        pan_offset=(pan_y, pan_x),
                        rotation_quadrants=rotation_quadrants,
                        flip_horizontal=flip_horizontal,
                        video_position=current_video_position,
                        video_seek_step=video_seek_step if current_video_position is not None else None,
                        video_playback=video_playback_active and current_video_position is not None,
                        video_playback_fps=args.fps,
                        ascii_mode=args.ascii,
                        discard_queued_key=discard_queued_key,
                    )
                    discard_queued_key = None
                except Exception as e:
                    stdscr.clear()
                    stdscr.addstr(0, 0, f"Error: {e}")
                    stdscr.refresh()
                    stdscr.getch()
                    return
                # Navigation
                if key in ('video_seek_backward', 'video_seek_forward', 'video_previous_frame', 'video_next_frame'):
                    offsets = {
                        'video_seek_backward': -video_seek_step,
                        'video_seek_forward': video_seek_step,
                        'video_previous_frame': -video_playback_frame_seconds,
                        'video_next_frame': video_playback_frame_seconds,
                    }
                    move_video_position(idx, offsets[key])
                    discard_queued_key = {
                        'video_seek_backward': ord('['),
                        'video_seek_forward': ord(']'),
                        'video_previous_frame': ord(','),
                        'video_next_frame': ord('.'),
                    }.get(key)
                    video_playback_active = False
                    continue
                if key == 'video_seek_step_down':
                    video_seek_step = _next_video_seek_step(video_seek_step, -1)
                    continue
                if key == 'video_seek_step_up':
                    video_seek_step = _next_video_seek_step(video_seek_step, 1)
                    continue
                if key == 'toggle_video_playback':
                    video_path = video_path_for_index(idx)
                    if video_path is not None:
                        video_playback_active = not video_playback_active
                        if video_playback_active:
                            video_precise_paths.add(os.fspath(video_path))
                            slideshow_active = False
                            slideshow_reverse = False
                    continue
                if key == 'video_playback_next':
                    if video_playback_active and move_video_position(idx, video_playback_frame_seconds):
                        continue
                    video_playback_active = False
                    continue
                if key == 'show_metadata':
                    _show_metadata_panel(stdscr, _metadata_lines(image_files[idx]))
                    continue
                if key == 'show_help':
                    _show_help_panel(stdscr)
                    continue
                if key == 'zoom_in':
                    zoom_factor = min(4.0, zoom_factor + 0.25)
                    continue
                if key == 'zoom_out':
                    zoom_factor = max(0.25, zoom_factor - 0.25)
                    if zoom_factor <= 1.0:
                        pan_y = 0.0
                        pan_x = 0.0
                    continue
                if key == 'reset_zoom':
                    zoom_factor = 1.0
                    pan_y = 0.0
                    pan_x = 0.0
                    continue
                if key == 'rotate_clockwise':
                    rotation_quadrants = (rotation_quadrants + 1) % 4
                    pan_y = 0.0
                    pan_x = 0.0
                    continue
                if key == 'flip_horizontal':
                    flip_horizontal = not flip_horizontal
                    pan_y = 0.0
                    pan_x = 0.0
                    continue
                if key in ('pan_left', 'pan_right', 'pan_up', 'pan_down'):
                    if zoom_factor > 1.0:
                        pan_y, pan_x = _update_pan_offset(pan_y, pan_x, key)
                    continue
                if key == 'increase_delay':
                    current_delay = _clamp_delay(current_delay + 1)
                    continue
                if key == 'decrease_delay':
                    current_delay = _clamp_delay(current_delay - 1)
                    continue
                if key in (ord('t'), ord('T')) or key in _ENTER_KEYS:
                    thumb_mode = True
                    if key in (ord('t'), ord('T')):
                        thumb_count = 4 if key == ord('t') else 9
                    slideshow_active = False
                    slideshow_reverse = False
                    video_playback_active = False
                    continue
                slideshow_active, slideshow_reverse = _update_slideshow_state(slideshow_active, slideshow_reverse, key)
                if key in ('toggle_slideshow', 'toggle_slideshow_reverse'):
                    continue
                if slideshow_active and key not in ('slideshow_next', -1, 'toggle_slideshow', 'toggle_slideshow_reverse'):
                    # Any other key disables slideshow
                    slideshow_active = False
                    slideshow_reverse = False
                if key == 'slideshow_next':
                    if slideshow_reverse:
                        idx = (idx - 1) % n
                    else:
                        idx = (idx + 1) % n
                elif key in (curses.KEY_RIGHT, ord('n'), ord(' ')):
                    idx = (idx + 1) % n
                elif key in (curses.KEY_LEFT, ord('p')):
                    idx = (idx - 1) % n
                elif key == curses.KEY_UP:
                    idx = 0
                elif key == curses.KEY_DOWN:
                    idx = n - 1
                elif key in (ord('q'), 27):
                    break

    curses.wrapper(render_with_video_support, image_files, idx, not args.no_sharpen, args.dither, not args.no_color, wait_time=slideshow_delay, slideshow=slideshow)


if __name__ == "__main__":
    main()
