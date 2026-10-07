<h2 align="center">dotz - <i>Braille and ASCII art previews in the terminal</i></h2>

<p align="center">
<a href="https://github.com/jarun/dotz/releases/latest"><img src="https://img.shields.io/github/release/jarun/dotz.svg?maxAge=600" alt="Latest release" /></a>
<a href="https://pypi.org/project/dotz/"><img src="https://img.shields.io/pypi/v/dotz.svg?maxAge=600" alt="PyPI" /></a>
<a href="https://github.com/jarun/dotz/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-MIT-yellowgreen.svg?maxAge=2592000" alt="License" /></a>
</p>

Render image and video previews as Braille and ASCII art in the terminal with xterm-256 color and ncurses dim/normal/bold attributes.

Initially written for [`nnn`](https://github.com/jarun/nnn), it evolved as an independent feature-rich project.

## Features

- Braille art rendering for image and video previews
- Video playback with seek controls
- ASCII density fallback mode (for terminals without Braille font support)
- Thumbnail gallery with navigation
- Animated GIF support
- xterm-256 color and grayscale
- Dithering options (ordered, error diffusion, atkinson)
- Automatic aspect ratio correction for both Braille and ASCII modes
- File metadata panel
- Zoom in, zoom out, pan while zoom
- Rotate clockwise, flip horizontally
- Bounded background preloading
- Keyboard navigation and slideshow mode

<table width="100%">
    <tr>
        <td width="50%"><img src="https://github.com/user-attachments/assets/81d8de38-540e-4fb6-8698-07e625f349c2" alt="image_01" width="100%"></td>
        <td width="50%"><img src="https://github.com/user-attachments/assets/96320b6a-283b-4cab-b2ba-5ab04214673c" alt="image_02" width="100%"></td>
    </tr>
    <tr>
        <td width="50%"><img src="https://github.com/user-attachments/assets/688e2f1c-7719-4d67-beb5-82c8eddec82e" alt="image_03" width="100%"></td>
        <td width="50%"><img src="https://github.com/user-attachments/assets/9fc9a617-d55d-408b-81e9-450004f1644b" alt="image_04" width="100%"></td>
    </tr>
</table>

#### Supported formats

- **Image:** PNG, JPG, JPEG, BMP, GIF, TIFF, WEBP
- **Video:** MP4, MKV, AVI, MOV, WEBM, FLV, WMV, MPEG, MPG

## Installation

Install from PyPI:

```sh
pip3 install dotz
```

Or install from the source repository:

```sh
# Install system dependencies (e.g., ffmpeg)
sudo apt-get install ffmpeg  # or use your OS package manager

# Install Python dependencies and the CLI tool
sudo pip3 install .
```

After installation, you can run the tool using:

```sh
dotz [options] <file-or-directory>
```

You can also run the tool directly from the source directory:

```sh
python3 dotz.py [options] <file-or-directory>
```

#### Dependencies

| Package   | Version | Usage                                      |
|-----------|---------|--------------------------------------------|
| python    | >=3.10  | Required Python version                    |
| numpy     | >=1.20  | Fast array operations for image processing |
| Pillow    | >=8.0   | Image loading and manipulation             |
| ffmpeg    | >=4.2   | Video frame extraction                     |

## Usage

```
usage: dotz [-h] [-S] [-C] [-d {ordered,error,atkinson,none}] [-a] [-s [DELAY]] [-k SEEK] [-f {jpeg,png}] [-F {5,6,7,8,9,10}] [path]

Render an image or all images/videos in a directory as Braille and ASCII cells using ncurses with optional xterm-256 color.

positional arguments:
  path                  Path to the image/video file or directory (optional)

options:
  -h, --help            show this help message and exit
  -S, --no-sharpen      Disable edge sharpening
  -C, --no-color        Disable color (greyscale only with dim/normal/bold)
  -d {ordered,error,atkinson,none}, --dither {ordered,error,atkinson,none}
                        Dithering mode: ordered (default, clean), error (Floyd-Steinberg, smooth gradients), atkinson (preserves brightness), none
  -a, --ascii           Use ASCII characters instead of Braille (for terminals without Braille font support)
  -t [N], --thumbnails [N]
                        Show N thumbnails per page: 4 (2x2) or 9 (3x3), default: 4; press Enter to open an image.
  -s [DELAY], --slideshow [DELAY]
                        Enable slideshow mode with optional integer delay in seconds (default: 5).
  -k SEEK, --seek SEEK  Seek position to extract frame from videos in seconds (default: 10)
  -f {jpeg,png}, --format {jpeg,png}
                        Format for extracted video frames: jpeg (default) or png
  -F {5,6,7,8,9,10}, --fps {5,6,7,8,9,10}
                        Video playback frame rate between 5 and 10 FPS (default: 5)
```

#### Examples

- Syntax:
    ```sh
    python3 -m dotz <file-or-directory>
    ```
- To render a single image:
    ```sh
    python3 -m dotz path/to/image.jpg
    ```
- To render all images and videos in a directory:
    ```sh
    python3 -m dotz path/to/directory/
    ```
- To run a slideshow with a custom delay (e.g. 3 seconds):
    ```sh
    python3 -m dotz -s 3 path/to/directory/
    ```
- To use a custom video playback frame rate (e.g. 5 FPS):
    ```sh
    python3 -m dotz -F 5 path/to/video.mp4
    ```
- To use Atkinson dithering (preserves brightness better):
    ```sh
    python3 -m dotz -d atkinson path/to/image.jpg
    ```
- To use ASCII mode (for terminals without Braille font):
    ```sh
    python3 -m dotz -a path/to/image.jpg
    ```
- To combine ASCII mode with Atkinson dithering:
    ```sh
    python3 -m dotz -a -d atkinson path/to/image.jpg
    ```

## Navigation

| Key             | Action   |
|-----------------|----------|
| <kbd>Right</kbd>, <kbd>n</kbd>, <kbd>Space</kbd> | Next     |
| <kbd>Left</kbd>, <kbd>p</kbd>         | Previous |
| <kbd>Up</kbd>, <kbd>Down</kbd>        | First, Last    |
| <kbd>s</kbd>, <kbd>S</kbd>            | Toggle forward/reverse slideshow |
| <kbd>+</kbd>, <kbd>-</kbd>, <kbd>0</kbd>         | Zoom in, zoom out, zoom reset |
| <kbd>h</kbd>, <kbd>j</kbd>, <kbd>k</kbd>, <kbd>l</kbd>      | Pan left, down, up, right while zoomed |
| <kbd>r</kbd>               | Rotate clockwise |
| <kbd>f</kbd>               | Flip horizontally |
| <kbd>t</kbd>, <kbd>T</kbd> | Show 4 / 9 thumbnails |
| <kbd>Enter</kbd>           | Toggle between thumbnails and the selected image |
| <kbd>i</kbd>               | Show file metadata |
| <kbd>d</kbd>, <kbd>D</kbd>            | Decrease/increase slideshow delay by 1 sec |
| <kbd>[</kbd>, <kbd>]</kbd>            | Seek backward/forward in a video by the current seek step |
| <kbd>{</kbd>, <kbd>}</kbd>            | Decrease/increase the video seek step: 1, 2, 5, 10, 30, or 60 sec |
| <kbd>,</kbd>, <kbd>.</kbd>            | Move to the previous/next video playback frame |
| <kbd>v</kbd>               | Toggle video playback |
| <kbd>q</kbd>, <kbd>Esc</kbd>          | Quit     |
| <kbd>?</kbd>               | Show keyboard help |

The two-line status bar shows the current item and filename first, followed by zoom, slideshow, and video state on the second line.

## License

MIT
