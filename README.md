<h2 align="center">dotz - <i>terminal image and video previewer in Braille art</i></h2>

Render image and video previews as Braille art in the terminal with xterm-256 color and ncurses dim/normal/bold attributes.

It was written to be used as a terminal image viewer with [`nnn`](https://github.com/jarun/nnn). Works independently too.

## Features

- Braille art rendering for images
- Animated GIF support
- xterm-256 color and grayscale
- Dithering options (ordered, error diffusion, atkinson)
- Block character fallback mode (for terminals without Braille font support)
- Automatic aspect ratio correction for both Braille and block modes
- Video preview (frame extraction with ffmpeg)
- Video playback with seek controls
- File metadata panel
- Zoom in, zoom out, pan while zoom
- Rotate clockwise, flip horizontally
- Bounded background preloading
- Keyboard navigation and slideshow mode

<br>
<img width="1323" height="826" alt="image_01" src="https://github.com/user-attachments/assets/f2becbbc-cfeb-42b3-bd92-3882ff3fb570" />
<br><br>
<img width="1333" height="827" alt="image_02" src="https://github.com/user-attachments/assets/62bc16a8-246b-4b5a-a11e-fd0faa5c8066" />
<br><br>
<img width="1301" height="954" alt="image_03" src="https://github.com/user-attachments/assets/609805be-c0c5-4815-bb33-3bc70d69c152" />


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

Render an image or all images/videos in a directory as Braille cells using ncurses with optional xterm-256 color.

positional arguments:
  path                  Path to the image/video file or directory (optional)

options:
  -h, --help            show this help message and exit
  -S, --no-sharpen      Disable edge sharpening
  -C, --no-color        Disable color (greyscale only with dim/normal/bold)
  -d {ordered,error,atkinson,none}, --dither {ordered,error,atkinson,none}
                        Dithering mode: ordered (default, clean), error (Floyd-Steinberg, smooth gradients), atkinson (preserves brightness), none
  -a, --ascii           Use block characters instead of Braille (for terminals without Braille font support)
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
- To use block character mode (for terminals without Braille font):
    ```sh
    python3 -m dotz -a path/to/image.jpg
    ```
- To combine block character mode with Atkinson dithering:
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
