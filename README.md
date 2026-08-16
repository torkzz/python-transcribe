# python-transcribe

Transcribe and translate audio/video files to English text using `faster-whisper`.

## Requirements

- Python 3.8+
- CUDA / GPU support

## Setup

```bash
pip install faster-whisper
```

## Usage

```bash
python auto_transcribe.py <path-to-audio-or-video>
```

Output is saved to `<path-to-audio-or-video>.txt`.
