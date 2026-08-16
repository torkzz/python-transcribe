# python-transcribe

> Batch transcription and translation script leveraging `faster-whisper` for fast GPU-accelerated audio/video processing.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Features

- **Fast Inference**: Uses [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) with `float16` precision on CUDA.
- **Auto Translation**: Transcribes and translates speech directly into English text.
- **Simple Output**: Writes transcripts alongside source files with `.txt` extensions.

## Prerequisites

- Linux / Windows with an NVIDIA GPU
- CUDA drivers installed
- Python 3.8+

## Quick Start

### 1. Clone the repository

```bash
git clone git@github.com:torkzz/python-transcribe.git
cd python-transcribe
```

### 2. Set up virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install faster-whisper
```

### 4. Run transcription

```bash
python auto_transcribe.py /path/to/meeting.wav
```

Output saved to `/path/to/meeting.wav.txt`.

### Running Tests

```bash
python3 -m unittest discover
```

Default settings in `auto_transcribe.py`:

- **Model size**: `large-v3`
- **Device**: `cuda`
- **Compute type**: `float16`
- **Task**: `translate`

## License

[MIT](LICENSE)
