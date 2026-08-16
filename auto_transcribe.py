def transcribe_file(input_file: str, model_size: str = "large-v3", device: str = "cuda", compute_type: str = "float16", task: str = "translate") -> str:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, info = model.transcribe(input_file, task=task)

    output_file = input_file + ".txt"
    with open(output_file, "w") as f:
        for segment in segments:
            f.write(segment.text + "\n")

    return output_file


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python auto_transcribe.py <input_file>")
        sys.exit(1)
    transcribe_file(sys.argv[1])
