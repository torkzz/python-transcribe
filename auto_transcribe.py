from faster_whisper import WhisperModel
import sys

input_file = sys.argv[1]

model = WhisperModel(
    "large-v3",
    device="cuda",
    compute_type="float16"
)

segments, info = model.transcribe(
    input_file,
    task="translate"
)

output_file = input_file + ".txt"

with open(output_file, "w") as f:
    for segment in segments:
        f.write(segment.text + "\n")

print("Saved:", output_file)
