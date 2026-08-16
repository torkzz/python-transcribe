import sys
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Mock faster_whisper in sys.modules if not installed locally
if "faster_whisper" not in sys.modules:
    sys.modules["faster_whisper"] = MagicMock()

from auto_transcribe import transcribe_file


class TestAutoTranscribe(unittest.TestCase):
    @patch("faster_whisper.WhisperModel")
    def test_transcribe_file_writes_output(self, mock_whisper_model):
        mock_segment_1 = MagicMock()
        mock_segment_1.text = "Hello world"
        mock_segment_2 = MagicMock()
        mock_segment_2.text = "Testing transcription"

        mock_instance = MagicMock()
        mock_instance.transcribe.return_value = ([mock_segment_1, mock_segment_2], None)
        mock_whisper_model.return_value = mock_instance

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            output_path = transcribe_file(tmp_path, device="cpu", compute_type="int8")
            self.assertEqual(output_path, tmp_path + ".txt")
            self.assertTrue(os.path.exists(output_path))

            with open(output_path, "r") as f:
                content = f.read()

            self.assertEqual(content, "Hello world\nTesting transcription\n")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            out_file = tmp_path + ".txt"
            if os.path.exists(out_file):
                os.remove(out_file)


if __name__ == "__main__":
    unittest.main()
