"""Tests for the zero_talk package. Run from the repository root, after installing the package: ``python -m unittest``.

Tests that compare the built-in model with a folder of model files need that folder: set ``ZERO_TALK_MODEL_DIR`` to a folder that holds
``model.bin`` and ``frontend.bin`` (the engine's model files). Without it, those tests are skipped.
"""

import hashlib
import platform
import sys
import os
import pathlib
import tempfile
import unittest

import zero_talk

VOICE_ID = "labzero-ai/zero-talk-11m"
RATE = 24000
MODEL_BYTES = 7053248
MODEL_SHA256 = "f0e885226e5fab84d9a800926479e420fc19161a5b4b435eb39d4ee102431ac5"
FRONTEND_BYTES = 816248
FRONTEND_SHA256 = "5ade509c060811a5fe9525ac554d6edc2229c2fca66581e093463b9d9b8eaffd"

# The AI-generated marker that every WAV and Ogg file carries: the comment, and the Ogg tag DIGITALSOURCETYPE.
MARKER_COMMENT = "AI-generated speech from Zero Talk 11M (labzero-ai/zero-talk-11m)"
MARKER_SOURCE = "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"

# The WAV files the Rust engine makes for these texts, by SHA-256 of the file.
ENGINE_WAV_SHA256 = {
    "Hello.": "c4607d7ac4e842dde2f860871b444a51e00fad8180f2a64933f1b71f9996ea27",
    "It opened on the 20th of February in 1940, and 48 people came.": "5ae072e8473c224d473730fd90e45bd5cb85d653ea6fdc61638e2bffea401be4",
    "Built in 1998, it weighs 84.7 kilograms and costs $1,090.61.": "a2ee1850af2cd80a5b33f48d99238c3661f082c80eb6f54732d502d1a5589abd",
}

LONG_TEXT = "It opened on the 20th of February in 1940, and 48 people came. Built in 1998, it weighs 84.7 kilograms."


def model_folder():
    """The folder named by ZERO_TALK_MODEL_DIR when it holds both model files, else None."""
    folder = os.environ.get("ZERO_TALK_MODEL_DIR")
    if folder and all(os.path.isfile(os.path.join(folder, name)) for name in ("model.bin", "frontend.bin")):
        return folder
    return None


def float_bytes(samples):
    """The raw bytes of a samples object, read through the buffer protocol."""
    return memoryview(samples).tobytes()


class PackageTests(unittest.TestCase):
    def test_version(self):
        self.assertEqual(zero_talk.__version__, "1.0.0")

    def test_voices_lists_the_one_voice(self):
        self.assertEqual(zero_talk.voices(), [VOICE_ID])

    def test_load_voice_by_id(self):
        self.assertIsInstance(zero_talk.load_voice(VOICE_ID), zero_talk.Voice)

    def test_unknown_voice_is_a_value_error_that_lists_the_voices(self):
        with self.assertRaises(ValueError) as caught:
            zero_talk.load_voice("someone/else")
        self.assertIn(VOICE_ID, str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            zero_talk.model_info("someone/else")
        self.assertIn(VOICE_ID, str(caught.exception))

    def test_model_info_gives_the_sizes_and_hashes(self):
        info = zero_talk.model_info(VOICE_ID)
        self.assertEqual(info["id"], VOICE_ID)
        self.assertEqual(info["files"]["model.bin"], {"bytes": MODEL_BYTES, "sha256": MODEL_SHA256})
        self.assertEqual(info["files"]["frontend.bin"], {"bytes": FRONTEND_BYTES, "sha256": FRONTEND_SHA256})

    def test_model_dir_without_the_files_is_a_file_not_found_error(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError) as caught:
                zero_talk.load_voice(VOICE_ID, model_dir=folder)
        self.assertIn("model.bin", str(caught.exception))


class SpeechTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.voice = zero_talk.load_voice(VOICE_ID)

    def test_speak_hello_at_24_khz(self):
        audio = self.voice.speak("Hello.")
        self.assertIsInstance(audio, zero_talk.Audio)
        self.assertEqual(audio.sample_rate, RATE)
        self.assertGreater(len(audio.samples), 0)
        self.assertAlmostEqual(audio.seconds, len(audio.samples) / RATE, places=6)
        self.assertEqual(audio.warnings, [])

    def test_samples_are_a_read_only_float32_memoryview(self):
        samples = self.voice.speak("Hello.").samples
        self.assertIsInstance(samples, memoryview)
        self.assertEqual(samples.format, "f")
        self.assertEqual(samples.itemsize, 4)
        self.assertEqual(samples.ndim, 1)
        self.assertGreater(samples.shape[0], 0)
        self.assertTrue(samples.readonly)

    def test_samples_share_one_block_of_memory(self):
        audio = self.voice.speak("Hello.")
        self.assertIs(audio.samples, audio.samples)
        self.assertIs(memoryview(audio.samples).obj, audio.samples.obj)

    def test_wav_bytes_equal_the_command_for_the_same_texts(self):
        # The package and the engine's own command run the same code, so their WAV bytes match on every platform.
        import os, shutil, subprocess, tempfile
        exe = shutil.which("zero-talk")
        self.assertIsNotNone(exe, "the zero-talk console script is not installed")
        with tempfile.TemporaryDirectory() as d:
            for i, text in enumerate(ENGINE_WAV_SHA256):
                with self.subTest(text=text):
                    out = os.path.join(d, f"{i}.wav")
                    done = subprocess.run([exe, text, "-o", out], capture_output=True)
                    self.assertEqual(done.returncode, 0, done.stderr)
                    with open(out, "rb") as f:
                        self.assertEqual(self.voice.speak(text).to_wav(), f.read())

    @unittest.skipUnless(os.environ.get("ZERO_TALK_REFERENCE_HASHES") == "1",
                         "set ZERO_TALK_REFERENCE_HASHES=1 on the reference machine (macOS 15, Apple silicon); native audio "
                         "differs slightly between systems and system versions (floating-point rounding)")
    def test_wav_bytes_equal_the_reference(self):
        for text, expected in ENGINE_WAV_SHA256.items():
            with self.subTest(text=text):
                digest = hashlib.sha256(self.voice.speak(text).to_wav()).hexdigest()
                self.assertEqual(digest, expected)

    def test_steering_tags_are_removed_with_a_warning(self):
        tagged = self.voice.speak("[M2] Hello.")
        self.assertEqual(len(tagged.warnings), 1)
        self.assertIn("steering is not supported", tagged.warnings[0])
        self.assertIn("[M2]", tagged.warnings[0])
        self.assertEqual(bytes(memoryview(tagged.samples)), bytes(memoryview(self.voice.speak("Hello.").samples)))

    def test_to_wav_and_to_ogg_return_bytes(self):
        audio = self.voice.speak("Hello.")
        wav = audio.to_wav()
        ogg = audio.to_ogg()
        self.assertIsInstance(wav, bytes)
        self.assertEqual(wav[:4], b"RIFF")
        self.assertEqual(wav[8:12], b"WAVE")
        self.assertIsInstance(ogg, bytes)
        self.assertEqual(ogg[:4], b"OggS")

    def test_wav_and_ogg_carry_the_ai_generated_marker(self):
        audio = self.voice.speak("Hello.")
        wav = audio.to_wav()
        self.assertEqual(wav[36:40], b"LIST")  # the marker is in the header, before the samples
        self.assertIn(MARKER_COMMENT.encode("ascii") + b"\0", wav)
        ogg = audio.to_ogg()
        self.assertIn(b"COMMENT=" + MARKER_COMMENT.encode("ascii"), ogg)
        self.assertIn(b"DIGITALSOURCETYPE=" + MARKER_SOURCE.encode("ascii"), ogg)

    def test_save_writes_wav_or_ogg_by_extension(self):
        audio = self.voice.speak("Hello.")
        with tempfile.TemporaryDirectory() as folder:
            wav_path = os.path.join(folder, "hello.wav")
            ogg_path = pathlib.Path(folder) / "hello.ogg"
            audio.save(wav_path)
            audio.save(ogg_path)
            with open(wav_path, "rb") as f:
                self.assertEqual(f.read(), audio.to_wav())
            with open(ogg_path, "rb") as f:
                ogg_file = f.read()
            self.assertEqual(ogg_file[:4], b"OggS")
            self.assertTrue(ogg_file)

    def test_save_with_another_extension_is_a_value_error(self):
        audio = self.voice.speak("Hello.")
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                audio.save(os.path.join(folder, "hello.mp3"))
            self.assertFalse(os.path.exists(os.path.join(folder, "hello.mp3")))

    def test_speak_with_nothing_to_say_is_a_value_error(self):
        with self.assertRaises(ValueError):
            self.voice.speak("")
        with self.assertRaises(ValueError):
            self.voice.stream("")

    def test_warnings_name_what_was_left_out(self):
        audio = self.voice.speak("Hello. 你好.")
        self.assertTrue(audio.warnings)
        self.assertTrue(all(isinstance(message, str) for message in audio.warnings))

    def test_stream_chunks_concatenate_to_the_speech(self):
        spoken = float_bytes(self.voice.speak(LONG_TEXT).samples)
        chunks = list(self.voice.stream(LONG_TEXT))
        self.assertTrue(chunks)
        for chunk in chunks:
            self.assertIsInstance(chunk, zero_talk.Chunk)
            self.assertIsInstance(chunk.samples, memoryview)
        self.assertEqual(b"".join(float_bytes(chunk.samples) for chunk in chunks), spoken)

    def test_stream_is_an_iterator(self):
        stream = self.voice.stream("Hello.")
        self.assertIsInstance(stream, zero_talk.Stream)
        self.assertIs(iter(stream), stream)
        chunks = list(stream)
        self.assertTrue(chunks)
        self.assertIsInstance(chunks[0], zero_talk.Chunk)
        self.assertGreater(len(chunks[0].samples), 0)
        with self.assertRaises(StopIteration):
            next(stream)


@unittest.skipUnless(model_folder(), "set ZERO_TALK_MODEL_DIR to a folder with model.bin and frontend.bin")
class ModelFolderTests(unittest.TestCase):
    def test_model_files_have_the_listed_sizes_and_hashes(self):
        folder = model_folder()
        for name, entry in zero_talk.model_info(VOICE_ID)["files"].items():
            with self.subTest(file=name):
                with open(os.path.join(folder, name), "rb") as f:
                    data = f.read()
                self.assertEqual(len(data), entry["bytes"])
                self.assertEqual(hashlib.sha256(data).hexdigest(), entry["sha256"])

    def test_model_dir_gives_the_same_speech_as_the_built_in_model(self):
        built_in = zero_talk.load_voice(VOICE_ID)
        from_folder = zero_talk.load_voice(VOICE_ID, model_dir=model_folder())
        for text in ENGINE_WAV_SHA256:
            with self.subTest(text=text):
                self.assertEqual(float_bytes(from_folder.speak(text).samples), float_bytes(built_in.speak(text).samples))
                self.assertEqual(from_folder.speak(text).to_wav(), built_in.speak(text).to_wav())


class CommandTests(unittest.TestCase):
    def test_console_script_writes_a_wav(self):
        import shutil, subprocess, tempfile, os
        exe = shutil.which("zero-talk")
        self.assertIsNotNone(exe, "the zero-talk console script is not installed")
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "hello.wav")
            done = subprocess.run([exe, "Hello.", "-o", out], capture_output=True)
            self.assertEqual(done.returncode, 0, done.stderr)
            with open(out, "rb") as f:
                self.assertEqual(f.read(4), b"RIFF")


if __name__ == "__main__":
    unittest.main()
