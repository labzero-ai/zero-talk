"""Real-time factor of the Python package: python bench/python.py [text file]"""
import json, pathlib, sys, time
import zero_talk

path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).with_name("text.txt")
text = path.read_text(encoding="utf-8").strip()
t0 = time.perf_counter()
voice = zero_talk.load_voice("labzero-ai/zero-talk-11m")
t1 = time.perf_counter()
audio = voice.speak(text)
t2 = time.perf_counter()
print(json.dumps({"load_s": round(t1 - t0, 2), "speak_s": round(t2 - t1, 2), "audio_s": round(audio.seconds, 1), "rtf": round(audio.seconds / (t2 - t1), 2)}))
