# Benchmark

Speed and memory of each package on your machine, with the text used for the figures in the READMEs (`text.txt`, 503
characters, 32.8 s of speech). "Speed" is seconds of speech made per second of compute; each script prints it as `rtf`.

```sh
zero-talk --file bench/text.txt -o out.wav            # the command prints its own speed
python bench/python.py                                # after pip install zero-talk
node bench/node.mjs                                   # after npm install in js/ or npm install @labzero-ai/zero-talk
```

Peak memory: put `/usr/bin/time -l` (macOS) or `/usr/bin/time -v` (Linux) in front of the command. The `bench` workflow
(Actions, manual) runs the command on each kind of CI runner.
