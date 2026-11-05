# Zero Talk

Zero Talk 11M is an English text-to-speech model with about 11M parameters that runs on the user's device. The model ships
inside each package: no server, no API key, no network access at runtime.

| Folder | Package | For |
| --- | --- | --- |
| [`rust/`](rust/) | `zero-talk`, `zero-talk-model-11m` on crates.io | Rust, and the `zero-talk` command |
| [`python/`](python/) | `zero-talk` on PyPI | Python |
| [`js/`](js/) | `@labzero-ai/zero-talk` on npm | Browsers and Node.js |

Every package loads the model by its ID, `labzero-ai/zero-talk-11m`. Model page and live demo:
https://labzero.ai/zero-talk-11m/

Apache-2.0. Zero Talk builds on Kokoro-82M (Apache-2.0); third-party notices are in each package.

Lab Zero · https://labzero.ai
