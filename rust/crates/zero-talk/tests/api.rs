//! The public interface, end to end: loading, speaking, streaming and saving. These tests run offline.
use std::path::PathBuf;

use zero_talk::Voice;

const ID: &str = "labzero-ai/zero-talk-11m";
const LONG: &str = "It opened on the 20th of February in 1940, and 48 people came. Built in 1998, it weighs 84.7 kilograms.";

fn scratch_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("zero-talk-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a scratch folder");
    dir
}

fn model_files(name: &str) -> PathBuf {
    let dir = scratch_dir(name);
    std::fs::write(dir.join("model.bin"), zero_talk_model_11m::MODEL).unwrap();
    std::fs::write(dir.join("frontend.bin"), zero_talk_model_11m::FRONTEND).unwrap();
    dir
}

#[test]
fn loads_by_id_and_speaks_hello_at_24_khz() {
    let voice = Voice::load(ID).expect("the built-in voice loads");
    let audio = voice.speak("Hello.").expect("a short text speaks");
    assert!(!audio.samples().is_empty());
    assert_eq!(audio.sample_rate(), 24_000);
    assert!(audio.seconds() > 0.1, "{} s", audio.seconds());
    assert!(audio.warnings().is_empty());
}

#[test]
fn an_unknown_id_is_an_error_that_lists_the_valid_ids() {
    let err = Voice::load("labzero-ai/zero-talk-99m").err().expect("an unknown id is an error");
    let message = err.to_string();
    assert!(message.contains("labzero-ai/zero-talk-99m"), "{message}");
    assert!(message.contains(ID), "the message lists the valid ids: {message}");
}

#[test]
fn lists_the_one_voice() {
    assert_eq!(zero_talk::voices(), [ID]);
}

#[test]
fn saves_wav_and_ogg_by_extension() {
    let voice = Voice::load(ID).unwrap();
    let audio = voice.speak("Hello.").unwrap();
    let dir = scratch_dir("save");
    audio.save(dir.join("hello.wav")).expect("a .wav file");
    audio.save(dir.join("hello.ogg")).expect("a .ogg file");
    let wav = std::fs::read(dir.join("hello.wav")).unwrap();
    assert_eq!(&wav[..4], b"RIFF");
    assert_eq!(&wav[8..12], b"WAVE");
    assert_eq!(wav, audio.encode(zero_talk::Format::Wav).unwrap());
    let ogg = std::fs::read(dir.join("hello.ogg")).unwrap();
    assert_eq!(&ogg[..4], b"OggS");
    assert_eq!(ogg, audio.encode(zero_talk::Format::OGG).unwrap());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn other_extensions_are_an_error_that_names_both_formats() {
    let audio = Voice::load(ID).unwrap().speak("Hello.").unwrap();
    let err = audio.save("hello.mp3").err().expect("mp3 is not written");
    let message = err.to_string();
    assert!(message.contains(".wav") && message.contains(".ogg"), "{message}");
}

#[test]
fn stream_chunks_concatenate_to_the_speak_output() {
    let voice = Voice::load(ID).unwrap();
    let whole = voice.speak(LONG).unwrap();
    let mut joined: Vec<f32> = Vec::new();
    let mut chunks = 0;
    for chunk in voice.stream(LONG).unwrap() {
        joined.extend_from_slice(chunk.unwrap().samples());
        chunks += 1;
    }
    assert!(chunks > 1, "a long text comes in several chunks");
    assert_eq!(joined.len(), whole.samples().len());
    assert!(joined == whole.samples(), "the chunks are the speak output, sample for sample");
}

#[test]
fn a_text_with_nothing_to_say_is_an_error() {
    // No Latin letters: every word is left out, and no phonemes come out.
    let voice = Voice::load(ID).unwrap();
    assert!(voice.speak("東京 大阪").is_err());
    assert!(voice.stream("東京 大阪").is_err());
}

#[test]
fn unreadable_words_are_left_out_with_a_warning() {
    let audio = Voice::load(ID).unwrap().speak("Hello. 東京 is far.").unwrap();
    assert!(audio.warnings().iter().any(|w| w.contains("could not be read")), "{:?}", audio.warnings());
}

#[test]
fn encode_ogg_takes_a_bit_rate_and_names_the_fix_for_a_wrong_one() {
    let audio = Voice::load(ID).unwrap().speak("Hello from a small voice.").unwrap();
    let [small, default, large] = [8, 24, 128].map(|k| audio.encode(zero_talk::Format::Ogg { bitrate: k }).unwrap());
    assert_eq!(default, audio.encode(zero_talk::Format::OGG).unwrap());
    assert!(small.len() < default.len() && default.len() < large.len());
    let message = audio.encode(zero_talk::Format::Ogg { bitrate: 999 }).unwrap_err().to_string();
    assert_eq!(message, "bitrate must be a whole number of kbit/s from 8 to 128, not 999");
}

#[test]
fn save_into_a_missing_folder_names_the_file_and_the_fix() {
    let audio = Voice::load(ID).unwrap().speak("Hello.").unwrap();
    let path = std::env::temp_dir().join("zero-talk-missing-folder").join("nested").join("hello.wav");
    let message = audio.save(&path).unwrap_err().to_string();
    assert!(message.contains(&path.display().to_string()) && message.contains("create the folder first"), "{message}");
}

#[test]
fn steering_tags_are_left_out_with_a_warning() {
    let voice = Voice::load(ID).unwrap();
    let tagged = voice.speak("[M2] Hello.").unwrap();
    assert!(tagged.warnings().iter().any(|w| w.contains("steering is not supported") && w.contains("[M2]")), "{:?}", tagged.warnings());
    assert!(tagged.samples() == voice.speak("Hello.").unwrap().samples());
}

#[test]
fn files_give_the_same_voice_as_the_built_in_one() {
    let data = model_files("from-files");
    let from_files = Voice::from_files(data.join("model.bin"), data.join("frontend.bin")).expect("the model files load");
    let built_in = Voice::load(ID).unwrap();
    assert!(from_files.speak("Hello.").unwrap().samples() == built_in.speak("Hello.").unwrap().samples());
    std::fs::remove_dir_all(&data).ok();
}

#[test]
fn load_from_dir_gives_the_same_voice_and_names_a_missing_file() {
    let data = model_files("from-dir");
    let from_dir = Voice::load_from_dir(ID, &data).expect("the folder loads");
    assert!(from_dir.speak("Hello.").unwrap().samples() == Voice::load(ID).unwrap().speak("Hello.").unwrap().samples());
    std::fs::remove_file(data.join("frontend.bin")).unwrap();
    let message = Voice::load_from_dir(ID, &data).err().unwrap().to_string();
    assert!(message.contains("frontend.bin: not found; the folder must hold model.bin and frontend.bin"), "{message}");
    assert!(Voice::load_from_dir("labzero-ai/unknown", &data).err().unwrap().to_string().contains("the voices are: labzero-ai/zero-talk-11m"));
    std::fs::remove_dir_all(&data).ok();
}

#[test]
fn a_file_of_the_wrong_kind_is_an_error() {
    let data = model_files("wrong-kind");
    let swapped = Voice::from_files(data.join("frontend.bin"), data.join("model.bin"));
    assert!(swapped.is_err());
    let missing = Voice::from_files("no-such-model.bin", data.join("frontend.bin"));
    assert!(missing.err().unwrap().to_string().contains("no-such-model.bin"));
    std::fs::remove_dir_all(&data).ok();
}
