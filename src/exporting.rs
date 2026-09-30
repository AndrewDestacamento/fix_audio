use std::{io::Error, path::Path};

use hound::{SampleFormat, WavSpec, WavWriter};
use lofty::{
    config::WriteOptions,
    tag::{Tag, TagExt as _},
};

/// Export processed audio to the output using `hound`.
pub fn export_audio(
    file_path: &Path,
    audio: (Box<[f32]>, Box<[f32]>),
    sample_rate: u32,
) -> Result<(), Error> {
    // TODO: add simple functionality for mono signals?
    let spec = WavSpec {
        channels: 2,
        sample_rate,
        bits_per_sample: 32, // hound only supports 32-bit float
        sample_format: SampleFormat::Float,
    };
    let mut writer = WavWriter::create(file_path, spec).map_err(Error::other)?;

    // Interleaving the audio beforehand doesn't improve performance.
    audio.0.into_iter().zip(audio.1).for_each(|(left, right)| {
        // SAFETY: can only return an error if we are NOT writing an f32 sample
        unsafe {
            writer.write_sample(left).unwrap_unchecked();
        }
        // SAFETY: can only return an error if we are NOT writing an f32 sample
        unsafe {
            writer.write_sample(right).unwrap_unchecked();
        }
    });
    writer.finalize().map_err(Error::other)?;
    Ok(())
}

/// Write tags to exported audio.
/// Requires `export_audio()` to be executed first.
///
/// Unfortunately doubles Exporting time and memory since `hound` clears all tags when calling `.finalize()`.
pub fn write_tags(file_path: &Path, tags: Box<[Tag]>) {
    // We don't particularly care if tags are written
    for tag in tags {
        _ = tag.save_to_path(file_path, WriteOptions::default());
    }
}
