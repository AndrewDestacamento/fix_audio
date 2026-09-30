use core::hint;
use std::{fs, io, path};

use lofty::{
    file::{AudioFile as _, TaggedFileExt as _},
    probe::Probe,
    tag::Tag,
};
use rayon::iter::{IntoParallelRefMutIterator as _, ParallelIterator as _};
use symphonia::{
    core::{
        codecs::audio::AudioDecoderOptions,
        errors::Error,
        formats::{FormatOptions, TrackType, probe::Hint},
        io::{MediaSourceStream, MediaSourceStreamOptions},
        meta::MetadataOptions,
    },
    default,
};

/// Filler type being used because of a Clippy lint.
type SamplesResult = Result<(Box<[f32]>, Box<[f32]>), Error>;

/// Get tags and sample rate for a given file using `lofty-rs`.
///
/// Good to use after using `get_samples()` to verify that audio was found.
pub fn get_metadata(path: &path::PathBuf) -> Result<(Box<[Tag]>, u32, usize), io::Error> {
    // Early exit if file doesn't have an extension indicating audio, but could still be read by Symphonia
    // .png files could have .jpg data if converted from that
    // .zip files with no compression would have Symphonia decode the first track it sees
    if let Some(extension) = path.extension()
        && let Some("zip" | "png") = extension.to_str()
    {
        return Err(io::Error::other("Not an audio file."));
    }

    let tagged_file = Probe::open(path)
        .map_err(io::Error::other)?
        .read()
        .map_err(io::Error::other)?;
    let tags = tagged_file.tags().iter().cloned().collect();
    let sample_rate = tagged_file
        .properties()
        .sample_rate()
        .ok_or_else(|| io::Error::other("ERROR: file has no sample rate"))?;

    let sample_count = (tagged_file.properties().duration().as_secs_f64() * f64::from(sample_rate))
        .ceil() as usize;
    Ok((tags, sample_rate, sample_count))
}

/// Get samples for a given file using `Symphonia`.
pub fn get_samples(path: &path::PathBuf, sample_count: usize) -> SamplesResult {
    // Based on `Symphonia`'s docs.rs page and example code (mix of 0.5.4 and dev-0.6)
    // Numbers are from the `Symphonia` basic proceedures in its docs.rs

    // Early exit if file doesn't have an extension indicating audio, but could still be read by Symphonia
    // .png files could have .jpg data if converted from that
    // .zip files with no compression would have Symphonia decode the first track it sees
    if let Some(extension) = path.extension()
        && let Some("zip" | "png") = extension.to_str()
    {
        return Err(Error::Unsupported("Not an audio file."));
    }
    // 1
    let codec_registry = default::get_codecs();
    // 2
    let probe = default::get_probe();

    // 3
    // 4
    let mss = MediaSourceStream::new(
        Box::new(fs::File::open(path)?),
        MediaSourceStreamOptions::default(),
    );

    // 5
    // 6
    let mut format = probe.probe(
        Hint::new().with_extension("flac"),
        mss,
        FormatOptions::default(),
        MetadataOptions::default(),
    )?;

    // 7
    let track = format
        .default_track(TrackType::Audio)
        .ok_or_else(|| io::Error::other("ERROR: no tracks found"))?;

    // 8
    let mut decoder = codec_registry.make_audio_decoder(
        track
            .codec_params
            .as_ref()
            .ok_or_else(|| io::Error::other("ERROR: unplayable file"))?
            .audio()
            .ok_or_else(|| io::Error::other("ERROR: unknown audio parameters"))?,
        &AudioDecoderOptions::default(),
    )?;

    let track_id = track.id;

    let mut left_samples: Vec<f32> = Vec::with_capacity(sample_count);
    let mut right_samples: Vec<f32> = Vec::with_capacity(sample_count);

    // No need to determine the capacity
    let mut sample_buf: Vec<Vec<f32>> = vec![];
    let mut channel_count = -1_isize;

    // 9
    // 10
    // 11
    while let Ok(Some(packet)) = format.next_packet() {
        if packet.track_id() == track_id {
            match decoder.decode(&packet) {
                Ok(audio_buf) => {
                    // Unsure how to get rid of this statement since it'll be run once, but it might be optimized out
                    if channel_count == -1 {
                        hint::cold_path(); // Called once, so technically still a cold path
                        channel_count = audio_buf.num_planes().cast_signed();

                        // .copy_to_vecs_planar() requires a Vec containing the channels
                        match channel_count {
                            1 => {
                                sample_buf = vec![vec![0_f32; audio_buf.samples_planar()]];
                            }
                            2 => {
                                sample_buf = vec![vec![0_f32; audio_buf.samples_planar()]; 2];
                            }
                            _ => {
                                return Err(Error::Unsupported(
                                    "Neither 1 nor 2 channels detected.",
                                ));
                            }
                        }
                    }

                    audio_buf.copy_to_vecs_planar(&mut sample_buf);
                    // SAFETY: At least one channel must exist.
                    left_samples.extend(unsafe { sample_buf.get_unchecked(0) });
                    match channel_count {
                        1 => {}
                        2 => {
                            // SAFETY: At least two channel must exist.
                            right_samples.extend(unsafe { sample_buf.get_unchecked(1) });
                        }
                        _ => {
                            // SAFETY: channel count can only be 1 or 2
                            unsafe {
                                hint::unreachable_unchecked();
                            }
                        }
                    }
                }
                // For some reason, `Symphonia` is fine if the decode doesn't work?
                // like with malformed data or something
                Err(Error::DecodeError(_)) => hint::cold_path(),
                Err(_) => {
                    hint::cold_path();
                    break;
                }
            }
        }
    }

    if left_samples.is_empty() {
        return Err(Error::Unsupported("No audio found"));
    }

    // Change infinite or NaN samples to silence.
    // TODO: research whether other programs upsample from the existing data instead
    // TODO: check if it's better to flush subnormal values to zero as well with f32::MIN_POSITIVE
    left_samples
        .par_iter_mut()
        .chain(right_samples.par_iter_mut())
        .for_each(|samp| {
            if !samp.is_finite() {
                *samp = 0_f32;
            }
        });

    // TODO: return error if fft_total would be larger than usize::MAX
    if channel_count == 2 {
        Ok((
            left_samples.into_boxed_slice(),
            right_samples.into_boxed_slice(),
        ))
    } else {
        // channel_count = 1
        // Upmixing mono audio to two channels
        // TODO: add speecial functionality for mono audio
        Ok((
            left_samples.clone().into_boxed_slice(),
            left_samples.into_boxed_slice(),
        ))
    }
}
