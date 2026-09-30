use std::io::{self, Error, Write as _};

use ebur128::{EbuR128, Mode};
use lofty::tag::{ItemKey, Tag, TagExt, TagType};
use rayon::iter::{
    IndexedParallelIterator as _, IntoParallelIterator as _, IntoParallelRefIterator as _,
    IntoParallelRefMutIterator as _, ParallelIterator as _,
};
use realfft::RealFftPlanner;

use crate::fft;

/// Force minimum reconstructed frequency to `MIN_FREQ` hertz.
///
/// Thimeo Stereo Tool suggests that it uses 4096 samples, which is about 11hz between 44.1khz and 48khz audio.
// For some reason, the 11hz creates smearing here, but not in Stereo Tool?
const MIN_FREQ: f64 = 20.;

/// 10^(1/20), for `gated_rms()`.
///
/// `10_f64.powf(loudness * 0.05_f64)` == `LOUDNESS_BASE.powf(loudness)`.
const LOUDNESS_BASE: f64 = 1.122_018_454_301_963_3_f64;

/// Filler type being used because of a Clippy lint.
pub type AudioPeak = ((Box<[f32]>, Box<[f32]>), f32);

/// RMS calculation.
fn plain_rms(channel: &[f32]) -> f64 {
    channel
        .par_iter()
        .fold(
            || 0_f64,
            |acc, &samp| f64::from(samp).mul_add(f64::from(samp), acc),
        )
        .reduce(|| 0_f64, f64::algebraic_add)
        .sqrt()
}

/// EBU R 128 Integrated Loudness calculation.
///
/// Basically a two-pass windowed RMS.
///   First pass is used to detect and ignore silence at -70dB.
///   Second pass is used to detect and ignore audio that's 10dB below the first-pass result.
fn gated_rms(samples: &[f32], sample_rate: u32) -> Result<f64, Error> {
    // Will fail on sample rate greater than 2.8224Mhz
    let mut ebur128 = EbuR128::new(1_u32, sample_rate, Mode::I).map_err(Error::other)?;
    // .add_frames_f32_planar() sucks since it requires an array of channel arrays, so for one channel, it needs an array around it
    // Result is dropped since it'll only panic if channels == 0, which does not happen above
    _ = ebur128.add_frames_f32(samples);

    // SAFETY: `loudness_global()` panics if not using Mode::I, which we are doing above
    let loudness_db = unsafe { ebur128.loudness_global().unwrap_unchecked() };
    Ok(LOUDNESS_BASE.powf(loudness_db)) //`loudness_global()` always returns an f64, so we just truncate at the end
}

/// Parallel DC removal.
// A high-pass filter isn't being used here in order to preserve the shape of the waveform.
fn remove_dc(channel: &mut [f32]) {
    let length = channel.len() as f64;
    let dc = (channel
        .into_par_iter()
        .fold(
            || 0_f64,
            |acc, &mut samp| acc.algebraic_add(f64::from(samp)),
        )
        .reduce(|| 0_f64, f64::algebraic_add)
        / length) as f32;

    channel.par_iter_mut().for_each(|samp| *samp -= dc);
}

/// All processing steps into one function.
pub fn process_samples(
    realfft_planner: &mut RealFftPlanner<f32>,
    data: (Box<[f32]>, Box<[f32]>),
    sample_rate: u32,
    cached_window: &mut Box<[f32]>,
) -> Result<AudioPeak, Error> {
    let mut left_channel = data.0;
    let mut right_channel = data.1;

    // Remove DC before processing
    // DC might affect magnitude of `MIN_FREQ` Hz and interpolated values close to it
    remove_dc(&mut left_channel);
    remove_dc(&mut right_channel);

    // Integrated Loudness shouldn't be affected by DC noise, but this is placed after DC removal just in case
    let true_mean_rms = gated_rms(&left_channel, sample_rate)?.sqrt()
        * gated_rms(&right_channel, sample_rate)?.sqrt();

    // Average out plain RMS of left and right channels before processing
    // Might help in phase conflicts
    // Human hearing doesn't matter here?
    let plain_left_rms = plain_rms(&left_channel);
    let plain_right_rms = plain_rms(&right_channel);
    let plain_mean_rms = plain_left_rms.sqrt() * plain_right_rms.sqrt();
    let left_mult = (plain_mean_rms / plain_left_rms) as f32;
    let right_mult = (plain_mean_rms / plain_right_rms) as f32;

    left_channel
        .par_iter_mut()
        .zip(right_channel.par_iter_mut())
        .for_each(|(left_samp, right_samp)| {
            *left_samp *= left_mult;
            *right_samp *= right_mult;
        });

    let pre_fft_correlation = {
        let covar = left_channel
            .par_iter()
            .zip(right_channel.par_iter())
            .fold(
                || 0_f64,
                |acc, (&lc, &rc)| f64::from(lc).mul_add(f64::from(rc), acc),
            )
            .reduce(|| 0_f64, f64::algebraic_add);

        covar / (plain_left_rms * plain_right_rms)
    };

    print!(" ({:.3?}% ->", 100_f64 * pre_fft_correlation);
    io::stdout().flush()?;

    let sample_count = f64::from(sample_rate) / MIN_FREQ; // actually in number of samples

    let (mut processed_left, mut processed_right) = fft::overlapping_fft(
        realfft_planner,
        sample_count,
        left_channel,
        right_channel,
        cached_window,
    )?;

    // STFT will generate sub-MIN_FREQ noise
    // As such, DC noise is likely added and should be removed since we'll multiply the signals later
    remove_dc(&mut processed_left);
    remove_dc(&mut processed_right);

    // Average out the loudness of the left and right channels
    // This handles amplification by RustFFT and overlap-adding, assuming there isn't much precision loss
    // TODO: This might make the first -70dB pass redundant
    let processed_left_rms = gated_rms(&processed_left, sample_rate)?;
    let processed_right_rms = gated_rms(&processed_right, sample_rate)?;
    let processed_left_mult = (true_mean_rms / processed_left_rms) as f32;
    let processed_right_mult = (true_mean_rms / processed_right_rms) as f32;
    processed_left
        .par_iter_mut()
        .zip(processed_right.par_iter_mut())
        .for_each(|(left_samp, right_samp)| {
            *left_samp *= processed_left_mult;
            *right_samp *= processed_right_mult;
        });

    let post_fft_correlation = {
        let post_plain_left_rms = plain_rms(&processed_left);
        let post_plain_right_rms = plain_rms(&processed_right);
        let covar = processed_left
            .par_iter()
            .zip(processed_right.par_iter())
            .fold(
                || 0_f64,
                |acc, (&lc, &rc)| f64::from(lc).mul_add(f64::from(rc), acc),
            )
            .reduce(|| 0_f64, f64::algebraic_add);

        covar / (post_plain_left_rms * post_plain_right_rms)
    };

    print!(" {:.3?}%)", 100_f64 * post_fft_correlation);
    io::stdout().flush()?;

    // Rotating the phase of a signal should not (significantly) change the observed RMS and integrated loudness

    #[cfg(feature = "final_rotation")]
    return Ok(fft::minimize_peak(processed_left, processed_right));
    #[cfg(not(feature = "final_rotation"))]
    {
        let peak = processed_left
            .par_iter()
            .chain(processed_right.par_iter())
            .fold(|| f32::NEG_INFINITY, |acc, samp| acc.max(samp.abs()))
            .reduce(|| f32::NEG_INFINITY, f32::max);
        Ok(((processed_left, processed_right), peak))
    }
}

/// Modify tags to remove outdated info.
pub fn process_metadata(mut tags: Box<[Tag]>, peak: f32) -> Box<[Tag]> {
    for tag in &mut tags {
        tag.remove_empty();

        // File peak will change due to processing
        tag.remove_key(ItemKey::ReplayGainAlbumPeak);

        // Paste the new peak value
        tag.insert_text(ItemKey::ReplayGainTrackPeak, peak.to_string());
    }
    let mut id3v2_tags = tags.clone();
    id3v2_tags.iter_mut().for_each(|tag| {
        if tag.tag_type() != TagType::Id3v2 {
            tag.re_map(TagType::Id3v2);
        }
    });
    if let Some(id3v2_fields) = id3v2_tags.iter().map(TagExt::len).max() {
        id3v2_tags = id3v2_tags
            .into_iter()
            .filter(|tag| tag.len() >= id3v2_fields)
            .collect();
    }

    let mut riff_tags = tags.clone();
    riff_tags.iter_mut().for_each(|tag| {
        if tag.tag_type() != TagType::RiffInfo {
            tag.re_map(TagType::RiffInfo);
        }
    });
    if let Some(riff_fields) = riff_tags.iter().map(TagExt::len).max() {
        riff_tags = riff_tags
            .into_iter()
            .filter(|tag| tag.len() >= riff_fields)
            .collect();
    }

    id3v2_tags.into_iter().chain(riff_tags).collect()
}
