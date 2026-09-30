use core::iter::once;
#[cfg(feature = "final_rotation")]
use rayon::iter::{IndexedParallelIterator as _, IntoParallelRefIterator as _};
use rayon::iter::{IntoParallelIterator as _, ParallelIterator as _};
use realfft::{RealFftPlanner, num_complex::Complex};
use std::{io::Error, sync::Mutex};

#[cfg(feature = "final_rotation")]
use crate::processing;
#[cfg(feature = "final_rotation")]
use core::f64::consts::PI;

/// Base-2 logarithm of 3 for `get_full_fft_size()` for an arbitrary FFT size.
const LB_3: f64 = 1.584_962_500_721_156;

/// Filler type being used because of a Clippy lint.
type FftResult = Result<(Box<[f32]>, Box<[f32]>), Error>;

/// Windowing is used to make the signal chunk fade in and out
///   to prevent discontinuities, which causes spectral leakage (noise tuned to the music).
// This is a periodic window, where the first point is zero and the ending point is non-zero, with the idea that the next window has the next zero point.
// Choosing a symmetric window with both or neither endpoints being zero doesn't seem to matter much here.
// The specific window being used is an approximation of a cosh window (alpha = 16.0), which is then an approximation of the DPSS/Slepian window.
//   One paper I read had a max alpha = 15.0 with about -100dB sidelobes, so 16.0 can't hurt.
pub fn window(sample_count: usize) -> Box<[f32]> {
    let f64_rate_recip = (sample_count as f64).recip();

    (0..sample_count)
        .into_par_iter()
        .map(|n| {
            let point = n as f64 * f64_rate_recip;
            point
                .mul_add(-point, point)
                .sqrt()
                .mul_add(32.0, -16.0)
                .exp() as f32
        })
        .collect()
}

/// Returns an STFT frame size in the form of `2^n * 3^m` since `rustfft` claims to work the fastest on these types.
fn get_stft_frame_size(length: usize) -> usize {
    match length {
        2_205 => 2_304,                 // 44100 / 20
        2_400 => 2_592,                 // 48000 / 20
        4_410 => 4_608,                 // 88200 / 20
        4_800 => 5_184,                 // 96000 / 20
        8_820 => 9_216,                 // 176400 / 20
        9_600 => 10_368, // 192000 / 20, probably the highest sample rate still in distribution (e.g. Frank Zappa's 'Hot Rats' on Qobuz)
        17_640 => 18_432, // 352800 / 20, VVV
        19_200 => 19_683, // 384000 / 20, These rates are here if the length is doubled for more zero-padding
        35_280 => 36_864, // 705600 / 20, ^^^
        38_400 => 39_366, // 768000 / 20, ^^^
        _ => get_full_fft_size(length), // Better if your code called `get_full_fft_size` directly instead of using `get_stft_frame_size`
    }
}

/// Returns an FFT size in the form of `2^n * 3^m` since `rustfft` claims to work the fastest on these types.
fn get_full_fft_size(length: usize) -> usize {
    let lb_length = (length as f64).log2();
    let mut saved_pow_of_2 = 0;
    let mut saved_pow_of_3 = 0;
    let mut lowest_remainder = 1_f64;

    let max_pow_of_3 = (lb_length / LB_3).ceil() as u32;

    for test_pow_of_3 in 0..=max_pow_of_3 {
        let criteria = (-LB_3).mul_add(f64::from(test_pow_of_3), lb_length); // lb_length - saved_pow_of_3 * LB_3 
        let test_remainder = criteria.ceil() - criteria;

        if test_remainder < lowest_remainder {
            lowest_remainder = test_remainder;
            saved_pow_of_2 = (criteria).ceil() as u32;
            saved_pow_of_3 = test_pow_of_3;
        }
    }

    2_usize
        .saturating_pow(saved_pow_of_2)
        .saturating_mul(3_usize.saturating_pow(saved_pow_of_3))
}

/// Faster `.norm_sqr()` or power calculation compared to `num_complex`.
///
/// Main reason is that the default uses `.hypot()`, which is then `hypotf` in libc.
const fn norm_sqr(point: Complex<f32>) -> f32 {
    point.re.mul_add(point.re, point.im * point.im)
}

/// Faster complex multiplication compared to `num_complex`.
///
/// Less useful on CPUs where the compiler already converts a `num_complex` multiplication to a `.mul_add()`.
const fn comp_mult(lhs: Complex<f32>, rhs: Complex<f32>) -> Complex<f32> {
    Complex::new(
        lhs.re.mul_add(rhs.re, -lhs.im * rhs.im),
        lhs.re.mul_add(rhs.im, lhs.im * rhs.re),
    )
}

/// Aligns the phase angle of the left and right channels.
// According to Intel VTune Profiler, this is the hottest function since it's in a hot loop.
// I've tried a branchless and an SIMD version, but they pretty much compile to the same peformance.
#[expect(
    clippy::arithmetic_side_effects,
    reason = "clippy thinks the operations done on Complex<f32> are for integers"
)]
fn align(original_left: &mut Complex<f32>, original_right: &mut Complex<f32>) {
    // Check if the points are more than 90 degrees away from each other
    // Equal to real(left * conj(right))
    if original_left.re * original_right.re < -original_left.im * original_right.im {
        // The original idea for this was to multiply the quieter channel by -1 to get closer to the louder channel while maintaining some distance.
        // e.g. If the quieter channel is -91 degrees away from the louder channel, then it'll become +89 degrees away.

        // Luckily, there is another point with the same distance to the louder channel, but requires a lower rotation angle.
        // Let A, B be complex numbers, with |A| > |B|.
        // Using the out_of_phase_checker above, suppose that A.re * B.re + A.im * B.im is negative.
        // This can become positive with A.re * (-B).re + A.im * (-B).im.
        // However, there is a second point that can be obtained by reflecting across the line connecting A with the origin (y = (A.im / A.re) * x).
        // This is equivalent to A_new = A, and B_new = -A * conj(B/A).
        // e.g. with the example above, the quieter channel will become -89 degrees away.

        // However, this changes the angle of the mid/sum channel. Something interesting occurs when we try to correct for this.
        // Let M = A + B, and M_2 = A + (-A*conj(B/A)) in accordance to the previous constraints.
        // Then A_new = (M/M_2) / |M/M_2| * A, and B_new = (M/M_2) / |M/M_2| * -A*conj(B/A).
        // Due to the repeated use of A, B, and then conj(A) and conj(B), this can all be simplified,
        //     mainly by using conj(B/A) = conj(B)/conj(A), conj(A) - conj(B) = conj(A-B), and (A-B)*conj(A-B) = |A-B|^2.
        // A_new = (A+B)(A-B)/|(A+B)(A-B)| * conj(A), and B_new = (A+B)(A-B)/|(A+B)(A-B)| * -conj(B).
        // Due to this construction, we don't actually have to check for the larger channel anymore.

        // Upon further analysis, it can be seen that the above is equivalent to converting to mid/side / sum/diff channels,
        //     swapping the magnitudes of the mid and side channels, then converting back to left/right.
        // This seems to produce more calculations, likely due to having to calucalte the power of mid and side separately.

        let line = comp_mult(
            *original_left + *original_right,
            *original_left - *original_right,
        );
        let line_power = norm_sqr(line);

        // We know that line_power >= 0_f32.
        // Since we'll use the inverse square root to calculate `align` (i.e. normalize a vector),
        //     we only need to check that line_power != 0_f32 to make sure .sqrt().recip() doesn't produce f32::INFINITY
        if line_power > 0_f32 {
            let align = line * line_power.sqrt().recip();
            *original_left = comp_mult(align, original_left.conj());
            *original_right = -comp_mult(align, original_right.conj());
        } else {
            // If this happens, then *original_left + *original_right == 0.0
            // Can't be *original_left - *original_right == 0.0, as that would imply *original_left == *original_right, and that is in-phase

            // This causes mathematically undefined behavior (divide by zero)
            // Hence, we just flip the right channel.
            *original_right = -*original_right;
        }
    }
}

/// STFT that, in each frame, aligns each frequency to the louder channel's phase angle.
#[expect(
    clippy::arithmetic_side_effects,
    reason = "clippy thinks the operations done on Complex<f32> are for integers"
)]
pub fn overlapping_fft(
    realfft_planner: &mut RealFftPlanner<f32>,
    sample_count: f64,
    left_channel: Box<[f32]>,
    right_channel: Box<[f32]>,
    cached_window: &mut Box<[f32]>,
) -> FftResult {
    // Idea is that sample_count gives us the amount of samples (possibly fractional) that we need to FFT
    let rounded_sample_count = sample_count.round_ties_even() as usize;
    // We should pad with half-a-second of silence to allow for half-windows at the beginning and end
    let half_sample_count = (sample_count * 0.5_f64).round_ties_even() as usize;

    // Adding more padding will increase the runtime non-linearly. In this case, it's probably better to add more overlaps instead, since that is linear.
    let fft_size = get_stft_frame_size(rounded_sample_count);
    let fft_sqrt_norm = (fft_size as f64).sqrt().recip() as f32;

    // We need a bit of silence at the beginning
    // This consumes left_channel and right_channel
    let build = |channel: Box<[f32]>| {
        vec![0_f32; half_sample_count]
            .into_iter()
            .chain(channel)
            .collect::<Box<[f32]>>()
    };
    let extended_left = build(left_channel);
    let extended_right = build(right_channel);
    let extended_length = extended_left.len();

    // `.into_boxed_slice()` is here to prevent overallocation if it stayed as a Vec
    let r2c = realfft_planner.plan_fft_forward(fft_size);
    let c2r = realfft_planner.plan_fft_inverse(fft_size);
    let holding_left = Mutex::new(vec![0_f32; extended_length].into_boxed_slice());
    let holding_right = Mutex::new(vec![0_f32; extended_length].into_boxed_slice());

    // Use pre-calculated window if possible. Otherwise, just overwrite it.
    if cached_window.len() != rounded_sample_count {
        *cached_window = window(rounded_sample_count);
    }

    // Windows need a bunch of hops.
    // A zipper noise is heard without more overlaps, likely because of phase discontinuities between frames. Fixing this would require dependence on the previous frame.
    // This dependence would then require the function to run in serial instead of parallel, so we'll just do a lot of overlaps to smooth these out.
    // More overlaps means more discontinuities, but at a reduced amplitude, so it's like a high-frequency noise that's increasing in pitch but decreasing in volume.
    let hop_size = sample_count * 0.001_f64; // equal to sample_rate/20_000, so ideally the generated noise is at 20khz and thus inaudible
    let hop_indexes = {
        let max_index = (extended_length as f64 / hop_size).round_ties_even() as usize;

        (0..max_index)
            .into_par_iter()
            .map(|index| (index as f64 * hop_size).round_ties_even() as usize)
    };

    // Function moved due to Clippy lint
    let multiply_and_pad = |channel: &[f32], hold_pos: usize| {
        channel
            .iter()
            .skip(hold_pos)
            .take(rounded_sample_count)
            .zip(cached_window.iter())
            .map(|(&samp, &mult)| samp * mult)
            .chain(once(0_f32).cycle()) // Extend iterator by cycling 0
            .take(fft_size)
            .collect::<Box<[f32]>>()
    };
    // Up until the end, which should be basically a half-window
    hop_indexes.try_for_each(|holding_position| {
        // Surprisingly, these don't take much memory per thread
        let mut left_chunk = multiply_and_pad(&extended_left, holding_position);
        let mut right_chunk = multiply_and_pad(&extended_right, holding_position);

        let mut scratch = c2r.make_scratch_vec().into_boxed_slice();

        let mut left_complex = {
            let mut pre_left_complex = r2c.make_output_vec().into_boxed_slice();
            _ = r2c.process_with_scratch(&mut left_chunk, &mut pre_left_complex, &mut scratch);
            pre_left_complex
        };
        let mut right_complex = {
            let mut pre_right_complex = r2c.make_output_vec().into_boxed_slice();
            _ = r2c.process_with_scratch(&mut right_chunk, &mut pre_right_complex, &mut scratch);
            pre_right_complex
        };

        #[cfg(feature = "subsonic_removal")]
        {
            // The first two elements correspond to the 0Hz and 20Hz frequencies, where inbetween frequencies are rounded to the nearest one.
            // As a crude low-cut/high-pass filter, the 0Hz frequency is removed, thus removing frequencies from 0Hz to 10Hz, randomly inclusive or exclusive.
            // SAFETY: fft_size is never zero.
            *unsafe { left_complex.get_unchecked_mut(0) } = Complex::ZERO;
            // SAFETY: fft_size is never zero.
            *unsafe { right_complex.get_unchecked_mut(0) } = Complex::ZERO;
        }

        left_complex
            .iter_mut()
            .zip(right_complex.iter_mut())
            .for_each(|(left_point, right_point)| {
                align(left_point, right_point);
                *left_point *= fft_sqrt_norm; // Do normalization after so the above points are using larger numbers
                *right_point *= fft_sqrt_norm;
            });

        // left_chunk and right_chunk will be overwritten
        _ = c2r.process_with_scratch(&mut left_complex, &mut left_chunk, &mut scratch);
        drop(left_complex);
        _ = c2r.process_with_scratch(&mut right_complex, &mut right_chunk, &mut scratch);
        drop(right_complex);
        drop(scratch);

        // RealFFT, which uses RustFFT, amplifies the signal by fft_size
        // Normalization happens later in processing.rs

        let add_to_hold =
            |chunk: Box<[f32]>, hold_channel: &Mutex<Box<[f32]>>| -> Result<(), Error> {
                hold_channel
                    .lock()
                    .map_err(|err| Error::other(err.to_string()))?
                    .iter_mut()
                    .skip(holding_position)
                    .zip(chunk)
                    .for_each(|(hold_samp, new_samp)| {
                        *hold_samp = new_samp.mul_add(fft_sqrt_norm, *hold_samp);
                    });
                Ok(())
            };

        // left_chunk is done first so less time is used locking the mutexes
        add_to_hold(left_chunk, &holding_left)?;
        add_to_hold(right_chunk, &holding_right)?;
        Ok::<(), Error>(())
    })?;

    drop(extended_left);
    drop(extended_right);

    // Normalization happens later in processing.rs

    let collect = |channel: Mutex<Box<[f32]>>| -> Result<Box<[f32]>, Error> {
        let check = channel.into_inner().map_err(Error::other)?;
        Ok(check
            .into_iter() // Don't think doing .into_par_iter() does anything
            .skip(half_sample_count)
            .collect())
    };
    Ok((collect(holding_left)?, collect(holding_right)?))
}

#[cfg(feature = "final_rotation")]
/// Minimize peaks by interpreting the analytic signal as a polygon and finding the rotating angle that will minimize the width on the real axis.
// Memory usage: a bit more than STFT's usage
// Luckily faster than STFT above
#[expect(
    clippy::arithmetic_side_effects,
    reason = "clippy thinks the operations done on Complex<f32> are for integers"
)]
pub fn minimize_peak(left_channel: Box<[f32]>, right_channel: Box<[f32]>) -> processing::AudioPeak {
    let fft_sqrt_norm;

    // Get a 90-degree rotated version of the audio signal to make it easier to rotate
    // We have to renormalize everything since we already did that in the previous processing.rs step (unless I want to do it again?)
    let (rotated_left, rotated_right) = {
        let original_length = left_channel.len();
        let fft_size = get_full_fft_size(original_length);
        // The planner has an internal cache to store different FFT sizes. This makes sense for reusing songs with the same sample rate as in the STFT,
        //   but each song probably have different lengths to each other, so it's more efficient to make a specific FFT planner for each song
        let mut long_realfft_planner = RealFftPlanner::new();
        let r2c = long_realfft_planner.plan_fft_forward(fft_size);
        let c2r = long_realfft_planner.plan_fft_inverse(fft_size);
        let mut scratch = c2r.make_scratch_vec().into_boxed_slice();

        fft_sqrt_norm = (fft_size as f64).sqrt().recip() as f32;

        // Function reduced
        let mut compute_rotated_channel = |channel: &[f32]| {
            let mut complex = r2c.make_output_vec().into_boxed_slice();
            _ = r2c.process_with_scratch(
                &mut channel
                    .iter()
                    .cycle() // Pad signal by cycling it
                    .take(fft_size)
                    .copied()
                    .collect::<Box<[f32]>>(),
                &mut complex,
                &mut scratch,
            );

            complex
                .iter_mut()
                .for_each(|point| *point = Complex::new(-point.im, point.re) * fft_sqrt_norm); // Equivalent to multiplying by i
            let mut finished = c2r.make_output_vec().into_boxed_slice();
            _ = c2r.process_with_scratch(&mut complex, &mut finished, &mut scratch);
            finished.into_iter()
        };
        (
            compute_rotated_channel(&left_channel).map(|samp| samp * fft_sqrt_norm),
            compute_rotated_channel(&right_channel).map(|samp| samp * fft_sqrt_norm),
        )
    };

    // The current optimal method of minimizing the peak value would be to use the rotating calipers method and calculate the angle of a bounding box.
    // However, this doubles our memory usage as we have to sort the polar angles of the analytic signal and its opposite (A and -A).
    // Instead, we'll just approximate it with uniformly-spaced angles.
    // 32 angles seems to be enough, but it can be increased to 1_024 if we want.
    let candidate_angle = (1_i32..32_i32).into_par_iter().map(|numerator| {
        let (sine, cosine) = (f64::from(numerator) * PI / 32_f64).sin_cos();
        Complex::new(cosine as f32, -sine as f32)
    });
    let normal_peak = left_channel
        .par_iter()
        .chain(right_channel.par_iter())
        .fold(|| f32::NEG_INFINITY, |acc, samp| acc.max(samp.abs()))
        .reduce(|| f32::NEG_INFINITY, f32::max);

    let analytic_left = left_channel
        .into_iter()
        .zip(rotated_left)
        .map(|(left, rot_left)| Complex::new(left, rot_left))
        .collect::<Box<[Complex<f32>]>>();
    let analytic_right = right_channel
        .into_iter()
        .zip(rotated_right)
        .map(|(right, rot_right)| Complex::new(right, rot_right))
        .collect::<Box<[Complex<f32>]>>();

    // Short-circuiting parallel loop
    // .by_exponential_blocks() might do nothing? We're using a sequential parallel .fold(), but inside we're using a serial .try_fold()
    let (new_angle, new_peak) = candidate_angle
        .by_exponential_blocks()
        .fold(
            || (Complex::new(1_f32, 0_f32), normal_peak),
            |angle_acc, test_angle| {
                // try_fold() allows us to exit early if the test_peak is already higher than our previous best
                let test_peak = analytic_left.iter().chain(analytic_right.iter()).try_fold(
                    f32::NEG_INFINITY,
                    |point_acc, point| {
                        let point_peak = point
                            .re
                            .mul_add(test_angle.re, point.im * test_angle.im)
                            .abs();

                        if point_peak > angle_acc.1 {
                            return None;
                        }
                        Some(point_acc.max(point_peak))
                    },
                );
                if let Some(inner_test_peak) = test_peak {
                    return (test_angle, inner_test_peak);
                }
                angle_acc
            },
        )
        .reduce(
            || (Complex::new(1_f32, 0_f32), normal_peak),
            |acc, new| {
                if new.1 < acc.1 {
                    return new;
                }
                acc
            },
        );

    (
        (
            analytic_left
                .into_par_iter()
                .map(|point| point.re.mul_add(new_angle.re, point.im * new_angle.im))
                .collect(),
            analytic_right
                .into_par_iter()
                .map(|point| point.re.mul_add(new_angle.re, point.im * new_angle.im))
                .collect(),
        ),
        new_peak,
    )
}
