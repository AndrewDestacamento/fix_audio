/// Decoding module.
mod decoding;
/// Exporting module.
mod exporting;
/// FFT-specific module for processing.rs.
mod fft;
/// Processing module.
mod processing;

use std::{
    fs,
    io::{self, Write as _},
    path, thread, time,
};

use realfft::RealFftPlanner;
use symphonia::core::errors::Error;

#[global_allocator]
/// Change to the `MiMalloc` allocator.
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

// Hard-coded directories
/// Input directory, created on first run.
const INPUT_DIR: &str = "./inputs/";
/// Output directory.
const OUTPUT_DIR: &str = "./outputs/";

/// Recursive file path retriever from `StackOverflow`.
///
/// TODO: check if there's a newer std function or crate to do this.
fn get_paths(directory: path::PathBuf) -> io::Result<Box<[path::PathBuf]>> {
    let mut entries: Vec<path::PathBuf> = vec![];
    let folder_read = fs::read_dir(directory)?;

    for entry in folder_read {
        let unwrapped_entry = entry?;
        let meta = unwrapped_entry.metadata()?;

        if meta.is_dir() {
            let subdir = get_paths(unwrapped_entry.path())?;
            entries.extend(subdir);
        }

        if meta.is_file() {
            entries.push(unwrapped_entry.path());
        }
    }
    Ok(entries.into_boxed_slice())
}

/// Main function to execute.
fn main() -> Result<(), Error> {
    // Prevent the system from locking up by using all of the cores
    rayon::ThreadPoolBuilder::new()
        .num_threads({
            // Cannot use rayon::current_num_threads() since that will initialize early
            usize::from(thread::available_parallelism()?)
                .saturating_sub(2_usize)
                .max(1_usize)
        })
        .build_global()
        .map_err(|err| Error::IoError(io::Error::other(err)))?;

    // Keeping the time for benchmarking
    let time = time::Instant::now();

    // Check if INPUT_DIR exists, or create it if it doesn't
    match fs::exists(INPUT_DIR) {
        Ok(true) => {}
        Ok(false) => {
            _ = fs::create_dir_all(INPUT_DIR);
            println!("Notice: Inputs folder created. Copy audio files here to process them.");
            return Ok(());
        }
        // `Symphonia` has a wrapper for IoErrors
        Err(err) => return Err(Error::IoError(err)),
    }

    // Get list of files in INPUT_DIR
    let entries = get_paths(INPUT_DIR.into())?;

    // Because the STFT size shouldn't be that large (9600 max), we can reuse it, even for smaller sizes
    let mut realfft_planner = RealFftPlanner::new();
    let mut cached_window = fft::window(2400); // 48000 / 20

    println!("Setup and file-exploring time: {:#.3?}", time.elapsed());
    for entry in entries {
        // SAFETY: `INPUT_DIR` should always be there
        let stripped_entry = unsafe { entry.strip_prefix(INPUT_DIR).unwrap_unchecked() };
        println!("Found file: {}", stripped_entry.display());
        print!("	Decoding...");
        io::stdout().flush()?; // Show print instantly
        let mut output_path = path::PathBuf::from(OUTPUT_DIR).join(stripped_entry);

        // Tags are collected first in order to calculate the number of samples needed to collect in `decoding::get_samples()`
        let Ok((tags, sample_rate, sample_count)) = decoding::get_metadata(&entry) else {
            // Catches most non-audio files such as pictures
            println!("	Not audio or just unsupported; sent to output.");
            // SAFETY: output_path is defined, so it cannot be on the root
            fs::create_dir_all(unsafe { output_path.parent().unwrap_unchecked() })?;
            fs::rename(entry, output_path)?;
            continue;
        };

        // If we can't properly decode the data, it's likely not audio data
        // In this case, we just send it to OUTPUT_DIR so it's not spitting the warning every run
        let channels = match decoding::get_samples(&entry, sample_count) {
            Ok(data) => data,
            // Also catches detected-but-unimplemented audio like Opus files
            Err(Error::Unsupported(_)) => {
                println!("	Not audio or just unsupported; sent to output.");
                // SAFETY: output_path is defined, so it cannot be on the root
                fs::create_dir_all(unsafe { output_path.parent().unwrap_unchecked() })?;
                fs::rename(entry, output_path)?;
                continue;
            }
            // Catches a few non-audio files such as pictures with interesting metadata (.png or .gif with .jpg data?)
            Err(Error::IoError(err)) => {
                if err.kind() == io::ErrorKind::UnexpectedEof {
                    println!("	Not audio; sent to output.");
                    // SAFETY: output_path is defined, so it cannot be on the root
                    fs::create_dir_all(unsafe { output_path.parent().unwrap_unchecked() })?;
                    fs::rename(entry, output_path)?;
                    continue;
                }
                return Err(Error::IoError(err)); // Except here, where an actual audio file failed decoding
            }
            // Unsure if this even happens
            Err(Error::DecodeError(err)) => {
                println!("	Decoding error ({err}); sent to output.");
                // SAFETY: output_path is defined, so it cannot be on the root
                fs::create_dir_all(unsafe { output_path.parent().unwrap_unchecked() })?;
                fs::rename(entry, output_path)?;
                continue;
            }
            Err(other) => return Err(other), // some other unknown error
        };

        print!(" (T+{:#.3?}),", time.elapsed());
        io::stdout().flush()?;

        print!("	Processing...");
        io::stdout().flush()?;
        let (modified_audio, modified_audio_peak) = processing::process_samples(
            &mut realfft_planner,
            channels,
            sample_rate,
            &mut cached_window,
        )?;
        let modified_tags = processing::process_metadata(tags, modified_audio_peak);
        print!(" (T+{:#.3?}),", time.elapsed());
        io::stdout().flush()?;

        print!("	Exporting...");
        io::stdout().flush()?;
        output_path.set_extension("wav");
        // SAFETY: output_path is defined, so it cannot be on the root
        fs::create_dir_all(unsafe { output_path.parent().unwrap_unchecked() })?;
        exporting::export_audio(&output_path, modified_audio, sample_rate)?;
        // Unfortunately doubles Exporting time and memory since `hound` clears all tags when calling `.finalize()`
        exporting::write_tags(&output_path, modified_tags);

        println!(" (T+{:#.3?})", time.elapsed());
    }
    Ok(())
}
