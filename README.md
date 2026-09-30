# Some set of DSP effects
Pet project of processing audio files by and for Andrew Destacamento to learn FFT processing

Currently, this program takes in stereo audio files (input folder created on first run) and:
* Reduces the phase angle difference with the left and right channels to ±90°.
  * Concept based on Thimeo Stereo Tool, automated through Thimeo WatchCat
    * Repair > Phase issues > Single instrument phase errors > \[self-titled\]
      * Limit maximum phase offset below: +180 deg
      * Fully move to center if above: 0 deg
    * Processing > Stereo > Stereo Image
      * Limit phase differences: +90 deg
    * Both of these ways have a similar effect, where the audio is modified if the phase angle difference is greater than ±90°
  * Use case: switching between stereo-speaker and mono-speaker setups (e.g. single earbud, headphones)
    * Prevents per-frequency phase cancellation for a better downmix to mono
    * Reduces the perceived stereo width, but instrument placement / channel-specific sounds are preserved
* Low-cuts audio 10Hz and below / High-passes audio 10Hz and above (Enabled by default, feature flag `subsonic_removal`)
  * Not based on anything in particular
  * Use case: Remove subsonic/rumble noise (<20Hz audio) that is not worth capturing in music.
    * Very few stereo systems and headphones allow audio <20Hz audio playback, which would be played back as vibrations rather than smooth tones.
    * Will obviously change the shape of the waveform (e.g. when viewed with an oscilloscope), but the actual heard content should be the same
* Rotates the phase of the result from the above step (Enabled by default, feature flag `final_rotation`)
  * Concept based on iZotope RX 12's "Phase" module, can't be automated
  * Use case: Reduce signal peak levels, especially ones amplified due to the above steps
    * RX 12's algorithm usually increases peak levels for no good reason
    * May be removed if the alignment algorithm is changed to one that inherently produces lower signal levels, making this step redundant
* Averages the loudness of the left and right channel
  * Concept based on iZotope RX 12's "Azimuth" module, can't be automated
  * Use case: ensure that one channel doesn't overpower the other over the course of a track
    * Uses the EBU R 128 Integrated Loudness, while RX 12 uses plain RMS
    * Plain RMS is affected by DC bias and does not account for human hearing

Processed audio files are sent to the output folder as 32-bit floating-point .wav files with tags and embedded covers transfered over. Non-audio files (covers, documents, etc.) are transfered to the output folder. The original audio files are kept in the input folder, so remember to delete them if you don't need to re-run the program with changes.

## Reflection
### Known problems I can't seem to fix:
* Symphonia dev-0.6 doesn't support certain codecs and features
  * Try converting unsupported music files to 32-bit .wav
    * Video files with an audio track
    * .opus files
    * .mp3 files: does not support invalid CRC checksums, so output files will have added silence or are cut prematurely based on padding
* Lofty is used to remaps tags to .wav's ID3v2 and RIFF INFO tags, so the conversion is usually lossy; non-standard tags like LYRICS and UNSYNCEDLYRICS/UNSYNCED_LYRICS/'UNSYNCED LYRICS' are likely not copied over.
  * Since this project exports files as .wav, you can try converting input files to 32-bit .wav while keeping tags using another program.
* FFT produces relatively minor frequency smearing / pre-echo depending on chosen frequency
  * Mainly affects very short hi-hats and sounds delayed in one channel
  * Stereo Tool suggests that it uses ~11Hz, but no frequency smearing is detected?

### Things to do:
* Add option and confirmation to delete input files after processing
* Make all steps optional through feature flags or command-line options
* Improve program efficiency
  * Approximate performance on my workstation:
    * ~3.46 minutes of runtime per 1 hour of 44.1kHz audio
    * Decoding time seems to grow linearly and is increased due to I/O (i.e. 2.91-3.76 seconds to decode 1 hour)
    * Processing time seems to grow linearly (i.e. 2.75 minutes to process 1 hour)
    * Exporting time seems to grow non-linearly and is increased due to I/O (i.e. a 3.14-minute track takes 1.24 seconds, a 24.9-minute track takes 13.9 seconds, and a 57.0-minute track takes 41.5 seconds)
  * Possible slowdown due to CPU affinity (`rayon` does not implement CPU pinning or similar) or other applications
  * (Windows only) Set the program's priority class (Idle -> Above Normal) and I/O priority (Normal -> High)
    * Approximate 50% speedup (90s to 60s on an old test suite) using System Informer to apply priorities
  * Add shortcut for mono files (remove DC noise only)
* Add more error-checking
  * Handle all existing `.unwrap()`s and `.expect()`s
  * Vec memory allocation on 32-bit builds for long files of audio
    * Could just suggest cutting down the audio into smaller bits
  * Test files that shorter than FFT (sound effects?)
  * Mono files are converted to stereo files

![performance](<Screenshot 2026-09-18 143652.png>)