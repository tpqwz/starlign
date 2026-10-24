//! Decode an audiobook to 16 kHz mono PCM without ffmpeg.
//!
//! Symphonia reads m4b/mp4/aac, mp3, flac, ogg vorbis, and wav. Rubato resamples
//! to 16 kHz. Samples are written as little-endian f32 so a long book is not held
//! in memory twice. Clip times later divide the 16 kHz sample index by 16000,
//! which matches original playback time after the resampler delay is removed.

use anyhow::{anyhow, bail, Context, Result};
use rubato::{FftFixedIn, Resampler};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error as SymError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

pub const SAMPLE_RATE: u32 = 16_000;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Track {
    pub source: PathBuf,
    /// File name inside the EPUB `Audio/` directory.
    pub epub_name: String,
    pub media_type: String,
    /// First sample of this file on the 16 kHz timeline.
    pub start_sample: u64,
    pub samples: u64,
}

impl Track {
    pub fn start_secs(&self) -> f64 {
        self.start_sample as f64 / SAMPLE_RATE as f64
    }

    pub fn duration_secs(&self) -> f64 {
        self.samples as f64 / SAMPLE_RATE as f64
    }
}

#[derive(Clone, Debug)]
pub struct Timeline {
    pub pcm_path: Option<PathBuf>,
    pub tracks: Vec<Track>,
}

impl Timeline {
    pub fn samples(&self) -> u64 {
        self.tracks.last().map(|t| t.start_sample + t.samples).unwrap_or(0)
    }

    pub fn duration_secs(&self) -> f64 {
        self.samples() as f64 / SAMPLE_RATE as f64
    }

    /// Map a global playback time to per-file clip ranges.
    pub fn clips(&self, start: f64, end: f64) -> Vec<(usize, f64, f64)> {
        if self.tracks.is_empty() || end <= start {
            return Vec::new();
        }
        let mut out = Vec::new();
        for (index, track) in self.tracks.iter().enumerate() {
            let file_start = track.start_secs();
            let file_end = file_start + track.duration_secs();
            let lo = start.max(file_start);
            let hi = end.min(file_end);
            if hi - lo > 1e-4 {
                out.push((index, lo - file_start, hi - file_start));
            }
        }
        out
    }
}

pub fn decode_audiobook(input: &Path, pcm_path: &Path, max_seconds: Option<f64>) -> Result<Timeline> {
    let files = audio_files(input)?;
    if let Some(parent) = pcm_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let file = File::create(pcm_path).with_context(|| format!("creating {}", pcm_path.display()))?;
    let mut writer = PcmWriter::new(file, max_seconds.map(|s| (s * SAMPLE_RATE as f64).round() as u64))?;
    let mut tracks = Vec::new();
    let mut names = Vec::new();
    for path in files {
        if writer.full() {
            break;
        }
        let start = writer.written;
        let epub_name = unique_name(&path, &mut names);
        let media_type = media_type_of(&path);
        crate::say!("decoding {}", path.display());
        let samples_before = writer.written;
        decode_file(&path, &mut writer)?;
        let samples = writer.written - samples_before;
        tracks.push(Track {
            source: path,
            epub_name,
            media_type,
            start_sample: start,
            samples,
        });
    }
    // The resampler flush lands after the last packet. Count it on the last file.
    writer.finish()?;
    if let Some(last) = tracks.last_mut() {
        if writer.written >= last.start_sample {
            last.samples = writer.written - last.start_sample;
        }
    }
    if tracks.is_empty() || tracks.iter().all(|t| t.samples == 0) {
        bail!("no audio samples decoded from {}", input.display());
    }
    let timeline = Timeline {
        pcm_path: Some(pcm_path.to_path_buf()),
        tracks,
    };
    crate::say!(
        "audio {:.1}s, {} file(s), 16 kHz mono",
        timeline.duration_secs(),
        timeline.tracks.len()
    );
    Ok(timeline)
}

fn audio_files(input: &Path) -> Result<Vec<PathBuf>> {
    if input.is_file() {
        return Ok(vec![input.to_path_buf()]);
    }
    if !input.is_dir() {
        bail!("audiobook path does not exist: {}", input.display());
    }
    let mut files = Vec::new();
    for entry in std::fs::read_dir(input).with_context(|| format!("reading {}", input.display()))? {
        let path = entry?.path();
        if path.is_file() && is_audio(&path) {
            files.push(path);
        }
    }
    files.sort();
    if files.is_empty() {
        bail!("no audio files in {}", input.display());
    }
    Ok(files)
}

fn is_audio(path: &Path) -> bool {
    matches!(
        extension(path).as_str(),
        "m4b" | "m4a" | "mp4" | "mp3" | "flac" | "ogg" | "oga" | "wav" | "aac" | "opus"
    )
}

pub fn media_type_of(path: &Path) -> String {
    match extension(path).as_str() {
        "mp3" => "audio/mpeg",
        "flac" => "audio/flac",
        "ogg" | "oga" | "opus" => "audio/ogg",
        "wav" => "audio/wav",
        _ => "audio/mp4",
    }
    .to_string()
}

fn extension(path: &Path) -> String {
    path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
}

fn unique_name(path: &Path, used: &mut Vec<String>) -> String {
    let base = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| format!("audio-{}.m4a", used.len()));
    if !used.iter().any(|n| n == &base) {
        used.push(base.clone());
        return base;
    }
    let name = format!("{}-{base}", used.len());
    used.push(name.clone());
    name
}

struct PcmWriter {
    out: BufWriter<File>,
    resampler: Option<FftFixedIn<f32>>,
    pending: Vec<f32>,
    pending_rate: u32,
    delay_left: usize,
    input_frames: u64,
    written: u64,
    max_samples: Option<u64>,
}

impl PcmWriter {
    fn new(file: File, max_samples: Option<u64>) -> Result<Self> {
        Ok(Self {
            out: BufWriter::new(file),
            resampler: None,
            pending: Vec::new(),
            pending_rate: SAMPLE_RATE,
            delay_left: 0,
            input_frames: 0,
            written: 0,
            max_samples,
        })
    }

    fn full(&self) -> bool {
        self.max_samples.is_some_and(|max| self.written >= max)
    }

    fn ensure_rate(&mut self, rate: u32) -> Result<()> {
        let started = self.written > 0 || self.input_frames > 0 || !self.pending.is_empty();
        if started && rate != self.pending_rate {
            bail!("audio files have different sample rates ({rate} vs {})", self.pending_rate);
        }
        self.pending_rate = rate;
        if rate == SAMPLE_RATE {
            return Ok(());
        }
        if self.resampler.is_none() {
            let resampler = FftFixedIn::<f32>::new(rate as usize, SAMPLE_RATE as usize, 1024, 2, 1)
                .map_err(|err| anyhow!("resampler: {err}"))?;
            self.delay_left = resampler.output_delay();
            self.resampler = Some(resampler);
        }
        Ok(())
    }

    fn push(&mut self, rate: u32, mono: &[f32]) -> Result<()> {
        if mono.is_empty() || self.full() {
            return Ok(());
        }
        self.ensure_rate(rate)?;
        if rate == SAMPLE_RATE {
            self.write_samples(mono)?;
            return Ok(());
        }
        self.pending.extend_from_slice(mono);
        self.input_frames += mono.len() as u64;
        let mut offset = 0usize;
        while !self.full() {
            let need = self.resampler.as_ref().unwrap().input_frames_next();
            if self.pending.len() - offset < need {
                break;
            }
            let chunk = self.pending[offset..offset + need].to_vec();
            let produced = self
                .resampler
                .as_mut()
                .unwrap()
                .process(&[&chunk], None)
                .map_err(|err| anyhow!("resample: {err}"))?;
            self.emit(&produced[0])?;
            offset += need;
        }
        if offset > 0 {
            self.pending.drain(0..offset);
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        if self.resampler.is_some() {
            let rest = std::mem::take(&mut self.pending);
            let mut chunks = Vec::new();
            if !rest.is_empty() {
                let produced = self
                    .resampler
                    .as_mut()
                    .unwrap()
                    .process_partial(Some(&[&rest[..]]), None)
                    .map_err(|err| anyhow!("resample tail: {err}"))?;
                chunks.push(produced.into_iter().next().unwrap_or_default());
            }
            let produced = self
                .resampler
                .as_mut()
                .unwrap()
                .process_partial(None::<&[Vec<f32>]>, None)
                .map_err(|err| anyhow!("resample flush: {err}"))?;
            chunks.push(produced.into_iter().next().unwrap_or_default());
            for chunk in &chunks {
                self.emit(chunk)?;
            }
            let expected = ((self.input_frames as f64) * SAMPLE_RATE as f64 / self.pending_rate as f64).round() as u64;
            if self.written > expected {
                // The flush can overshoot by a handful of samples. The caller
                // uses `written`, so trim the file back to the input duration.
                let keep = expected * 4;
                self.out.flush()?;
                let file = self.out.get_mut();
                file.set_len(keep)?;
                self.written = expected;
            }
        }
        self.out.flush()?;
        Ok(())
    }

    fn emit(&mut self, samples: &[f32]) -> Result<()> {
        let mut samples = samples;
        if self.delay_left > 0 {
            if samples.len() <= self.delay_left {
                self.delay_left -= samples.len();
                return Ok(());
            }
            samples = &samples[self.delay_left..];
            self.delay_left = 0;
        }
        self.write_samples(samples)
    }

    fn write_samples(&mut self, samples: &[f32]) -> Result<()> {
        let mut samples = samples;
        if let Some(max) = self.max_samples {
            let room = max.saturating_sub(self.written) as usize;
            if room == 0 {
                return Ok(());
            }
            if samples.len() > room {
                samples = &samples[..room];
            }
        }
        for sample in samples {
            self.out.write_all(&sample.to_le_bytes())?;
        }
        self.written += samples.len() as u64;
        Ok(())
    }
}

fn decode_file(path: &Path, writer: &mut PcmWriter) -> Result<()> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .with_context(|| format!("probing {}", path.display()))?;
    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|track| track.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| anyhow!("no audio track in {}", path.display()))?
        .clone();
    // HE-AAC advertises the SBR rate (often 44100) on codec_params, but the
    // packets we actually decode are at the container timescale (often 22050).
    // Resampling from the advertised rate plays the book at double speed.
    let fallback_rate = track.codec_params.sample_rate.unwrap_or(0);
    let track_id = track.id;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .with_context(|| format!("decoder for {}", path.display()))?;
    let mut sample_buf: Option<SampleBuffer<f32>> = None;
    let mut channels = 1usize;

    loop {
        if writer.full() {
            break;
        }
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymError::ResetRequired) => {
                decoder.reset();
                continue;
            }
            Err(SymError::IoError(err)) if err.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(SymError::IoError(_)) => break,
            Err(_) => break,
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(SymError::DecodeError(_)) => continue,
            Err(SymError::IoError(_)) => break,
            Err(err) => return Err(anyhow!("decode {}: {err}", path.display())),
        };
        let spec = *decoded.spec();
        let rate = if spec.rate > 0 { spec.rate } else { fallback_rate };
        if rate == 0 {
            bail!("unknown sample rate in {}", path.display());
        }
        if sample_buf.is_none() {
            channels = spec.channels.count();
            sample_buf = Some(SampleBuffer::<f32>::new(decoded.capacity() as u64, spec));
            crate::say!("decoding {} at {rate} Hz", path.display());
        }
        let buf = sample_buf.as_mut().unwrap();
        buf.copy_interleaved_ref(decoded);
        let interleaved = buf.samples();
        if channels <= 1 {
            writer.push(rate, interleaved)?;
        } else {
            let frames = interleaved.len() / channels;
            let mut mono = Vec::with_capacity(frames);
            for frame in 0..frames {
                let mut sum = 0.0f32;
                for channel in 0..channels {
                    sum += interleaved[frame * channels + channel];
                }
                mono.push(sum / channels as f32);
            }
            writer.push(rate, &mono)?;
        }
    }
    Ok(())
}

pub fn read_pcm_range(path: &Path, start: u64, end: u64) -> Result<Vec<f32>> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = File::open(path)?;
    let start = start.min(end);
    file.seek(SeekFrom::Start(start * 4))?;
    let count = (end - start) as usize;
    let mut bytes = vec![0u8; count * 4];
    file.read_exact(&mut bytes)?;
    let mut out = vec![0f32; count];
    for (index, slot) in out.iter_mut().enumerate() {
        let raw = bytes[index * 4..index * 4 + 4].try_into().unwrap();
        *slot = f32::from_le_bytes(raw);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_wav(path: &Path, rate: u32, samples: &[f32]) {
        let mut bytes = Vec::new();
        let data_len = (samples.len() * 2) as u32;
        let mut header = Vec::new();
        header.extend_from_slice(b"RIFF");
        header.extend_from_slice(&(36 + data_len).to_le_bytes());
        header.extend_from_slice(b"WAVEfmt ");
        header.extend_from_slice(&16u32.to_le_bytes());
        header.extend_from_slice(&1u16.to_le_bytes());
        header.extend_from_slice(&1u16.to_le_bytes());
        header.extend_from_slice(&rate.to_le_bytes());
        header.extend_from_slice(&(rate * 2).to_le_bytes());
        header.extend_from_slice(&2u16.to_le_bytes());
        header.extend_from_slice(&16u16.to_le_bytes());
        header.extend_from_slice(b"data");
        header.extend_from_slice(&data_len.to_le_bytes());
        bytes.extend_from_slice(&header);
        for sample in samples {
            let clipped = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
            bytes.extend_from_slice(&clipped.to_le_bytes());
        }
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn decodes_wav_and_resamples_to_16k() {
        let dir = std::env::temp_dir().join(format!("stalign-wav-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let wav = dir.join("tone.wav");
        let mut samples = Vec::new();
        for n in 0..8000 {
            samples.push((n as f32 * 0.01).sin() * 0.2);
        }
        write_wav(&wav, 8000, &samples);
        let pcm = dir.join("out.f32");
        let timeline = decode_audiobook(&wav, &pcm, None).unwrap();
        let expected = 16000.0;
        assert!(
            (timeline.samples() as f64 - expected).abs() < 80.0,
            "got {} samples",
            timeline.samples()
        );
        assert_eq!(timeline.tracks.len(), 1);
        assert_eq!(timeline.tracks[0].media_type, "audio/wav");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn clips_split_across_files() {
        let timeline = Timeline {
            pcm_path: None,
            tracks: vec![
                Track {
                    source: PathBuf::from("a.mp3"),
                    epub_name: "a.mp3".into(),
                    media_type: "audio/mpeg".into(),
                    start_sample: 0,
                    samples: 16_000,
                },
                Track {
                    source: PathBuf::from("b.mp3"),
                    epub_name: "b.mp3".into(),
                    media_type: "audio/mpeg".into(),
                    start_sample: 16_000,
                    samples: 16_000,
                },
            ],
        };
        let clips = timeline.clips(0.5, 1.5);
        assert_eq!(clips.len(), 2);
        assert!((clips[0].1 - 0.5).abs() < 1e-6);
        assert!((clips[0].2 - 1.0).abs() < 1e-6);
        assert!((clips[1].1).abs() < 1e-6);
        assert!((clips[1].2 - 0.5).abs() < 1e-6);
    }
}
