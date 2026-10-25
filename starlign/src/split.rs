//! Cut an audiobook without re-encoding.
//!
//! Overlay times use the decoded playback clock. For m4b, m4a, mp4, and mp3
//! that clock matches the encoded frames, so a range is a run of those frames
//! copied into a new file. The file starts on the first frame of the range.
//! Callers shift clip times by that origin.
//!
//! [`Split::Chapters`] is one file per aligned chapter. [`Split::Seconds`] and
//! [`Split::Parts`] cut the whole audiobook on a time grid. A span that crosses
//! a grid line stays on the file where it starts, and that file runs through
//! the end of the span. The next file still starts on the grid, so the two
//! files share that short overhang.

use crate::audio::{Track, SAMPLE_RATE};
use anyhow::{anyhow, bail, Context, Result};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// How the written EPUB stores the audiobook.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Split {
    /// One stored copy of each source file. Clip times seek into that file.
    Whole,
    /// One file per aligned chapter, covering that chapter's spans.
    Chapters,
    /// Files of at most this many seconds. The last file is the remainder.
    Seconds(f64),
    /// This many files of equal length across the audiobook.
    Parts(u32),
}

impl Split {
    /// One of the three split flags, or none of them.
    pub fn resolve(chapters: bool, seconds: Option<f64>, parts: Option<u32>) -> Result<Self> {
        let chosen =
            usize::from(chapters) + usize::from(seconds.is_some()) + usize::from(parts.is_some());
        if chosen > 1 {
            bail!("pass only one of --split-chapters, --split-seconds, and --split-parts");
        }
        if let Some(seconds) = seconds {
            if !(seconds.is_finite() && seconds > 0.0) {
                bail!("--split-seconds must be greater than 0");
            }
            return Ok(Self::Seconds(seconds));
        }
        if let Some(parts) = parts {
            if parts < 2 {
                bail!("--split-parts must be at least 2");
            }
            return Ok(Self::Parts(parts));
        }
        if chapters {
            Ok(Self::Chapters)
        } else {
            Ok(Self::Whole)
        }
    }
}

/// Grid lines for a duration or equal-parts split, covering `[start, end]`.
pub fn nominal_slices(split: Split, start: f64, end: f64) -> Result<Vec<(f64, f64)>> {
    if !(end > start) {
        bail!("the audiobook has no duration to split");
    }
    match split {
        Split::Seconds(seconds) => {
            if !(seconds.is_finite() && seconds > 0.0) {
                bail!("split duration must be greater than 0 seconds");
            }
            let mut out = Vec::new();
            let mut cursor = start;
            while cursor < end - 1e-6 {
                let next = (cursor + seconds).min(end);
                if next - cursor <= 1e-4 {
                    break;
                }
                out.push((cursor, next));
                cursor = next;
                if out.len() > 100_000 {
                    bail!("split duration is too small for this audiobook");
                }
            }
            if out.is_empty() {
                bail!("split duration is too small for this audiobook");
            }
            Ok(out)
        }
        Split::Parts(count) => {
            if count < 2 {
                bail!("split count must be at least 2");
            }
            let step = (end - start) / f64::from(count);
            if step <= 1e-3 {
                bail!("each part would be shorter than a millisecond");
            }
            let mut out = Vec::with_capacity(count as usize);
            for index in 0..count {
                let lo = start + step * f64::from(index);
                let hi = if index + 1 == count {
                    end
                } else {
                    start + step * f64::from(index + 1)
                };
                out.push((lo, hi));
            }
            Ok(out)
        }
        Split::Whole | Split::Chapters => bail!("not a timed audio split"),
    }
}

/// Slice index whose start is at or before `time`. A time on a grid line belongs
/// to the later slice. Times before the first slice belong to it.
pub fn slice_index(nominal: &[(f64, f64)], time: f64) -> usize {
    let mut index = 0usize;
    for (i, &(start, _)) in nominal.iter().enumerate() {
        if time + 1e-9 >= start {
            index = i;
        } else {
            break;
        }
    }
    index
}

/// Extend each slice so every span that starts in it fits in that file.
///
/// Returns the file ranges and, parallel to `spans`, the slice each span uses.
pub fn assign_slices(
    nominal: &[(f64, f64)],
    spans: &[(f64, f64)],
) -> (Vec<(f64, f64)>, Vec<usize>) {
    let mut covered = nominal.to_vec();
    let mut owners = Vec::with_capacity(spans.len());
    if nominal.is_empty() {
        return (covered, owners);
    }
    for &(start, end) in spans {
        let index = slice_index(nominal, start);
        owners.push(index);
        if start < covered[index].0 {
            covered[index].0 = start;
        }
        if end > covered[index].1 {
            covered[index].1 = end;
        }
    }
    (covered, owners)
}

pub struct Cache {
    open: HashMap<PathBuf, OpenAudio>,
}

struct OpenAudio {
    file: File,
    kind: Kind,
}

enum Kind {
    Mp4(Mp4Index),
    Mp3(Mp3Index),
}

struct Mp4Index {
    timescale: u32,
    language: u16,
    /// Playback seconds added to reach this track's media timeline.
    shift_secs: f64,
    stsd: Vec<u8>,
    runs: Vec<Run>,
    total_samples: usize,
    total_ticks: u64,
    sizes: Vec<u32>,
    offsets: Vec<u64>,
}

struct Run {
    count: u32,
    delta: u32,
    first: usize,
    start_tick: u64,
}

struct Mp3Index {
    frames: Vec<(u64, u32)>,
    samples_per_frame: u32,
    rate: u32,
}

/// A temporary chapter file. Removed when dropped.
#[derive(Debug)]
pub struct Piece {
    pub epub_name: String,
    pub media_type: String,
    path: PathBuf,
    pub origin_secs: f64,
    pub duration_secs: f64,
}

impl Piece {
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn track(&self) -> Track {
        let start_sample = (self.origin_secs * SAMPLE_RATE as f64).floor() as u64;
        let end_sample =
            ((self.origin_secs + self.duration_secs) * SAMPLE_RATE as f64).ceil() as u64;
        Track {
            source: PathBuf::new(),
            epub_name: self.epub_name.clone(),
            media_type: self.media_type.clone(),
            start_sample,
            samples: end_sample.saturating_sub(start_sample).max(1),
        }
    }
}

impl Drop for Piece {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

impl Cache {
    pub fn new() -> Self {
        Self {
            open: HashMap::new(),
        }
    }
}

/// Cut `[start, end)` on the global playback clock into one file per source track it touches.
pub fn cut_range(
    cache: &mut Cache,
    tracks: &[Track],
    start: f64,
    end: f64,
    stem: &str,
    used: &mut HashSet<String>,
) -> Result<Vec<Piece>> {
    if end <= start {
        bail!("audio range is empty ({start:.3}s–{end:.3}s)");
    }
    let stem = safe_stem(stem);
    let mut pieces = Vec::new();
    for track in tracks {
        let file_start = track.start_secs();
        let file_end = file_start + track.duration_secs();
        let lo = start.max(file_start);
        let hi = end.min(file_end);
        if hi - lo <= 1e-4 {
            continue;
        }
        let local_lo = lo - file_start;
        let local_hi = hi - file_start;
        let cut = cache.cut(&track.source, local_lo, local_hi)?;
        let name = unique_name(&stem, &cut.extension, used);
        let path = cache.temp_path(&cut.extension);
        std::fs::write(&path, &cut.bytes).with_context(|| format!("writing {}", path.display()))?;
        let origin = file_start + cut.origin_secs;
        crate::say!(
            "split audio {name} {:.3}s–{:.3}s",
            origin,
            origin + cut.duration_secs
        );
        pieces.push(Piece {
            epub_name: name,
            media_type: cut.media_type,
            path,
            origin_secs: origin,
            duration_secs: cut.duration_secs,
        });
    }
    Ok(pieces)
}

struct CutBytes {
    bytes: Vec<u8>,
    origin_secs: f64,
    duration_secs: f64,
    media_type: String,
    extension: String,
}

impl Cache {
    fn temp_path(&self, extension: &str) -> PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let seq = NEXT.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "starlign-split-{}-{seq}.{extension}",
            std::process::id()
        ))
    }

    fn cut(&mut self, source: &Path, start: f64, end: f64) -> Result<CutBytes> {
        if !self.open.contains_key(source) {
            let indexed = open_audio(source)?;
            self.open.insert(source.to_path_buf(), indexed);
        }
        let audio = self.open.get_mut(source).unwrap();
        match &audio.kind {
            Kind::Mp4(index) => cut_mp4(&mut audio.file, index, start, end),
            Kind::Mp3(index) => cut_mp3(&mut audio.file, index, start, end),
        }
        .with_context(|| format!("splitting {}", source.display()))
    }
}

fn open_audio(path: &Path) -> Result<OpenAudio> {
    let mut file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let ext = extension(path);
    let kind = match ext.as_str() {
        "m4b" | "m4a" | "mp4" | "m4p" => {
            Kind::Mp4(parse_mp4(&mut file).with_context(|| format!("reading {}", path.display()))?)
        }
        "mp3" => {
            Kind::Mp3(parse_mp3(&mut file).with_context(|| format!("reading {}", path.display()))?)
        }
        _ => bail!(
            "splitting supports m4b, m4a, mp4, and mp3. {} is a {ext} file",
            path.display()
        ),
    };
    Ok(OpenAudio { file, kind })
}

fn extension(path: &Path) -> String {
    path.extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
}

fn safe_stem(stem: &str) -> String {
    let mut out = String::new();
    for ch in stem.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            out.push(ch);
        } else if !out.ends_with('_') {
            out.push('_');
        }
    }
    let out = out.trim_matches('_').to_string();
    if out.is_empty() {
        "chapter".to_string()
    } else {
        out
    }
}

fn unique_name(stem: &str, extension: &str, used: &mut HashSet<String>) -> String {
    let mut name = format!("{stem}.{extension}");
    let mut n = 2u32;
    while used.contains(&name) {
        name = format!("{stem}-{n}.{extension}");
        n += 1;
    }
    used.insert(name.clone());
    name
}

fn cut_mp4(file: &mut File, index: &Mp4Index, start: f64, end: f64) -> Result<CutBytes> {
    let total = index.total_ticks as f64 / index.timescale as f64;
    let mut media_start = start + index.shift_secs;
    let mut media_end = end + index.shift_secs;
    if media_start < 0.0 {
        media_start = 0.0;
    }
    if media_start >= total {
        bail!("audio starts at {start:.3}s, past the end of the file ({total:.3}s)");
    }
    if media_end > total + 0.05 {
        bail!(
            "audio runs to {end:.3}s, past the end of the file ({:.3}s)",
            total - index.shift_secs
        );
    }
    media_end = media_end.min(total);
    let start_tick = (media_start * index.timescale as f64).round() as u64;
    let end_tick = (media_end * index.timescale as f64).round() as u64;
    let first = index.sample_containing(start_tick);
    let last = index.first_sample_at_or_after(end_tick.max(start_tick + 1));
    if first >= last || last > index.total_samples {
        bail!("no audio frames in {start:.3}s–{end:.3}s");
    }
    let origin_media = index.start_tick(first) as f64 / index.timescale as f64;
    let end_media = index.start_tick(last) as f64 / index.timescale as f64;
    let durations = index.durations(first..last);
    let sizes = index.sizes[first..last].to_vec();
    let bytes = read_span(file, &index.offsets[first..last], &sizes)?;
    let mp4 = write_mp4(
        &index.stsd,
        index.timescale,
        index.language,
        &durations,
        &sizes,
        &bytes,
    )?;
    Ok(CutBytes {
        bytes: mp4,
        origin_secs: origin_media - index.shift_secs,
        duration_secs: end_media - origin_media,
        media_type: "audio/mp4".to_string(),
        extension: "m4a".to_string(),
    })
}

fn cut_mp3(file: &mut File, index: &Mp3Index, start: f64, end: f64) -> Result<CutBytes> {
    let frame_secs = index.samples_per_frame as f64 / index.rate as f64;
    let total = index.frames.len() as f64 * frame_secs;
    if start >= total {
        bail!("chapter starts at {start:.3}s, past the end of the audio ({total:.3}s)");
    }
    if end > total + 0.05 {
        bail!("chapter runs to {end:.3}s, past the end of the audio ({total:.3}s)");
    }
    let end = end.min(total);
    let first = (start / frame_secs).floor() as usize;
    let mut last = (end / frame_secs).ceil() as usize;
    if last <= first {
        last = first + 1;
    }
    last = last.min(index.frames.len());
    let offsets: Vec<u64> = index.frames[first..last]
        .iter()
        .map(|frame| frame.0)
        .collect();
    let sizes: Vec<u32> = index.frames[first..last]
        .iter()
        .map(|frame| frame.1)
        .collect();
    let bytes = read_span(file, &offsets, &sizes)?;
    let count = (last - first) as f64;
    Ok(CutBytes {
        bytes,
        origin_secs: first as f64 * frame_secs,
        duration_secs: count * frame_secs,
        media_type: "audio/mpeg".to_string(),
        extension: "mp3".to_string(),
    })
}

fn read_span(file: &mut File, offsets: &[u64], sizes: &[u32]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let mut index = 0usize;
    while index < offsets.len() {
        let mut next = offsets[index] + sizes[index] as u64;
        let mut end = index + 1;
        while end < offsets.len() && offsets[end] == next {
            next += sizes[end] as u64;
            end += 1;
        }
        file.seek(SeekFrom::Start(offsets[index]))?;
        let at = out.len();
        let len = (next - offsets[index]) as usize;
        out.resize(at + len, 0);
        file.read_exact(&mut out[at..])
            .with_context(|| format!("reading audio at {}", offsets[index]))?;
        index = end;
    }
    Ok(out)
}

impl Mp4Index {
    fn start_tick(&self, index: usize) -> u64 {
        if index >= self.total_samples {
            return self.total_ticks;
        }
        let run = self
            .runs
            .partition_point(|run| run.first + run.count as usize <= index);
        let run = &self.runs[run];
        run.start_tick + (index - run.first) as u64 * run.delta as u64
    }

    fn delta(&self, index: usize) -> u32 {
        let run = self
            .runs
            .partition_point(|run| run.first + run.count as usize <= index);
        self.runs[run].delta
    }

    fn sample_containing(&self, tick: u64) -> usize {
        if tick >= self.total_ticks {
            return self.total_samples;
        }
        let mut lo = 0usize;
        let mut hi = self.total_samples;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.start_tick(mid) + self.delta(mid) as u64 <= tick {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    fn first_sample_at_or_after(&self, tick: u64) -> usize {
        let mut lo = 0usize;
        let mut hi = self.total_samples;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.start_tick(mid) < tick {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    fn durations(&self, range: std::ops::Range<usize>) -> Vec<u32> {
        let mut out = Vec::with_capacity(range.len());
        let mut index = range.start;
        if index >= self.total_samples {
            return out;
        }
        let mut run_index = self
            .runs
            .partition_point(|run| run.first + run.count as usize <= index);
        while index < range.end {
            let run = &self.runs[run_index];
            let run_end = run.first + run.count as usize;
            while index < range.end && index < run_end {
                out.push(run.delta);
                index += 1;
            }
            run_index += 1;
        }
        out
    }
}

struct BoxRef {
    typ: [u8; 4],
    header: usize,
    start: usize,
    end: usize,
}

fn parse_mp4(file: &mut File) -> Result<Mp4Index> {
    let len = file.seek(SeekFrom::End(0))?;
    let mut pos = 0u64;
    let mut moov = None;
    while pos + 8 <= len {
        file.seek(SeekFrom::Start(pos))?;
        let (size, typ, _header) = read_box_header(file, pos, len)?;
        if typ == *b"moov" {
            if size > 64 * 1024 * 1024 {
                bail!("audio movie header is larger than 64 MB");
            }
            let mut buf = vec![0u8; size as usize];
            file.seek(SeekFrom::Start(pos))?;
            file.read_exact(&mut buf)?;
            moov = Some(buf);
            break;
        }
        pos += size;
    }
    let moov = moov.ok_or_else(|| anyhow!("no movie header in the audio file"))?;
    parse_moov(&moov)
}

fn read_box_header(file: &mut File, pos: u64, len: u64) -> Result<(u64, [u8; 4], u64)> {
    let mut hdr = [0u8; 8];
    file.read_exact(&mut hdr)?;
    let mut size = u32::from_be_bytes(hdr[0..4].try_into().unwrap()) as u64;
    let typ = [hdr[4], hdr[5], hdr[6], hdr[7]];
    let mut header = 8u64;
    if size == 1 {
        let mut wide = [0u8; 8];
        file.read_exact(&mut wide)?;
        size = u64::from_be_bytes(wide);
        header = 16;
    } else if size == 0 {
        size = len - pos;
    }
    if size < header || pos + size > len {
        bail!("broken audio container at {pos}");
    }
    Ok((size, typ, header))
}

fn parse_moov(data: &[u8]) -> Result<Mp4Index> {
    let boxes = read_boxes(data, 0, data.len())?;
    let moov = find(&boxes, b"moov").ok_or_else(|| anyhow!("no movie header"))?;
    let kids = children(data, moov)?;
    let mvhd = find(&kids, b"mvhd").ok_or_else(|| anyhow!("no movie header timescale"))?;
    let movie_timescale = mvhd_timescale(payload(data, mvhd))?;
    let mut sound = None;
    for trak in kids.iter().filter(|child| child.typ == *b"trak") {
        if let Some(index) = parse_trak(data, trak, movie_timescale)? {
            if sound.is_some() {
                bail!("the audio file has more than one sound track");
            }
            sound = Some(index);
        }
    }
    sound.ok_or_else(|| anyhow!("no sound track"))
}

fn parse_trak(data: &[u8], trak: &BoxRef, _movie_timescale: u32) -> Result<Option<Mp4Index>> {
    let kids = children(data, trak)?;
    let Some(mdia) = find(&kids, b"mdia") else {
        return Ok(None);
    };
    let mdia_kids = children(data, mdia)?;
    let hdlr = find(&mdia_kids, b"hdlr").ok_or_else(|| anyhow!("sound track has no handler"))?;
    if handler_type(payload(data, hdlr))? != *b"soun" {
        return Ok(None);
    }
    let mdhd =
        find(&mdia_kids, b"mdhd").ok_or_else(|| anyhow!("sound track has no media header"))?;
    let (timescale, language) = mdhd_fields(payload(data, mdhd))?;
    if timescale == 0 {
        bail!("sound track timescale is 0");
    }
    let shift_secs = if let Some(edts) = find(&kids, b"edts") {
        let edits = children(data, edts)?;
        if let Some(elst) = find(&edits, b"elst") {
            edit_shift(payload(data, elst), timescale)?
        } else {
            0.0
        }
    } else {
        0.0
    };
    let minf = find(&mdia_kids, b"minf").ok_or_else(|| anyhow!("sound track has no media info"))?;
    let minf_kids = children(data, minf)?;
    let stbl =
        find(&minf_kids, b"stbl").ok_or_else(|| anyhow!("sound track has no sample table"))?;
    let table = children(data, stbl)?;
    let stsd =
        find(&table, b"stsd").ok_or_else(|| anyhow!("sound track has no sample description"))?;
    let stts = find(&table, b"stts").ok_or_else(|| anyhow!("sound track has no sample times"))?;
    let stsc = find(&table, b"stsc").ok_or_else(|| anyhow!("sound track has no sample chunks"))?;
    let stsz = find(&table, b"stsz").ok_or_else(|| anyhow!("sound track has no sample sizes"))?;
    let chunk_offsets = if let Some(stco) = find(&table, b"stco") {
        chunk_offsets(payload(data, stco), false)?
    } else if let Some(co64) = find(&table, b"co64") {
        chunk_offsets(payload(data, co64), true)?
    } else {
        bail!("sound track has no chunk offsets");
    };
    let stsd_bytes = data[stsd.start..stsd.end].to_vec();
    let sizes = sample_sizes(payload(data, stsz))?;
    let runs = time_runs(payload(data, stts), sizes.len())?;
    let offsets = sample_offsets(&sizes, &chunk_offsets, payload(data, stsc))?;
    let total_ticks = runs
        .last()
        .map(|run| run.start_tick + run.count as u64 * run.delta as u64)
        .unwrap_or(0);
    Ok(Some(Mp4Index {
        timescale,
        language,
        shift_secs,
        stsd: stsd_bytes,
        total_samples: sizes.len(),
        total_ticks,
        runs,
        sizes,
        offsets,
    }))
}

fn children(data: &[u8], parent: &BoxRef) -> Result<Vec<BoxRef>> {
    read_boxes(data, parent.start + parent.header, parent.end)
}

fn payload<'a>(data: &'a [u8], box_ref: &BoxRef) -> &'a [u8] {
    &data[box_ref.start + box_ref.header..box_ref.end]
}

fn find<'a>(boxes: &'a [BoxRef], typ: &[u8; 4]) -> Option<&'a BoxRef> {
    boxes.iter().find(|box_ref| box_ref.typ == *typ)
}

fn read_boxes(data: &[u8], mut pos: usize, end: usize) -> Result<Vec<BoxRef>> {
    let mut boxes = Vec::new();
    while pos + 8 <= end {
        let mut size = u32::from_be_bytes(data[pos..pos + 4].try_into().unwrap()) as u64;
        let typ = [data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]];
        let mut header = 8usize;
        if size == 1 {
            if pos + 16 > end {
                bail!("broken audio container");
            }
            size = u64::from_be_bytes(data[pos + 8..pos + 16].try_into().unwrap());
            header = 16;
        } else if size == 0 {
            size = (end - pos) as u64;
        }
        if size < header as u64 || pos + size as usize > end {
            bail!("broken audio container");
        }
        let next = pos + size as usize;
        boxes.push(BoxRef {
            typ,
            header,
            start: pos,
            end: next,
        });
        pos = next;
    }
    Ok(boxes)
}

fn mvhd_timescale(payload: &[u8]) -> Result<u32> {
    if payload.is_empty() {
        bail!("empty movie header");
    }
    let timescale_at = if payload[0] == 0 { 12 } else { 20 };
    if payload.len() < timescale_at + 4 {
        bail!("short movie header");
    }
    Ok(u32::from_be_bytes(
        payload[timescale_at..timescale_at + 4].try_into().unwrap(),
    ))
}

fn mdhd_fields(payload: &[u8]) -> Result<(u32, u16)> {
    if payload.is_empty() {
        bail!("empty media header");
    }
    let (timescale_at, language_at) = if payload[0] == 0 { (12, 20) } else { (20, 32) };
    if payload.len() < language_at + 2 {
        bail!("short media header");
    }
    let timescale = u32::from_be_bytes(payload[timescale_at..timescale_at + 4].try_into().unwrap());
    let language = u16::from_be_bytes(payload[language_at..language_at + 2].try_into().unwrap());
    Ok((timescale, language))
}

fn handler_type(payload: &[u8]) -> Result<[u8; 4]> {
    if payload.len() < 12 {
        bail!("short media handler");
    }
    Ok(payload[8..12].try_into().unwrap())
}

fn edit_shift(payload: &[u8], media_timescale: u32) -> Result<f64> {
    if payload.len() < 8 {
        bail!("short audio edit list");
    }
    let version = payload[0];
    let count = u32::from_be_bytes(payload[4..8].try_into().unwrap());
    if count == 0 {
        return Ok(0.0);
    }
    if count != 1 {
        bail!("audio edit list has {count} entries; chapter splitting needs one edit or none");
    }
    let (media_time, rate, fraction) = if version == 0 {
        if payload.len() < 20 {
            bail!("short audio edit list");
        }
        let media_time = i32::from_be_bytes(payload[12..16].try_into().unwrap()) as i64;
        let rate = u16::from_be_bytes(payload[16..18].try_into().unwrap());
        let fraction = u16::from_be_bytes(payload[18..20].try_into().unwrap());
        (media_time, rate, fraction)
    } else {
        if payload.len() < 28 {
            bail!("short audio edit list");
        }
        let media_time = i64::from_be_bytes(payload[16..24].try_into().unwrap());
        let rate = u16::from_be_bytes(payload[24..26].try_into().unwrap());
        let fraction = u16::from_be_bytes(payload[26..28].try_into().unwrap());
        (media_time, rate, fraction)
    };
    if media_time < 0 {
        bail!("audio edit list is empty; chapter splitting cannot use this file");
    }
    if rate != 1 || fraction != 0 {
        bail!("audio edit list changes the playback speed; chapter splitting cannot use this file");
    }
    Ok(media_time as f64 / media_timescale as f64)
}

fn sample_sizes(payload: &[u8]) -> Result<Vec<u32>> {
    if payload.len() < 12 {
        bail!("short sample-size table");
    }
    let sample_size = u32::from_be_bytes(payload[4..8].try_into().unwrap());
    let count = u32::from_be_bytes(payload[8..12].try_into().unwrap()) as usize;
    if sample_size > 0 {
        return Ok(vec![sample_size; count]);
    }
    if payload.len() < 12 + count * 4 {
        bail!("sample-size table is truncated");
    }
    let mut sizes = Vec::with_capacity(count);
    for index in 0..count {
        let at = 12 + index * 4;
        sizes.push(u32::from_be_bytes(payload[at..at + 4].try_into().unwrap()));
    }
    Ok(sizes)
}

fn time_runs(payload: &[u8], samples: usize) -> Result<Vec<Run>> {
    if payload.len() < 8 {
        bail!("short sample-time table");
    }
    let entries = u32::from_be_bytes(payload[4..8].try_into().unwrap()) as usize;
    if payload.len() < 8 + entries * 8 {
        bail!("sample-time table is truncated");
    }
    let mut runs = Vec::new();
    let mut seen = 0usize;
    let mut tick = 0u64;
    for index in 0..entries {
        let at = 8 + index * 8;
        let count = u32::from_be_bytes(payload[at..at + 4].try_into().unwrap());
        let delta = u32::from_be_bytes(payload[at + 4..at + 8].try_into().unwrap());
        if delta == 0 {
            bail!("audio frame duration is 0");
        }
        let take = count as usize;
        if seen >= samples {
            break;
        }
        let take = take.min(samples - seen);
        runs.push(Run {
            count: take as u32,
            delta,
            first: seen,
            start_tick: tick,
        });
        seen += take;
        tick += take as u64 * delta as u64;
        if seen == samples {
            break;
        }
    }
    if seen < samples {
        let delta = runs.last().map(|run| run.delta).unwrap_or(0);
        if delta == 0 {
            bail!("audio frame duration is missing");
        }
        runs.push(Run {
            count: (samples - seen) as u32,
            delta,
            first: seen,
            start_tick: tick,
        });
    }
    if runs.is_empty() {
        bail!("audio has no frames");
    }
    Ok(runs)
}

fn chunk_offsets(payload: &[u8], wide: bool) -> Result<Vec<u64>> {
    if payload.len() < 8 {
        bail!("short chunk-offset table");
    }
    let count = u32::from_be_bytes(payload[4..8].try_into().unwrap()) as usize;
    let width = if wide { 8 } else { 4 };
    if payload.len() < 8 + count * width {
        bail!("chunk-offset table is truncated");
    }
    let mut offsets = Vec::with_capacity(count);
    for index in 0..count {
        let at = 8 + index * width;
        let offset = if wide {
            u64::from_be_bytes(payload[at..at + 8].try_into().unwrap())
        } else {
            u32::from_be_bytes(payload[at..at + 4].try_into().unwrap()) as u64
        };
        offsets.push(offset);
    }
    Ok(offsets)
}

fn sample_offsets(sizes: &[u32], chunks: &[u64], stsc: &[u8]) -> Result<Vec<u64>> {
    if stsc.len() < 8 {
        bail!("short sample-to-chunk table");
    }
    let entries = u32::from_be_bytes(stsc[4..8].try_into().unwrap()) as usize;
    if entries == 0 || stsc.len() < 8 + entries * 12 {
        bail!("sample-to-chunk table is truncated");
    }
    let mut map = Vec::with_capacity(entries);
    for index in 0..entries {
        let at = 8 + index * 12;
        let first = u32::from_be_bytes(stsc[at..at + 4].try_into().unwrap());
        let per_chunk = u32::from_be_bytes(stsc[at + 4..at + 8].try_into().unwrap());
        let desc = u32::from_be_bytes(stsc[at + 8..at + 12].try_into().unwrap());
        if desc != 1 {
            bail!("audio file uses more than one sample description");
        }
        if per_chunk == 0 || first == 0 {
            bail!("broken sample-to-chunk table");
        }
        map.push((first, per_chunk));
    }
    if map[0].0 != 1 {
        bail!("sample-to-chunk table does not start at the first chunk");
    }
    let mut offsets = vec![0u64; sizes.len()];
    let mut sample = 0usize;
    let mut map_index = 0usize;
    for (chunk_index, chunk_offset) in chunks.iter().copied().enumerate() {
        let chunk_no = (chunk_index + 1) as u32;
        while map_index + 1 < map.len() && map[map_index + 1].0 <= chunk_no {
            map_index += 1;
        }
        let mut offset = chunk_offset;
        for _ in 0..map[map_index].1 {
            if sample >= sizes.len() {
                bail!("chunk table has more frames than the sample-size table");
            }
            offsets[sample] = offset;
            offset += sizes[sample] as u64;
            sample += 1;
        }
    }
    if sample != sizes.len() {
        bail!(
            "chunk table lists {sample} frames, sample-size table lists {}",
            sizes.len()
        );
    }
    Ok(offsets)
}

fn parse_mp3(file: &mut File) -> Result<Mp3Index> {
    let mut data = Vec::new();
    file.read_to_end(&mut data)?;
    let mut pos = skip_id3v2(&data);
    let mut frames = Vec::new();
    let mut samples_per_frame = None;
    let mut rate = None;
    while pos + 4 <= data.len() {
        if let Some(frame) = mp3_header(&data[pos..pos + 4]) {
            let next = pos + frame.size as usize;
            if next <= data.len() {
                let followed = next == data.len()
                    || (next + 4 <= data.len() && mp3_header(&data[next..next + 4]).is_some())
                    || (next + 128 == data.len() && data.get(next..next + 3) == Some(b"TAG"));
                if followed {
                    if let Some(seen) = samples_per_frame {
                        if seen != frame.samples || rate != Some(frame.rate) {
                            bail!("mp3 frame size changes mid-file");
                        }
                    }
                    samples_per_frame = Some(frame.samples);
                    rate = Some(frame.rate);
                    frames.push((pos as u64, frame.size));
                    pos = if next + 128 == data.len() && data.get(next..next + 3) == Some(b"TAG") {
                        data.len()
                    } else {
                        next
                    };
                    continue;
                }
            }
        }
        pos += 1;
    }
    let (Some(samples_per_frame), Some(rate)) = (samples_per_frame, rate) else {
        bail!("no mp3 frames");
    };
    if frames.is_empty() {
        bail!("no mp3 frames");
    }
    Ok(Mp3Index {
        frames,
        samples_per_frame,
        rate,
    })
}

fn skip_id3v2(data: &[u8]) -> usize {
    if data.len() < 10 || &data[0..3] != b"ID3" {
        return 0;
    }
    let size = synchsafe(&data[6..10]);
    let footer = if data[5] & 0x10 != 0 { 10 } else { 0 };
    (10 + size + footer).min(data.len())
}

fn synchsafe(bytes: &[u8]) -> usize {
    ((bytes[0] as usize & 0x7f) << 21)
        | ((bytes[1] as usize & 0x7f) << 14)
        | ((bytes[2] as usize & 0x7f) << 7)
        | (bytes[3] as usize & 0x7f)
}

struct Mp3Frame {
    size: u32,
    samples: u32,
    rate: u32,
}

fn mp3_header(header: &[u8]) -> Option<Mp3Frame> {
    if header.len() < 4 || header[0] != 0xff || header[1] & 0xe0 != 0xe0 {
        return None;
    }
    let version = (header[1] >> 3) & 0x03;
    let layer = (header[1] >> 1) & 0x03;
    if version == 1 || layer != 1 {
        return None;
    }
    let bitrate_index = (header[2] >> 4) & 0x0f;
    let rate_index = (header[2] >> 2) & 0x03;
    let padding = ((header[2] >> 1) & 0x01) as u32;
    if bitrate_index == 0 || bitrate_index == 15 || rate_index == 3 {
        return None;
    }
    let mpeg1 = version == 3;
    let bitrates = if mpeg1 {
        [
            0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0,
        ]
    } else {
        [
            0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0,
        ]
    };
    let rates = match version {
        3 => [44100, 48000, 32000],
        2 => [22050, 24000, 16000],
        _ => [11025, 12000, 8000],
    };
    let rate = rates[rate_index as usize];
    let bits = bitrates[bitrate_index as usize] * 1000;
    let size = if mpeg1 {
        (144 * bits) / rate + padding
    } else {
        (72 * bits) / rate + padding
    };
    if size < 4 {
        return None;
    }
    Some(Mp3Frame {
        size,
        samples: if mpeg1 { 1152 } else { 576 },
        rate,
    })
}

fn write_mp4(
    stsd: &[u8],
    timescale: u32,
    language: u16,
    durations: &[u32],
    sizes: &[u32],
    samples: &[u8],
) -> Result<Vec<u8>> {
    if durations.len() != sizes.len() || durations.is_empty() {
        bail!("cannot write an empty chapter");
    }
    let duration: u64 = durations.iter().map(|delta| *delta as u64).sum();
    if duration > u32::MAX as u64 {
        bail!("chapter is too long to store in one m4a file");
    }
    let mut out = Vec::new();
    write_ftyp(&mut out);
    let moov_at = begin_box(&mut out, b"moov");
    write_mvhd(&mut out, timescale, duration as u32);
    let trak_at = begin_box(&mut out, b"trak");
    write_tkhd(&mut out, duration as u32);
    let mdia_at = begin_box(&mut out, b"mdia");
    write_mdhd(&mut out, timescale, duration as u32, language);
    write_hdlr(&mut out);
    let minf_at = begin_box(&mut out, b"minf");
    write_smhd(&mut out);
    write_dinf(&mut out);
    let stbl_at = begin_box(&mut out, b"stbl");
    out.extend_from_slice(stsd);
    write_stts(&mut out, durations);
    write_stsc(&mut out);
    write_stsz(&mut out, sizes);
    let stco_entries = begin_stco(&mut out, sizes.len());
    end_box(&mut out, stbl_at);
    end_box(&mut out, minf_at);
    end_box(&mut out, mdia_at);
    end_box(&mut out, trak_at);
    end_box(&mut out, moov_at);
    let mdat_at = begin_box(&mut out, b"mdat");
    let data_at = out.len();
    if data_at as u64 + samples.len() as u64 > u32::MAX as u64 {
        bail!("chapter audio is too large to store");
    }
    out.extend_from_slice(samples);
    end_box(&mut out, mdat_at);
    let mut cursor = data_at as u32;
    for (index, size) in sizes.iter().enumerate() {
        let at = stco_entries + index * 4;
        out[at..at + 4].copy_from_slice(&cursor.to_be_bytes());
        cursor += size;
    }
    Ok(out)
}

fn write_ftyp(out: &mut Vec<u8>) {
    let at = begin_box(out, b"ftyp");
    out.extend_from_slice(b"M4A ");
    be32(out, 0);
    out.extend_from_slice(b"M4A mp42isom");
    end_box(out, at);
}

fn write_mvhd(out: &mut Vec<u8>, timescale: u32, duration: u32) {
    let at = begin_box(out, b"mvhd");
    be32(out, 0);
    be32(out, 0);
    be32(out, 0);
    be32(out, timescale);
    be32(out, duration);
    be32(out, 0x0001_0000);
    be16(out, 0x0100);
    be16(out, 0);
    out.extend_from_slice(&[0u8; 8]);
    write_matrix(out);
    out.extend_from_slice(&[0u8; 24]);
    be32(out, 2);
    end_box(out, at);
}

fn write_tkhd(out: &mut Vec<u8>, duration: u32) {
    let at = begin_box(out, b"tkhd");
    be32(out, 0x0000_0007);
    be32(out, 0);
    be32(out, 0);
    be32(out, 1);
    be32(out, 0);
    be32(out, duration);
    out.extend_from_slice(&[0u8; 8]);
    be16(out, 0);
    be16(out, 0);
    be16(out, 0x0100);
    be16(out, 0);
    write_matrix(out);
    be32(out, 0);
    be32(out, 0);
    end_box(out, at);
}

fn write_mdhd(out: &mut Vec<u8>, timescale: u32, duration: u32, language: u16) {
    let at = begin_box(out, b"mdhd");
    be32(out, 0);
    be32(out, 0);
    be32(out, 0);
    be32(out, timescale);
    be32(out, duration);
    be16(out, language);
    be16(out, 0);
    end_box(out, at);
}

fn write_hdlr(out: &mut Vec<u8>) {
    let at = begin_box(out, b"hdlr");
    be32(out, 0);
    be32(out, 0);
    out.extend_from_slice(b"soun");
    out.extend_from_slice(&[0u8; 12]);
    out.extend_from_slice(b"SoundHandler\0");
    end_box(out, at);
}

fn write_smhd(out: &mut Vec<u8>) {
    let at = begin_box(out, b"smhd");
    be32(out, 0);
    be16(out, 0);
    be16(out, 0);
    end_box(out, at);
}

fn write_dinf(out: &mut Vec<u8>) {
    let at = begin_box(out, b"dinf");
    let dref = begin_box(out, b"dref");
    be32(out, 0);
    be32(out, 1);
    let url = begin_box(out, b"url ");
    be32(out, 1);
    end_box(out, url);
    end_box(out, dref);
    end_box(out, at);
}

fn write_stts(out: &mut Vec<u8>, durations: &[u32]) {
    let at = begin_box(out, b"stts");
    be32(out, 0);
    let count_at = out.len();
    be32(out, 0);
    let mut entries = 0u32;
    let mut index = 0usize;
    while index < durations.len() {
        let delta = durations[index];
        let mut count = 1u32;
        index += 1;
        while index < durations.len() && durations[index] == delta {
            count += 1;
            index += 1;
        }
        be32(out, count);
        be32(out, delta);
        entries += 1;
    }
    out[count_at..count_at + 4].copy_from_slice(&entries.to_be_bytes());
    end_box(out, at);
}

fn write_stsc(out: &mut Vec<u8>) {
    let at = begin_box(out, b"stsc");
    be32(out, 0);
    be32(out, 1);
    be32(out, 1);
    be32(out, 1);
    be32(out, 1);
    end_box(out, at);
}

fn write_stsz(out: &mut Vec<u8>, sizes: &[u32]) {
    let at = begin_box(out, b"stsz");
    be32(out, 0);
    let constant = sizes.iter().all(|size| *size == sizes[0]);
    if constant {
        be32(out, sizes[0]);
        be32(out, sizes.len() as u32);
    } else {
        be32(out, 0);
        be32(out, sizes.len() as u32);
        for size in sizes {
            be32(out, *size);
        }
    }
    end_box(out, at);
}

fn begin_stco(out: &mut Vec<u8>, count: usize) -> usize {
    let at = begin_box(out, b"stco");
    be32(out, 0);
    be32(out, count as u32);
    let entries = out.len();
    out.extend(std::iter::repeat(0u8).take(count * 4));
    end_box(out, at);
    entries
}

fn write_matrix(out: &mut Vec<u8>) {
    be32(out, 0x0001_0000);
    be32(out, 0);
    be32(out, 0);
    be32(out, 0);
    be32(out, 0x0001_0000);
    be32(out, 0);
    be32(out, 0);
    be32(out, 0);
    be32(out, 0x4000_0000);
}

fn begin_box(out: &mut Vec<u8>, typ: &[u8; 4]) -> usize {
    let at = out.len();
    be32(out, 0);
    out.extend_from_slice(typ);
    at
}

fn end_box(out: &mut Vec<u8>, at: usize) {
    let size = (out.len() - at) as u32;
    out[at..at + 4].copy_from_slice(&size.to_be_bytes());
}

fn be32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn be16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_be_bytes());
}

#[cfg(test)]
pub(crate) fn write_marker_fixture(path: &Path) {
    tests::marker_fixture(path);
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn marker_fixture(path: &Path) {
        let markers = [0xA1u8, 0xB2, 0xC3, 0xD4];
        let mut bytes = Vec::new();
        let mut sizes = Vec::new();
        for marker in markers {
            bytes.extend(std::iter::repeat(marker).take(8));
            sizes.push(8u32);
        }
        let durations = vec![16_000u32; 4];
        let file = write_mp4(&minimal_stsd(), 16_000, 0x55C4, &durations, &sizes, &bytes).unwrap();
        std::fs::write(path, file).unwrap();
    }

    fn minimal_stsd() -> Vec<u8> {
        let mut out = Vec::new();
        let at = begin_box(&mut out, b"stsd");
        be32(&mut out, 0);
        be32(&mut out, 1);
        let entry = begin_box(&mut out, b"mp4a");
        out.extend_from_slice(&[0u8; 28]);
        end_box(&mut out, entry);
        end_box(&mut out, at);
        out
    }

    #[test]
    fn an_m4a_cut_keeps_the_frames_inside_the_range() {
        let dir =
            std::env::temp_dir().join(format!("starlign-split-fixture-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let source = dir.join("book.m4a");
        marker_fixture(&source);
        let track = Track {
            source: source.clone(),
            epub_name: "book.m4a".into(),
            media_type: "audio/mp4".into(),
            start_sample: 0,
            samples: 4 * SAMPLE_RATE as u64,
        };
        let mut cache = Cache::new();
        let mut used = HashSet::new();
        let pieces = cut_range(&mut cache, &[track], 1.25, 2.10, "chapter", &mut used).unwrap();
        assert_eq!(pieces.len(), 1);
        assert_eq!(pieces[0].epub_name, "chapter.m4a");
        assert!(
            (pieces[0].origin_secs - 1.0).abs() < 1e-6,
            "{}",
            pieces[0].origin_secs
        );
        assert!(
            (pieces[0].duration_secs - 2.0).abs() < 1e-6,
            "{}",
            pieces[0].duration_secs
        );
        let bytes = std::fs::read(pieces[0].path()).unwrap();
        assert!(
            bytes.windows(8).any(|window| window == [0xB2; 8]),
            "missing second frame"
        );
        assert!(
            bytes.windows(8).any(|window| window == [0xC3; 8]),
            "missing third frame"
        );
        assert!(
            !bytes.windows(8).any(|window| window == [0xA1; 8]),
            "kept the first frame"
        );
        assert!(
            !bytes.windows(8).any(|window| window == [0xD4; 8]),
            "kept the last frame"
        );
        let track = pieces[0].track();
        assert!((track.start_secs() - 1.0).abs() < 1e-6);
        assert!(track.duration_secs() >= 2.0);
        let temp = pieces[0].path().to_path_buf();
        drop(pieces);
        assert!(!temp.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_mp3_cut_keeps_the_frames_inside_the_range() {
        let dir = std::env::temp_dir().join(format!("starlign-split-mp3-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let source = dir.join("book.mp3");
        let header = [0xffu8, 0xfb, 0x90, 0x00];
        let frame = mp3_header(&header).unwrap();
        assert_eq!(frame.size, 417);
        let mut bytes = Vec::new();
        for marker in [0x11u8, 0x22, 0x33, 0x44] {
            bytes.extend_from_slice(&header);
            bytes.extend(std::iter::repeat(marker).take(frame.size as usize - 4));
        }
        std::fs::write(&source, &bytes).unwrap();
        let frame_secs = 1152.0 / 44100.0;
        let track = Track {
            source: source.clone(),
            epub_name: "book.mp3".into(),
            media_type: "audio/mpeg".into(),
            start_sample: 0,
            samples: (4.0 * frame_secs * SAMPLE_RATE as f64).round() as u64,
        };
        let mut cache = Cache::new();
        let pieces = cut_range(
            &mut cache,
            &[track],
            frame_secs * 1.2,
            frame_secs * 2.2,
            "ch",
            &mut HashSet::new(),
        )
        .unwrap();
        assert_eq!(pieces[0].epub_name, "ch.mp3");
        assert_eq!(pieces[0].media_type, "audio/mpeg");
        let cut = std::fs::read(pieces[0].path()).unwrap();
        assert_eq!(cut.len(), frame.size as usize * 2);
        assert_eq!(cut[4], 0x22);
        assert_eq!(cut[frame.size as usize + 4], 0x33);
        assert!((pieces[0].origin_secs - frame_secs).abs() < 1e-9);
        drop(pieces);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn other_audio_formats_say_what_can_be_split() {
        let dir = std::env::temp_dir().join(format!("starlign-split-flac-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let source = dir.join("book.flac");
        std::fs::write(&source, b"fLaC").unwrap();
        let track = Track {
            source,
            epub_name: "book.flac".into(),
            media_type: "audio/flac".into(),
            start_sample: 0,
            samples: SAMPLE_RATE as u64,
        };
        let err = cut_range(
            &mut Cache::new(),
            &[track],
            0.0,
            0.5,
            "ch",
            &mut HashSet::new(),
        )
        .unwrap_err();
        let text = format!("{err:#}");
        assert!(text.contains("m4b"), "{text}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_recovery_m4b_slice_decodes() {
        let source = PathBuf::from(
            "/home/yy/code/st/Recovery - the Sacred Art: The Twelve Steps as Spiritual Practice Rami Shapiro, Joan Borysenko PhD[B0735SJH91].m4b",
        );
        if !source.is_file() {
            return;
        }
        let track = Track {
            source: source.clone(),
            epub_name: "book.m4b".into(),
            media_type: "audio/mp4".into(),
            start_sample: 0,
            samples: 468_152_924,
        };
        let mut cache = Cache::new();
        let pieces = cut_range(
            &mut cache,
            &[track],
            12.0,
            14.0,
            "part",
            &mut HashSet::new(),
        )
        .unwrap();
        assert_eq!(pieces.len(), 1);
        assert!(pieces[0].origin_secs <= 12.0, "{}", pieces[0].origin_secs);
        assert!(
            pieces[0].duration_secs >= 2.0,
            "{}",
            pieces[0].duration_secs
        );
        assert!(pieces[0].duration_secs < 2.2, "{}", pieces[0].duration_secs);
        let pcm = std::env::temp_dir().join(format!("starlign-split-pcm-{}", std::process::id()));
        let decoded = crate::audio::decode_audiobook(pieces[0].path(), &pcm, None).unwrap();
        let drift = (decoded.duration_secs() - pieces[0].duration_secs).abs();
        assert!(
            drift < 0.08,
            "decode {:.3}s slice {:.3}s",
            decoded.duration_secs(),
            pieces[0].duration_secs
        );
        let _ = std::fs::remove_file(&pcm);
        drop(pieces);
    }

    #[test]
    fn seconds_and_parts_share_a_grid_and_a_crossing_span_stays_put() {
        let seconds = nominal_slices(Split::Seconds(10.0), 0.0, 25.0).unwrap();
        assert_eq!(seconds, vec![(0.0, 10.0), (10.0, 20.0), (20.0, 25.0)]);
        let parts = nominal_slices(Split::Parts(4), 0.0, 100.0).unwrap();
        assert_eq!(parts.len(), 4);
        assert!((parts[0].0 - 0.0).abs() < 1e-9);
        assert!((parts[1].0 - 25.0).abs() < 1e-9);
        assert!((parts[3].1 - 100.0).abs() < 1e-9);
        assert_eq!(
            nominal_slices(Split::Parts(2), 0.0, 4.0).unwrap(),
            nominal_slices(Split::Seconds(2.0), 0.0, 4.0).unwrap()
        );

        let nominal = nominal_slices(Split::Seconds(10.0), 0.0, 30.0).unwrap();
        let (covered, owners) = assign_slices(&nominal, &[(8.0, 12.5), (12.5, 18.0), (10.0, 11.0)]);
        assert_eq!(owners, vec![0, 1, 1]);
        assert!((covered[0].1 - 12.5).abs() < 1e-9, "{covered:?}");
        assert!((covered[1].0 - 10.0).abs() < 1e-9, "{covered:?}");
        assert!((covered[1].1 - 20.0).abs() < 1e-9, "{covered:?}");
        assert_eq!(slice_index(&nominal, 10.0), 1);
        assert_eq!(slice_index(&nominal, 9.999), 0);

        assert!(Split::resolve(true, Some(10.0), None).is_err());
        assert!(Split::resolve(false, Some(0.0), None).is_err());
        assert!(Split::resolve(false, None, Some(1)).is_err());
        assert_eq!(
            Split::resolve(false, Some(30.0), None).unwrap(),
            Split::Seconds(30.0)
        );
        assert_eq!(
            Split::resolve(false, None, Some(8)).unwrap(),
            Split::Parts(8)
        );
        assert_eq!(Split::resolve(true, None, None).unwrap(), Split::Chapters);
        assert_eq!(Split::resolve(false, None, None).unwrap(), Split::Whole);
    }
}
