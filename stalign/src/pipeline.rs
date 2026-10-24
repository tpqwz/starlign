//! Decode, emit, align, and write an EPUB 3 media overlay.

use crate::audio::{self, Timeline};
use crate::ctc::{self, Anchor, Emissions, TokenAlignment};
use crate::emit::{self, Device};
use crate::epub::{self, Book, ChapterOutput, OverlaySpan, SpanKind};
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

pub use crate::emit::Device as DevicePref;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Granularity {
    Sentence,
    Word,
}

#[derive(Clone, Debug)]
pub struct RunOptions {
    pub audiobook: PathBuf,
    pub epub: PathBuf,
    pub output: PathBuf,
    pub emissions: Option<PathBuf>,
    pub device: Device,
    pub granularity: Granularity,
    pub model: PathBuf,
    pub gap_seconds: f64,
    pub drop_unread: bool,
    pub max_seconds: Option<f64>,
    pub reports: Option<PathBuf>,
    pub download: bool,
}

pub const MODEL_URL: &str = "https://huggingface.co/onnx-community/mms-300m-1130-forced-aligner-ONNX/resolve/main/onnx/model_fp16.onnx";

pub fn run(opts: &RunOptions) -> Result<()> {
    ensure_model(&opts.model, opts.download)?;
    let loaded = load_alignment_inputs(opts, true)?;
    let _pcm_guard = loaded.pcm.map(PcmGuard);
    align_loaded(opts, &loaded.timeline, &loaded.emissions, &loaded.device)
}

pub fn emit_audio(
    audiobook: &Path,
    emissions_path: &Path,
    model: &Path,
    device: Device,
    max_seconds: Option<f64>,
    download: bool,
) -> Result<()> {
    ensure_model(model, download)?;
    let pcm = temp_pcm();
    let guard = PcmGuard(pcm.clone());
    let timeline = audio::decode_audiobook(audiobook, &pcm, max_seconds)?;
    let (emissions, device_name) = emit::emit_pcm(
        &pcm,
        timeline.samples(),
        model,
        device,
    )?;
    if let Some(parent) = emissions_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    emissions.save(emissions_path)?;
    save_tracks(emissions_path, &timeline)?;
    crate::say!(
        "wrote {} ({:.1}s, {device_name})",
        emissions_path.display(),
        timeline.duration_secs()
    );
    drop(guard);
    Ok(())
}

pub fn align_book(opts: &RunOptions) -> Result<()> {
    let loaded = load_alignment_inputs(opts, false)?;
    let _pcm_guard = loaded.pcm.map(PcmGuard);
    align_loaded(opts, &loaded.timeline, &loaded.emissions, &loaded.device)
}

struct Loaded {
    timeline: Timeline,
    emissions: Emissions,
    device: String,
    pcm: Option<PathBuf>,
}

fn load_alignment_inputs(opts: &RunOptions, allow_emit: bool) -> Result<Loaded> {
    if let Some(path) = &opts.emissions {
        if path.is_file() {
            let emissions = Emissions::load(path)?;
            if let Some(timeline) = load_tracks(path)? {
                crate::say!("using emissions {}", path.display());
                return Ok(Loaded {
                    timeline,
                    emissions,
                    device: "cached".into(),
                    pcm: None,
                });
            }
            if !allow_emit {
                let timeline = timeline_only(&opts.audiobook, opts.max_seconds)?;
                return Ok(Loaded {
                    timeline,
                    emissions,
                    device: "cached".into(),
                    pcm: None,
                });
            }
        }
    }
    if !allow_emit {
        let Some(path) = &opts.emissions else {
            bail!("align needs --emissions");
        };
        bail!("emissions file not found: {}", path.display());
    }
    ensure_model(&opts.model, opts.download)?;
    let pcm = temp_pcm();
    let timeline = audio::decode_audiobook(&opts.audiobook, &pcm, opts.max_seconds)?;
    let (emissions, device) = emit::emit_pcm(&pcm, timeline.samples(), &opts.model, opts.device)?;
    if let Some(path) = &opts.emissions {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        emissions.save(path)?;
        save_tracks(path, &timeline)?;
        crate::say!("wrote {}", path.display());
    }
    Ok(Loaded {
        timeline,
        emissions,
        device,
        pcm: Some(pcm),
    })
}

fn timeline_only(audiobook: &Path, max_seconds: Option<f64>) -> Result<Timeline> {
    let pcm = temp_pcm();
    let guard = PcmGuard(pcm.clone());
    let timeline = audio::decode_audiobook(audiobook, &pcm, max_seconds)?;
    drop(guard);
    Ok(Timeline {
        pcm_path: None,
        tracks: timeline.tracks,
    })
}

fn align_loaded(opts: &RunOptions, timeline: &Timeline, emissions: &Emissions, device: &str) -> Result<()> {
    let book = Book::open(&opts.epub)?;
    let sources = book.chapters()?;
    let mut excluded: Vec<(i64, i64)> = Vec::new();
    let mut outputs = Vec::new();
    let mut reports = Vec::new();
    let mut covered_until = 0.0f64;
    let word_level = opts.granularity == Granularity::Word;
    for source in &sources {
        if source.skip {
            crate::say!("skip {} ({})", source.href, source.skip_reason);
            continue;
        }
        let (xhtml, text) = epub::markup_xml(&source.xml, &id_prefix(&source.href), word_level)?;
        if text.letters.len() < 10 {
            crate::say!("skip {} (too little text)", source.href);
            continue;
        }
        let Some(found) = align_query(&text.letters, emissions, &excluded, opts.drop_unread) else {
            crate::say!("no alignment for {}", source.href);
            reports.push(serde_json::json!({
                "href": source.href,
                "aligned": false,
                "letters": text.letters.len(),
            }));
            continue;
        };
        let (raw, claimed) = time_units(&text, &found, emissions, word_level);
        let spans = finalize_spans(&raw, opts.gap_seconds, opts.drop_unread, covered_until);
        let voiced = spans.iter().any(|span| {
            span.kind != crate::epub::SpanKind::AudioOnly && span.end - span.start > 0.05
        });
        if !voiced {
            crate::say!("no alignment for {}", source.href);
            reports.push(serde_json::json!({
                "href": source.href,
                "aligned": false,
                "letters": text.letters.len(),
                "confidence": found.confidence,
            }));
            continue;
        }
        // Reserve the audio this chapter actually highlighted. The search
        // window runs past that, into the next chapter's pad, and must stay
        // available for the next chapter.
        if let Some(span) = claimed {
            excluded.push(span);
        }
        if let Some(last) = spans.iter().map(|span| span.end).max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)) {
            covered_until = covered_until.max(last);
        }
        let matched = spans.iter().filter(|span| span.kind == SpanKind::Matched).count();
        crate::say!(
            "{}  confidence {:.2}  {} spans  {:.1}s–{:.1}s",
            source.href,
            found.confidence,
            matched,
            spans.first().map(|span| span.start).unwrap_or(0.0),
            spans.last().map(|span| span.end).unwrap_or(0.0),
        );
        reports.push(serde_json::json!({
            "href": source.href,
            "aligned": true,
            "confidence": found.confidence,
            "letters": text.letters.len(),
            "spans": spans.len(),
            "matched": matched,
        }));
        outputs.push(ChapterOutput {
            href: source.href.clone(),
            zip_path: source.zip_path.clone(),
            xhtml,
            spans,
        });
    }
    if outputs.is_empty() {
        bail!("no chapter aligned. The audio may not contain this text, or --max-seconds cut it off before the chapter.");
    }
    book.write(&opts.output, &outputs, &timeline.tracks, timeline.duration_secs())?;
    crate::say!("wrote {}", opts.output.display());
    if let Some(path) = &opts.reports {
        let report = serde_json::json!({
            "device": device,
            "audio_seconds": timeline.duration_secs(),
            "chapters": reports,
        });
        std::fs::write(path, serde_json::to_string_pretty(&report)?)?;
        crate::say!("wrote {}", path.display());
    }
    Ok(())
}

struct Found {
    confidence: f64,
    frame_start: i64,
    ev0: usize,
    tokens: Vec<TokenAlignment>,
}

fn align_query(query: &str, emissions: &Emissions, excluded: &[(i64, i64)], drop_unread: bool) -> Option<Found> {
    let mut boundary = ctc::find_boundaries(query, emissions, excluded)?;
    let (start, end) = narrow_to_free(excluded, boundary.start, boundary.end)?;
    boundary.start = start;
    boundary.end = end;
    let ev0 = boundary.evidence.0.min(query.len());
    let ev1 = boundary.evidence.1.clamp(ev0, query.len());
    if ev1 - ev0 < 10 {
        return None;
    }
    let slice_end = ((boundary.end as usize) + 1).min(emissions.frames());
    let slice_start = (boundary.start as usize).min(slice_end);
    let slice = emissions.slice(slice_start, slice_end);
    if slice.frames() == 0 {
        return None;
    }
    let labels: Vec<i32> = query[ev0..ev1]
        .chars()
        .map(|ch| ctc::label_of(ch).unwrap_or(ctc::CLASSES as i32))
        .collect();
    let mut anchors = Vec::new();
    for anchor in &boundary.anchors {
        if anchor.offset < ev0 || anchor.offset > ev1 {
            continue;
        }
        let position = anchor.position;
        if position < boundary.start || position > boundary.end {
            continue;
        }
        anchors.push(Anchor {
            offset: anchor.offset - ev0,
            position: position - boundary.start,
        });
    }
    let aligned = ctc::forced_align(&slice, &labels, &anchors, ctc::CLASSES as i32, drop_unread);
    Some(Found {
        confidence: boundary.confidence,
        frame_start: boundary.start,
        ev0,
        tokens: aligned,
    })
}

/// Keep the widest part of `start..end` that does not sit inside audio an
/// earlier chapter already claimed. Mirrors Storyteller's
/// `narrowToAvailableBoundary`.
fn narrow_to_free(excluded: &[(i64, i64)], start: i64, end: i64) -> Option<(i64, i64)> {
    if end <= start {
        return None;
    }
    let mut ranges: Vec<(i64, i64)> = excluded.iter().copied().filter(|(lo, hi)| hi > lo).collect();
    if ranges.is_empty() {
        return Some((start, end));
    }
    ranges.sort_by_key(|range| range.0);
    let mut merged: Vec<(i64, i64)> = Vec::new();
    for (lo, hi) in ranges {
        if let Some(prev) = merged.last_mut() {
            if lo <= prev.1 {
                prev.1 = prev.1.max(hi);
                continue;
            }
        }
        merged.push((lo, hi));
    }
    let mut points = vec![-1i64];
    for (lo, hi) in &merged {
        points.push(*lo);
        points.push(*hi);
    }
    points.push(i64::MAX / 4);
    let mut best: Option<(i64, i64)> = None;
    let mut index = 0;
    while index + 1 < points.len() {
        let gap_lo = points[index];
        let gap_hi = points[index + 1];
        let overlaps = (start <= gap_lo && end >= gap_lo) || (start <= gap_hi && end >= gap_hi);
        if overlaps {
            let lo = start.max(gap_lo + 1);
            let hi = end.min(gap_hi - 1);
            if lo <= hi && best.map(|(a, b)| hi - lo > b - a).unwrap_or(true) {
                best = Some((lo, hi));
            }
        }
        index += 2;
    }
    if let Some(window) = best {
        return Some(window);
    }
    let buried = merged.iter().any(|(lo, hi)| *lo <= start && end <= *hi);
    if buried { None } else { Some((start, end)) }
}

/// Longest silence, in frames, tolerated inside one sentence. A wider gap means
/// the characters past it were pinned onto some other speech.
const MAX_INTRA_SENTENCE_GAP_FRAMES: i64 = 150;
/// Pull a sentence start back over the attack the spike misses. 25 frames is 0.5s.
const SENTENCE_PAD_FRAMES: i64 = 25;

pub(crate) struct RawSpan {
    id: String,
    start: Option<f64>,
    end: Option<f64>,
    dropped: bool,
    interpolated: bool,
}

struct KeptFrames {
    start_frame: i64,
    end_frame: i64,
    dropped: bool,
}

fn time_units(
    text: &epub::ChapterText,
    found: &Found,
    emissions: &Emissions,
    word_level: bool,
) -> (Vec<RawSpan>, Option<(i64, i64)>) {
    let hop = emissions.hop().max(1e-4);
    let start_bias = (ctc::MMS_START_OFFSET / hop).round() as i64;
    let end_bias = (ctc::MMS_END_OFFSET / hop).round() as i64;
    let origin = found.frame_start;
    let mut prev_end = -1i64;
    let mut raw = Vec::new();
    let mut claimed: Option<(i64, i64)> = None;
    for sentence in &text.sentences {
        if sentence.letter_range.is_empty() {
            continue;
        }
        let kept = keep_sentence(&sentence.letter_range, found, &mut prev_end, start_bias, end_bias);
        if let Some(span) = &kept {
            let lo = origin + span.start_frame;
            let hi = origin + span.end_frame;
            claimed = Some(match claimed {
                Some((start, end)) => (start.min(lo), end.max(hi)),
                None => (lo, hi),
            });
        }
        if word_level {
            let mut prev_word = kept.as_ref().map(|span| span.start_frame - 1).unwrap_or(-1);
            for word in &sentence.words {
                raw.push(time_word(
                    word,
                    found,
                    kept.as_ref(),
                    &mut prev_word,
                    origin,
                    emissions,
                    start_bias,
                    end_bias,
                ));
            }
        } else if let Some(span) = &kept {
            let (start, end) = frame_span_to_time(emissions, origin, span.start_frame, span.end_frame);
            raw.push(RawSpan {
                id: sentence.id.clone(),
                start: Some(start),
                end: Some(end),
                dropped: span.dropped,
                interpolated: false,
            });
        } else {
            raw.push(RawSpan {
                id: sentence.id.clone(),
                start: None,
                end: None,
                dropped: false,
                interpolated: true,
            });
        }
    }
    (raw, claimed)
}

fn keep_sentence(
    range: &std::ops::Range<usize>,
    found: &Found,
    prev_end: &mut i64,
    start_bias: i64,
    end_bias: i64,
) -> Option<KeptFrames> {
    let ev1 = found.ev0 + found.tokens.len();
    if range.end <= found.ev0 || range.start >= ev1 {
        return None;
    }
    let local_start = range.start.max(found.ev0) - found.ev0;
    let local_end = range.end.min(ev1) - found.ev0 - 1;
    if local_end >= found.tokens.len() || local_start > local_end {
        return None;
    }
    let letters_n = local_end + 1 - local_start;
    let (run_start, run_end) = find_dominant_run(
        &found.tokens,
        local_start,
        local_end,
        MAX_INTRA_SENTENCE_GAP_FRAMES,
    );
    let run_len = run_end + 1 - run_start;
    if run_len * 2 < letters_n {
        return None;
    }
    let good = found.tokens[run_start..=run_end]
        .iter()
        .filter(|token| token.score > -4.0 && token.start_frame >= 0)
        .count();
    if good * 2 < run_len {
        return None;
    }
    let run_start_frame = found.tokens[run_start].start_frame;
    let run_end_frame = found.tokens[run_end].end_frame;
    if run_start_frame < 0 || run_end_frame < 0 {
        return None;
    }
    let run_frames_per = (run_end_frame - run_start_frame) as f64 / run_len as f64;
    // Eight letters is "thank you". A run that slow has left its own spike.
    if run_len > 4 && !(2.0..15.0).contains(&run_frames_per) {
        return None;
    }
    let (extended_start, extended_end) =
        extend_dominant_run(&found.tokens, local_start, local_end, run_start, run_end);
    let spiked = extended_start + start_bias;
    let start_frame = (*prev_end + 1).max(spiked - SENTENCE_PAD_FRAMES).max(0);
    let end_frame = (extended_end + end_bias).max(start_frame);
    *prev_end = end_frame;
    Some(KeptFrames {
        start_frame,
        end_frame,
        dropped: false,
    })
}

fn time_word(
    word: &epub::WordMark,
    found: &Found,
    sentence: Option<&KeptFrames>,
    prev_word: &mut i64,
    origin: i64,
    emissions: &Emissions,
    start_bias: i64,
    end_bias: i64,
) -> RawSpan {
    let missing = RawSpan {
        id: word.id.clone(),
        start: None,
        end: None,
        dropped: false,
        interpolated: true,
    };
    let Some(sentence) = sentence else {
        return missing;
    };
    let ev1 = found.ev0 + found.tokens.len();
    if word.letter_range.end <= found.ev0 || word.letter_range.start >= ev1 {
        return missing;
    }
    let local_start = word.letter_range.start.max(found.ev0) - found.ev0;
    let local_end = word.letter_range.end.min(ev1) - found.ev0 - 1;
    if local_start > local_end || local_end >= found.tokens.len() {
        return missing;
    }
    let letters_n = local_end + 1 - local_start;
    // One stolen letter (the "m" of "meeting" latching onto an earlier word)
    // must not drag the highlight back to the sentence pad. Time the dense run.
    let (run_start, run_end) = find_dominant_run(
        &found.tokens,
        local_start,
        local_end,
        MAX_INTRA_SENTENCE_GAP_FRAMES,
    );
    let run_len = run_end + 1 - run_start;
    if run_len * 2 < letters_n {
        return missing;
    }
    let good = found.tokens[run_start..=run_end]
        .iter()
        .filter(|token| token.score > -4.0 && token.start_frame >= 0)
        .count();
    if good * 2 < run_len {
        return missing;
    }
    let first = found.tokens[run_start].start_frame;
    let last = found.tokens[run_end].end_frame;
    if first < 0 || last < 0 {
        return missing;
    }
    let clamp = |frame: i64| frame.clamp(sentence.start_frame, sentence.end_frame);
    let start_frame = (*prev_word + 1).max(clamp(first + start_bias));
    let end_frame = clamp(last + end_bias).max(start_frame);
    *prev_word = end_frame;
    let (start, end) = frame_span_to_time(emissions, origin, start_frame, end_frame);
    RawSpan {
        id: word.id.clone(),
        start: Some(start),
        end: Some(end),
        dropped: false,
        interpolated: false,
    }
}

fn frame_span_to_time(emissions: &Emissions, origin: i64, start_frame: i64, end_frame: i64) -> (f64, f64) {
    let start = emissions.frame_start(origin + start_frame).max(0.0);
    let end = emissions.frame_end(origin + end_frame.max(start_frame));
    (start, end.max(start))
}

/// The run with the most characters. `end` is inclusive.
fn find_dominant_run(tokens: &[TokenAlignment], start: usize, end: usize, max_gap: i64) -> (usize, usize) {
    let mut best = (start, start);
    let mut run_start = start;
    let mut index = start + 1;
    while index <= end + 1 {
        let gap = if index <= end {
            let previous = tokens[index - 1].end_frame;
            let current = tokens[index].start_frame;
            if previous < 0 || current < 0 {
                i64::MAX
            } else {
                current - previous
            }
        } else {
            i64::MAX
        };
        if gap > max_gap {
            if index - 1 - run_start > best.1 - best.0 {
                best = (run_start, index - 1);
            }
            run_start = index;
        }
        index += 1;
    }
    best
}

fn extend_dominant_run(
    tokens: &[TokenAlignment],
    start: usize,
    end: usize,
    run_start: usize,
    run_end: usize,
) -> (i64, i64) {
    let run_start_frame = tokens[run_start].start_frame;
    let run_end_frame = tokens[run_end].end_frame;
    if run_start_frame < 0 || run_end_frame < 0 {
        return (0, 0);
    }
    let frames_per = (run_end_frame - run_start_frame) as f64 / (run_end + 1 - run_start) as f64;
    let mut start_frame = run_start_frame;
    if run_start > start {
        let previous = tokens[run_start - 1].end_frame;
        if previous >= 0 {
            let gap = run_start_frame - previous;
            let estimate = ((run_start - start) as f64 * frames_per).round() as i64;
            start_frame -= estimate.min((gap - 1).max(0));
        }
    }
    let mut end_frame = run_end_frame;
    if run_end < end {
        let next = tokens[run_end + 1].start_frame;
        if next >= 0 {
            let gap = next - run_end_frame;
            let estimate = ((end - run_end) as f64 * frames_per).round() as i64;
            end_frame += estimate.min((gap - 1).max(0));
        }
    }
    (start_frame, end_frame)
}

pub(crate) fn finalize_spans(raw: &[RawSpan], gap_seconds: f64, drop_unread: bool, origin: f64) -> Vec<OverlaySpan> {
    let mut placed: Vec<Option<(f64, f64)>> = raw
        .iter()
        .map(|span| match (span.start, span.end) {
            (Some(start), Some(end)) => Some((start, end.max(start))),
            _ => None,
        })
        .collect();
    let mut index = 0usize;
    while index < placed.len() {
        if placed[index].is_some() {
            index += 1;
            continue;
        }
        let mut end = index;
        while end < placed.len() && placed[end].is_none() {
            end += 1;
        }
        let left = if index > 0 { placed[index - 1].map(|(_, value)| value) } else { None };
        let right = if end < placed.len() { placed[end].map(|(value, _)| value) } else { None };
        let count = (end - index) as f64;
        match (left, right) {
            (Some(left), Some(right)) => {
                let step = (right - left).max(0.0) / count;
                for step_index in 0..(end - index) {
                    let start = left + step * step_index as f64;
                    placed[index + step_index] = Some((start, start + step));
                }
            }
            (Some(left), None) => {
                for slot in index..end {
                    placed[slot] = Some((left, left));
                }
            }
            (None, Some(right)) => {
                for slot in index..end {
                    placed[slot] = Some((right, right));
                }
            }
            (None, None) => {
                for slot in index..end {
                    placed[slot] = Some((origin, origin));
                }
            }
        }
        index = end;
    }

    let mut kept: Vec<OverlaySpan> = Vec::new();
    for (span, time) in raw.iter().zip(placed) {
        if drop_unread && span.dropped {
            continue;
        }
        let Some((start, end)) = time else { continue };
        let kind = if span.dropped {
            SpanKind::Unmatched
        } else if span.interpolated {
            SpanKind::Interpolated
        } else {
            SpanKind::Matched
        };
        kept.push(OverlaySpan {
            id: span.id.clone(),
            start,
            end,
            kind,
        });
    }
    if kept.is_empty() {
        return kept;
    }
    for pair in 0..kept.len() - 1 {
        if kept[pair].end > kept[pair + 1].start {
            if kept[pair + 1].start > kept[pair].start {
                kept[pair].end = kept[pair + 1].start;
            } else {
                kept[pair + 1].start = kept[pair].end;
                if kept[pair + 1].end < kept[pair + 1].start {
                    kept[pair + 1].end = kept[pair + 1].start;
                }
            }
        }
    }
    if kept[0].start - origin <= gap_seconds && kept[0].start > origin {
        kept[0].start = origin.max(0.0);
    }
    for pair in 0..kept.len() - 1 {
        let gap = kept[pair + 1].start - kept[pair].end;
        if gap > 0.0 && gap <= gap_seconds {
            kept[pair].end = kept[pair + 1].start;
        }
    }
    let mut with_silence = Vec::new();
    if kept[0].start - origin > gap_seconds {
        with_silence.push(OverlaySpan {
            id: kept[0].id.clone(),
            start: origin.max(0.0),
            end: kept[0].start,
            kind: SpanKind::AudioOnly,
        });
    }
    for (index, span) in kept.iter().enumerate() {
        with_silence.push(span.clone());
        if let Some(next) = kept.get(index + 1) {
            let gap = next.start - span.end;
            if gap > gap_seconds {
                with_silence.push(OverlaySpan {
                    id: next.id.clone(),
                    start: span.end,
                    end: next.start,
                    kind: SpanKind::AudioOnly,
                });
            }
        }
    }
    with_silence
}

fn id_prefix(href: &str) -> String {
    href.rsplit('/').next().unwrap_or(href).to_string()
}

pub fn default_model_path() -> PathBuf {
    let mut candidates = vec![
        PathBuf::from("models/mms-fp16.onnx"),
        PathBuf::from("../models/mms-fp16.onnx"),
    ];
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join("../../../models/mms-fp16.onnx"));
            candidates.push(dir.join("../../models/mms-fp16.onnx"));
        }
    }
    if let Ok(cache) = std::env::var("XDG_CACHE_HOME") {
        candidates.push(PathBuf::from(&cache).join("starlign/mms-fp16.onnx"));
        candidates.push(PathBuf::from(&cache).join("stalign/mms-fp16.onnx"));
    }
    if let Ok(home) = std::env::var("HOME") {
        candidates.push(PathBuf::from(&home).join(".cache/starlign/mms-fp16.onnx"));
        candidates.push(PathBuf::from(&home).join(".cache/stalign/mms-fp16.onnx"));
    }
    for candidate in &candidates {
        if candidate.is_file() {
            return candidate.clone();
        }
    }
    candidates[0].clone()
}

pub fn ensure_model(path: &Path, download: bool) -> Result<()> {
    if path.is_file() {
        let len = std::fs::metadata(path)?.len();
        if len > 1_000_000 {
            return Ok(());
        }
    }
    if !download {
        bail!(
            "MMS-300M model not found at {}.\nThe weights are CC-BY-NC 4.0. Pass --download, or save the fp16 ONNX at that path.\n{MODEL_URL}",
            path.display()
        );
    }
    crate::say!("MMS-300M forced aligner weights are CC-BY-NC 4.0.");
    crate::say!("downloading {MODEL_URL}");
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let status = std::process::Command::new("curl")
        .args(["-L", "--fail", "--retry", "3", "-o"])
        .arg(path)
        .arg(MODEL_URL)
        .status()
        .context("running curl")?;
    if !status.success() {
        bail!("model download failed");
    }
    Ok(())
}

fn temp_pcm() -> PathBuf {
    std::env::temp_dir().join(format!("stalign-{}-{}.f32", std::process::id(), std::process::id()))
}

struct PcmGuard(PathBuf);

impl Drop for PcmGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn tracks_sidecar(emissions: &Path) -> PathBuf {
    let name = emissions
        .file_name()
        .map(|name| format!("{}.tracks.json", name.to_string_lossy()))
        .unwrap_or_else(|| "tracks.json".into());
    emissions.with_file_name(name)
}

fn save_tracks(emissions: &Path, timeline: &Timeline) -> Result<()> {
    let path = tracks_sidecar(emissions);
    std::fs::write(&path, serde_json::to_string_pretty(&timeline.tracks)?)?;
    Ok(())
}

fn load_tracks(emissions: &Path) -> Result<Option<Timeline>> {
    let path = tracks_sidecar(emissions);
    if !path.is_file() {
        return Ok(None);
    }
    let tracks = serde_json::from_slice(&std::fs::read(&path)?)?;
    Ok(Some(Timeline { pcm_path: None, tracks }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaps_longer_than_the_threshold_stay_unhighlighted() {
        let raw = vec![
            RawSpan {
                id: "s0".into(),
                start: Some(1.0),
                end: Some(2.0),
                dropped: false,
                interpolated: false,
            },
            RawSpan {
                id: "s1".into(),
                start: Some(8.0),
                end: Some(9.0),
                dropped: false,
                interpolated: false,
            },
        ];
        let spans = finalize_spans(&raw, 3.0, false, 0.0);
        assert!(spans.iter().any(|span| span.kind == SpanKind::AudioOnly));
        // One second of leading silence is inside the 3s threshold, so it joins
        // the first sentence. The six-second hole between sentences stays audio-only.
        assert!((spans[0].start - 0.0).abs() < 1e-6, "{:?}", spans[0]);
        assert!((spans[0].end - 2.0).abs() < 1e-6, "{:?}", spans[0]);
        let hole = spans.iter().find(|span| span.kind == SpanKind::AudioOnly).unwrap();
        assert!((hole.start - 2.0).abs() < 1e-6 && (hole.end - 8.0).abs() < 1e-6, "{hole:?}");
        let short = finalize_spans(&raw, 10.0, false, 0.0);
        assert!(short.iter().all(|span| span.kind != SpanKind::AudioOnly));
        assert!((short[0].end - 8.0).abs() < 1e-6 || (short[0].start - 0.0).abs() < 1e-6);
    }

    #[test]
    fn romanized_letters_align_on_synthetic_spikes() {
        let letters = "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcd";
        let mut data = Vec::new();
        let mut times = Vec::new();
        for ch in letters.chars() {
            for _ in 0..4 {
                let mut row = vec![-8.0f32; ctc::CLASSES];
                row[0] = -0.2;
                data.extend_from_slice(&row);
                times.push(times.len() as f64 * 0.02);
            }
            let mut row = vec![-8.0f32; ctc::CLASSES];
            row[ctc::label_of(ch).unwrap() as usize] = 0.0;
            data.extend_from_slice(&row);
            times.push(times.len() as f64 * 0.02);
        }
        let emissions = Emissions {
            classes: ctc::CLASSES,
            data,
            times,
        };
        let found = align_query(letters, &emissions, &[], false).expect("boundary");
        let finite = found.tokens.iter().filter(|token| token.start_frame >= 0).count();
        assert!(finite > letters.len() / 2, "{finite}");
    }

    #[test]
    fn dominant_run_drops_a_tail_pinned_across_a_silence() {
        let mut tokens = Vec::new();
        for index in 0..10 {
            tokens.push(TokenAlignment {
                label: 1,
                start_frame: index * 4,
                end_frame: index * 4 + 3,
                score: -1.0,
            });
        }
        tokens.push(TokenAlignment {
            label: 1,
            start_frame: 500,
            end_frame: 510,
            score: -1.0,
        });
        let (start, end) = find_dominant_run(&tokens, 0, 10, MAX_INTRA_SENTENCE_GAP_FRAMES);
        assert_eq!((start, end), (0, 9));
        let (_start_frame, end_frame) = extend_dominant_run(&tokens, 0, 10, start, end);
        assert!(end_frame < 80, "{end_frame}");
    }

    fn found_with(frames: &[i64]) -> Found {
        Found {
            confidence: 1.0,
            frame_start: 0,
            ev0: 0,
            tokens: frames
                .iter()
                .map(|frame| TokenAlignment {
                    label: 1,
                    start_frame: *frame,
                    end_frame: *frame,
                    score: 0.0,
                })
                .collect(),
        }
    }

    #[test]
    fn an_eight_letter_thanks_stretched_over_later_speech_is_dropped() {
        let frames: Vec<i64> = (0..8).map(|index| 1000 + index * 50).collect();
        let found = found_with(&frames);
        let mut prev_end = 900i64;
        let kept = keep_sentence(&(0..8), &found, &mut prev_end, -2, 2);
        assert!(kept.is_none());
    }

    #[test]
    fn a_tight_eight_letter_thanks_is_kept() {
        let frames: Vec<i64> = (0..8).map(|index| 1000 + index * 3).collect();
        let found = found_with(&frames);
        let mut prev_end = 900i64;
        let kept = keep_sentence(&(0..8), &found, &mut prev_end, -2, 2).expect("kept");
        assert!(kept.end_frame < 1100, "{}", kept.end_frame);
        assert!(kept.start_frame < kept.end_frame);
    }
}
