//! CTC greedy decode, anchor fitting, and forced alignment.
//!
//! Anchor fitting follows the MIT-licensed Storyteller aligner
//! (libraries/align, Shane Friedman and contributors):
//! https://gitlab.com/storyteller-platform/storyteller
//!
//! Between anchors, each spelled letter is matched to the greedy transcript.
//! MMS emits pronunciation, not spelling, so a silent vowel must stay between
//! its neighbors instead of walking forward into later speech.

use anyhow::{bail, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

pub const BLANK_ID: i32 = 0;
pub const CLASSES: usize = 31;
#[cfg(test)]
pub const STAR_SCORE: f32 = -3.0;
/// Median spike-to-boundary offsets measured for MMS, in seconds.
pub const MMS_START_OFFSET: f64 = -0.04;
pub const MMS_END_OFFSET: f64 = 0.04;

const NGRAM: usize = 10;
const RANSAC_ITERS: usize = 500;
const MIN_INLIERS: usize = 10;
const MAX_PAIR_SEPARATION: usize = 200;
const MIN_CONFIDENCE: f64 = 0.5;
const MIN_COVERAGE: f64 = 0.3;
const REFINE_COUNT: usize = 25;
const ANCHOR_SPACING: usize = 2000;
const BOUNDARY_PAD_FRAMES: f64 = 500.0;
const EVIDENCE_GAP_MULTIPLIER: f64 = 3.0;
const EVIDENCE_GAP_FLOOR: f64 = 500.0;
const EVIDENCE_GAP_PERCENTILE: f64 = 0.99;
const MIN_EDGE_CLUSTER: usize = 10;
const EDGE_EXTENSION_MAX_GAP: usize = 1000;
const EDGE_EXTENSION_WINDOW: usize = 250;
const EDGE_EXTENSION_PAIR_SEPARATION: usize = 50;
const ANCHOR_CONTEXT: usize = 40;
const ANCHOR_AGREEMENT_FRACTION: f64 = 0.4;
const ANCHOR_AGREEMENT_FLOOR: f64 = 0.05;
const ANCHOR_CALIBRATION_MINIMUM: usize = 5;
/// On-disk header. The bytes stay STALIGN1 so emissions written under the old name still load.
const MAGIC: &[u8; 8] = b"STALIGN1";

#[derive(Clone, Copy, Debug)]
pub struct TokenAlignment {
    pub label: i32,
    pub start_frame: i64,
    pub end_frame: i64,
    pub score: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct Anchor {
    /// Offset into the letter query.
    pub offset: usize,
    /// Emission frame.
    pub position: i64,
}

#[derive(Clone, Debug)]
pub struct Boundary {
    pub start: i64,
    pub end: i64,
    pub confidence: f64,
    pub anchors: Vec<Anchor>,
    pub evidence: (usize, usize),
}

/// Frame-level log-probabilities, row-major `[frame, class]`.
#[derive(Clone, Debug)]
pub struct Emissions {
    pub classes: usize,
    pub data: Vec<f32>,
    /// Time, in seconds, of the center of each frame.
    pub times: Vec<f64>,
}

impl Emissions {
    pub fn frames(&self) -> usize {
        self.times.len()
    }

    pub fn row(&self, frame: usize) -> &[f32] {
        let c = self.classes;
        &self.data[frame * c..(frame + 1) * c]
    }

    pub fn hop(&self) -> f64 {
        if self.times.len() < 2 {
            0.02
        } else {
            (self.times[self.times.len() - 1] - self.times[0]) / (self.times.len() - 1) as f64
        }
    }

    pub fn frame_start(&self, frame: i64) -> f64 {
        if self.times.is_empty() {
            return 0.0;
        }
        let hop = self.hop();
        let idx = frame.clamp(0, self.times.len() as i64 - 1) as usize;
        self.times[idx] - hop * 0.5
    }

    pub fn frame_end(&self, frame: i64) -> f64 {
        if self.times.is_empty() {
            return 0.0;
        }
        let hop = self.hop();
        let idx = frame.clamp(0, self.times.len() as i64 - 1) as usize;
        self.times[idx] + hop * 0.5
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if self.data.len() != self.frames() * self.classes {
            bail!("emissions shape does not match frame count");
        }
        let mut file = File::create(path)?;
        file.write_all(MAGIC)?;
        file.write_all(&(self.classes as u32).to_le_bytes())?;
        file.write_all(&(self.frames() as u64).to_le_bytes())?;
        write_f32s(&mut file, &self.data)?;
        write_f64s(&mut file, &self.times)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let mut file = File::open(path)?;
        let mut magic = [0u8; 8];
        file.read_exact(&mut magic)?;
        if &magic != MAGIC {
            bail!("not an emissions file: {}", path.display());
        }
        let classes = read_u32(&mut file)? as usize;
        let frames = read_u64(&mut file)? as usize;
        let mut data = vec![0f32; frames * classes];
        read_f32s(&mut file, &mut data)?;
        let mut times = vec![0f64; frames];
        read_f64s(&mut file, &mut times)?;
        Ok(Self { classes, data, times })
    }

    pub fn slice(&self, start: usize, end: usize) -> Self {
        let start = start.min(self.frames());
        let end = end.clamp(start, self.frames());
        let c = self.classes;
        Self {
            classes: c,
            data: self.data[start * c..end * c].to_vec(),
            times: self.times[start..end].to_vec(),
        }
    }
}

fn write_f32s(file: &mut File, values: &[f32]) -> Result<()> {
    let mut buf = Vec::with_capacity(values.len() * 4);
    for value in values {
        buf.extend_from_slice(&value.to_le_bytes());
    }
    file.write_all(&buf)?;
    Ok(())
}

fn write_f64s(file: &mut File, values: &[f64]) -> Result<()> {
    let mut buf = Vec::with_capacity(values.len() * 8);
    for value in values {
        buf.extend_from_slice(&value.to_le_bytes());
    }
    file.write_all(&buf)?;
    Ok(())
}

fn read_u32(file: &mut File) -> Result<u32> {
    let mut buf = [0u8; 4];
    file.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(file: &mut File) -> Result<u64> {
    let mut buf = [0u8; 8];
    file.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_f32s(file: &mut File, out: &mut [f32]) -> Result<()> {
    let mut buf = vec![0u8; out.len() * 4];
    file.read_exact(&mut buf)?;
    for (i, slot) in out.iter_mut().enumerate() {
        let bytes = buf[i * 4..i * 4 + 4].try_into().unwrap();
        *slot = f32::from_le_bytes(bytes);
    }
    Ok(())
}

fn read_f64s(file: &mut File, out: &mut [f64]) -> Result<()> {
    let mut buf = vec![0u8; out.len() * 8];
    file.read_exact(&mut buf)?;
    for (i, slot) in out.iter_mut().enumerate() {
        let bytes = buf[i * 8..i * 8 + 8].try_into().unwrap();
        *slot = f64::from_le_bytes(bytes);
    }
    Ok(())
}

pub fn label_of(ch: char) -> Option<i32> {
    Some(match ch {
        'a' => 4,
        'i' => 5,
        'e' => 6,
        'n' => 7,
        'o' => 8,
        'u' => 9,
        't' => 10,
        's' => 11,
        'r' => 12,
        'm' => 13,
        'k' => 14,
        'l' => 15,
        'd' => 16,
        'g' => 17,
        'h' => 18,
        'y' => 19,
        'b' => 20,
        'p' => 21,
        'w' => 22,
        'c' => 23,
        'v' => 24,
        'j' => 25,
        'z' => 26,
        'f' => 27,
        '\'' => 28,
        'q' => 29,
        'x' => 30,
        _ => return None,
    })
}

pub fn char_of(id: i32) -> Option<char> {
    Some(match id {
        4 => 'a',
        5 => 'i',
        6 => 'e',
        7 => 'n',
        8 => 'o',
        9 => 'u',
        10 => 't',
        11 => 's',
        12 => 'r',
        13 => 'm',
        14 => 'k',
        15 => 'l',
        16 => 'd',
        17 => 'g',
        18 => 'h',
        19 => 'y',
        20 => 'b',
        21 => 'p',
        22 => 'w',
        23 => 'c',
        24 => 'v',
        25 => 'j',
        26 => 'z',
        27 => 'f',
        28 => '\'',
        29 => 'q',
        30 => 'x',
        _ => return None,
    })
}

pub fn log_softmax_rows(data: &mut [f32], classes: usize) {
    if classes == 0 {
        return;
    }
    for row in data.chunks_mut(classes) {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0f32;
        for value in row.iter() {
            sum += (value - max).exp();
        }
        let log_z = max + sum.ln();
        for value in row.iter_mut() {
            *value -= log_z;
        }
    }
}

struct Positioned {
    text: String,
    positions: Vec<i64>,
}

pub fn greedy_decode(emissions: &Emissions) -> (String, Vec<i64>) {
    let mut chars = String::new();
    let mut frames = Vec::new();
    let mut previous = -1i32;
    for (t, row) in emissions.data.chunks(emissions.classes).enumerate() {
        let mut best = 0usize;
        for (i, value) in row.iter().enumerate().skip(1) {
            if *value > row[best] {
                best = i;
            }
        }
        let best = best as i32;
        if best != previous {
            if let Some(ch) = char_of(best) {
                chars.push(ch);
                frames.push(t as i64);
            }
        }
        previous = best;
    }
    (chars, frames)
}

pub fn find_boundaries(
    query: &str,
    emissions: &Emissions,
    excluded: &[(i64, i64)],
) -> Option<Boundary> {
    let (text, positions) = greedy_decode(emissions);
    let document = Positioned { text, positions };
    let tolerance = if query.len() > 15_000 { 5000.0 } else { 2500.0 };
    find_in_document(
        query,
        &document,
        excluded,
        2.0,
        15.0,
        tolerance,
        emissions.frames() as i64,
    )
}

fn find_in_document(
    query: &str,
    document: &Positioned,
    excluded: &[(i64, i64)],
    min_slope: f64,
    max_slope: f64,
    inlier_tolerance: f64,
    frame_limit: i64,
) -> Option<Boundary> {
    if query.len() < NGRAM || document.text.len() < NGRAM {
        return None;
    }
    let gram_count = query.len() - NGRAM + 1;
    let pair_separation = ((gram_count / 4).max(20)).min(MAX_PAIR_SEPARATION);
    let (matches, query_ngrams) = collect_matches(query, document, excluded);
    if matches.len() < MIN_INLIERS {
        return None;
    }
    let line = fit_line(&matches, pair_separation, min_slope, max_slope, inlier_tolerance)?;
    // The first fit can keep a second copy of an n-gram that sits on a later
    // chapter. The refined line is the one the chapter actually follows; a hit
    // more than `inlier_tolerance` off it, or a worse duplicate of the same
    // offset, must not stretch the window into the next chapter.
    let mut sorted_inliers = line.inliers;
    sorted_inliers.retain(|m| {
        let predicted = line.intercept + line.slope * m.offset as f64;
        (m.position as f64 - predicted).abs() <= inlier_tolerance
    });
    closest_per_offset(&mut sorted_inliers, line.slope, line.intercept);
    if sorted_inliers.len() < MIN_INLIERS {
        return None;
    }
    let extended = extend_inlier_edges(&matches, &sorted_inliers, line.slope, inlier_tolerance);
    let mut inlier_offsets: Vec<usize> = extended.iter().map(|m| m.offset).collect();
    inlier_offsets.sort_unstable();
    inlier_offsets.dedup();
    let (evidence, _matched) = evidence_range(&inlier_offsets, query.len());

    let head = &extended[..extended.len().min(REFINE_COUNT)];
    let tail = &extended[extended.len().saturating_sub(REFINE_COUNT)..];
    if head.is_empty() || tail.is_empty() {
        return None;
    }
    let extrapolated_start = head
        .iter()
        .map(|m| m.position as f64 - line.slope * m.offset as f64)
        .fold(f64::INFINITY, f64::min);
    let extrapolated_end = tail
        .iter()
        .map(|m| m.position as f64 + line.slope * (query.len() - m.offset) as f64)
        .fold(f64::NEG_INFINITY, f64::max);
    let start_pad = BOUNDARY_PAD_FRAMES.max(0.25 * (head[0].position as f64 - extrapolated_start));
    let end_pad = BOUNDARY_PAD_FRAMES
        .max(0.25 * (extrapolated_end - tail[tail.len() - 1].position as f64));
    let start = extrapolated_start - start_pad;
    let end = extrapolated_end + end_pad;
    if end <= start {
        return None;
    }

    let matched_grams = matches.iter().map(|m| m.offset).collect::<std::collections::HashSet<_>>().len();
    if matched_grams == 0 {
        return None;
    }
    let confidence = inlier_offsets.len() as f64 / matched_grams as f64;
    if confidence < MIN_CONFIDENCE {
        return None;
    }
    let first = extended[0];
    let last = extended[extended.len() - 1];
    let coverage = (last.offset - first.offset + NGRAM) as f64 / query.len() as f64;
    if coverage < MIN_COVERAGE {
        return None;
    }
    let anchors = select_anchors(
        query,
        &sorted_inliers,
        &query_ngrams,
        document,
        line.slope,
        line.intercept,
    );
    let start = start.round().clamp(0.0, (frame_limit - 1).max(0) as f64) as i64;
    let end = end.round().clamp(0.0, (frame_limit - 1).max(0) as f64) as i64;
    if end <= start {
        return None;
    }
    Some(Boundary {
        start,
        end,
        confidence,
        anchors,
        evidence,
    })
}

fn collect_matches(
    query: &str,
    document: &Positioned,
    excluded: &[(i64, i64)],
) -> (Vec<Anchor>, HashMap<String, Vec<usize>>) {
    let index = ngram_index(document);
    let mut query_ngrams: HashMap<String, Vec<usize>> = HashMap::new();
    let mut matches = Vec::new();
    let q = query.as_bytes();
    for i in 0..=query.len().saturating_sub(NGRAM) {
        let gram = &query[i..i + NGRAM];
        query_ngrams.entry(gram.to_string()).or_default().push(i);
        let Some(positions) = index.get(gram) else {
            continue;
        };
        for &position in positions {
            if excluded.iter().any(|&(start, end)| start <= position && position < end) {
                continue;
            }
            let _ = q;
            matches.push(Anchor { offset: i, position });
        }
    }
    (matches, query_ngrams)
}

fn ngram_index(document: &Positioned) -> HashMap<String, Vec<i64>> {
    let mut index: HashMap<String, Vec<i64>> = HashMap::new();
    if document.text.len() < NGRAM {
        return index;
    }
    for i in 0..=document.text.len() - NGRAM {
        let gram = document.text[i..i + NGRAM].to_string();
        index.entry(gram).or_default().push(document.positions[i]);
    }
    index
}

struct LineFit {
    slope: f64,
    intercept: f64,
    inliers: Vec<Anchor>,
}

fn fit_line(
    matches: &[Anchor],
    pair_separation: usize,
    min_slope: f64,
    max_slope: f64,
    tolerance: f64,
) -> Option<LineFit> {
    let mut seed: u32 = 0x2545_f491;
    let mut random = || {
        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345) & 0x7fff_ffff;
        seed as f64 / (0x8000_0000u32 as f64)
    };
    let mut best: Option<(f64, f64, usize)> = None;
    for _ in 0..RANSAC_ITERS {
        let a = matches[(random() * matches.len() as f64) as usize];
        let b = matches[(random() * matches.len() as f64) as usize];
        if a.offset.abs_diff(b.offset) < pair_separation {
            continue;
        }
        let slope = (b.position - a.position) as f64 / (b.offset as f64 - a.offset as f64);
        if slope < min_slope || slope > max_slope {
            continue;
        }
        let intercept = a.position as f64 - slope * a.offset as f64;
        let mut count = 0usize;
        for m in matches {
            let predicted = intercept + slope * m.offset as f64;
            if (m.position as f64 - predicted).abs() <= tolerance {
                count += 1;
            }
        }
        if best.map(|(_, _, c)| count > c).unwrap_or(true) {
            best = Some((slope, intercept, count));
        }
    }
    let (slope0, intercept0, count) = best?;
    if count < MIN_INLIERS {
        return None;
    }
    let mut inliers: Vec<Anchor> = matches
        .iter()
        .copied()
        .filter(|m| {
            let predicted = intercept0 + slope0 * m.offset as f64;
            (m.position as f64 - predicted).abs() <= tolerance
        })
        .collect();
    inliers.sort_by_key(|m| m.offset);
    let stride = (inliers.len() / 2).max(1);
    let mut slopes = Vec::new();
    for i in 0..inliers.len().saturating_sub(stride) {
        let lo = inliers[i];
        let hi = inliers[i + stride];
        if hi.offset - lo.offset < pair_separation {
            continue;
        }
        slopes.push((hi.position - lo.position) as f64 / (hi.offset - lo.offset) as f64);
    }
    let slope = if slopes.is_empty() { slope0 } else { median(&mut slopes) };
    let intercept = if inliers.is_empty() {
        intercept0
    } else {
        let mut values: Vec<f64> = inliers
            .iter()
            .map(|m| m.position as f64 - slope * m.offset as f64)
            .collect();
        median(&mut values)
    };
    Some(LineFit { slope, intercept, inliers })
}

fn closest_per_offset(inliers: &mut Vec<Anchor>, slope: f64, intercept: f64) {
    if inliers.is_empty() {
        return;
    }
    inliers.sort_by_key(|m| m.offset);
    let mut kept: Vec<Anchor> = Vec::with_capacity(inliers.len());
    for anchor in inliers.drain(..) {
        if let Some(previous) = kept.last_mut() {
            if previous.offset == anchor.offset {
                let predicted = intercept + slope * anchor.offset as f64;
                let previous_error = (previous.position as f64 - predicted).abs();
                let error = (anchor.position as f64 - predicted).abs();
                if error < previous_error {
                    *previous = anchor;
                }
                continue;
            }
        }
        kept.push(anchor);
    }
    *inliers = kept;
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    values[values.len() / 2]
}

/// Signed so the head-extension mirror (`offset: -offset`) cannot underflow.
#[derive(Clone, Copy)]
struct Point {
    offset: i64,
    position: i64,
}

fn point_of(anchor: Anchor) -> Point {
    Point {
        offset: anchor.offset as i64,
        position: anchor.position,
    }
}

fn anchor_of(point: Point) -> Option<Anchor> {
    if point.offset < 0 {
        return None;
    }
    Some(Anchor {
        offset: point.offset as usize,
        position: point.position,
    })
}

fn extend_edge(sorted_matches: &[Point], sorted_inliers: &[Point], global_slope: f64, tolerance: f64) -> Vec<Point> {
    let Some(last_inlier) = sorted_inliers.last().copied() else {
        return Vec::new();
    };
    let mut window: Vec<Point> = sorted_inliers
        .iter()
        .copied()
        .rev()
        .take(EDGE_EXTENSION_WINDOW)
        .collect();
    window.reverse();
    let mut accepted = Vec::new();
    let mut last_offset = last_inlier.offset;
    let mut local = fit_local_line(&window, global_slope);
    for m in sorted_matches {
        if m.offset <= last_inlier.offset {
            continue;
        }
        if m.offset - last_offset > EDGE_EXTENSION_MAX_GAP as i64 {
            break;
        }
        let predicted = local.0 + local.1 * m.offset as f64;
        if (m.position as f64 - predicted).abs() > tolerance {
            continue;
        }
        accepted.push(*m);
        window.push(*m);
        if window.len() > EDGE_EXTENSION_WINDOW {
            window.remove(0);
        }
        last_offset = m.offset;
        local = fit_local_line(&window, global_slope);
    }
    accepted
}

fn fit_local_line(points: &[Point], fallback: f64) -> (f64, f64) {
    let stride = (points.len() / 2).max(1);
    let mut slopes = Vec::new();
    let mut i = 0;
    while i + stride < points.len() {
        let lo = points[i];
        let hi = points[i + stride];
        let delta = hi.offset - lo.offset;
        if delta >= EDGE_EXTENSION_PAIR_SEPARATION as i64 {
            slopes.push((hi.position - lo.position) as f64 / delta as f64);
        }
        i += stride;
    }
    let slope = if slopes.is_empty() { fallback } else { median(&mut slopes) };
    let mut intercepts: Vec<f64> = points
        .iter()
        .map(|m| m.position as f64 - slope * m.offset as f64)
        .collect();
    (median(&mut intercepts), slope)
}

fn extend_inlier_edges(
    matches: &[Anchor],
    sorted_inliers: &[Anchor],
    slope: f64,
    tolerance: f64,
) -> Vec<Anchor> {
    let mut by_offset: Vec<Point> = matches.iter().copied().map(point_of).collect();
    by_offset.sort_by_key(|m| m.offset);
    let inliers: Vec<Point> = sorted_inliers.iter().copied().map(point_of).collect();
    let tail = extend_edge(&by_offset, &inliers, slope, tolerance);
    let mirror = |m: Point| Point {
        offset: -m.offset,
        position: -m.position,
    };
    let mut mirrored_matches: Vec<Point> = by_offset.iter().copied().map(mirror).collect();
    mirrored_matches.reverse();
    let mut mirrored_inliers: Vec<Point> = inliers.iter().copied().map(mirror).collect();
    mirrored_inliers.reverse();
    let mut head: Vec<Point> = extend_edge(&mirrored_matches, &mirrored_inliers, slope, tolerance)
        .into_iter()
        .map(mirror)
        .collect();
    head.reverse();
    head.into_iter()
        .chain(inliers)
        .chain(tail)
        .filter_map(anchor_of)
        .collect()
}

fn evidence_range(offsets: &[usize], query_len: usize) -> ((usize, usize), (usize, usize)) {
    if offsets.len() < 2 {
        return ((0, query_len), (0, query_len));
    }
    let mut gaps = Vec::new();
    for i in 1..offsets.len() {
        gaps.push((offsets[i] - offsets[i - 1]) as f64);
    }
    let mut sorted = gaps.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p99_index = ((sorted.len() as f64) * EVIDENCE_GAP_PERCENTILE).floor() as usize;
    let p99 = sorted[p99_index.min(sorted.len() - 1)];
    let threshold = (EVIDENCE_GAP_MULTIPLIER * p99).max(EVIDENCE_GAP_FLOOR);
    let mut clusters: Vec<(usize, usize)> = Vec::new();
    let mut cluster_start = 0usize;
    for (i, gap) in gaps.iter().enumerate() {
        if *gap <= threshold {
            continue;
        }
        clusters.push((cluster_start, i));
        cluster_start = i + 1;
    }
    clusters.push((cluster_start, offsets.len() - 1));
    let mut first_cluster = 0usize;
    let mut last_cluster = clusters.len() - 1;
    while first_cluster < last_cluster
        && clusters[first_cluster].1 - clusters[first_cluster].0 + 1 < MIN_EDGE_CLUSTER
    {
        first_cluster += 1;
    }
    while last_cluster > first_cluster
        && clusters[last_cluster].1 - clusters[last_cluster].0 + 1 < MIN_EDGE_CLUSTER
    {
        last_cluster -= 1;
    }
    let first = offsets[clusters[first_cluster].0];
    let last = offsets[clusters[last_cluster].1] + NGRAM;
    let threshold_us = threshold as usize;
    let evidence = (
        if first > threshold_us { first - threshold_us } else { 0 },
        if query_len - last > threshold_us { last + threshold_us } else { query_len },
    );
    ((evidence.0, evidence.1.min(query_len)), (first, last.min(query_len)))
}

fn select_anchors(
    query: &str,
    inliers: &[Anchor],
    query_ngrams: &HashMap<String, Vec<usize>>,
    document: &Positioned,
    slope: f64,
    intercept: f64,
) -> Vec<Anchor> {
    let document_index = ngram_index(document);
    let mut agreements: HashMap<(usize, i64), f64> = HashMap::new();
    let agreement = |m: Anchor, cache: &mut HashMap<(usize, i64), f64>| -> f64 {
        if let Some(score) = cache.get(&(m.offset, m.position)) {
            return *score;
        }
        let score = context_agreement(m, query, document);
        cache.insert((m.offset, m.position), score);
        score
    };
    let select = |min_agreement: Option<f64>, cache: &mut HashMap<(usize, i64), f64>| -> Vec<Anchor> {
        let mut anchors = Vec::new();
        let mut i = 0usize;
        while i < query.len() {
            let window_start = i.saturating_sub(ANCHOR_SPACING / 2);
            let window_end = (i + ANCHOR_SPACING / 2).min(query.len());
            let mut candidates: Vec<Anchor> = inliers
                .iter()
                .copied()
                .filter(|m| window_start <= m.offset && m.offset < window_end)
                .collect();
            candidates.sort_by_key(|m| m.offset.abs_diff(i));
            for c in candidates {
                let gram = &query[c.offset..(c.offset + NGRAM).min(query.len())];
                if gram.len() < NGRAM {
                    continue;
                }
                if document_index.get(gram).map(|v| v.len()) != Some(1) {
                    continue;
                }
                if query_ngrams.get(gram).map(|v| v.len()) != Some(1) {
                    continue;
                }
                if let Some(min) = min_agreement {
                    if agreement(c, cache) < min {
                        continue;
                    }
                }
                anchors.push(c);
                break;
            }
            i += ANCHOR_SPACING;
        }
        anchors
    };
    let mut anchors = select(None, &mut agreements);
    if anchors.len() >= ANCHOR_CALIBRATION_MINIMUM {
        let mut scores: Vec<f64> = anchors.iter().copied().map(|a| agreement(a, &mut agreements)).collect();
        let threshold = (ANCHOR_AGREEMENT_FRACTION * median(&mut scores)).max(ANCHOR_AGREEMENT_FLOOR);
        anchors = select(Some(threshold), &mut agreements);
    }
    let mut monotonic = Vec::new();
    for a in anchors {
        loop {
            let Some(b) = monotonic.last().copied() else {
                monotonic.push(a);
                break;
            };
            if a.position - b.position >= (a.offset as i64) - (b.offset as i64) {
                monotonic.push(a);
                break;
            }
            let a_dist = (a.position as f64 - (intercept + slope * a.offset as f64)).abs();
            let b_dist = (b.position as f64 - (intercept + slope * b.offset as f64)).abs();
            if a_dist > b_dist {
                break;
            }
            monotonic.pop();
        }
    }
    monotonic
}

fn context_agreement(m: Anchor, query: &str, document: &Positioned) -> f64 {
    let index = position_to_index(document, m.position);
    let q_left = &query[m.offset.saturating_sub(ANCHOR_CONTEXT)..m.offset];
    let d_left = &document.text[index.saturating_sub(ANCHOR_CONTEXT)..index];
    let q_right_end = (m.offset + NGRAM + ANCHOR_CONTEXT).min(query.len());
    let q_right_start = (m.offset + NGRAM).min(query.len());
    let q_right = &query[q_right_start..q_right_end];
    let d_right_end = (index + NGRAM + ANCHOR_CONTEXT).min(document.text.len());
    let d_right_start = (index + NGRAM).min(document.text.len());
    let d_right = &document.text[d_right_start..d_right_end];
    bigram_similarity(q_left, d_left).max(bigram_similarity(q_right, d_right))
}

fn position_to_index(document: &Positioned, position: i64) -> usize {
    let mut low = 0usize;
    let mut high = document.positions.len().saturating_sub(1);
    while low < high {
        let mid = (low + high) / 2;
        if document.positions[mid] < position {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    low
}

fn bigram_similarity(a: &str, b: &str) -> f64 {
    if a.len() < 2 || b.len() < 2 {
        return 0.0;
    }
    let mut counts: HashMap<&str, i32> = HashMap::new();
    for i in 0..a.len() - 1 {
        *counts.entry(&a[i..i + 2]).or_default() += 1;
    }
    let mut shared = 0usize;
    for i in 0..b.len() - 1 {
        let gram = &b[i..i + 2];
        if let Some(left) = counts.get_mut(gram) {
            if *left > 0 {
                shared += 1;
                *left -= 1;
            }
        }
    }
    shared as f64 / (a.len() - 1).max(b.len() - 1) as f64
}

#[cfg(test)]
pub fn ctc_viterbi(
    emissions: &Emissions,
    labels: &[i32],
    blank_id: i32,
    star_id: i32,
    star_score: f32,
) -> Vec<TokenAlignment> {
    if labels.is_empty() || emissions.frames() == 0 {
        return Vec::new();
    }
    let classes = emissions.classes;
    let frames = emissions.frames();
    let mut tokens = vec![blank_id; labels.len() * 2 + 1];
    for (i, label) in labels.iter().enumerate() {
        tokens[2 * i + 1] = *label;
    }
    let mut skip_legal = vec![0u8; tokens.len()];
    for s in 2..tokens.len() {
        skip_legal[s] = if tokens[s] != tokens[s - 2] || tokens[s] == star_id {
            1
        } else {
            0
        };
    }
    let columns = classes + 1;
    let star_column = classes;
    let mut scores = vec![0f32; frames * columns];
    for t in 0..frames {
        scores[t * columns..t * columns + classes].copy_from_slice(emissions.row(t));
        scores[t * columns + star_column] = star_score;
    }
    let state_column: Vec<usize> = tokens
        .iter()
        .map(|token| if *token == star_id { star_column } else { *token as usize })
        .collect();
    let num_states = tokens.len();
    let mut prev = vec![f32::NEG_INFINITY; num_states];
    let mut curr = vec![f32::NEG_INFINITY; num_states];
    let mut back = vec![0u8; frames * num_states];
    prev[0] = scores[state_column[0]];
    if num_states > 1 {
        prev[1] = scores[state_column[1]];
    }
    if labels[0] == star_id && num_states > 3 {
        prev[2] = scores[state_column[2]];
        prev[3] = scores[state_column[3]];
    }
    for t in 1..frames {
        let row = t * columns;
        for s in 0..num_states {
            let mut best = prev[s];
            let mut op = 0u8;
            if s > 0 && prev[s - 1] > best {
                best = prev[s - 1];
                op = 1;
            }
            if skip_legal[s] == 1 && prev[s - 2] > best {
                best = prev[s - 2];
                op = 2;
            }
            curr[s] = scores[row + state_column[s]] + best;
            back[t * num_states + s] = op;
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    let mut terminals = vec![num_states - 1];
    if num_states >= 2 {
        terminals.push(num_states - 2);
    }
    if labels[labels.len() - 1] == star_id {
        if num_states >= 3 {
            terminals.push(num_states - 3);
        }
        if num_states >= 4 {
            terminals.push(num_states - 4);
        }
    }
    let mut terminal = terminals[0];
    for s in terminals {
        if prev[s] > prev[terminal] {
            terminal = s;
        }
    }
    if prev[terminal] == f32::NEG_INFINITY {
        return labels
            .iter()
            .map(|label| TokenAlignment {
                label: *label,
                start_frame: -1,
                end_frame: -1,
                score: f64::NEG_INFINITY,
            })
            .collect();
    }
    let mut path = vec![0i32; frames];
    path[frames - 1] = terminal as i32;
    for t in (1..frames).rev() {
        let s = path[t] as usize;
        let op = back[t * num_states + s] as i32;
        path[t - 1] = s as i32 - op;
    }
    let mut first = vec![-1i64; num_states];
    let mut last = vec![-1i64; num_states];
    let mut score_sums = vec![0f64; num_states];
    let mut counts = vec![0i32; num_states];
    for t in 0..frames {
        let s = path[t] as usize;
        if first[s] < 0 {
            first[s] = t as i64;
        }
        last[s] = t as i64;
        score_sums[s] += scores[t * columns + state_column[s]] as f64;
        counts[s] += 1;
    }
    let mut alignments = Vec::with_capacity(labels.len());
    for (l, label) in labels.iter().enumerate() {
        let state = 2 * l + 1;
        let count = counts[state];
        let mean = if count > 0 { score_sums[state] / count as f64 } else { f64::NEG_INFINITY };
        alignments.push(TokenAlignment {
            label: *label,
            start_frame: first[state],
            end_frame: last[state],
            score: mean,
        });
    }
    alignments
}

/// Align `labels` to emissions.
///
/// Anchors are increasing `(label offset, frame)` pairs inside this slice.
/// `star_id` marks a character the model has no class for; those letters are
/// interpolated between their neighbors. `drop_skipped` is retained so callers
/// keep a stable signature; unspoken stretches are scored as insertions and
/// the sentence timer drops them.
pub fn forced_align(
    emissions: &Emissions,
    labels: &[i32],
    anchors: &[Anchor],
    _star_id: i32,
    _drop_skipped: bool,
) -> Vec<TokenAlignment> {
    if labels.is_empty() {
        return Vec::new();
    }
    let frames = emissions.frames();
    let mut spans = Vec::new();
    let mut label_at = 0usize;
    let mut frame_at = 0usize;
    for anchor in anchors {
        let next_label = anchor.offset.min(labels.len());
        let next_frame = (anchor.position.max(0) as usize).min(frames);
        if next_label < label_at || next_frame < frame_at {
            continue;
        }
        spans.push((label_at, next_label, frame_at, next_frame));
        label_at = next_label;
        frame_at = next_frame;
    }
    spans.push((label_at, labels.len(), frame_at, frames));

    let mut out = vec![
        TokenAlignment {
            label: BLANK_ID,
            start_frame: -1,
            end_frame: -1,
            score: f64::NEG_INFINITY,
        };
        labels.len()
    ];
    for &(ls, le, fs, fe) in &spans {
        if le <= ls || fe <= fs {
            continue;
        }
        let query: Vec<char> = labels[ls..le].iter().map(|id| char_of(*id).unwrap_or('?')).collect();
        let segment = emissions.slice(fs, fe);
        let (greedy, gframes) = greedy_decode(&segment);
        let heard: Vec<(char, i64)> = greedy.chars().zip(gframes).collect();
        let local = monotonic_align(&query, &heard);
        for (k, (frame, matched)) in local.into_iter().enumerate() {
            if ls + k >= out.len() {
                break;
            }
            out[ls + k] = TokenAlignment {
                label: labels[ls + k],
                start_frame: frame + fs as i64,
                end_frame: frame + fs as i64,
                // A letter the audio never said sits between its neighbors.
                // A negative score keeps a wholly unspoken sentence from
                // counting as a match.
                score: if matched { 0.0 } else { -10.0 },
            };
        }
    }
    // A silent letter can earn an extra match by jumping into the next
    // phrase ("foreword" → the "o" in "Joan"). That stretches a short heading
    // across the credits. Snap only those stretched phrases back onto the
    // burst that holds their first spikes; ordinary sentences stay put.
    tighten_stretched(&mut out, labels, emissions);
    fill_unread_gaps(&mut out);
    out
}

/// Gap, in frames, that still belongs to the same spoken burst (~0.36s).
const TIGHT_BURST_FRAMES: i64 = 18;
/// A phrase slower than this many frames per letter has latched onto later speech.
const STRETCHED_FRAMES_PER_LETTER: f64 = 8.0;
const MAX_TIGHTEN_LETTERS: usize = 40;

fn tighten_stretched(tokens: &mut [TokenAlignment], labels: &[i32], emissions: &Emissions) {
    let matched: Vec<usize> = tokens
        .iter()
        .enumerate()
        .filter(|(_, token)| token.score > -4.0 && token.start_frame >= 0)
        .map(|(index, _)| index)
        .collect();
    let mut cursor = 0usize;
    while cursor < matched.len() {
        let mut end = cursor;
        while end + 1 < matched.len() {
            let gap = tokens[matched[end + 1]].start_frame - tokens[matched[end]].start_frame;
            if gap > MAX_INTRA_PHRASE_FRAMES {
                break;
            }
            end += 1;
        }
        tighten_phrase(tokens, labels, emissions, matched[cursor], matched[end]);
        cursor = end + 1;
    }
}

/// Same silence that splits a sentence. A wider hole is a new phrase.
const MAX_INTRA_PHRASE_FRAMES: i64 = 150;

fn tighten_phrase(
    tokens: &mut [TokenAlignment],
    labels: &[i32],
    emissions: &Emissions,
    first: usize,
    last: usize,
) {
    if last < first {
        return;
    }
    let letters = last - first + 1;
    if letters < 4 || letters > MAX_TIGHTEN_LETTERS {
        return;
    }
    let start_frame = tokens[first].start_frame;
    let end_frame = tokens[last].start_frame;
    if start_frame < 0 || end_frame <= start_frame {
        return;
    }
    let frames_per = (end_frame - start_frame) as f64 / letters as f64;
    if frames_per <= STRETCHED_FRAMES_PER_LETTER {
        return;
    }
    let hits: Vec<usize> = (first..=last).filter(|&index| tokens[index].score > -4.0).collect();
    let mut last_tight = None;
    for pair in hits.windows(2) {
        let gap = tokens[pair[1]].start_frame - tokens[pair[0]].start_frame;
        if gap > TIGHT_BURST_FRAMES {
            last_tight = Some(tokens[pair[0]].start_frame);
            break;
        }
    }
    let Some(burst_end) = last_tight else {
        return;
    };
    let window_start = start_frame as usize;
    let window_end = (burst_end + TIGHT_BURST_FRAMES) as usize;
    if window_end <= window_start {
        return;
    }
    let segment = emissions.slice(window_start, window_end.min(emissions.frames()));
    if segment.frames() == 0 {
        return;
    }
    let query: Vec<char> = labels[first..=last]
        .iter()
        .map(|id| char_of(*id).unwrap_or('?'))
        .collect();
    let (greedy, frames) = greedy_decode(&segment);
    let heard: Vec<(char, i64)> = greedy.chars().zip(frames).collect();
    let local = monotonic_align(&query, &heard);
    let tight_hits = local.iter().filter(|(_, hit)| *hit).count();
    // Keep the original when the burst cannot explain the phrase. A heading
    // loses one stolen vowel; a real second clause loses many letters.
    if tight_hits + 2 < hits.len() || tight_hits * 2 < query.len() {
        return;
    }
    for (offset, (frame, hit)) in local.into_iter().enumerate() {
        let index = first + offset;
        if frame < 0 {
            tokens[index] = TokenAlignment {
                label: labels[index],
                start_frame: -1,
                end_frame: -1,
                score: -10.0,
            };
            continue;
        }
        let abs = frame + window_start as i64;
        tokens[index] = TokenAlignment {
            label: labels[index],
            start_frame: abs,
            end_frame: abs,
            score: if hit { 0.0 } else { -10.0 },
        };
    }
}

/// Place each query character on the greedy transcript.
///
/// MMS writes what it hears (`forwrd`), not the spelling (`foreword`). Plain
/// Viterbi will walk a silent vowel forward until it finds that letter in
/// later, unrelated speech. Matching the spelling onto the heard characters,
/// and skipping extra audio cheaply, keeps the word on its own spikes.
///
/// Returns `(frame, exact_match)` per query character. Silent letters get a
/// frame interpolated between their neighbors. When several full embeddings
/// exist, the earlier one wins.
fn monotonic_align(query: &[char], heard: &[(char, i64)]) -> Vec<(i64, bool)> {
    let mut assigned = vec![None; query.len()];
    let mut matched = vec![false; query.len()];
    if !query.is_empty() && !heard.is_empty() {
        assign_monotonic(query, heard, &mut assigned, &mut matched);
    }
    interpolate_frames(&mut assigned);
    assigned
        .into_iter()
        .zip(matched)
        .map(|(frame, hit)| (frame.unwrap_or(-1), hit))
        .collect()
}

/// Largest edit matrix kept in one piece. Bigger spans are cut in half.
const MONOTONIC_CELL_LIMIT: usize = 8_000_000;
const MONOTONIC_SUB: i32 = 250;
const MONOTONIC_INS: i32 = 100;
const MONOTONIC_DEL: i32 = 15;

fn assign_monotonic(
    query: &[char],
    heard: &[(char, i64)],
    assigned: &mut [Option<i64>],
    matched: &mut [bool],
) {
    if query.is_empty() || heard.is_empty() {
        return;
    }
    if query.len().saturating_mul(heard.len()) > MONOTONIC_CELL_LIMIT && query.len() > 1 && heard.len() > 1 {
        let mid = query.len() / 2;
        let target = heard[0].1 + (heard[heard.len() - 1].1 - heard[0].1) * mid as i64 / query.len() as i64;
        let split = heard.partition_point(|item| item.1 < target).clamp(1, heard.len() - 1);
        assign_monotonic(&query[..mid], &heard[..split], &mut assigned[..mid], &mut matched[..mid]);
        assign_monotonic(&query[mid..], &heard[split..], &mut assigned[mid..], &mut matched[mid..]);
        return;
    }
    let n = query.len();
    let m = heard.len();
    let cols = m + 1;
    let mut cost = vec![0i32; (n + 1) * cols];
    let mut sum = vec![0i64; (n + 1) * cols];
    let mut op = vec![0u8; (n + 1) * cols];
    for j in 0..=m {
        cost[j] = j as i32 * MONOTONIC_DEL;
        op[j] = 3;
    }
    for i in 0..=n {
        cost[i * cols] = i as i32 * MONOTONIC_INS;
        op[i * cols] = 2;
    }
    for i in 1..=n {
        for j in 1..=m {
            let same = query[i - 1] == heard[j - 1].0;
            let diag_cost = cost[(i - 1) * cols + (j - 1)] + if same { 0 } else { MONOTONIC_SUB };
            // Tie-break on the sum of matched frames, so an earlier spike wins.
            let diag_sum = sum[(i - 1) * cols + (j - 1)] + if same { heard[j - 1].1 } else { 0 };
            let up_cost = cost[(i - 1) * cols + j] + MONOTONIC_INS;
            let up_sum = sum[(i - 1) * cols + j];
            let left_cost = cost[i * cols + (j - 1)] + MONOTONIC_DEL;
            let left_sum = sum[i * cols + (j - 1)];
            let mut best_c = left_cost;
            let mut best_s = left_sum;
            let mut best_op = 3u8;
            if up_cost < best_c || (up_cost == best_c && up_sum < best_s) {
                best_c = up_cost;
                best_s = up_sum;
                best_op = 2;
            }
            if diag_cost < best_c || (diag_cost == best_c && diag_sum < best_s) {
                best_c = diag_cost;
                best_s = diag_sum;
                best_op = 1;
            }
            let cell = i * cols + j;
            cost[cell] = best_c;
            sum[cell] = best_s;
            op[cell] = best_op;
        }
    }
    // Audio after the query is the next sentence or the next chapter. Paying
    // to consume it makes a slightly mismatched line spend its letters on
    // that later speech, and the following short line ("thank you") follows
    // them. Ending at the cheapest column leaves that tail unexplained.
    let mut best_j = 0usize;
    let mut best_c = cost[n * cols];
    let mut best_s = sum[n * cols];
    for j in 1..=m {
        let c = cost[n * cols + j];
        let s = sum[n * cols + j];
        if c < best_c || (c == best_c && s < best_s) {
            best_c = c;
            best_s = s;
            best_j = j;
        }
    }
    let mut i = n;
    let mut j = best_j;
    while i > 0 || j > 0 {
        let which = if i == 0 {
            3
        } else if j == 0 {
            2
        } else {
            op[i * cols + j]
        };
        match which {
            1 => {
                if query[i - 1] == heard[j - 1].0 {
                    assigned[i - 1] = Some(heard[j - 1].1);
                    matched[i - 1] = true;
                }
                i -= 1;
                j -= 1;
            }
            2 => i -= 1,
            _ => j -= 1,
        }
    }
}

fn interpolate_frames(frames: &mut [Option<i64>]) {
    let mut index = 0usize;
    while index < frames.len() {
        if frames[index].is_some() {
            index += 1;
            continue;
        }
        let mut end = index;
        while end < frames.len() && frames[end].is_none() {
            end += 1;
        }
        let left = if index > 0 { frames[index - 1] } else { None };
        let right = if end < frames.len() { frames[end] } else { None };
        match (left, right) {
            (Some(start), Some(stop)) => {
                let den = (end - (index - 1)) as i64;
                for slot in index..end {
                    let num = (slot - (index - 1)) as i64;
                    frames[slot] = Some(start + (stop - start) * num / den);
                }
            }
            (Some(start), None) => {
                for slot in index..end {
                    frames[slot] = Some(start);
                }
            }
            (None, Some(stop)) => {
                for slot in index..end {
                    frames[slot] = Some(stop);
                }
            }
            (None, None) => {}
        }
        index = end;
    }
}

fn fill_unread_gaps(tokens: &mut [TokenAlignment]) {
    let mut last_end = 0i64;
    for token in tokens.iter_mut() {
        if token.start_frame < 0 {
            token.start_frame = last_end;
            token.end_frame = last_end;
            continue;
        }
        last_end = token.end_frame.max(token.start_frame);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_emissions() -> Emissions {
        // Compact 3-class fake: blank=0, a=1, b=2. label_of is not used.
        let data = vec![
            -6.0, 0.0, -6.0, // a
            0.0, -6.0, -6.0, // blank
            -6.0, -6.0, 0.0, // b
        ];
        Emissions {
            classes: 3,
            data,
            times: vec![0.0, 0.02, 0.04],
        }
    }

    #[test]
    fn viterbi_places_letters_on_their_spikes() {
        let emissions = tiny_emissions();
        let aligned = ctc_viterbi(&emissions, &[1, 2], 0, 3, STAR_SCORE);
        assert_eq!(aligned.len(), 2);
        assert_eq!(aligned[0].start_frame, 0);
        assert_eq!(aligned[1].start_frame, 2);
        assert!(aligned[0].score > -1.0);
        assert!(aligned[1].score > -1.0);
    }

    #[test]
    fn a_silent_vowel_stays_inside_the_spoken_word() {
        // The model hears "forwrd" and then, seconds later, another "o".
        let heard = [
            ('f', 10),
            ('o', 12),
            ('r', 14),
            ('w', 16),
            ('a', 18),
            ('r', 20),
            ('d', 22),
            ('b', 80),
            ('y', 90),
            ('o', 120),
        ];
        let placed = monotonic_align(&"foreword".chars().collect::<Vec<_>>(), &heard);
        assert!(
            placed.iter().all(|(frame, _)| (10..=22).contains(frame)),
            "{placed:?}"
        );
        assert!(placed[0].1 && placed[4].1 && placed[7].1, "{placed:?}");
        assert!(!placed[3].1 && !placed[5].1, "{placed:?}");
    }

    #[test]
    fn a_later_copy_does_not_steal_a_tail_the_early_audio_already_matches() {
        // The early line drops one letter. A later chapter then says the whole
        // line. Consuming that later audio would "fix" the missing letter and
        // drag the tail with it; the tail stays on the early spike.
        let query: Vec<char> = "abcdefthankyou".chars().collect();
        let mut heard = Vec::new();
        let mut t = 10i64;
        for ch in "abdefthankyou".chars() {
            heard.push((ch, t));
            t += 5;
        }
        let early_end = t;
        for _ in 0..40 {
            heard.push(('q', t));
            t += 5;
        }
        for ch in "abcdefthankyou".chars() {
            heard.push((ch, t));
            t += 5;
        }
        let placed = monotonic_align(&query, &heard);
        let thank = query.len() - 8;
        assert!(
            placed[thank].1 && placed[thank].0 < early_end,
            "thankyou left the early spike: {placed:?}"
        );
        assert!(
            placed.iter().take(thank).all(|(frame, _)| *frame < early_end),
            "{placed:?}"
        );
    }

    #[test]
    fn speech_after_a_short_pause_stays_matched() {
        let query: Vec<char> = "abcdefwxyz".chars().collect();
        let mut heard = Vec::new();
        let mut t = 0i64;
        for ch in "abcdef".chars() {
            heard.push((ch, t));
            t += 4;
        }
        for _ in 0..10 {
            heard.push(('q', t));
            t += 4;
        }
        let after = t;
        for ch in "wxyz".chars() {
            heard.push((ch, t));
            t += 4;
        }
        let placed = monotonic_align(&query, &heard);
        assert!(placed.iter().all(|(_, hit)| *hit), "{placed:?}");
        assert!(placed[6].0 >= after, "{placed:?}");
    }

    #[test]
    fn forced_align_pins_foreword_to_the_spoken_spikes() {
        // f o r w a r d, then the narrator credit, then a decoy "o".
        let spikes = [
            ('f', 10),
            ('o', 12),
            ('r', 14),
            ('w', 16),
            ('a', 18),
            ('r', 20),
            ('d', 22),
            ('b', 80),
            ('y', 90),
            ('o', 120),
        ];
        let last = 130usize;
        let mut data = vec![-8.0f32; last * CLASSES];
        let mut times = Vec::with_capacity(last);
        for frame in 0..last {
            data[frame * CLASSES] = 0.0;
            times.push(frame as f64 * 0.02);
        }
        for (ch, frame) in spikes {
            let id = label_of(ch).unwrap() as usize;
            data[frame * CLASSES] = -1.0;
            data[frame * CLASSES + id] = 0.0;
        }
        let emissions = Emissions { classes: CLASSES, data, times };
        let labels: Vec<i32> = "foreword".chars().map(|ch| label_of(ch).unwrap()).collect();
        let aligned = forced_align(&emissions, &labels, &[], CLASSES as i32, false);
        assert!(
            aligned.iter().all(|token| (10..=22).contains(&token.start_frame)),
            "{aligned:?}"
        );
        assert!(aligned[0].score > -1.0 && aligned[4].score > -1.0 && aligned[7].score > -1.0);
        assert!(aligned[3].score < -4.0 && aligned[5].score < -4.0, "{aligned:?}");
    }

    #[test]
    fn a_heading_does_not_latch_onto_the_following_credit() {
        // "forward" then, a little later, "bores...d" which contains o,r,d.
        let spikes = [
            ('f', 10),
            ('o', 12),
            ('r', 14),
            ('w', 16),
            ('a', 18),
            ('r', 20),
            ('d', 22),
            ('b', 30),
            ('o', 48),
            ('r', 52),
            ('e', 56),
            ('d', 90),
        ];
        let last = 100usize;
        let mut data = vec![-8.0f32; last * CLASSES];
        let mut times = Vec::with_capacity(last);
        for frame in 0..last {
            data[frame * CLASSES] = 0.0;
            times.push(frame as f64 * 0.02);
        }
        for (ch, frame) in spikes {
            let id = label_of(ch).unwrap() as usize;
            data[frame * CLASSES] = -1.0;
            data[frame * CLASSES + id] = 0.0;
        }
        let emissions = Emissions { classes: CLASSES, data, times };
        let labels: Vec<i32> = "foreword".chars().map(|ch| label_of(ch).unwrap()).collect();
        let aligned = forced_align(&emissions, &labels, &[], CLASSES as i32, false);
        assert!(
            aligned.iter().all(|token| (10..=22).contains(&token.start_frame)),
            "{aligned:?}"
        );
    }

    #[test]
    fn greedy_collapses_repeats_and_blanks() {
        // char_of only knows the MMS ids, not 1 and 2. Build MMS-shaped rows instead.
        let mut data = vec![-8.0; 3 * CLASSES];
        data[0 * CLASSES + 4] = 0.0; // a
        data[1 * CLASSES + 0] = 0.0; // blank
        data[2 * CLASSES + 4] = 0.0; // a again, collapsed
        data[2 * CLASSES + 0] = -1.0;
        data[0 * CLASSES + 0] = -1.0;
        // third frame should be 'b' = 20, replace frame 2
        for value in &mut data[2 * CLASSES..(3 * CLASSES)] {
            *value = -8.0;
        }
        data[2 * CLASSES + 20] = 0.0;
        let emissions = Emissions {
            classes: CLASSES,
            data,
            times: vec![0.0, 0.02, 0.04],
        };
        let (text, frames) = greedy_decode(&emissions);
        assert_eq!(text, "ab");
        assert_eq!(frames, vec![0, 2]);
    }

    #[test]
    fn identical_text_finds_a_boundary() {
        let mut query = String::new();
        for i in 0..80 {
            query.push(char::from(b'a' + (i % 23) as u8));
        }
        // Emissions whose greedy decode equals `query`, one letter every 5 frames.
        let frames = query.len() * 5;
        let mut data = vec![-8.0; frames * CLASSES];
        let mut times = Vec::with_capacity(frames);
        for (i, ch) in query.chars().enumerate() {
            let id = label_of(ch).unwrap() as usize;
            let frame = i * 5;
            data[frame * CLASSES + id] = 0.0;
            data[frame * CLASSES] = -2.0;
            for pad in 1..5 {
                data[(frame + pad) * CLASSES] = 0.0;
            }
        }
        for t in 0..frames {
            times.push(t as f64 * 0.02);
        }
        let emissions = Emissions { classes: CLASSES, data, times };
        let found = find_boundaries(&query, &emissions, &[]).expect("boundary");
        assert!(found.confidence >= MIN_CONFIDENCE);
        assert!(found.end > found.start);
        assert!(!found.anchors.is_empty() || query.len() < ANCHOR_SPACING);
    }

    #[test]
    fn a_later_copy_of_the_same_words_does_not_stretch_the_window() {
        let mut query = String::new();
        for i in 0..80 {
            query.push(char::from(b'a' + (i % 23) as u8));
        }
        // The last 20 letters are spoken again ~36s later, still inside the
        // 50s inlier tolerance. That second hit must not extend the window.
        let late_origin = 2200usize;
        let frames = late_origin + 20 * 5 + 5;
        let mut data = vec![-8.0f32; frames * CLASSES];
        let mut times = Vec::with_capacity(frames);
        for t in 0..frames {
            data[t * CLASSES] = 0.0;
            times.push(t as f64 * 0.02);
        }
        for (i, ch) in query.chars().enumerate() {
            let id = label_of(ch).unwrap() as usize;
            let frame = i * 5;
            data[frame * CLASSES] = -2.0;
            data[frame * CLASSES + id] = 0.0;
        }
        for (i, ch) in query.chars().skip(60).enumerate() {
            let id = label_of(ch).unwrap() as usize;
            let frame = late_origin + i * 5;
            data[frame * CLASSES] = -2.0;
            data[frame * CLASSES + id] = 0.0;
        }
        let emissions = Emissions { classes: CLASSES, data, times };
        let found = find_boundaries(&query, &emissions, &[]).expect("boundary");
        assert!(
            found.end < late_origin as i64,
            "window end {} reached the later copy at {late_origin}",
            found.end
        );
    }

    #[test]
    fn emissions_roundtrip() {
        let emissions = Emissions {
            classes: 2,
            data: vec![0.0, -1.0, -1.0, 0.0],
            times: vec![0.1, 0.3],
        };
        let path = std::env::temp_dir().join("stalign-emissions-test.bin");
        emissions.save(&path).unwrap();
        let loaded = Emissions::load(&path).unwrap();
        assert_eq!(loaded.data, emissions.data);
        assert_eq!(loaded.times, emissions.times);
        let _ = std::fs::remove_file(path);
    }
}
