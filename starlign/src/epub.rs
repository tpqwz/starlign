//! Read an EPUB, wrap sentences and words, and write Media Overlays.
//!
//! Sentence spans bubble out through inline tags (`em`, links, page styles) so
//! one highlight covers the whole sentence. A later fragment that cannot share
//! that element keeps the same timing under `id-pN` and `data-s`.

use crate::audio::Track;
use crate::xmltree::{local_name, Document, NodeData};
use any_ascii::any_ascii;
use anyhow::{bail, Context, Result};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use unicode_segmentation::UnicodeSegmentation;
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

pub const READALOUD_CSS: &str = "\
.-epub-media-overlay-active {\n\
  background-color: #ffff99;\n\
  color: inherit;\n\
}\n";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpanKind {
    Matched,
    Interpolated,
    Unmatched,
    AudioOnly,
}

#[derive(Clone, Debug)]
pub struct OverlaySpan {
    pub id: String,
    pub start: f64,
    pub end: f64,
    pub kind: SpanKind,
}

#[derive(Clone, Debug)]
pub struct WordMark {
    pub id: String,
    pub letter_range: std::ops::Range<usize>,
}

#[derive(Clone, Debug)]
pub struct SentenceMark {
    pub id: String,
    pub letter_range: std::ops::Range<usize>,
    pub words: Vec<WordMark>,
}

#[derive(Clone, Debug)]
pub struct ChapterText {
    pub letters: String,
    pub sentences: Vec<SentenceMark>,
}

#[derive(Clone, Debug)]
pub struct SourceChapter {
    pub id: String,
    pub href: String,
    pub zip_path: String,
    pub properties: String,
    pub xml: String,
    pub skip: bool,
    /// Why `skip` is set: nav, title, structural, guide, filename, or linear.
    pub skip_reason: String,
}

pub struct Book {
    opf_path: String,
    files: BTreeMap<String, Vec<u8>>,
}

#[derive(Clone, Debug)]
pub struct ChapterOutput {
    pub href: String,
    pub zip_path: String,
    pub xhtml: String,
    pub spans: Vec<OverlaySpan>,
}

impl Book {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let mut archive = ZipArchive::new(file).context("reading epub zip")?;
        let mut files = BTreeMap::new();
        for index in 0..archive.len() {
            let mut entry = archive.by_index(index)?;
            if entry.is_dir() {
                continue;
            }
            let name = entry.name().replace('\\', "/");
            let mut bytes = Vec::new();
            entry.read_to_end(&mut bytes)?;
            files.insert(name, bytes);
        }
        let opf_path = opf_path(&files)?;
        Ok(Self { opf_path, files })
    }

    pub fn chapters(&self) -> Result<Vec<SourceChapter>> {
        let opf = self
            .files
            .get(&self.opf_path)
            .ok_or_else(|| anyhow::anyhow!("missing {}", self.opf_path))?;
        let xml = std::str::from_utf8(opf).context("opf encoding")?;
        let doc = Document::parse(xml)?;
        let manifest = child_named(&doc, doc.root, "manifest").context("opf has no manifest")?;
        let spine = child_named(&doc, doc.root, "spine").context("opf has no spine")?;
        let mut items = HashMap::new();
        for child in doc.children(manifest) {
            if local_name(doc.name(*child)) != "item" {
                continue;
            }
            let Some(id) = doc.attr(*child, "id") else {
                continue;
            };
            items.insert(id, *child);
        }
        let opf_dir = parent_dir(&self.opf_path);
        let mut nav_href = String::new();
        for child in doc.children(manifest) {
            if local_name(doc.name(*child)) != "item" {
                continue;
            }
            let properties = doc.attr(*child, "properties").unwrap_or_default();
            if properties.split_whitespace().any(|token| token == "nav") {
                nav_href = doc.attr(*child, "href").unwrap_or_default();
                break;
            }
        }
        let (toc_titles, landmarks) = nav_labels(&self.files, &opf_dir, &nav_href);
        let guides = guide_types(&doc, doc.root);
        let mut pending = Vec::new();
        for child in doc.children(spine) {
            if local_name(doc.name(*child)) != "itemref" {
                continue;
            }
            let Some(idref) = doc.attr(*child, "idref") else {
                continue;
            };
            let Some(&item) = items.get(&idref) else {
                continue;
            };
            let href = doc.attr(item, "href").unwrap_or_default();
            let media = doc.attr(item, "media-type").unwrap_or_default();
            let properties = doc.attr(item, "properties").unwrap_or_default();
            if !media.contains("html") {
                continue;
            }
            let decoded = url_decode(&href);
            let zip_path = join_path(&opf_dir, &decoded);
            let Some(bytes) = self.files.get(&zip_path) else {
                crate::say!("spine item {href} is not in the epub");
                continue;
            };
            let xml = String::from_utf8_lossy(bytes).into_owned();
            let linear = doc
                .attr(*child, "linear")
                .unwrap_or_default()
                .to_ascii_lowercase()
                != "no";
            pending.push((idref.to_string(), href, zip_path, properties, xml, linear));
        }
        let spine_len = pending.len();
        let mut chapters = Vec::new();
        for (index, (idref, href, zip_path, properties, xml, linear)) in
            pending.into_iter().enumerate()
        {
            let key = href_key(&href);
            let nav = properties.split_whitespace().any(|token| token == "nav");
            let content_types = content_types_of(&xml);
            let heading = first_heading(&xml);
            let title = toc_titles.get(&key).cloned().or(heading);
            let empty_marks: Vec<String> = Vec::new();
            let landmark_list = landmarks.get(&key).unwrap_or(&empty_marks);
            let guide_list = guides.get(&key).unwrap_or(&empty_marks);
            let facts = crate::exclude::ChapterFacts {
                index,
                spine_len,
                href: &href,
                linear,
                nav,
                title: title.as_deref(),
                landmarks: landmark_list,
                guides: guide_list,
                content_types: &content_types,
            };
            let skip_reason = crate::exclude::skip_reason(&facts).unwrap_or_default();
            let skip = !skip_reason.is_empty();
            chapters.push(SourceChapter {
                id: items
                    .get(idref.as_str())
                    .and_then(|item| doc.attr(*item, "id"))
                    .unwrap_or(idref),
                href,
                zip_path,
                properties,
                xml,
                skip,
                skip_reason,
            });
        }
        Ok(chapters)
    }

    pub fn write(
        &self,
        output: &Path,
        chapters: &[ChapterOutput],
        tracks: &[Track],
        total_seconds: f64,
        split: crate::split::Split,
    ) -> Result<()> {
        if let Some(parent) = output.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let opf_dir = parent_dir(&self.opf_path);
        let mut replaced: HashMap<String, Vec<u8>> = HashMap::new();
        let mut smils: Vec<(String, String, f64)> = Vec::new();
        let mut kept: Vec<crate::split::Piece> = Vec::new();
        let mut cache = crate::split::Cache::new();
        let mut used_audio = HashSet::new();
        let timed = match split {
            crate::split::Split::Seconds(_) | crate::split::Split::Parts(_) => {
                Some(prepare_timed_split(
                    &mut cache,
                    tracks,
                    chapters,
                    split,
                    &mut kept,
                    &mut used_audio,
                )?)
            }
            _ => None,
        };
        let mut chapter_tracks: Vec<Track> = Vec::new();
        for chapter in chapters {
            if chapter.spans.is_empty() {
                continue;
            }
            replaced.insert(chapter.zip_path.clone(), chapter.xhtml.as_bytes().to_vec());
            let smil_name = smil_name(&chapter.href);
            let smil_zip = join_path(&opf_dir, &format!("MediaOverlays/{smil_name}"));
            let smil_dir = parent_dir(&smil_zip);
            let text_rel = url_encode_path(&relative_from(&smil_dir, &chapter.zip_path));
            let duration = chapter_duration(&chapter.spans);
            let smil = if let Some(plan) = &timed {
                let windows: Vec<std::ops::Range<usize>> = chapter
                    .spans
                    .iter()
                    .map(|span| {
                        let index = crate::split::slice_index(&plan.nominal, span.start);
                        plan.ranges[index].clone()
                    })
                    .collect();
                render_smil(
                    &chapter.href,
                    &text_rel,
                    &chapter.spans,
                    &plan.tracks,
                    &smil_dir,
                    &opf_dir,
                    Some(&windows),
                )
            } else if split == crate::split::Split::Chapters {
                let start = chapter
                    .spans
                    .iter()
                    .map(|span| span.start)
                    .fold(f64::INFINITY, f64::min);
                let end = chapter
                    .spans
                    .iter()
                    .map(|span| span.end)
                    .fold(f64::NEG_INFINITY, f64::max);
                let from = chapter_tracks.len();
                let pieces = crate::split::cut_range(
                    &mut cache,
                    tracks,
                    start,
                    end,
                    &audio_stem(&chapter.href),
                    &mut used_audio,
                )
                .with_context(|| format!("splitting audio for {}", chapter.href))?;
                if pieces.is_empty() {
                    bail!(
                        "no audiobook audio covers {start:.3}s–{end:.3}s for {}",
                        chapter.href
                    );
                }
                for piece in pieces {
                    chapter_tracks.push(piece.track());
                    kept.push(piece);
                }
                render_smil(
                    &chapter.href,
                    &text_rel,
                    &chapter.spans,
                    &chapter_tracks[from..],
                    &smil_dir,
                    &opf_dir,
                    None,
                )
            } else {
                render_smil(
                    &chapter.href,
                    &text_rel,
                    &chapter.spans,
                    tracks,
                    &smil_dir,
                    &opf_dir,
                    None,
                )
            };
            smils.push((smil_zip, smil, duration));
        }
        let manifest_ref: &[Track] = if let Some(plan) = &timed {
            &plan.tracks
        } else if split == crate::split::Split::Chapters {
            &chapter_tracks
        } else {
            tracks
        };
        let css_zip = join_path(&opf_dir, "Styles/storyteller-readaloud.css");
        let opf_xml = rewrite_opf(self, chapters, &smils, manifest_ref, total_seconds)?;
        replaced.insert(self.opf_path.clone(), opf_xml.into_bytes());
        replaced.insert(css_zip.clone(), READALOUD_CSS.as_bytes().to_vec());

        let copying_sources = split == crate::split::Split::Whole;
        let audio_paths: HashMap<String, &Track> = if copying_sources {
            tracks
                .iter()
                .map(|track| {
                    (
                        join_path(&opf_dir, &format!("Audio/{}", track.epub_name)),
                        track,
                    )
                })
                .collect()
        } else {
            HashMap::new()
        };

        let file =
            File::create(output).with_context(|| format!("creating {}", output.display()))?;
        let mut zip = ZipWriter::new(file);
        let stored = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
        let deflated = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
        zip.start_file("mimetype", stored)?;
        zip.write_all(b"application/epub+zip")?;

        for (name, bytes) in &self.files {
            if name == "mimetype" || audio_paths.contains_key(name) || replaced.contains_key(name) {
                continue;
            }
            zip.start_file(name, deflated)?;
            zip.write_all(bytes)?;
        }
        for (name, bytes) in &replaced {
            if name == "mimetype" {
                continue;
            }
            zip.start_file(name, deflated)?;
            zip.write_all(bytes)?;
        }
        for (name, xml, _) in &smils {
            zip.start_file(name, deflated)?;
            zip.write_all(xml.as_bytes())?;
        }
        if copying_sources {
            for (name, track) in &audio_paths {
                zip.start_file(name, stored)?;
                let mut audio = File::open(&track.source)
                    .with_context(|| format!("embedding {}", track.source.display()))?;
                std::io::copy(&mut audio, &mut zip)?;
            }
        } else {
            for piece in &kept {
                let name = join_path(&opf_dir, &format!("Audio/{}", piece.epub_name));
                zip.start_file(name, stored)?;
                let mut audio = File::open(piece.path())
                    .with_context(|| format!("embedding {}", piece.epub_name))?;
                std::io::copy(&mut audio, &mut zip)?;
            }
        }
        zip.finish()?;
        Ok(())
    }
}

struct TimedSplit {
    tracks: Vec<Track>,
    ranges: Vec<std::ops::Range<usize>>,
    nominal: Vec<(f64, f64)>,
}

fn prepare_timed_split(
    cache: &mut crate::split::Cache,
    tracks: &[Track],
    chapters: &[ChapterOutput],
    split: crate::split::Split,
    kept: &mut Vec<crate::split::Piece>,
    used: &mut HashSet<String>,
) -> Result<TimedSplit> {
    let (timeline_start, timeline_end) = timeline_bounds(tracks)?;
    let nominal = crate::split::nominal_slices(split, timeline_start, timeline_end)?;
    let span_times: Vec<(f64, f64)> = chapters
        .iter()
        .flat_map(|chapter| chapter.spans.iter().map(|span| (span.start, span.end)))
        .collect();
    let (covered, _) = crate::split::assign_slices(&nominal, &span_times);
    let width = nominal.len().to_string().len().max(2);
    let mut out_tracks = Vec::new();
    let mut ranges = Vec::new();
    for (index, (raw_start, raw_end)) in covered.iter().enumerate() {
        let lo = (*raw_start).max(timeline_start);
        let hi = (*raw_end).min(timeline_end);
        let from = out_tracks.len();
        if hi - lo <= 1e-4 {
            ranges.push(from..from);
            continue;
        }
        let stem = format!("audio-{n:0width$}", n = index + 1, width = width);
        let pieces = crate::split::cut_range(cache, tracks, lo, hi, &stem, used)
            .with_context(|| format!("splitting audio {stem} {lo:.3}s–{hi:.3}s"))?;
        if pieces.is_empty() {
            crate::say!("split {stem} {lo:.3}s–{hi:.3}s has no audio; skipping");
            ranges.push(from..from);
            continue;
        }
        for piece in pieces {
            out_tracks.push(piece.track());
            kept.push(piece);
        }
        ranges.push(from..out_tracks.len());
    }
    for &(span_start, span_end) in &span_times {
        let index = crate::split::slice_index(&nominal, span_start);
        if ranges
            .get(index)
            .map(|range| range.is_empty())
            .unwrap_or(true)
        {
            bail!("no audiobook audio covers {span_start:.3}s–{span_end:.3}s");
        }
    }
    Ok(TimedSplit {
        tracks: out_tracks,
        ranges,
        nominal,
    })
}

fn timeline_bounds(tracks: &[Track]) -> Result<(f64, f64)> {
    if tracks.is_empty() {
        bail!("the audiobook has no audio to split");
    }
    let start = tracks
        .iter()
        .map(|track| track.start_secs())
        .fold(f64::INFINITY, f64::min);
    let end = tracks
        .iter()
        .map(|track| track.start_secs() + track.duration_secs())
        .fold(f64::NEG_INFINITY, f64::max);
    if !(end > start) {
        bail!("the audiobook has no duration to split");
    }
    Ok((start, end))
}

fn audio_stem(href: &str) -> String {
    let base = href.rsplit('/').next().unwrap_or(href);
    base.rsplit_once('.')
        .map(|(stem, _)| stem)
        .unwrap_or(base)
        .to_string()
}

#[cfg(test)]
#[test]
fn split_chapters_shifts_clips_onto_a_chapter_file() {
    let dir = std::env::temp_dir().join(format!("starlign-epub-split-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let audio = dir.join("book.m4a");
    crate::split::write_marker_fixture(&audio);
    let mut files = BTreeMap::new();
    files.insert(
        "content.opf".into(),
        br#"<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="uid">
<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
<dc:identifier id="uid">id</dc:identifier>
<dc:title>T</dc:title>
<dc:language>en</dc:language>
</metadata>
<manifest>
<item id="c1" href="chapter.xhtml" media-type="application/xhtml+xml"/>
</manifest>
<spine>
<itemref idref="c1"/>
</spine>
</package>"#
            .to_vec(),
    );
    files.insert(
        "chapter.xhtml".into(),
        b"<?xml version=\"1.0\"?><html xmlns=\"http://www.w3.org/1999/xhtml\"><head><title>c</title></head><body><p>Hello.</p></body></html>".to_vec(),
    );
    let book = Book {
        opf_path: "content.opf".into(),
        files,
    };
    let output = dir.join("out.epub");
    let chapters = vec![ChapterOutput {
        href: "chapter.xhtml".into(),
        zip_path: "chapter.xhtml".into(),
        xhtml: "<html xmlns=\"http://www.w3.org/1999/xhtml\"></html>".into(),
        spans: vec![OverlaySpan {
            id: "chapter.xhtml-s0".into(),
            start: 1.25,
            end: 2.10,
            kind: SpanKind::Matched,
        }],
    }];
    let tracks = vec![Track {
        source: audio,
        epub_name: "whole.m4a".into(),
        media_type: "audio/mp4".into(),
        start_sample: 0,
        samples: 4 * 16_000,
    }];
    book.write(
        &output,
        &chapters,
        &tracks,
        4.0,
        crate::split::Split::Chapters,
    )
    .unwrap();
    let mut archive = ZipArchive::new(File::open(&output).unwrap()).unwrap();
    let mut smil = String::new();
    archive
        .by_name("MediaOverlays/chapter.smil")
        .unwrap()
        .read_to_string(&mut smil)
        .unwrap();
    assert!(smil.contains("clipBegin=\"0:00:00.250\""), "{smil}");
    assert!(smil.contains("clipEnd=\"0:00:01.100\""), "{smil}");
    assert!(smil.contains("../Audio/chapter.m4a"), "{smil}");
    assert!(!smil.contains("whole.m4a"), "{smil}");
    let mut audio_bytes = Vec::new();
    archive
        .by_name("Audio/chapter.m4a")
        .unwrap()
        .read_to_end(&mut audio_bytes)
        .unwrap();
    assert!(
        audio_bytes.windows(8).any(|window| window == [0xB2; 8]),
        "missing kept frame"
    );
    assert!(
        !audio_bytes.windows(8).any(|window| window == [0xA1; 8]),
        "kept audio from before the chapter"
    );
    assert!(
        !audio_bytes.windows(8).any(|window| window == [0xD4; 8]),
        "kept audio from after the chapter"
    );
    let mut opf = String::new();
    archive
        .by_name("content.opf")
        .unwrap()
        .read_to_string(&mut opf)
        .unwrap();
    assert!(opf.contains("Audio/chapter.m4a"), "{opf}");
    assert!(opf.contains("media-overlay"), "{opf}");
    let whole = dir.join("whole.epub");
    book.write(&whole, &chapters, &tracks, 4.0, crate::split::Split::Whole)
        .unwrap();
    let mut archive = ZipArchive::new(File::open(&whole).unwrap()).unwrap();
    let mut smil = String::new();
    archive
        .by_name("MediaOverlays/chapter.smil")
        .unwrap()
        .read_to_string(&mut smil)
        .unwrap();
    assert!(smil.contains("clipBegin=\"0:00:01.250\""), "{smil}");
    assert!(smil.contains("../Audio/whole.m4a"), "{smil}");
    assert!(archive.by_name("Audio/chapter.m4a").is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[cfg(test)]
#[test]
fn a_duration_split_keeps_a_crossing_span_on_the_file_where_it_starts() {
    let dir = std::env::temp_dir().join(format!("starlign-epub-duration-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let audio = dir.join("book.m4a");
    crate::split::write_marker_fixture(&audio);
    let mut files = BTreeMap::new();
    files.insert(
        "content.opf".into(),
        br#"<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="uid">
<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
<dc:identifier id="uid">id</dc:identifier>
<dc:title>T</dc:title>
<dc:language>en</dc:language>
</metadata>
<manifest>
<item id="c1" href="chapter.xhtml" media-type="application/xhtml+xml"/>
</manifest>
<spine>
<itemref idref="c1"/>
</spine>
</package>"#
            .to_vec(),
    );
    files.insert(
        "chapter.xhtml".into(),
        b"<?xml version=\"1.0\"?><html xmlns=\"http://www.w3.org/1999/xhtml\"><body><p>Hello.</p></body></html>".to_vec(),
    );
    let book = Book {
        opf_path: "content.opf".into(),
        files,
    };
    let chapters = vec![ChapterOutput {
        href: "chapter.xhtml".into(),
        zip_path: "chapter.xhtml".into(),
        xhtml: "<html xmlns=\"http://www.w3.org/1999/xhtml\"></html>".into(),
        spans: vec![
            OverlaySpan {
                id: "chapter.xhtml-s0".into(),
                start: 1.25,
                end: 2.10,
                kind: SpanKind::Matched,
            },
            OverlaySpan {
                id: "chapter.xhtml-s1".into(),
                start: 2.20,
                end: 3.40,
                kind: SpanKind::Matched,
            },
        ],
    }];
    let tracks = vec![Track {
        source: audio,
        epub_name: "whole.m4a".into(),
        media_type: "audio/mp4".into(),
        start_sample: 0,
        samples: 4 * 16_000,
    }];
    for (split, name) in [
        (crate::split::Split::Seconds(2.0), "seconds.epub"),
        (crate::split::Split::Parts(2), "parts.epub"),
    ] {
        let output = dir.join(name);
        book.write(&output, &chapters, &tracks, 4.0, split).unwrap();
        let mut archive = ZipArchive::new(File::open(&output).unwrap()).unwrap();
        let mut smil = String::new();
        archive
            .by_name("MediaOverlays/chapter.smil")
            .unwrap()
            .read_to_string(&mut smil)
            .unwrap();
        assert_eq!(smil.matches("<par ").count(), 2, "{name} {smil}");
        let first = par_text(&smil, "chapter.xhtml-s0-0");
        assert!(
            first.contains("clipBegin=\"0:00:01.250\""),
            "{name} {first}"
        );
        assert!(first.contains("clipEnd=\"0:00:02.100\""), "{name} {first}");
        assert!(first.contains("../Audio/audio-01.m4a"), "{name} {first}");
        assert!(!first.contains("audio-02"), "{name} {first}");
        let second = par_text(&smil, "chapter.xhtml-s1-1");
        assert!(
            second.contains("clipBegin=\"0:00:00.200\""),
            "{name} {second}"
        );
        assert!(
            second.contains("clipEnd=\"0:00:01.400\""),
            "{name} {second}"
        );
        assert!(second.contains("../Audio/audio-02.m4a"), "{name} {second}");
        assert!(!second.contains("audio-01"), "{name} {second}");
        let mut early = Vec::new();
        archive
            .by_name("Audio/audio-01.m4a")
            .unwrap()
            .read_to_end(&mut early)
            .unwrap();
        assert!(early.windows(8).any(|window| window == [0xA1; 8]), "{name}");
        assert!(early.windows(8).any(|window| window == [0xC3; 8]), "{name}");
        assert!(
            !early.windows(8).any(|window| window == [0xD4; 8]),
            "{name}"
        );
        let mut later = Vec::new();
        archive
            .by_name("Audio/audio-02.m4a")
            .unwrap()
            .read_to_end(&mut later)
            .unwrap();
        assert!(later.windows(8).any(|window| window == [0xC3; 8]), "{name}");
        assert!(later.windows(8).any(|window| window == [0xD4; 8]), "{name}");
        assert!(
            !later.windows(8).any(|window| window == [0xA1; 8]),
            "{name}"
        );
        assert!(
            !later.windows(8).any(|window| window == [0xB2; 8]),
            "{name}"
        );
        let mut opf = String::new();
        archive
            .by_name("content.opf")
            .unwrap()
            .read_to_string(&mut opf)
            .unwrap();
        assert!(opf.contains("Audio/audio-01.m4a"), "{name} {opf}");
        assert!(opf.contains("Audio/audio-02.m4a"), "{name} {opf}");
        assert!(!opf.contains("whole.m4a"), "{name} {opf}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[cfg(test)]
fn par_text<'a>(smil: &'a str, id: &str) -> &'a str {
    let key = format!("id=\"{id}\"");
    let start = smil
        .find(&key)
        .unwrap_or_else(|| panic!("missing {id} in {smil}"));
    let end = smil[start..].find("</par>").expect("par") + start;
    &smil[start..end]
}

pub fn markup_xml(xml: &str, id_prefix: &str, word_level: bool) -> Result<(String, ChapterText)> {
    let mut doc = Document::parse(xml)?;
    ensure_epub_namespace(&mut doc);
    let mut flat = String::new();
    let mut pieces = Vec::new();
    flatten(&doc, doc.root, &mut flat, &mut pieces);
    let ranges = sentence_ranges(&flat);
    let mut segs: HashMap<usize, Vec<(usize, usize, Option<usize>)>> = HashMap::new();
    for piece in &pieces {
        let cuts = cut_piece(&flat, piece, |pos| sentence_index(&ranges, pos));
        segs.insert(piece.node, cuts);
    }
    apply_segments(&mut doc, &segs, "data-st-sent");
    let root = doc.root;
    bubble(&mut doc, root, "data-st-sent", false);
    strip_attr(&mut doc, root, "data-st-own");
    assign_sentence_ids(&mut doc, root, id_prefix, &mut HashMap::new());
    if word_level {
        let spans = sentence_spans(&doc);
        let mut word_counts: HashMap<String, usize> = HashMap::new();
        for span in spans {
            wrap_words(&mut doc, span, &mut word_counts);
        }
    }
    ensure_stylesheet(&mut doc);
    let text = collect_text(&doc, word_level);
    let xml = doc.to_xml()?;
    Ok((xml, text))
}

pub fn push_letters(text: &str, out: &mut String) {
    let stripped: String = text.chars().filter(|ch| *ch != '\u{00ad}').collect();
    let ascii = any_ascii(&stripped);
    let chars: Vec<char> = ascii.chars().collect();
    let mut index = 0usize;
    while index < chars.len() {
        match chars[index] {
            'A'..='Z' => {
                out.push(chars[index].to_ascii_lowercase());
                index += 1;
            }
            'a'..='z' => {
                out.push(chars[index]);
                index += 1;
            }
            '\'' | '\u{2019}' | '\u{2018}' | '\u{02bc}' => {
                out.push('\'');
                index += 1;
            }
            '0'..='9' => {
                let mut number = String::new();
                while index < chars.len() && (chars[index].is_ascii_digit() || chars[index] == ',')
                {
                    if chars[index] != ',' {
                        number.push(chars[index]);
                    } else if index + 1 < chars.len() && chars[index + 1].is_ascii_digit() {
                        // thousands separator
                    } else {
                        break;
                    }
                    index += 1;
                }
                let mut fraction = String::new();
                if index < chars.len()
                    && chars[index] == '.'
                    && index + 1 < chars.len()
                    && chars[index + 1].is_ascii_digit()
                {
                    index += 1;
                    while index < chars.len() && chars[index].is_ascii_digit() {
                        fraction.push(chars[index]);
                        index += 1;
                    }
                }
                // "3rd" stays with the suffix letters; a year or a standalone count is spoken.
                let glued = index < chars.len() && chars[index].is_ascii_alphabetic();
                if !glued {
                    if let Some(spoken) = spell_number(&number, &fraction) {
                        push_letters(&spoken, out);
                    }
                }
            }
            _ => index += 1,
        }
    }
}

/// American English cardinals, the way `to-words` speaks `en-US` ("2003" → "two thousand three").
fn spell_number(digits: &str, fraction: &str) -> Option<String> {
    if digits.is_empty() || digits.len() > 15 {
        return None;
    }
    let value: u64 = digits.parse().ok()?;
    let mut spoken = spell_cardinal(value);
    if !fraction.is_empty() {
        spoken.push_str(" point");
        for ch in fraction.chars() {
            spoken.push(' ');
            spoken.push_str(&spell_cardinal(u64::from(ch.to_digit(10)?)));
        }
    }
    Some(spoken)
}

fn spell_cardinal(value: u64) -> String {
    if value == 0 {
        return "zero".into();
    }
    const SCALES: [(u64, &str); 4] = [
        (1_000_000_000_000, "trillion"),
        (1_000_000_000, "billion"),
        (1_000_000, "million"),
        (1_000, "thousand"),
    ];
    let mut rest = value;
    let mut parts = Vec::new();
    for (scale, name) in SCALES {
        if rest >= scale {
            let count = rest / scale;
            rest %= scale;
            parts.push(format!("{} {name}", spell_below_thousand(count as u32)));
        }
    }
    if rest > 0 {
        parts.push(spell_below_thousand(rest as u32));
    }
    parts.join(" ")
}

fn spell_below_thousand(value: u32) -> String {
    const ONES: [&str; 20] = [
        "",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ];
    const TENS: [&str; 10] = [
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    ];
    let mut parts = Vec::new();
    let mut rest = value;
    if rest >= 100 {
        parts.push(format!("{} hundred", ONES[(rest / 100) as usize]));
        rest %= 100;
    }
    if rest >= 20 {
        let ten = TENS[(rest / 10) as usize];
        let one = rest % 10;
        if one == 0 {
            parts.push(ten.to_string());
        } else {
            parts.push(format!("{ten} {}", ONES[one as usize]));
        }
    } else if rest > 0 {
        parts.push(ONES[rest as usize].to_string());
    }
    parts.join(" ")
}

struct Piece {
    node: usize,
    flat_start: usize,
    flat_end: usize,
}

fn flatten(doc: &Document, idx: usize, flat: &mut String, pieces: &mut Vec<Piece>) {
    if !doc.is_element(idx) || should_skip(doc, idx) {
        return;
    }
    let name = local_name(doc.name(idx)).to_string();
    if is_block(&name) && !flat.is_empty() && !flat.ends_with('\n') {
        flat.push('\n');
    }
    // A line break inside a heading is still one spoken sentence
    // ("THE GIFT OF / POWERLESSNESS"). A horizontal rule separates them.
    if name == "br" {
        if !flat.is_empty() && !flat.ends_with(|ch: char| ch.is_whitespace()) {
            flat.push(' ');
        }
        return;
    }
    if name == "hr" {
        if !flat.ends_with('\n') {
            flat.push('\n');
        }
        return;
    }
    for child in doc.children(idx).to_vec() {
        match &doc.nodes[child].data {
            NodeData::Text(text) => {
                let start = flat.len();
                flat.push_str(text);
                pieces.push(Piece {
                    node: child,
                    flat_start: start,
                    flat_end: flat.len(),
                });
            }
            NodeData::Element { .. } => flatten(doc, child, flat, pieces),
            NodeData::Comment(_) => {}
        }
    }
}

fn sentence_ranges(flat: &str) -> Vec<(usize, usize)> {
    // `unicode_sentences` drops bounds that contain no letters, so adding
    // their lengths walks the cursor off the real byte offsets. Bound indices
    // stay aligned with the flat string, including newlines inserted between
    // blocks and whitespace text nodes the segmenter skips.
    let ranges: Vec<(usize, usize)> = flat
        .split_sentence_bound_indices()
        .filter(|(_, sentence)| sentence_has_words(sentence))
        .map(|(start, sentence)| (start, start + sentence.len()))
        .collect();
    // UAX #29 ends a sentence at "C." and at "1." when the next word is
    // capitalized. Spoken initials and numbered steps are one phrase.
    let mut merged = Vec::new();
    let mut index = 0usize;
    while index < ranges.len() {
        let start = ranges[index].0;
        let mut end = ranges[index].1;
        while index + 1 < ranges.len() && is_spoken_prefix(&flat[ranges[index].0..ranges[index].1])
        {
            index += 1;
            end = ranges[index].1;
        }
        merged.push((start, end));
        index += 1;
    }
    merged
}

/// A fragment the narrator says as part of the next words.
/// "C." and "It is to C." both continue into "S. Lewis". "12." continues into the step.
fn is_spoken_prefix(sentence: &str) -> bool {
    let trimmed = sentence.trim();
    let bytes = trimmed.as_bytes();
    if bytes.is_empty() {
        return false;
    }
    if (2..=3).contains(&bytes.len())
        && bytes[..bytes.len() - 1].iter().all(u8::is_ascii_digit)
        && bytes[bytes.len() - 1] == b'.'
    {
        let number: u32 = trimmed[..trimmed.len() - 1].parse().unwrap_or(0);
        return (1..=31).contains(&number);
    }
    let chars: Vec<char> = trimmed.chars().collect();
    if chars.len() >= 2
        && chars[chars.len() - 1] == '.'
        && chars[chars.len() - 2].is_ascii_uppercase()
    {
        return chars.len() == 2 || chars[chars.len() - 3].is_whitespace();
    }
    false
}

fn sentence_has_words(text: &str) -> bool {
    text.chars().any(|ch| ch.is_alphanumeric())
}

fn sentence_index(ranges: &[(usize, usize)], pos: usize) -> Option<usize> {
    ranges
        .iter()
        .position(|(start, end)| pos >= *start && pos < *end)
}

fn cut_piece(
    flat: &str,
    piece: &Piece,
    label_at: impl Fn(usize) -> Option<usize>,
) -> Vec<(usize, usize, Option<usize>)> {
    let mut out = Vec::new();
    if piece.flat_end <= piece.flat_start {
        return out;
    }
    let mut cursor = piece.flat_start;
    while cursor < piece.flat_end {
        let label = label_at(cursor);
        let mut next = next_boundary(flat, cursor);
        while next < piece.flat_end && label_at(next) == label {
            next = next_boundary(flat, next);
        }
        let b0 = cursor - piece.flat_start;
        let b1 = next - piece.flat_start;
        out.push((b0, b1, label));
        cursor = next;
    }
    out
}

fn next_boundary(text: &str, pos: usize) -> usize {
    if pos >= text.len() {
        return text.len();
    }
    let mut next = pos + 1;
    while next < text.len() && !text.is_char_boundary(next) {
        next += 1;
    }
    next
}

fn apply_segments(
    doc: &mut Document,
    segs: &HashMap<usize, Vec<(usize, usize, Option<usize>)>>,
    attr: &str,
) {
    fn rewrite(
        doc: &mut Document,
        idx: usize,
        segs: &HashMap<usize, Vec<(usize, usize, Option<usize>)>>,
        attr: &str,
    ) {
        let children = doc.children(idx).to_vec();
        let mut new_children = Vec::new();
        for child in children {
            if doc.is_element(child) {
                rewrite(doc, child, segs, attr);
                new_children.push(child);
            } else if let Some(parts) = segs.get(&child) {
                let text = doc.text(child).to_string();
                for (start, end, label) in parts {
                    if *start >= *end || *end > text.len() {
                        continue;
                    }
                    let slice = text[*start..*end].to_string();
                    if slice.is_empty() {
                        continue;
                    }
                    if let Some(label) = label {
                        let span =
                            doc.create_element("span", vec![(attr.to_string(), label.to_string())]);
                        let text_node = doc.create_text(slice);
                        doc.set_children(span, vec![text_node]);
                        new_children.push(span);
                    } else {
                        new_children.push(doc.create_text(slice));
                    }
                }
            } else {
                new_children.push(child);
            }
        }
        doc.set_children(idx, new_children);
    }
    rewrite(doc, doc.root, segs, attr);
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Own {
    Neutral,
    Blocker,
    Sent(usize),
}

fn ownership(doc: &Document, idx: usize, attr: &str) -> Own {
    match &doc.nodes[idx].data {
        NodeData::Comment(_) => Own::Neutral,
        NodeData::Text(text) => {
            if text.chars().all(|ch| ch.is_whitespace()) {
                Own::Neutral
            } else {
                Own::Blocker
            }
        }
        NodeData::Element { .. } => {
            if let Some(value) = doc.attr(idx, attr) {
                if let Ok(sent) = value.parse::<usize>() {
                    return Own::Sent(sent);
                }
            }
            if let Some(value) = doc.attr(idx, "data-st-own") {
                if let Ok(sent) = value.parse::<usize>() {
                    return Own::Sent(sent);
                }
            }
            let mut found: Option<usize> = None;
            let mut any = false;
            for child in doc.children(idx) {
                match ownership(doc, *child, attr) {
                    Own::Neutral => {}
                    Own::Blocker => return Own::Blocker,
                    Own::Sent(sent) => {
                        any = true;
                        if let Some(previous) = found {
                            if previous != sent {
                                return Own::Blocker;
                            }
                        }
                        found = Some(sent);
                    }
                }
            }
            if any {
                Own::Sent(found.unwrap())
            } else {
                Own::Neutral
            }
        }
    }
}

fn bubble(doc: &mut Document, idx: usize, attr: &str, allow_lift: bool) {
    if !doc.is_element(idx) {
        return;
    }
    for child in doc.children(idx).to_vec() {
        if doc.is_element(child) {
            bubble(doc, child, attr, true);
        }
    }
    regroup(doc, idx, attr);
    if allow_lift && doc.attr(idx, attr).is_none() && !is_block(local_name(doc.name(idx))) {
        lift_inline(doc, idx, attr);
    }
}

fn regroup(doc: &mut Document, idx: usize, attr: &str) {
    let children = doc.children(idx).to_vec();
    if children.is_empty() {
        return;
    }
    let owns: Vec<Own> = children
        .iter()
        .map(|child| ownership(doc, *child, attr))
        .collect();
    let mut runs: Vec<(Option<usize>, Vec<usize>)> = Vec::new();
    let mut pending: Vec<usize> = Vec::new();
    for (child, own) in children.iter().zip(owns) {
        match own {
            Own::Neutral => pending.push(*child),
            Own::Blocker => {
                if !pending.is_empty() {
                    runs.push((None, std::mem::take(&mut pending)));
                }
                runs.push((None, vec![*child]));
            }
            Own::Sent(sent) => {
                if let Some((Some(previous), nodes)) = runs.last_mut() {
                    if *previous == sent {
                        nodes.append(&mut pending);
                        nodes.push(*child);
                        continue;
                    }
                }
                if let Some((_, nodes)) = runs.last_mut() {
                    nodes.append(&mut pending);
                }
                let mut nodes = std::mem::take(&mut pending);
                nodes.push(*child);
                runs.push((Some(sent), nodes));
            }
        }
    }
    if !pending.is_empty() {
        if let Some((_, nodes)) = runs.last_mut() {
            nodes.append(&mut pending);
        } else {
            runs.push((None, pending));
        }
    }
    let mut new_children = Vec::new();
    for (sent, nodes) in runs {
        let Some(sent) = sent else {
            new_children.extend(nodes);
            continue;
        };
        let marker = sent.to_string();
        // A block already holds its own sentence spans. Wrapping the block in
        // another span nests the same sentence and counts its letters twice.
        let wraps_block = nodes
            .iter()
            .any(|node| doc.is_element(*node) && is_block(&local_name(doc.name(*node))));
        if wraps_block {
            new_children.extend(nodes);
            continue;
        }
        let single =
            nodes.len() == 1 && doc.attr(nodes[0], attr).as_deref() == Some(marker.as_str());
        if single {
            new_children.push(nodes[0]);
            continue;
        }
        let mut inner = Vec::new();
        for node in nodes {
            if doc.attr(node, attr).as_deref() == Some(marker.as_str()) {
                inner.extend(doc.take_children(node));
            } else {
                inner.push(node);
            }
        }
        let span = doc.create_element("span", vec![(attr.to_string(), marker)]);
        doc.set_children(span, inner);
        new_children.push(span);
    }
    doc.set_children(idx, new_children);
}

fn lift_inline(doc: &mut Document, idx: usize, attr: &str) {
    let children = doc.children(idx).to_vec();
    let mut spans = Vec::new();
    let mut non_neutral = 0usize;
    for child in &children {
        if matches!(ownership(doc, *child, attr), Own::Neutral) {
            continue;
        }
        non_neutral += 1;
        if doc.attr(*child, attr).is_some() {
            spans.push(*child);
        }
    }
    if spans.len() == 1 && non_neutral == 1 {
        let span = spans[0];
        let Some(sent) = doc.attr(span, attr) else {
            return;
        };
        let inner = doc.take_children(span);
        let mut rebuilt = Vec::new();
        for child in children {
            if child == span {
                rebuilt.extend(inner.clone());
            } else {
                rebuilt.push(child);
            }
        }
        doc.set_children(idx, rebuilt);
        doc.set_attr(idx, "data-st-own", &sent);
    }
}

fn assign_sentence_ids(
    doc: &mut Document,
    idx: usize,
    prefix: &str,
    counts: &mut HashMap<usize, usize>,
) {
    if doc.is_element(idx) {
        if let Some(value) = doc.attr(idx, "data-st-sent") {
            if let Ok(sent) = value.parse::<usize>() {
                let count = counts.entry(sent).or_insert(0);
                let id = if *count == 0 {
                    format!("{prefix}-s{sent}")
                } else {
                    format!("{prefix}-s{sent}-p{count}")
                };
                if *count > 0 {
                    doc.set_attr(idx, "data-s", &format!("{prefix}-s{sent}"));
                }
                *count += 1;
                doc.set_attr(idx, "id", &id);
                doc.set_attr(idx, "epub:type", "storyteller:sentence-span");
                doc.remove_attr(idx, "data-st-sent");
            }
        }
    }
    for child in doc.children(idx).to_vec() {
        assign_sentence_ids(doc, child, prefix, counts);
    }
}

fn sentence_spans(doc: &Document) -> Vec<usize> {
    let mut out = Vec::new();
    let root = doc.root;
    doc.walk(root, &mut |doc, idx| {
        if doc.attr(idx, "epub:type").as_deref() == Some("storyteller:sentence-span") {
            out.push(idx);
        }
    });
    out
}

fn wrap_words(doc: &mut Document, span: usize, counts: &mut HashMap<String, usize>) {
    let mut flat = String::new();
    let mut pieces = Vec::new();
    append_pieces(doc, span, &mut flat, &mut pieces);
    if flat.is_empty() {
        return;
    }
    let mut ranges = Vec::new();
    let mut byte = 0usize;
    let mut word_index = 0usize;
    for token in flat.split_word_bounds() {
        let start = byte;
        byte += token.len();
        let mut letters = String::new();
        push_letters(token, &mut letters);
        if !letters.is_empty() {
            ranges.push((start, byte, word_index));
            word_index += 1;
        }
    }
    if ranges.is_empty() {
        return;
    }
    let mut segs = HashMap::new();
    for piece in &pieces {
        let cuts = cut_piece(&flat, piece, |pos| {
            ranges
                .iter()
                .find(|(start, end, _)| pos >= *start && pos < *end)
                .map(|(_, _, index)| *index)
        });
        segs.insert(piece.node, cuts);
    }
    apply_segments_under(doc, span, &segs, "data-st-word");
    bubble(doc, span, "data-st-word", false);
    strip_attr(doc, span, "data-st-own");
    let sentence_id = canonical_id(doc, span);
    let count = counts.entry(sentence_id.clone()).or_insert(0);
    number_words(doc, span, &sentence_id, count);
}

fn append_pieces(doc: &Document, idx: usize, flat: &mut String, pieces: &mut Vec<Piece>) {
    match &doc.nodes[idx].data {
        NodeData::Text(_) => {}
        NodeData::Comment(_) => {}
        NodeData::Element { .. } => {
            for child in doc.children(idx).to_vec() {
                match &doc.nodes[child].data {
                    NodeData::Text(text) => {
                        let start = flat.len();
                        flat.push_str(text);
                        pieces.push(Piece {
                            node: child,
                            flat_start: start,
                            flat_end: flat.len(),
                        });
                    }
                    NodeData::Element { .. } => append_pieces(doc, child, flat, pieces),
                    NodeData::Comment(_) => {}
                }
            }
        }
    }
}

fn apply_segments_under(
    doc: &mut Document,
    root: usize,
    segs: &HashMap<usize, Vec<(usize, usize, Option<usize>)>>,
    attr: &str,
) {
    fn rewrite(
        doc: &mut Document,
        idx: usize,
        segs: &HashMap<usize, Vec<(usize, usize, Option<usize>)>>,
        attr: &str,
    ) {
        let children = doc.children(idx).to_vec();
        let mut new_children = Vec::new();
        for child in children {
            if doc.is_element(child) {
                rewrite(doc, child, segs, attr);
                new_children.push(child);
            } else if let Some(parts) = segs.get(&child) {
                let text = doc.text(child).to_string();
                for (start, end, label) in parts {
                    if *start >= *end || *end > text.len() {
                        continue;
                    }
                    let slice = text[*start..*end].to_string();
                    if slice.is_empty() {
                        continue;
                    }
                    if let Some(label) = label {
                        let span =
                            doc.create_element("span", vec![(attr.to_string(), label.to_string())]);
                        let text_node = doc.create_text(slice);
                        doc.set_children(span, vec![text_node]);
                        new_children.push(span);
                    } else {
                        new_children.push(doc.create_text(slice));
                    }
                }
            } else {
                new_children.push(child);
            }
        }
        doc.set_children(idx, new_children);
    }
    rewrite(doc, root, segs, attr);
}

fn number_words(doc: &mut Document, idx: usize, sentence_id: &str, count: &mut usize) {
    if doc.attr(idx, "data-st-word").is_some() {
        let id = format!("{sentence_id}w{count}");
        *count += 1;
        doc.set_attr(idx, "id", &id);
        doc.set_attr(idx, "epub:type", "storyteller:word-span");
        doc.remove_attr(idx, "data-st-word");
    }
    for child in doc.children(idx).to_vec() {
        if doc.is_element(child) {
            number_words(doc, child, sentence_id, count);
        }
    }
}

fn strip_attr(doc: &mut Document, idx: usize, key: &str) {
    if doc.is_element(idx) {
        doc.remove_attr(idx, key);
    }
    for child in doc.children(idx).to_vec() {
        strip_attr(doc, child, key);
    }
}

fn collect_text(doc: &Document, word_level: bool) -> ChapterText {
    let spans = sentence_spans(doc);
    let mut letters = String::new();
    let mut sentences = Vec::new();
    let mut index = 0usize;
    while index < spans.len() {
        let canonical = canonical_id(doc, spans[index]);
        let mut visible = String::new();
        while index < spans.len() && canonical_id(doc, spans[index]) == canonical {
            visible.push_str(&visible_text(doc, spans[index]));
            index += 1;
        }
        let start = letters.len();
        push_letters(&visible, &mut letters);
        let end = letters.len();
        let mut words = Vec::new();
        if word_level {
            let mut cursor = start;
            let mut word_index = 0usize;
            for token in visible.split_word_bounds() {
                let mut token_letters = String::new();
                push_letters(token, &mut token_letters);
                if token_letters.is_empty() {
                    continue;
                }
                let range = cursor..cursor + token_letters.len();
                cursor += token_letters.len();
                words.push(WordMark {
                    id: format!("{canonical}w{word_index}"),
                    letter_range: range,
                });
                word_index += 1;
            }
        }
        sentences.push(SentenceMark {
            id: canonical,
            letter_range: start..end,
            words,
        });
    }
    ChapterText { letters, sentences }
}

fn canonical_id(doc: &Document, idx: usize) -> String {
    if let Some(id) = doc.attr(idx, "data-s") {
        return id;
    }
    doc.attr(idx, "id").unwrap_or_else(|| "s".into())
}

fn visible_text(doc: &Document, idx: usize) -> String {
    let mut out = String::new();
    fn walk(doc: &Document, idx: usize, out: &mut String) {
        match &doc.nodes[idx].data {
            NodeData::Text(text) => out.push_str(text),
            NodeData::Comment(_) => {}
            NodeData::Element { .. } => {
                if should_skip(doc, idx)
                    && doc.attr(idx, "epub:type").as_deref() != Some("storyteller:sentence-span")
                {
                    return;
                }
                for child in doc.children(idx) {
                    walk(doc, *child, out);
                }
            }
        }
    }
    walk(doc, idx, &mut out);
    out
}

fn should_skip(doc: &Document, idx: usize) -> bool {
    let name = local_name(doc.name(idx));
    if matches!(name, "script" | "style" | "head" | "svg" | "math") {
        return true;
    }
    if let Some(kind) = doc.attr(idx, "epub:type") {
        let kind = kind.to_ascii_lowercase();
        if kind.split_whitespace().any(|token| {
            matches!(
                token,
                "noteref" | "footnote" | "rearnote" | "pagebreak" | "endnote"
            )
        }) {
            return true;
        }
    }
    false
}

fn is_block(name: &str) -> bool {
    matches!(
        name,
        "p" | "div"
            | "h1"
            | "h2"
            | "h3"
            | "h4"
            | "h5"
            | "h6"
            | "li"
            | "blockquote"
            | "section"
            | "article"
            | "tr"
            | "table"
            | "figcaption"
            | "body"
            | "html"
            | "head"
            | "ul"
            | "ol"
            | "figure"
            | "header"
            | "footer"
            | "nav"
    )
}

fn ensure_epub_namespace(doc: &mut Document) {
    if doc.attr(doc.root, "xmlns:epub").is_none() {
        doc.set_attr(doc.root, "xmlns:epub", "http://www.idpf.org/2007/ops");
    }
}

fn ensure_stylesheet(doc: &mut Document) {
    let heads = doc.elements_named("head");
    let Some(&head) = heads.first() else { return };
    for child in doc.children(head) {
        if local_name(doc.name(*child)) == "link" {
            if doc.attr(*child, "href").as_deref() == Some("Styles/storyteller-readaloud.css") {
                return;
            }
        }
    }
    let link = doc.create_element(
        "link",
        vec![
            ("rel".into(), "stylesheet".into()),
            ("href".into(), "Styles/storyteller-readaloud.css".into()),
            ("type".into(), "text/css".into()),
        ],
    );
    let mut children = doc.children(head).to_vec();
    children.push(link);
    doc.set_children(head, children);
}

fn rewrite_opf(
    book: &Book,
    chapters: &[ChapterOutput],
    smils: &[(String, String, f64)],
    tracks: &[Track],
    total_seconds: f64,
) -> Result<String> {
    let bytes = book.files.get(&book.opf_path).context("opf missing")?;
    let mut doc = Document::parse(std::str::from_utf8(bytes)?)?;
    let version = doc.attr(doc.root, "version").unwrap_or_default();
    if version.starts_with('2') {
        doc.set_attr(doc.root, "version", "3.0");
    }
    let metadata = child_named(&doc, doc.root, "metadata").context("opf metadata")?;
    let manifest = child_named(&doc, doc.root, "manifest").context("opf manifest")?;
    drop_media_metas(&mut doc, metadata);
    set_modified(&mut doc, metadata);
    let mut meta_children = doc.children(metadata).to_vec();
    for (index, chapter) in chapters.iter().enumerate() {
        if chapter.spans.is_empty() {
            continue;
        }
        let overlay_id = format!("{}_overlay", chapter_item_id(&doc, manifest, &chapter.href));
        let duration = smils.get(index).map(|item| item.2).unwrap_or(0.0);
        // smils only contains chapters that have spans, in the same order we pushed them.
        let _ = duration;
        meta_children.push(meta_text(
            &mut doc,
            &[
                ("property".into(), "media:duration".into()),
                ("refines".into(), format!("#{overlay_id}")),
            ],
            &format_clock(chapter_duration(&chapter.spans)),
        ));
    }
    meta_children.push(meta_text(
        &mut doc,
        &[("property".into(), "media:duration".into())],
        &format_clock(total_seconds),
    ));
    meta_children.push(meta_text(
        &mut doc,
        &[("property".into(), "media:active-class".into())],
        "-epub-media-overlay-active",
    ));
    doc.set_children(metadata, meta_children);

    let opf_dir = parent_dir(&book.opf_path);
    let mut manifest_children = doc.children(manifest).to_vec();
    for chapter in chapters {
        if chapter.spans.is_empty() {
            continue;
        }
        let Some(item) = find_item(&doc, manifest, &chapter.href) else {
            continue;
        };
        let overlay_id = format!(
            "{}_overlay",
            doc.attr(item, "id").unwrap_or_else(|| chapter.href.clone())
        );
        doc.set_attr(item, "media-overlay", &overlay_id);
        let smil_href = format!("MediaOverlays/{}", smil_name(&chapter.href));
        if find_item_by_href(&doc, &manifest_children, &smil_href).is_none() {
            manifest_children.push(doc.create_element(
                "item",
                vec![
                    ("id".into(), overlay_id),
                    ("href".into(), url_encode_path(&smil_href)),
                    ("media-type".into(), "application/smil+xml".into()),
                ],
            ));
        }
    }
    for (index, track) in tracks.iter().enumerate() {
        let href = format!("Audio/{}", track.epub_name);
        if find_item_by_href(&doc, &manifest_children, &href).is_none() {
            manifest_children.push(doc.create_element(
                "item",
                vec![
                    ("id".into(), format!("audio-{index}")),
                    ("href".into(), url_encode_path(&href)),
                    ("media-type".into(), track.media_type.clone()),
                ],
            ));
        }
    }
    let css_href = "Styles/storyteller-readaloud.css";
    if find_item_by_href(&doc, &manifest_children, css_href).is_none() {
        manifest_children.push(doc.create_element(
            "item",
            vec![
                ("id".into(), "storyteller_readaloud_styles".into()),
                ("href".into(), css_href.into()),
                ("media-type".into(), "text/css".into()),
            ],
        ));
    }
    doc.set_children(manifest, manifest_children);
    let _ = opf_dir;
    doc.to_xml()
}

fn chapter_item_id(doc: &Document, manifest: usize, href: &str) -> String {
    find_item(doc, manifest, href)
        .and_then(|item| doc.attr(item, "id"))
        .unwrap_or_else(|| href.to_string())
}

fn find_item(doc: &Document, manifest: usize, href: &str) -> Option<usize> {
    for child in doc.children(manifest) {
        if local_name(doc.name(*child)) != "item" {
            continue;
        }
        if doc.attr(*child, "href").as_deref() == Some(href) {
            return Some(*child);
        }
    }
    None
}

fn find_item_by_href(doc: &Document, children: &[usize], href: &str) -> Option<usize> {
    let encoded = url_encode_path(href);
    children.iter().copied().find(|child| {
        let value = doc.attr(*child, "href").unwrap_or_default();
        value == href || value == encoded || url_decode(&value) == href
    })
}

fn drop_media_metas(doc: &mut Document, metadata: usize) {
    let kept: Vec<usize> = doc
        .children(metadata)
        .iter()
        .copied()
        .filter(|child| {
            let property = doc.attr(*child, "property").unwrap_or_default();
            !property.starts_with("media:")
        })
        .collect();
    doc.set_children(metadata, kept);
}

fn set_modified(doc: &mut Document, metadata: usize) {
    let stamp = utc_now();
    for child in doc.children(metadata).to_vec() {
        if doc.attr(child, "property").as_deref() == Some("dcterms:modified") {
            let text = doc.create_text(stamp);
            doc.set_children(child, vec![text]);
            return;
        }
    }
    let meta = meta_text(
        doc,
        &[("property".into(), "dcterms:modified".into())],
        &stamp,
    );
    let mut children = doc.children(metadata).to_vec();
    children.push(meta);
    doc.set_children(metadata, children);
}

fn meta_text(doc: &mut Document, attrs: &[(String, String)], text: &str) -> usize {
    let meta = doc.create_element("meta", attrs.to_vec());
    let node = doc.create_text(text.to_string());
    doc.set_children(meta, vec![node]);
    meta
}

fn href_key(href: &str) -> String {
    let path = href.split('#').next().unwrap_or(href);
    let path = url_decode(path);
    path.rsplit('/').next().unwrap_or(&path).to_string()
}

fn nav_labels(
    files: &BTreeMap<String, Vec<u8>>,
    opf_dir: &str,
    nav_href: &str,
) -> (HashMap<String, String>, HashMap<String, Vec<String>>) {
    let mut toc = HashMap::new();
    let mut landmarks = HashMap::new();
    if nav_href.is_empty() {
        return (toc, landmarks);
    }
    let zip_path = join_path(opf_dir, &url_decode(nav_href));
    let Some(bytes) = files.get(&zip_path) else {
        return (toc, landmarks);
    };
    let Ok(xml) = std::str::from_utf8(bytes) else {
        return (toc, landmarks);
    };
    let Ok(doc) = Document::parse(xml) else {
        return (toc, landmarks);
    };
    for nav in doc.elements_named("nav") {
        let kind = doc.attr(nav, "epub:type").unwrap_or_default();
        let is_toc = kind.split_whitespace().any(|token| token == "toc");
        let is_land = kind.split_whitespace().any(|token| token == "landmarks");
        if !is_toc && !is_land {
            continue;
        }
        doc.walk(nav, &mut |doc, idx| {
            if !doc.is_element(idx) || local_name(doc.name(idx)) != "a" {
                return;
            }
            let Some(href) = doc.attr(idx, "href") else {
                return;
            };
            let key = href_key(&href);
            if key.is_empty() {
                return;
            }
            if is_toc && !toc.contains_key(&key) {
                let title = element_text(doc, idx);
                if !title.trim().is_empty() {
                    toc.insert(
                        key.clone(),
                        title.split_whitespace().collect::<Vec<_>>().join(" "),
                    );
                }
            }
            if is_land {
                if let Some(role) = doc.attr(idx, "epub:type") {
                    let list = landmarks.entry(key).or_default();
                    for token in role.split_whitespace() {
                        if !token.is_empty() {
                            list.push(token.to_string());
                        }
                    }
                }
            }
        });
    }
    (toc, landmarks)
}

fn guide_types(doc: &Document, root: usize) -> HashMap<String, Vec<String>> {
    let mut out = HashMap::new();
    let Some(guide) = child_named(doc, root, "guide") else {
        return out;
    };
    for child in doc.children(guide) {
        if local_name(doc.name(*child)) != "reference" {
            continue;
        }
        let Some(href) = doc.attr(*child, "href") else {
            continue;
        };
        let Some(kind) = doc.attr(*child, "type") else {
            continue;
        };
        out.entry(href_key(&href)).or_default().push(kind);
    }
    out
}

fn content_types_of(xml: &str) -> Vec<String> {
    let Ok(doc) = Document::parse(xml) else {
        return Vec::new();
    };
    let mut found = Vec::new();
    let Some(&body) = doc.elements_named("body").first() else {
        return found;
    };
    if let Some(kind) = doc.attr(body, "epub:type") {
        for token in kind.split_whitespace() {
            if !token.is_empty() {
                found.push(token.to_string());
            }
        }
    }
    let total = text_len(&doc, body);
    let mut typed = HashMap::new();
    outermost_types(&doc, body, &mut typed);
    for (kind, len) in typed {
        if total == 0 || len * 2 >= total {
            if !found.iter().any(|have| have == &kind) {
                found.push(kind);
            }
        }
    }
    found
}

fn outermost_types(doc: &Document, element: usize, into: &mut HashMap<String, usize>) {
    for child in doc.children(element).to_vec() {
        if !doc.is_element(child) {
            continue;
        }
        if let Some(kind) = doc.attr(child, "epub:type") {
            let len = text_len(doc, child);
            for token in kind.split_whitespace() {
                if token.is_empty() {
                    continue;
                }
                *into.entry(token.to_string()).or_default() += len;
            }
        } else {
            outermost_types(doc, child, into);
        }
    }
}

fn text_len(doc: &Document, idx: usize) -> usize {
    let mut count = 0usize;
    doc.walk(idx, &mut |doc, child| {
        count += doc
            .text(child)
            .chars()
            .filter(|ch| !ch.is_whitespace())
            .count();
    });
    count
}

fn element_text(doc: &Document, idx: usize) -> String {
    let mut out = String::new();
    doc.walk(idx, &mut |doc, child| {
        out.push_str(doc.text(child));
    });
    out
}

fn first_heading(xml: &str) -> Option<String> {
    let doc = Document::parse(xml).ok()?;
    let mut found = None;
    doc.walk(doc.root, &mut |doc, idx| {
        if found.is_some() || !doc.is_element(idx) {
            return;
        }
        let name = local_name(doc.name(idx));
        if name == "h1" || name == "h2" || name == "h3" {
            let text = element_text(doc, idx);
            let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
            if !text.is_empty() {
                found = Some(text);
            }
        }
    });
    found
}

fn child_named(doc: &Document, parent: usize, name: &str) -> Option<usize> {
    doc.children(parent)
        .iter()
        .copied()
        .find(|child| local_name(doc.name(*child)) == name)
}

fn render_smil(
    href: &str,
    text_rel: &str,
    spans: &[OverlaySpan],
    tracks: &[Track],
    smil_dir: &str,
    opf_dir: &str,
    windows: Option<&[std::ops::Range<usize>]>,
) -> String {
    let seq_id = format!("{}_overlay", href.rsplit('/').next().unwrap_or(href));
    let mut body = String::new();
    body.push_str(r#"<?xml version="1.0" encoding="UTF-8"?>"#);
    body.push_str(r#"<smil xmlns="http://www.w3.org/ns/SMIL" xmlns:epub="http://www.idpf.org/2007/ops" version="3.0"><body>"#);
    body.push_str(&format!(
        r#"<seq id="{}" epub:textref="{}">"#,
        xml_escape(&seq_id),
        xml_escape(text_rel)
    ));
    let mut pending = Vec::new();
    for (index, span) in spans.iter().enumerate() {
        let view: &[Track] = if let Some(windows) = windows {
            let range = windows.get(index).cloned().unwrap_or(0..0);
            tracks.get(range).unwrap_or(&[])
        } else {
            tracks
        };
        let clips = clip_ranges(view, span.start, span.end);
        if clips.is_empty() {
            continue;
        }
        for (clip_index, (track_index, begin, end)) in clips.iter().enumerate() {
            let track = &view[*track_index];
            let audio_zip = join_path(opf_dir, &format!("Audio/{}", track.epub_name));
            let audio_rel = url_encode_path(&relative_from(smil_dir, &audio_zip));
            let par_id = if clips.len() == 1 {
                format!("{}-{index}", span.id)
            } else {
                format!("{}-{index}-{clip_index}", span.id)
            };
            pending.push(SmilClip {
                text_id: (span.kind != SpanKind::AudioOnly).then(|| span.id.clone()),
                kind: kind_name(span.kind),
                par_id,
                audio: audio_rel,
                begin: *begin,
                end: *end,
            });
        }
    }
    for clip in collapse_audio_only(pending) {
        let begin = format_smil_clock(clip.begin);
        let end = format_smil_clock(clip.end);
        match clip.text_id {
            Some(text_id) => body.push_str(&format!(
                r#"<par id="{id}" epub:type="{kind}"><text src="{src}"/><audio src="{audio}" clipBegin="{begin}" clipEnd="{end}"/></par>"#,
                id = xml_escape(&clip.par_id),
                kind = clip.kind,
                src = xml_escape(&format!("{text_rel}#{text_id}")),
                audio = xml_escape(&clip.audio),
            )),
            None => body.push_str(&format!(
                r#"<par id="{id}"><audio src="{audio}" clipBegin="{begin}" clipEnd="{end}"/></par>"#,
                id = xml_escape(&clip.par_id),
                audio = xml_escape(&clip.audio),
            )),
        }
    }
    body.push_str("</seq></body></smil>");
    body
}

struct SmilClip {
    text_id: Option<String>,
    kind: &'static str,
    par_id: String,
    audio: String,
    begin: f64,
    end: f64,
}

/// One `<text>` per fragment, with clips that abut.
///
/// A silence par repeats the following sentence id. Readers that ignore
/// `epub:type` then highlight that sentence during the pause. The silence
/// before the first sentence is pulled into that sentence, and a later
/// silence stays on the sentence just spoken. That is the contiguous
/// clock a ReadBeyond book uses.
fn collapse_audio_only(clips: Vec<SmilClip>) -> Vec<SmilClip> {
    let mut merged: Vec<SmilClip> = Vec::new();
    for mut clip in clips {
        if clip.text_id.is_none() {
            if let Some(prev) = merged.last_mut() {
                if clips_abut(prev, &clip) {
                    prev.end = clip.end.max(prev.end);
                    continue;
                }
            }
            merged.push(clip);
            continue;
        }
        if let Some(prev) = merged.last_mut() {
            if prev.text_id.is_none() && clips_abut(prev, &clip) {
                clip.begin = prev.begin;
                *prev = clip;
                continue;
            }
        }
        merged.push(clip);
    }
    merged.retain(|clip| clip.text_id.is_some());
    merged
}

fn clips_abut(prev: &SmilClip, next: &SmilClip) -> bool {
    prev.audio == next.audio && (next.begin - prev.end).abs() <= 0.05
}

fn clip_ranges(tracks: &[Track], start: f64, end: f64) -> Vec<(usize, f64, f64)> {
    let timeline_tracks = tracks;
    if end <= start {
        return Vec::new();
    }
    let mut out = Vec::new();
    for (index, track) in timeline_tracks.iter().enumerate() {
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

fn kind_name(kind: SpanKind) -> &'static str {
    match kind {
        SpanKind::Matched => "storyteller:matched",
        SpanKind::Interpolated => "storyteller:interpolated",
        SpanKind::Unmatched => "storyteller:unmatched",
        SpanKind::AudioOnly => "storyteller:audio-only",
    }
}

fn chapter_duration(spans: &[OverlaySpan]) -> f64 {
    let Some(first) = spans.first() else {
        return 0.0;
    };
    let last = spans.last().unwrap();
    (last.end - first.start).max(0.0)
}

pub fn format_clock(seconds: f64) -> String {
    let total = (seconds.max(0.0) * 100.0).round() as u64;
    let centis = total % 100;
    let secs = (total / 100) % 60;
    let mins = (total / 6000) % 60;
    let hours = total / 360000;
    format!("{hours:02}:{mins:02}:{secs:02}.{centis:02}")
}

/// SMIL clip clock in the form ReadBeyond writes: `0:00:13.962`.
/// Hours are not zero-padded. Minutes and seconds are. The fraction is
/// milliseconds. There is no `s` suffix.
fn format_smil_clock(seconds: f64) -> String {
    let millis = (seconds.max(0.0) * 1000.0).round() as u64;
    let ms = millis % 1000;
    let total = millis / 1000;
    let secs = total % 60;
    let mins = (total / 60) % 60;
    let hours = total / 3600;
    format!("{hours}:{mins:02}:{secs:02}.{ms:03}")
}

fn smil_name(href: &str) -> String {
    let base = href.rsplit('/').next().unwrap_or(href);
    let stem = base.rsplit_once('.').map(|(stem, _)| stem).unwrap_or(base);
    format!("{stem}.smil")
}

fn opf_path(files: &BTreeMap<String, Vec<u8>>) -> Result<String> {
    let Some(container) = files.get("META-INF/container.xml") else {
        bail!("EPUB is missing META-INF/container.xml");
    };
    let xml = std::str::from_utf8(container)?;
    let doc = Document::parse(xml)?;
    let mut found = None;
    let root = doc.root;
    doc.walk(root, &mut |doc, idx| {
        if local_name(doc.name(idx)) == "rootfile" {
            if let Some(path) = doc.attr(idx, "full-path") {
                found = Some(path);
            }
        }
    });
    found.ok_or_else(|| anyhow::anyhow!("container.xml has no rootfile"))
}

fn parent_dir(path: &str) -> String {
    match path.rfind('/') {
        Some(index) => path[..index].to_string(),
        None => String::new(),
    }
}

fn join_path(dir: &str, rel: &str) -> String {
    if dir.is_empty() {
        rel.to_string()
    } else {
        format!("{dir}/{rel}")
    }
}

fn relative_from(from_dir: &str, target: &str) -> String {
    let from: Vec<&str> = from_dir
        .split('/')
        .filter(|part| !part.is_empty())
        .collect();
    let to: Vec<&str> = target.split('/').filter(|part| !part.is_empty()).collect();
    let mut common = 0usize;
    while common < from.len() && common < to.len() && from[common] == to[common] {
        common += 1;
    }
    let mut parts = Vec::new();
    for _ in common..from.len() {
        parts.push("..".to_string());
    }
    for part in &to[common..] {
        parts.push((*part).to_string());
    }
    if parts.is_empty() {
        return to.last().copied().unwrap_or("").to_string();
    }
    parts.join("/")
}

pub fn url_encode_path(path: &str) -> String {
    path.split('/')
        .map(url_encode_segment)
        .collect::<Vec<_>>()
        .join("/")
}

fn url_encode_segment(segment: &str) -> String {
    let mut out = String::new();
    for byte in segment.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char)
            }
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

fn url_decode(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut out = Vec::new();
    let mut index = 0usize;
    while index < bytes.len() {
        if bytes[index] == b'%' && index + 2 < bytes.len() {
            if let Ok(byte) = u8::from_str_radix(
                std::str::from_utf8(&bytes[index + 1..index + 3]).unwrap_or(""),
                16,
            ) {
                out.push(byte);
                index += 3;
                continue;
            }
        }
        out.push(bytes[index]);
        index += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn xml_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn utc_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let days = secs.div_euclid(86400);
    let tod = secs.rem_euclid(86400) as u64;
    let hour = tod / 3600;
    let minute = (tod % 3600) / 60;
    let second = tod % 60;
    let (year, month, day) = civil_from_days(days);
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

fn civil_from_days(days: i64) -> (i32, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = y + if month <= 2 { 1 } else { 0 };
    (year as i32, month as u32, day as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wraps_a_sentence_around_inline_tags() {
        let xml = r##"<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops"><head><title>t</title></head><body><p>This <em>is</em> the best. Next sentence.</p><p>Quit playing God.<a epub:type="noteref" href="#n">1</a></p></body></html>"##;
        let (out, text) = markup_xml(xml, "part.xhtml", false).unwrap();
        assert!(out.contains("id=\"part.xhtml-s0\""), "{out}");
        assert!(out.contains("id=\"part.xhtml-s1\""), "{out}");
        assert!(out.contains("<em>is</em>"), "{out}");
        assert!(
            out.contains("epub:type=\"storyteller:sentence-span\""),
            "{out}"
        );
        assert!(text.letters.contains("thisisthebest"), "{:?}", text.letters);
        assert!(
            text.letters.contains("quitplayinggod"),
            "{:?}",
            text.letters
        );
        assert!(!text.letters.contains('1'), "{:?}", text.letters);
        assert!(text.sentences.len() >= 3, "{:?}", text.sentences.len());
    }

    #[test]
    fn sentence_spans_keep_the_last_letter_and_the_terminator() {
        let xml = r##"<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
<head><title>t</title></head>
<body>
<div>
<p><a href="part0003.xhtml#a1VB">FOREWORD</a></p>
<p>MEETING THE HOLY RASCAL</p>
<p>But here is the truth. This <em>is</em> the best book I've ever read.</p>
</div>
</body>
</html>"##;
        let (out, text) = markup_xml(xml, "part0004.xhtml", false).unwrap();
        assert!(out.contains(">FOREWORD<"), "{out}");
        assert!(out.contains(">MEETING THE HOLY RASCAL<"), "{out}");
        assert!(out.contains("the truth."), "{out}");
        assert!(out.contains("<em>is</em>"), "{out}");
        assert!(!out.contains("FOREWO<"), "{out}");
        assert!(!out.contains(">RD<"), "{out}");
        assert!(!out.contains("RASC<"), "{out}");
        assert!(!out.contains(">AL<"), "{out}");
        assert!(!out.contains("trut<"), "{out}");
        assert!(!out.contains(">h."), "{out}");
        assert_eq!(
            text.letters.matches("foreword").count(),
            1,
            "{:?}",
            text.letters
        );
        assert_eq!(
            text.letters.matches("meetingtheholyrascal").count(),
            1,
            "{:?}",
            text.letters
        );
        assert!(!out.contains("sentence-span\"><p"), "{out}");
        assert_eq!(out.matches(">FOREWORD<").count(), 1, "{out}");
        assert!(text.letters.contains("thetruth"), "{:?}", text.letters);
        assert!(text.letters.contains("thisisthebest"), "{:?}", text.letters);
        let truth = text
            .sentences
            .iter()
            .find(|sentence| {
                let slice = &text.letters[sentence.letter_range.clone()];
                slice.contains("truth")
            })
            .expect("truth sentence");
        let slice = &text.letters[truth.letter_range.clone()];
        assert!(
            slice.ends_with("truth") || slice.contains("thetruth"),
            "{slice}"
        );
        assert!(!slice.contains("thisis"), "{slice}");
    }

    #[test]
    fn a_line_break_stays_inside_the_heading() {
        let xml = r##"<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
<head><title>t</title></head>
<body>
<p><span class="k">ADDICTION, SPIRITUALITY,</span><br/><span class="k">AND THE PROCESS</span><br/><span class="k">OF UNCOVERY</span></p>
<p>THE GIFT OF<br/>POWERLESSNESS</p>
<p>1. We admitted we were powerless over alcohol.</p>
<p>It is to C. S. Lewis that people pointed.</p>
<p>Bill W. knew that egotism was the heart.</p>
</body>
</html>"##;
        let (out, text) = markup_xml(xml, "part.xhtml", false).unwrap();
        let slices: Vec<&str> = text
            .sentences
            .iter()
            .map(|sentence| &text.letters[sentence.letter_range.clone()])
            .collect();
        assert_eq!(
            slices,
            vec![
                "addictionspiritualityandtheprocessofuncovery",
                "thegiftofpowerlessness",
                "oneweadmittedwewerepowerlessoveralcohol",
                "itistocslewisthatpeoplepointed",
                "billwknewthategotismwastheheart",
            ],
            "{slices:?}\n{out}"
        );
        assert!(out.contains("ADDICTION, SPIRITUALITY,"), "{out}");
        assert!(
            out.contains("<br/>") || out.contains("<br />") || out.contains("<br"),
            "{out}"
        );
        let heading = out.rsplit("ADDICTION").next().unwrap();
        let close = heading.find("OF UNCOVERY").unwrap();
        let region = &heading[..close];
        assert!(
            !region.contains("storyteller:sentence-span"),
            "the heading was split:\n{out}"
        );
    }

    #[test]
    fn a_year_is_spoken_in_the_letter_string() {
        let mut letters = String::new();
        push_letters("since 2003 when", &mut letters);
        assert_eq!(letters, "sincetwothousandthreewhen", "{letters}");
        let mut ordinal = String::new();
        push_letters("3rd", &mut ordinal);
        assert_eq!(ordinal, "rd", "{ordinal}");
        let mut comma = String::new();
        push_letters("1,003", &mut comma);
        assert_eq!(comma, "onethousandthree", "{comma}");
    }

    #[test]
    fn word_ids_follow_the_sentence() {
        let xml = r#"<?xml version="1.0"?><html xmlns="http://www.w3.org/1999/xhtml"><head></head><body><p>Don't stop.</p></body></html>"#;
        let (out, text) = markup_xml(xml, "c.xhtml", true).unwrap();
        assert!(out.contains("id=\"c.xhtml-s0w0\""), "{out}");
        assert!(text.letters.contains("don't"), "{:?}", text.letters);
        assert!(!text.sentences[0].words.is_empty());
    }

    #[test]
    fn smil_uses_local_clip_times() {
        let tracks = vec![Track {
            source: std::path::PathBuf::from("book.m4b"),
            epub_name: "book.m4b".into(),
            media_type: "audio/mp4".into(),
            start_sample: 0,
            samples: 160_000,
        }];
        let spans = vec![OverlaySpan {
            id: "part.xhtml-s0".into(),
            start: 1.25,
            end: 2.5,
            kind: SpanKind::Matched,
        }];
        let smil = render_smil(
            "part.xhtml",
            "../part.xhtml",
            &spans,
            &tracks,
            "OEBPS/MediaOverlays",
            "OEBPS",
            None,
        );
        assert!(smil.contains("clipBegin=\"0:00:01.250\""), "{smil}");
        assert!(smil.contains("clipEnd=\"0:00:02.500\""), "{smil}");
        assert!(smil.contains("part.xhtml#part.xhtml-s0"), "{smil}");
        assert!(smil.contains("../Audio/book.m4b"), "{smil}");
        assert!(!smil.contains("s\"/>"), "{smil}");
    }

    #[test]
    fn smil_holds_silence_on_one_fragment_per_sentence() {
        let tracks = vec![Track {
            source: std::path::PathBuf::from("book.m4a"),
            epub_name: "part.m4a".into(),
            media_type: "audio/mp4".into(),
            start_sample: 0,
            samples: 160_000,
        }];
        let spans = vec![
            OverlaySpan {
                id: "c.xhtml-s0".into(),
                start: 0.0,
                end: 1.5,
                kind: SpanKind::AudioOnly,
            },
            OverlaySpan {
                id: "c.xhtml-s0".into(),
                start: 1.5,
                end: 2.5,
                kind: SpanKind::Matched,
            },
            OverlaySpan {
                id: "c.xhtml-s1".into(),
                start: 2.5,
                end: 6.0,
                kind: SpanKind::AudioOnly,
            },
            OverlaySpan {
                id: "c.xhtml-s1".into(),
                start: 6.0,
                end: 7.25,
                kind: SpanKind::Matched,
            },
        ];
        let smil = render_smil(
            "c.xhtml",
            "../c.xhtml",
            &spans,
            &tracks,
            "OEBPS/MediaOverlays",
            "OEBPS",
            None,
        );
        assert_eq!(smil.matches("<par ").count(), 2, "{smil}");
        let first = par_text(&smil, "c.xhtml-s0-1");
        assert!(first.contains("clipBegin=\"0:00:00.000\""), "{first}");
        assert!(first.contains("clipEnd=\"0:00:06.000\""), "{first}");
        assert!(first.contains("../c.xhtml#c.xhtml-s0"), "{first}");
        assert!(first.contains("storyteller:matched"), "{first}");
        let second = par_text(&smil, "c.xhtml-s1-3");
        assert!(second.contains("clipBegin=\"0:00:06.000\""), "{second}");
        assert!(second.contains("clipEnd=\"0:00:07.250\""), "{second}");
        assert!(second.contains("../c.xhtml#c.xhtml-s1"), "{second}");
        assert_eq!(smil.matches("#c.xhtml-s0").count(), 1, "{smil}");
        assert_eq!(smil.matches("#c.xhtml-s1").count(), 1, "{smil}");
        assert!(!smil.contains("audio-only"), "{smil}");
        assert!(!smil.contains("s\"/>"), "{smil}");
    }
}
