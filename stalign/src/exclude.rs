//! Skip spine items that are not narrated.
//!
//! The rules follow Storyteller's CTC aligner (`libraries/align`, MIT):
//! structural `epub:type` values, the OPF guide, nav landmarks, and the
//! English title and filename lists. A "review" item (foreword, preface,
//! introduction) is still aligned. Only an "exclude" verdict is skipped.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Verdict {
    Exclude,
    Review,
    Include,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Zone {
    Front,
    Body,
    Back,
}

#[derive(Clone, Copy)]
struct Signal {
    /// Lower wins. Order matches Storyteller's `EXCLUSION_SOURCES`.
    source: u8,
    verdict: Verdict,
    /// Content-type prefers include over exclude. Other sources prefer exclude.
    content_type: bool,
    reason: &'static str,
}

const STRUCTURAL: &[&str] = &[
    "cover",
    "titlepage",
    "halftitlepage",
    "toc",
    "toc-brief",
    "landmarks",
    "page-list",
    "loi",
    "lot",
    "loa",
    "lov",
    "index",
    "glossary",
    "bibliography",
    "copyright-page",
    "colophon",
    "contributors",
    "seriespage",
    "imprint",
    "imprimatur",
    "errata",
    "revision-history",
    "other-credits",
    "footnote",
    "footnotes",
    "endnote",
    "endnotes",
    "rearnote",
    "rearnotes",
];

const SOMETIMES: &[&str] = &[
    "dedication",
    "epigraph",
    "acknowledgments",
    "acknowledgements",
    "foreword",
    "preface",
    "introduction",
    "appendix",
    "afterword",
    "preamble",
    "abstract",
];

const NARRATED: &[&str] = &[
    "chapter",
    "part",
    "volume",
    "division",
    "prologue",
    "epilogue",
    "conclusion",
];

const GUIDE_EXCLUDE: &[&str] = &[
    "cover",
    "title-page",
    "toc",
    "index",
    "glossary",
    "bibliography",
    "colophon",
    "copyright-page",
    "loi",
    "lot",
    "notes",
];

/// Facts collected from the package. Titles are the TOC label or the first heading.
pub struct ChapterFacts<'a> {
    pub index: usize,
    pub spine_len: usize,
    pub href: &'a str,
    pub linear: bool,
    pub nav: bool,
    pub title: Option<&'a str>,
    pub landmarks: &'a [String],
    pub guides: &'a [String],
    pub content_types: &'a [String],
}

/// `Some(reason)` when the chapter should not be aligned.
pub fn skip_reason(facts: &ChapterFacts<'_>) -> Option<String> {
    let zone = spine_zone(facts.index, facts.spine_len);
    let mut signals = Vec::new();
    if !facts.linear {
        signals.push(Signal {
            source: 0,
            verdict: Verdict::Exclude,
            content_type: false,
            reason: "linear",
        });
    }
    if facts.nav {
        signals.push(Signal {
            source: 1,
            verdict: Verdict::Exclude,
            content_type: false,
            reason: "nav",
        });
    }
    for value in facts.landmarks {
        if let Some(signal) = type_signal(3, value, false) {
            signals.push(signal);
        }
    }
    for value in facts.content_types {
        if let Some(signal) = type_signal(4, value, true) {
            signals.push(signal);
        }
    }
    for value in facts.guides {
        let key = value.trim().to_ascii_lowercase();
        if GUIDE_EXCLUDE.iter().any(|item| *item == key) {
            signals.push(Signal {
                source: 5,
                verdict: Verdict::Exclude,
                content_type: false,
                reason: "guide",
            });
        }
    }
    if let Some(title) = facts.title {
        if let Some(signal) = title_signal(title, zone) {
            signals.push(signal);
        }
    }
    if let Some(signal) = filename_signal(facts.href, zone) {
        signals.push(signal);
    }
    let best = signals.into_iter().min_by(|a, b| {
        a.source
            .cmp(&b.source)
            .then_with(|| severity(a).cmp(&severity(b)))
    })?;
    if best.verdict == Verdict::Exclude {
        Some(best.reason.to_string())
    } else {
        None
    }
}

fn severity(signal: &Signal) -> u8 {
    if signal.content_type {
        return match signal.verdict {
            Verdict::Include => 0,
            Verdict::Exclude => 1,
            Verdict::Review => 2,
        };
    }
    match signal.verdict {
        Verdict::Exclude => 0,
        Verdict::Review => 1,
        Verdict::Include => 2,
    }
}

fn spine_zone(index: usize, len: usize) -> Zone {
    if len < 2 {
        return Zone::Body;
    }
    let fraction = index as f64 / (len - 1) as f64;
    let floor = 6.min(len / 2);
    if fraction <= 0.15 || index < floor {
        Zone::Front
    } else if fraction >= 0.85 || index >= len.saturating_sub(floor) {
        Zone::Back
    } else {
        Zone::Body
    }
}

fn type_signal(source: u8, value: &str, content_type: bool) -> Option<Signal> {
    let key = value.trim().to_ascii_lowercase();
    let (verdict, reason) = if key == "frontmatter" || key == "backmatter" {
        (Verdict::Review, "division")
    } else if key == "bodymatter" {
        (Verdict::Include, "bodymatter")
    } else if STRUCTURAL.iter().any(|item| *item == key) {
        (Verdict::Exclude, "structural")
    } else if SOMETIMES.iter().any(|item| *item == key) {
        (Verdict::Review, "sometimes-narrated")
    } else if NARRATED.iter().any(|item| *item == key) {
        (Verdict::Include, "narrated")
    } else {
        return None;
    };
    Some(Signal {
        source,
        verdict,
        content_type,
        reason,
    })
}

fn title_signal(title: &str, zone: Zone) -> Option<Signal> {
    let normalized = normalize_title(title);
    if normalized.is_empty() {
        return None;
    }
    let verdict = title_verdict(&normalized, zone)?;
    Some(Signal {
        source: 6,
        verdict,
        content_type: false,
        reason: "title",
    })
}

pub fn normalize_title(title: &str) -> String {
    let mut out = String::new();
    let mut spaced = false;
    for ch in title.chars() {
        if matches!(ch, '\'' | '\u{2019}' | '\u{2018}' | '\u{02bc}') {
            continue;
        }
        if ch.is_alphanumeric() {
            for lower in ch.to_lowercase() {
                out.push(lower);
            }
            spaced = false;
        } else if !spaced && !out.is_empty() {
            out.push(' ');
            spaced = true;
        }
    }
    while out.ends_with(' ') {
        out.pop();
    }
    out
}

fn title_verdict(title: &str, zone: Zone) -> Option<Verdict> {
    let edge = zone != Zone::Body;
    let front = zone == Zone::Front;
    let back = zone == Zone::Back;
    if exact(
        title,
        &[
            "copyright",
            "copyright page",
            "copyright notice",
            "copyright information",
            "legal notice",
        ],
    ) || exact(title, &["colophon"])
    {
        return Some(Verdict::Exclude);
    }
    if front
        && (exact(title, &["cover", "front cover"])
            || exact(title, &["title page", "half title page"]))
    {
        return Some(Verdict::Exclude);
    }
    if edge && exact(title, &["contents", "table of contents"]) {
        return Some(Verdict::Exclude);
    }
    if edge && (exact(title, &["map", "maps"]) || title.starts_with("map of ")) {
        return Some(Verdict::Exclude);
    }
    if edge
        && (exact(title, &["praise", "advance praise"])
            || title.starts_with("praise for ")
            || title.starts_with("advance praise for "))
    {
        return Some(Verdict::Exclude);
    }
    if edge
        && (title.starts_with("also by ")
            || title.starts_with("other books by ")
            || title.starts_with("more books by ")
            || title == "by the same author")
    {
        return Some(Verdict::Exclude);
    }
    if edge
        && exact(
            title,
            &[
                "about the author",
                "about the authors",
                "meet the author",
                "about the publisher",
            ],
        )
    {
        return Some(Verdict::Exclude);
    }
    if edge
        && (exact(title, &["newsletter", "join our mailing list"])
            || title.starts_with("connect with ")
            || title.starts_with("connect on ")
            || title.starts_with("follow with ")
            || title.starts_with("follow on ")
            || title.starts_with("sign up for ")
            || title.starts_with("sign up to ")
            || title.starts_with("subscribe for ")
            || title.starts_with("subscribe to "))
    {
        return Some(Verdict::Exclude);
    }
    if back
        && (title == "sneak peek"
            || title.starts_with("excerpt from ")
            || title.starts_with("excerpt of ")
            || title.starts_with("an excerpt")
            || title.starts_with("extract from ")
            || title.starts_with("preview of ")
            || exact(title, &["read an excerpt", "a preview", "a sneak peek"]))
    {
        return Some(Verdict::Exclude);
    }
    if exact(
        title,
        &[
            "discussion questions",
            "questions for discussion",
            "reading group guide",
            "readers guide",
            "book club guide",
            "a reading group guide",
        ],
    ) {
        return Some(Verdict::Exclude);
    }
    if back
        && exact(
            title,
            &[
                "notes",
                "endnotes",
                "end notes",
                "source notes",
                "notes and sources",
            ],
        )
    {
        return Some(Verdict::Exclude);
    }
    if back && exact(title, &["index", "general index"]) {
        return Some(Verdict::Exclude);
    }
    if back
        && exact(
            title,
            &[
                "bibliography",
                "selected bibliography",
                "references",
                "works cited",
                "sources",
                "further reading",
            ],
        )
    {
        return Some(Verdict::Exclude);
    }
    if back && title == "glossary" {
        return Some(Verdict::Exclude);
    }
    if back
        && (exact(
            title,
            &[
                "credits",
                "photo credits",
                "image credits",
                "illustration credits",
                "text credits",
                "permissions",
            ],
        ) || title == "a note on the type")
    {
        return Some(Verdict::Exclude);
    }
    if edge
        && exact(
            title,
            &[
                "dedication",
                "epigraph",
                "epigraphs",
                "acknowledgments",
                "acknowledgements",
                "acknowledgment",
                "acknowledgement",
            ],
        )
    {
        return Some(Verdict::Review);
    }
    if edge && exact(title, &["foreword", "preface", "introduction"]) {
        return Some(Verdict::Review);
    }
    if edge
        && (exact(
            title,
            &[
                "authors note",
                "author note",
                "a note from the author",
                "a note by the author",
                "note from the author",
                "note by the author",
            ],
        ))
    {
        return Some(Verdict::Review);
    }
    if back
        && (title == "afterword"
            || title == "appendices"
            || title == "appendix"
            || title.starts_with("appendix "))
    {
        return Some(Verdict::Review);
    }
    None
}

fn exact(title: &str, options: &[&str]) -> bool {
    options.iter().any(|option| *option == title)
}

fn filename_signal(href: &str, zone: Zone) -> Option<Signal> {
    let tokens = filename_tokens(href);
    if tokens.is_empty() {
        return None;
    }
    let phrases: &[(&str, Verdict, u8)] = &[
        ("cover", Verdict::Exclude, 0b101),
        ("titlepage", Verdict::Exclude, 0b101),
        ("title page", Verdict::Exclude, 0b101),
        ("halftitle", Verdict::Exclude, 0b101),
        ("half title", Verdict::Exclude, 0b101),
        ("copyright", Verdict::Exclude, 0b101),
        ("toc", Verdict::Exclude, 0b101),
        ("contents", Verdict::Exclude, 0b101),
        ("index", Verdict::Exclude, 0b100),
        ("glossary", Verdict::Exclude, 0b101),
        ("bibliography", Verdict::Exclude, 0b101),
        ("colophon", Verdict::Exclude, 0b101),
        ("praise", Verdict::Exclude, 0b101),
        ("map", Verdict::Exclude, 0b101),
        ("maps", Verdict::Exclude, 0b101),
        ("about the author", Verdict::Exclude, 0b101),
        ("about author", Verdict::Exclude, 0b101),
        ("author bio", Verdict::Exclude, 0b101),
        ("about the publisher", Verdict::Exclude, 0b101),
        ("also by", Verdict::Exclude, 0b101),
        ("credits", Verdict::Exclude, 0b100),
        ("imprint", Verdict::Exclude, 0b101),
        ("legal", Verdict::Exclude, 0b101),
        ("notice", Verdict::Exclude, 0b001),
        ("endnotes", Verdict::Exclude, 0b100),
        ("footnotes", Verdict::Exclude, 0b100),
        ("dedication", Verdict::Review, 0b101),
        ("foreword", Verdict::Review, 0b101),
        ("preface", Verdict::Review, 0b101),
        ("introduction", Verdict::Review, 0b101),
        ("intro", Verdict::Review, 0b101),
        ("afterword", Verdict::Review, 0b100),
        ("appendix", Verdict::Review, 0b100),
    ];
    let zone_bit = match zone {
        Zone::Front => 0b001,
        Zone::Body => 0b010,
        Zone::Back => 0b100,
    };
    for (phrase, verdict, zones) in phrases {
        if zones & zone_bit == 0 {
            continue;
        }
        if tokens_include(&tokens, phrase) {
            return Some(Signal {
                source: 7,
                verdict: *verdict,
                content_type: false,
                reason: "filename",
            });
        }
    }
    None
}

fn filename_tokens(href: &str) -> Vec<String> {
    let path = href.split('#').next().unwrap_or(href);
    let name = path.rsplit('/').next().unwrap_or(path);
    let stem = name.rsplit_once('.').map(|(stem, _)| stem).unwrap_or(name);
    let mut spaced = String::new();
    let mut prev_lower = false;
    for ch in stem.chars() {
        if ch.is_ascii_uppercase() && prev_lower {
            spaced.push(' ');
        }
        spaced.push(ch);
        prev_lower = ch.is_ascii_lowercase() || ch.is_ascii_digit();
    }
    spaced
        .to_ascii_lowercase()
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(|token| token.to_string())
        .collect()
}

fn tokens_include(tokens: &[String], phrase: &str) -> bool {
    let haystack = format!("-{}-", tokens.join("-"));
    let needle = format!(
        "-{}-",
        phrase.split_whitespace().collect::<Vec<_>>().join("-")
    );
    haystack.contains(&needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts<'a>(
        index: usize,
        len: usize,
        title: Option<&'a str>,
        href: &'a str,
    ) -> ChapterFacts<'a> {
        ChapterFacts {
            index,
            spine_len: len,
            href,
            linear: true,
            nav: false,
            title,
            landmarks: &[],
            guides: &[],
            content_types: &[],
        }
    }

    #[test]
    fn copyright_and_contents_are_skipped_and_the_foreword_is_not() {
        let len = 25;
        assert_eq!(
            skip_reason(&facts(2, len, Some("Copyright"), "part0002.xhtml")).as_deref(),
            Some("title")
        );
        assert_eq!(
            skip_reason(&facts(3, len, Some("Contents"), "part0003.xhtml")).as_deref(),
            Some("title")
        );
        assert_eq!(
            skip_reason(&facts(0, len, Some("Cover"), "part0000.xhtml")).as_deref(),
            Some("title")
        );
        assert!(skip_reason(&facts(4, len, Some("Foreword"), "part0004.xhtml")).is_none());
        assert!(skip_reason(&facts(
            5,
            len,
            Some("Introduction: Addiction, Spirituality, and the Process of Uncovery"),
            "part0005.xhtml"
        ))
        .is_none());
        assert!(skip_reason(&facts(
            6,
            len,
            Some("A Note on Anonymity"),
            "part0006.xhtml"
        ))
        .is_none());
        assert_eq!(
            skip_reason(&facts(21, len, Some("Notes"), "part0021.xhtml")).as_deref(),
            Some("title")
        );
        assert!(skip_reason(&facts(
            7,
            len,
            Some("Chapter One: The Gift of Powerlessness"),
            "part0007.xhtml"
        ))
        .is_none());
    }

    #[test]
    fn a_contents_chapter_in_the_body_is_kept() {
        assert!(skip_reason(&facts(10, 25, Some("Contents"), "chapter.xhtml")).is_none());
    }
}
