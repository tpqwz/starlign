//! Window for the same options as the command line.
//!
//! The aligner runs as a child process, one queued pair at a time. That keeps
//! this window up while the worker loads ONNX Runtime, and Stop can end the
//! current book without taking the window with it.

use anyhow::Result;
use eframe::egui;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};

const AUDIO_EXTS: &[&str] = &["m4b", "m4a", "mp3", "flac", "ogg", "wav", "aac", "mp4"];

pub fn launch() -> Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("starlign")
            .with_inner_size([920.0, 860.0])
            .with_min_inner_size([640.0, 520.0]),
        ..Default::default()
    };
    eframe::run_native(
        "starlign",
        options,
        Box::new(|_cc| {
            let mut window = AlignerWindow::default();
            window.load_settings();
            Ok(Box::new(window))
        }),
    )
    .map_err(|err| anyhow::anyhow!("{err}"))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Align,
    Run,
    Emit,
}

impl Mode {
    fn label(self) -> &'static str {
        match self {
            Mode::Align => "Align",
            Mode::Run => "Full run",
            Mode::Emit => "Emit",
        }
    }

    fn idle_status(self) -> &'static str {
        match self {
            Mode::Emit => "Add an audiobook.",
            Mode::Align | Mode::Run => "Add an audiobook and an EPUB.",
        }
    }

    fn summary(self) -> &'static str {
        match self {
            Mode::Align => "Reads emissions that already exist. This does not decode the audiobook again.",
            Mode::Run => "Decodes each audiobook, writes emissions, and aligns. An existing emissions file is reused.",
            Mode::Emit => "Decodes each audiobook and writes its emissions file. The EPUB column is unused.",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DeviceChoice {
    Auto,
    Cuda,
    Cpu,
}

impl DeviceChoice {
    fn arg(self) -> &'static str {
        match self {
            DeviceChoice::Auto => "auto",
            DeviceChoice::Cuda => "cuda",
            DeviceChoice::Cpu => "cpu",
        }
    }
}

#[derive(Clone, Default)]
struct PairRow {
    audiobook: String,
    epub: String,
}

#[derive(Debug)]
struct Planned {
    row: usize,
    label: String,
    args: Vec<String>,
}

struct Job {
    child: Child,
    lines: Receiver<String>,
}

pub struct AlignerWindow {
    mode: Mode,
    pairs: Vec<PairRow>,
    emissions: String,
    granularity_words: bool,
    gap_seconds: String,
    max_seconds: String,
    device: DeviceChoice,
    model: String,
    download: bool,
    drop_unread: bool,
    reports: String,
    /// Scales text and controls together. 1.0 is the normal size.
    ui_scale: f32,
    /// Align and Emit stay hidden until this is turned on in Settings.
    show_advanced: bool,
    show_settings: bool,
    log: String,
    status: String,
    job: Option<Job>,
    pending: Vec<Planned>,
    running_index: Option<usize>,
    queue_stop: bool,
    queue_failed: usize,
    queue_total: usize,
}

impl Default for AlignerWindow {
    fn default() -> Self {
        Self {
            mode: Mode::Run,
            pairs: Vec::new(),
            emissions: String::new(),
            granularity_words: false,
            gap_seconds: "3".into(),
            max_seconds: String::new(),
            device: DeviceChoice::Auto,
            model: String::new(),
            download: false,
            drop_unread: false,
            reports: String::new(),
            ui_scale: 1.0,
            show_advanced: false,
            show_settings: false,
            log: String::new(),
            status: Mode::Run.idle_status().into(),
            job: None,
            pending: Vec::new(),
            running_index: None,
            queue_stop: false,
            queue_failed: 0,
            queue_total: 0,
        }
    }
}

impl Drop for AlignerWindow {
    fn drop(&mut self) {
        self.save_settings();
        self.stop();
    }
}

impl AlignerWindow {
    fn running(&self) -> bool {
        self.job.is_some()
    }

    fn queued(&self) -> usize {
        self.pairs
            .iter()
            .filter(|row| !row.audiobook.trim().is_empty() || !row.epub.trim().is_empty())
            .count()
    }

    fn run_label(&self) -> String {
        let n = self.queued();
        if n > 1 {
            format!("{} {n} books", self.mode.label())
        } else {
            self.mode.label().to_string()
        }
    }

    fn append(&mut self, line: &str) {
        self.log.push_str(line);
        if !line.ends_with('\n') {
            self.log.push('\n');
        }
        const LIMIT: usize = 200_000;
        if self.log.len() > LIMIT {
            let rest = self.log.len() - LIMIT;
            let cut = self.log[rest..]
                .find('\n')
                .map(|i| rest + i + 1)
                .unwrap_or(rest);
            self.log.drain(..cut);
        }
    }

    fn pump(&mut self) {
        let (incoming, wait) = {
            let Some(job) = self.job.as_mut() else { return };
            let mut incoming = Vec::new();
            while let Ok(line) = job.lines.try_recv() {
                incoming.push(line);
            }
            let wait = job.child.try_wait();
            (incoming, wait)
        };
        for line in incoming {
            self.append(&line);
        }
        match wait {
            Ok(Some(status)) => self.finish_current(status),
            Ok(None) => {}
            Err(err) => {
                self.job = None;
                self.append(&format!("Could not read the aligner: {err}"));
                self.queue_failed += 1;
                if self.queue_stop {
                    self.pending.clear();
                    self.running_index = None;
                    self.status = "Stopped.".into();
                    return;
                }
                self.spawn_next();
            }
        }
    }

    fn finish_current(&mut self, status: ExitStatus) {
        if let Some(mut job) = self.job.take() {
            let _ = job.child.wait();
            self.drain_job(&mut job);
        }
        if self.queue_stop {
            self.pending.clear();
            self.running_index = None;
            self.status = "Stopped.".into();
            self.append("stopped");
            return;
        }
        if !status.success() {
            self.queue_failed += 1;
        }
        self.spawn_next();
    }

    fn drain_job(&mut self, job: &mut Job) {
        let deadline = Instant::now() + Duration::from_millis(500);
        loop {
            match job.lines.recv_timeout(Duration::from_millis(20)) {
                Ok(line) => self.append(&line),
                Err(RecvTimeoutError::Disconnected) => break,
                Err(RecvTimeoutError::Timeout) => {
                    if Instant::now() >= deadline {
                        break;
                    }
                }
            }
        }
    }

    fn stop(&mut self) {
        self.queue_stop = true;
        self.pending.clear();
        self.running_index = None;
        let Some(mut job) = self.job.take() else {
            return;
        };
        let _ = job.child.kill();
        let _ = job.child.wait();
        self.drain_job(&mut job);
        self.status = "Stopped.".into();
        self.append("stopped");
    }

    fn start(&mut self) {
        if self.running() {
            return;
        }
        let planned = match plan_queue(
            self.mode,
            &self.pairs,
            &self.emissions,
            self.granularity_words,
            &self.gap_seconds,
            &self.max_seconds,
            self.device,
            &self.model,
            self.download,
            self.drop_unread,
            &self.reports,
        ) {
            Ok(planned) => planned,
            Err(err) => {
                self.status = err.clone();
                self.append(&err);
                return;
            }
        };
        self.log.clear();
        self.queue_stop = false;
        self.queue_failed = 0;
        self.queue_total = planned.len();
        self.pending = planned;
        self.spawn_next();
    }

    fn spawn_next(&mut self) {
        loop {
            if self.queue_stop {
                self.pending.clear();
                self.running_index = None;
                self.status = "Stopped.".into();
                return;
            }
            if self.pending.is_empty() {
                self.running_index = None;
                self.status = if self.queue_failed == 0 {
                    if self.queue_total == 1 {
                        "Finished.".into()
                    } else {
                        format!("Finished {} books.", self.queue_total)
                    }
                } else {
                    format!(
                        "Finished. {} of {} failed.",
                        self.queue_failed, self.queue_total
                    )
                };
                return;
            }
            let planned = self.pending.remove(0);
            let started = self.queue_total - self.pending.len();
            if started > 1 {
                self.append("");
            }
            self.append(&format!(
                "Row {} of {}: {}",
                planned.row + 1,
                self.queue_total,
                planned.label
            ));
            let exe = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("starlign"));
            match spawn(&exe, &planned.args) {
                Ok(job) => {
                    self.append(&format!("$ {}", display_command(&planned.args)));
                    self.running_index = Some(planned.row);
                    self.status = format!(
                        "Running {} of {}: {}",
                        started, self.queue_total, planned.label
                    );
                    self.job = Some(job);
                    return;
                }
                Err(err) => {
                    let message = format!("Could not start the aligner: {err}");
                    self.append(&message);
                    self.queue_failed += 1;
                    self.running_index = None;
                }
            }
        }
    }

    fn add_pair_from_dialog(&mut self) {
        let audio_seed = self
            .pairs
            .iter()
            .rev()
            .find_map(|row| nonempty(&row.audiobook))
            .unwrap_or("")
            .to_string();
        let Some(audio) = pick_path("Audiobook", &[("Audio", AUDIO_EXTS)], &audio_seed, false)
        else {
            return;
        };
        let mut row = PairRow {
            audiobook: audio.display().to_string(),
            epub: String::new(),
        };
        if self.mode != Mode::Emit {
            let epub_seed = self
                .pairs
                .iter()
                .rev()
                .find_map(|row| nonempty(&row.epub))
                .unwrap_or(row.audiobook.as_str())
                .to_string();
            if let Some(epub) = pick_path("EPUB", &[("EPUB", &["epub"])], &epub_seed, false) {
                row.epub = epub.display().to_string();
            }
        }
        self.pairs.push(row);
    }
}

impl eframe::App for AlignerWindow {
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.set_zoom_factor(self.ui_scale);
        self.pump();
        if self.running() {
            ctx.request_repaint_after(Duration::from_millis(100));
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let busy = self.running();

        egui::CentralPanel::default().show(ui, |ui| {
            self.header(ui);
            ui.add_space(6.0);

            // The form stays as short as its cards. Extra window height goes to the log.
            let action_h = 36.0;
            let log_min = 120.0;
            let form_max = (ui.available_height() - action_h - log_min - 16.0).max(160.0);
            egui::ScrollArea::vertical()
                .id_salt("form")
                .max_height(form_max)
                .auto_shrink([false, true])
                .show(ui, |ui| {
                    if self.show_settings {
                        self.settings_ui(ui, busy);
                    } else {
                        self.queue_ui(ui, busy);
                        ui.add_space(8.0);
                        egui::Frame::group(ui.style()).show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.strong("Options");
                                tip_button(ui, "These choices apply to every row in the queue.");
                            });
                            ui.add_space(4.0);
                            self.options(ui, busy);
                        });
                    }
                });

            ui.add_space(8.0);
            ui.horizontal(|ui| {
                let run = egui::Button::new(self.run_label()).fill(ui.visuals().selection.bg_fill);
                if ui.add_enabled(!busy, run).clicked() {
                    self.start();
                }
                if ui.add_enabled(busy, egui::Button::new("Stop")).clicked() {
                    self.stop();
                }
                ui.label(egui::RichText::new(&self.status).weak());
            });

            ui.add_space(6.0);
            ui.separator();
            ui.strong("Log");
            egui::ScrollArea::vertical()
                .id_salt("log")
                .stick_to_bottom(true)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.add(
                        egui::Label::new(egui::RichText::new(&self.log).monospace())
                            .selectable(true)
                            .wrap(),
                    );
                });
        });
    }
}

impl AlignerWindow {
    fn header(&mut self, ui: &mut egui::Ui) {
        let busy = self.running();
        ui.horizontal(|ui| {
            ui.heading("starlign");
            tip_button(
                ui,
                &format!(
                    "{}\n\nA book that fails is logged, and the next one starts. Stop ends the current book and leaves the rest.",
                    self.mode.summary()
                ),
            );
            ui.add_space(8.0);
            if self.show_advanced {
                for mode in [Mode::Run, Mode::Align, Mode::Emit] {
                    if ui
                        .selectable_label(self.mode == mode, mode.label())
                        .clicked()
                        && !busy
                    {
                        let idle = self.status == Mode::Align.idle_status()
                            || self.status == Mode::Emit.idle_status()
                            || self.status == Mode::Run.idle_status();
                        self.mode = mode;
                        if idle {
                            self.status = mode.idle_status().into();
                        }
                    }
                }
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .selectable_label(self.show_settings, "Settings")
                    .clicked()
                {
                    self.show_settings = !self.show_settings;
                }
            });
        });
    }
}

enum RowAction {
    Remove(usize),
    Up(usize),
    Down(usize),
}

impl AlignerWindow {
    fn queue_ui(&mut self, ui: &mut egui::Ui, busy: bool) {
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.strong("Queue");
                tip_button(
                    ui,
                    if self.mode == Mode::Emit {
                        "Each row is one audiobook. Add pair chooses the file. Add row leaves a blank path to type."
                    } else {
                        "Each row is one audiobook and one EPUB. Add pair chooses the files. Add row leaves blank paths to type. Up and Down change the order."
                    },
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .add_enabled(!busy, egui::Button::new("Add row"))
                        .clicked()
                    {
                        self.pairs.push(PairRow::default());
                    }
                    if ui
                        .add_enabled(!busy, egui::Button::new("Add pair…"))
                        .clicked()
                    {
                        self.add_pair_from_dialog();
                    }
                });
            });
            ui.add_space(4.0);

            let mut action = None;
            let count = self.pairs.len();
            let queued = self.queued();
            if count == 0 {
                ui.weak("The queue is empty.");
            }
            for i in 0..count {
                if self.running_index == Some(i) {
                    ui.scroll_to_cursor(Some(egui::Align::Center));
                }
                let preview = row_preview(self.mode, &self.pairs[i], &self.emissions, queued.max(1));
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        ui.label(format!("{}", i + 1));
                        if self.running_index == Some(i) {
                            ui.strong("Running");
                        }
                        if let Some(preview) = &preview {
                            tip_button(ui, preview);
                        }
                        if ui
                            .add_enabled(!busy && i > 0, egui::Button::new("Up").small())
                            .clicked()
                        {
                            action = Some(RowAction::Up(i));
                        }
                        if ui
                            .add_enabled(!busy && i + 1 < count, egui::Button::new("Down").small())
                            .clicked()
                        {
                            action = Some(RowAction::Down(i));
                        }
                        if ui
                            .add_enabled(!busy, egui::Button::new("Remove").small())
                            .clicked()
                        {
                            action = Some(RowAction::Remove(i));
                        }
                    });
                    let _ = path_row(
                        ui,
                        "Audio",
                        &mut self.pairs[i].audiobook,
                        busy,
                        "Audiobook",
                        &[("Audio", AUDIO_EXTS)],
                        false,
                        false,
                        None,
                    );
                    if self.mode != Mode::Emit {
                        let _ = path_row(
                            ui,
                            "EPUB",
                            &mut self.pairs[i].epub,
                            busy,
                            "EPUB",
                            &[("EPUB", &["epub"])],
                            false,
                            false,
                            None,
                        );
                    }
                });
            }

            match action {
                Some(RowAction::Remove(i)) => {
                    self.pairs.remove(i);
                }
                Some(RowAction::Up(i)) if i > 0 => self.pairs.swap(i, i - 1),
                Some(RowAction::Down(i)) if i + 1 < self.pairs.len() => self.pairs.swap(i, i + 1),
                _ => {}
            }

            ui.add_space(6.0);
            let save_emissions = self.mode != Mode::Align;
            let emissions_tip = emissions_tip(self.mode);
            let _ = path_row(
                ui,
                "Emissions",
                &mut self.emissions,
                busy,
                "Emissions",
                &[("Emissions", &["emissions"])],
                save_emissions,
                true,
                Some(&emissions_tip),
            );
        });
    }

    fn settings_ui(&mut self, ui: &mut egui::Ui, busy: bool) {
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.strong("Settings");
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                ui.label("Size");
                tip_button(
                    ui,
                    "Scales text and controls together. 100% is the normal size. The choice is remembered.",
                );
                let slider_width = (ui.available_width() - 52.0).max(80.0);
                ui.spacing_mut().slider_width = slider_width;
                let response = ui.add(
                    egui::Slider::new(&mut self.ui_scale, 0.75..=1.75)
                        .step_by(0.05)
                        .show_value(false),
                );
                ui.label(format!("{:.0}%", self.ui_scale * 100.0));
                if response.changed() {
                    self.ui_scale = self.ui_scale.clamp(0.75, 1.75);
                    self.save_settings();
                }
            });
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.add_enabled_ui(!busy, |ui| {
                    if ui
                        .checkbox(&mut self.show_advanced, "Show Align and Emit")
                        .changed()
                    {
                        if !self.show_advanced && self.mode != Mode::Run {
                            let idle = self.status == Mode::Align.idle_status()
                                || self.status == Mode::Emit.idle_status()
                                || self.status == Mode::Run.idle_status();
                            self.mode = Mode::Run;
                            if idle {
                                self.status = Mode::Run.idle_status().into();
                            }
                        }
                        self.save_settings();
                    }
                });
                tip_button(
                    ui,
                    "Full run is the usual job. It decodes the audiobook and writes the aligned EPUB. Align reuses an emissions file that already exists. Emit only writes that file. Turning this off returns the window to Full run.",
                );
            });
        });
    }

    fn load_settings(&mut self) {
        let Ok(text) = std::fs::read_to_string(settings_path()) else {
            return;
        };
        let (scale, advanced) = decode_settings(&text);
        self.ui_scale = scale;
        self.show_advanced = advanced;
    }

    fn save_settings(&self) {
        let path = settings_path();
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(path, encode_settings(self.ui_scale, self.show_advanced));
    }

    fn options(&mut self, ui: &mut egui::Ui, busy: bool) {
        ui.add_enabled_ui(!busy, |ui| {
            if self.mode != Mode::Emit {
                let wide = ui.available_width() > 760.0;
                ui.horizontal(|ui| {
                    ui.label("Granularity");
                    ui.selectable_value(&mut self.granularity_words, false, "Sentence");
                    ui.selectable_value(&mut self.granularity_words, true, "Word");
                    tip_button(
                        ui,
                        "Sentence is the usual highlight. Word highlights each word inside the sentence.",
                    );
                    if wide {
                        ui.add_space(16.0);
                        gap_field(ui, &mut self.gap_seconds);
                        ui.add_space(16.0);
                        max_field(ui, &mut self.max_seconds);
                    }
                });
                if !wide {
                    ui.horizontal(|ui| {
                        gap_field(ui, &mut self.gap_seconds);
                        ui.add_space(16.0);
                        max_field(ui, &mut self.max_seconds);
                    });
                }
                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.drop_unread, "Drop unplaced sentences");
                    tip_button(
                        ui,
                        "Leave sentences the aligner could not place out of the overlay. Off keeps them in, timed between their neighbors.",
                    );
                });
                let _ = path_row(
                    ui,
                    "Report",
                    &mut self.reports,
                    false,
                    "JSON report",
                    &[("JSON", &["json"])],
                    true,
                    true,
                    Some("Optional JSON report. One book writes a file. A queue needs a folder, with one JSON per audiobook."),
                );
            }
            if self.mode != Mode::Align {
                ui.horizontal(|ui| {
                    ui.label("Device");
                    ui.selectable_value(&mut self.device, DeviceChoice::Auto, "Auto");
                    ui.selectable_value(&mut self.device, DeviceChoice::Cuda, "CUDA");
                    ui.selectable_value(&mut self.device, DeviceChoice::Cpu, "CPU");
                    tip_button(
                        ui,
                        "Auto uses CUDA when the runtime can load it, and CPU otherwise. CUDA and CPU force that device.",
                    );
                });
                let _ = path_row(
                    ui,
                    "Model",
                    &mut self.model,
                    false,
                    "ONNX model",
                    &[("ONNX", &["onnx"])],
                    false,
                    false,
                    Some("Optional ONNX file. Empty uses a cached MMS-300M model, or models/mms-fp16.onnx."),
                );
                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.download, "Download the model if it is missing");
                    tip_button(
                        ui,
                        "Downloads the MMS-300M fp16 ONNX weights when the model file is missing. The weights are CC-BY-NC 4.0.",
                    );
                });
            }
        });
    }
}

/// Small "i" control. Hover shows the tip. Click keeps it open.
fn tip_button(ui: &mut egui::Ui, text: &str) {
    let response = ui
        .add(
            egui::Button::new(egui::RichText::new("i").small().strong())
                .small()
                .min_size(egui::vec2(18.0, 18.0))
                .corner_radius(egui::CornerRadius::same(9)),
        )
        .on_hover_text(text);
    egui::Popup::from_toggle_button_response(&response)
        .width(320.0)
        .gap(4.0)
        .show(|ui| {
            ui.set_max_width(300.0);
            ui.label(text);
        });
}

fn gap_field(ui: &mut egui::Ui, gap_seconds: &mut String) {
    ui.label("Gap seconds");
    ui.add(egui::TextEdit::singleline(gap_seconds).desired_width(64.0));
    tip_button(
        ui,
        "A pause longer than this stays dark. A shorter pause extends the previous highlight. 3 seconds is the usual value. 0 keeps every gap.",
    );
}

fn max_field(ui: &mut egui::Ui, max_seconds: &mut String) {
    ui.label("Max seconds");
    ui.add(
        egui::TextEdit::singleline(max_seconds)
            .desired_width(80.0)
            .hint_text("whole file"),
    );
    tip_button(
        ui,
        "Only the first N seconds of each audiobook. Empty uses the whole file.",
    );
}

fn settings_path() -> PathBuf {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))
        .unwrap_or_else(std::env::temp_dir);
    base.join("starlign").join("ui.txt")
}

fn encode_settings(scale: f32, advanced: bool) -> String {
    format!(
        "ui_scale {scale:.2}\nshow_advanced {}\n",
        if advanced { "true" } else { "false" }
    )
}

fn decode_settings(text: &str) -> (f32, bool) {
    let mut scale = 1.0;
    let mut advanced = false;
    for line in text.lines() {
        let mut parts = line.split_whitespace();
        match (parts.next(), parts.next()) {
            (Some("ui_scale"), Some(value)) => {
                if let Ok(parsed) = value.parse::<f32>() {
                    scale = ((parsed * 20.0).round() / 20.0).clamp(0.75, 1.75);
                }
            }
            (Some("show_advanced"), Some("true")) => advanced = true,
            (Some("show_advanced"), Some("false")) => advanced = false,
            _ => {}
        }
    }
    (scale, advanced)
}

fn emissions_tip(mode: Mode) -> String {
    let mut tip = emissions_hint(mode).to_string();
    if mode != Mode::Emit {
        tip.push_str(" Each aligned EPUB is written next to its source as Name-aligned.epub.");
    }
    tip
}

fn emissions_hint(mode: Mode) -> &'static str {
    match mode {
        Mode::Align => {
            "One book: choose that book's emissions file. A queue: choose a folder of Name.emissions files, one per audiobook."
        }
        Mode::Run => {
            "Optional. A file for one book, or a folder for a queue. Empty writes Name.emissions beside each audiobook and reuses it when it is already there."
        }
        Mode::Emit => "Optional folder. Empty writes Name.emissions beside each audiobook.",
    }
}

fn row_preview(mode: Mode, row: &PairRow, emissions: &str, count: usize) -> Option<String> {
    let audio_text = row.audiobook.trim();
    if audio_text.is_empty() {
        return None;
    }
    let audio = Path::new(audio_text);
    match mode {
        Mode::Emit => match emissions_dest(audio, emissions, count) {
            Ok(path) => Some(format!("Writes {}", path.display())),
            Err(err) => Some(err),
        },
        Mode::Align | Mode::Run => {
            let epub_text = row.epub.trim();
            if epub_text.is_empty() {
                return None;
            }
            let output = match output_epub(Path::new(epub_text)) {
                Ok(path) => path,
                Err(err) => return Some(err),
            };
            let mut line = format!("Writes {}", output.display());
            if mode == Mode::Align && emissions.trim().is_empty() {
                line.push_str(". Choose an emissions file, or a folder of them.");
                return Some(line);
            }
            match emissions_dest(audio, emissions, count) {
                Ok(path) if mode == Mode::Run => {
                    line.push_str(&format!(". Emissions {}", path.display()));
                    Some(line)
                }
                Ok(path) => {
                    line.push_str(&format!(". Reads {}", path.display()));
                    Some(line)
                }
                Err(err) => Some(err),
            }
        }
    }
}

fn plan_queue(
    mode: Mode,
    pairs: &[PairRow],
    emissions: &str,
    granularity_words: bool,
    gap_seconds: &str,
    max_seconds: &str,
    device: DeviceChoice,
    model: &str,
    download: bool,
    drop_unread: bool,
    reports: &str,
) -> Result<Vec<Planned>, String> {
    let rows: Vec<(usize, &PairRow)> = pairs
        .iter()
        .enumerate()
        .filter(|(_, row)| !row.audiobook.trim().is_empty() || !row.epub.trim().is_empty())
        .collect();
    if rows.is_empty() {
        return Err(match mode {
            Mode::Emit => "Add an audiobook.".into(),
            Mode::Align | Mode::Run => "Add an audiobook and an EPUB.".into(),
        });
    }
    let gap = gap_seconds
        .trim()
        .parse::<f64>()
        .map_err(|_| "Gap seconds must be a number.".to_string())?;
    if gap < 0.0 {
        return Err("Gap seconds cannot be negative.".into());
    }
    let max = parse_optional("Max seconds", max_seconds)?;
    let mut ready = Vec::new();
    for (row, pair) in rows {
        let audio = require_file(&pair.audiobook, "an audiobook")
            .map_err(|err| format!("Row {}: {err}", row + 1))?;
        let epub = if mode == Mode::Emit {
            None
        } else {
            Some(
                require_file(&pair.epub, "an EPUB")
                    .map_err(|err| format!("Row {}: {err}", row + 1))?,
            )
        };
        ready.push((row, audio, epub));
    }
    let count = ready.len();
    if count > 1 && !emissions.trim().is_empty() && !absolute(Path::new(emissions.trim())).is_dir()
    {
        return Err(
            "One emissions file cannot cover more than one audiobook. Choose a folder of Name.emissions files."
                .into(),
        );
    }
    if mode == Mode::Align && emissions.trim().is_empty() {
        return Err("Choose an emissions file, or a folder of Name.emissions files.".into());
    }
    if count > 1 {
        if let Some(field) = nonempty(reports) {
            if !absolute(Path::new(field)).is_dir() {
                return Err(
                    "Choose a folder for the reports when the queue has more than one book.".into(),
                );
            }
        }
    }

    let mut planned = Vec::with_capacity(count);
    let mut inputs = Vec::new();
    let mut written = Vec::new();
    for (row, audio, epub) in ready {
        if let Some(epub) = epub {
            let output = output_epub(&epub)?;
            let emissions_path = emissions_dest(&audio, emissions, count)?;
            if mode == Mode::Align && !emissions_path.is_file() {
                return Err(format!(
                    "Row {}: could not find emissions: {}",
                    row + 1,
                    emissions_path.display()
                ));
            }
            if emissions_path == audio
                || emissions_path == epub
                || output == audio
                || output == epub
            {
                return Err(format!(
                    "Row {}: the output would overwrite an input.",
                    row + 1
                ));
            }
            let report = report_dest(&audio, reports, count)?;
            inputs.push(audio.clone());
            inputs.push(epub.clone());
            written.push((row, output.clone()));
            if mode == Mode::Run {
                written.push((row, emissions_path.clone()));
            }
            if let Some(path) = &report {
                written.push((row, path.clone()));
            }
            let mut args = Vec::new();
            args.push(if mode == Mode::Align { "align" } else { "run" }.into());
            push_pair(&mut args, "--audiobook", &audio);
            push_pair(&mut args, "--epub", &epub);
            push_pair(&mut args, "--output", &output);
            push_pair(&mut args, "--emissions", &emissions_path);
            if mode == Mode::Run {
                args.push("--device".into());
                args.push(device.arg().into());
                push_model(&mut args, model);
            }
            push_align_flags(
                &mut args,
                granularity_words,
                gap,
                max.as_deref(),
                drop_unread,
                report.as_deref(),
            )?;
            if mode == Mode::Run && download {
                args.push("--download".into());
            }
            planned.push(Planned {
                row,
                label: file_label(&audio),
                args,
            });
        } else {
            let emissions_path = emissions_dest(&audio, emissions, count)?;
            if emissions_path == audio {
                return Err(format!(
                    "Row {}: the emissions path is the same as the audiobook.",
                    row + 1
                ));
            }
            inputs.push(audio.clone());
            written.push((row, emissions_path.clone()));
            let mut args = Vec::new();
            args.push("emit".into());
            push_pair(&mut args, "--audiobook", &audio);
            push_pair(&mut args, "--output", &emissions_path);
            args.push("--device".into());
            args.push(device.arg().into());
            push_model(&mut args, model);
            if let Some(seconds) = &max {
                args.push("--max-seconds".into());
                args.push(seconds.clone());
            }
            if download {
                args.push("--download".into());
            }
            planned.push(Planned {
                row,
                label: file_label(&audio),
                args,
            });
        }
    }
    unique_writes(&written)?;
    for (row, path) in &written {
        if inputs.iter().any(|input| input == path) {
            return Err(format!(
                "Row {} would overwrite an input file: {}",
                row + 1,
                path.display()
            ));
        }
    }
    Ok(planned)
}

fn unique_writes(written: &[(usize, PathBuf)]) -> Result<(), String> {
    for (i, (row, path)) in written.iter().enumerate() {
        if let Some((other, _)) = written.iter().take(i).find(|(_, earlier)| earlier == path) {
            return Err(format!(
                "Rows {} and {} would write the same file: {}",
                other + 1,
                row + 1,
                path.display()
            ));
        }
    }
    Ok(())
}

fn file_label(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

fn output_epub(epub: &Path) -> Result<PathBuf, String> {
    let epub = absolute(epub);
    let stem = file_stem(&epub)?;
    Ok(epub.with_file_name(format!("{stem}-aligned.epub")))
}

fn emissions_dest(audio: &Path, field: &str, count: usize) -> Result<PathBuf, String> {
    let audio = absolute(audio);
    let stem = file_stem(&audio)?;
    let name = format!("{stem}.emissions");
    let field = field.trim();
    if field.is_empty() {
        return Ok(audio.with_file_name(name));
    }
    let path = absolute(Path::new(field));
    if path.is_dir() {
        return Ok(path.join(name));
    }
    if count > 1 {
        return Err(
            "One emissions file cannot cover more than one audiobook. Choose a folder of Name.emissions files."
                .into(),
        );
    }
    if path == audio {
        return Err("The emissions path is the same as the audiobook.".into());
    }
    Ok(path)
}

fn report_dest(audio: &Path, field: &str, count: usize) -> Result<Option<PathBuf>, String> {
    let Some(field) = nonempty(field) else {
        return Ok(None);
    };
    let audio = absolute(audio);
    let stem = file_stem(&audio)?;
    let path = absolute(Path::new(field));
    if count > 1 {
        if !path.is_dir() {
            return Err(
                "Choose a folder for the reports when the queue has more than one book.".into(),
            );
        }
        return Ok(Some(path.join(format!("{stem}.json"))));
    }
    Ok(Some(path))
}

fn file_stem(path: &Path) -> Result<String, String> {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.is_empty())
        .map(str::to_string)
        .ok_or_else(|| format!("Could not use the file name: {}", path.display()))
}

fn push_align_flags(
    args: &mut Vec<String>,
    granularity_words: bool,
    gap: f64,
    max: Option<&str>,
    drop_unread: bool,
    reports: Option<&Path>,
) -> Result<(), String> {
    args.push("--granularity".into());
    args.push(
        if granularity_words {
            "word"
        } else {
            "sentence"
        }
        .into(),
    );
    args.push("--gap-seconds".into());
    args.push(gap.to_string());
    if let Some(seconds) = max {
        args.push("--max-seconds".into());
        args.push(seconds.to_string());
    }
    if drop_unread {
        args.push("--drop-unread".into());
    }
    if let Some(path) = reports {
        push_pair(args, "--reports", path);
    }
    Ok(())
}

fn push_model(args: &mut Vec<String>, model: &str) {
    if let Some(model) = nonempty(model) {
        push_pair(args, "--model", &absolute(Path::new(model)));
    }
}

fn parse_optional(label: &str, text: &str) -> Result<Option<String>, String> {
    let text = text.trim();
    if text.is_empty() {
        return Ok(None);
    }
    let value: f64 = text
        .parse()
        .map_err(|_| format!("{label} must be a number."))?;
    if value < 0.0 {
        return Err(format!("{label} cannot be negative."));
    }
    Ok(Some(value.to_string()))
}

fn require_file(text: &str, what: &str) -> Result<PathBuf, String> {
    let text = nonempty(text).ok_or_else(|| format!("Choose {what}."))?;
    let path = absolute(Path::new(text));
    if !path.is_file() {
        return Err(format!("Could not find {what}: {}", path.display()));
    }
    Ok(path)
}

fn nonempty(text: &str) -> Option<&str> {
    let text = text.trim();
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

fn absolute(path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    }
}

fn push_pair(args: &mut Vec<String>, flag: &str, path: &Path) {
    args.push(flag.into());
    args.push(path.display().to_string());
}

fn display_command(args: &[String]) -> String {
    args.iter()
        .map(|arg| {
            if arg.chars().any(char::is_whitespace) {
                format!("\"{arg}\"")
            } else {
                arg.clone()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Returns true when the path changed from a button or by typing.
fn path_row(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut String,
    busy: bool,
    title: &str,
    filters: &[(&str, &[&str])],
    save: bool,
    folder: bool,
    tip: Option<&str>,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.allocate_ui_with_layout(
            egui::vec2(128.0, ui.spacing().interact_size.y),
            egui::Layout::left_to_right(egui::Align::Center),
            |ui| {
                ui.label(label);
                if let Some(tip) = tip {
                    tip_button(ui, tip);
                }
            },
        );
        let buttons = if folder { 200.0 } else { 96.0 };
        let edit = egui::TextEdit::singleline(value)
            .desired_width((ui.available_width() - buttons).max(80.0));
        let response = ui.add_enabled(!busy, edit);
        if response.changed() {
            changed = true;
        }
        if ui
            .add_enabled(!busy, egui::Button::new("Browse…"))
            .clicked()
        {
            if let Some(path) = pick_path(title, filters, value, save) {
                *value = path.display().to_string();
                changed = true;
            }
        }
        if folder
            && ui
                .add_enabled(!busy, egui::Button::new("Folder…"))
                .clicked()
        {
            if let Some(path) = pick_folder(title, value) {
                *value = path.display().to_string();
                changed = true;
            }
        }
    });
    changed
}

fn pick_path(
    title: &str,
    filters: &[(&str, &[&str])],
    current: &str,
    save: bool,
) -> Option<PathBuf> {
    let mut dialog = rfd::FileDialog::new().set_title(title);
    for (name, extensions) in filters {
        dialog = dialog.add_filter(*name, extensions);
    }
    dialog = dialog.add_filter("All files", &["*"]);
    let current_path = Path::new(current.trim());
    if let Some(dir) = current_path.parent().filter(|dir| dir.is_dir()) {
        dialog = dialog.set_directory(dir);
    }
    if save {
        if let Some(name) = current_path.file_name() {
            dialog = dialog.set_file_name(name.to_string_lossy());
        }
        dialog.save_file()
    } else {
        dialog.pick_file()
    }
}

fn pick_folder(title: &str, current: &str) -> Option<PathBuf> {
    let mut dialog = rfd::FileDialog::new().set_title(title);
    let current_path = Path::new(current.trim());
    let dir = if current_path.is_dir() {
        Some(current_path)
    } else {
        current_path.parent().filter(|dir| dir.is_dir())
    };
    if let Some(dir) = dir {
        dialog = dialog.set_directory(dir);
    }
    dialog.pick_folder()
}

fn spawn(exe: &Path, args: &[String]) -> std::io::Result<Job> {
    let mut child = Command::new(exe)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let (tx, lines) = mpsc::channel();
    pipe_to(child.stdout.take(), tx.clone());
    pipe_to(child.stderr.take(), tx);
    Ok(Job { child, lines })
}

fn pipe_to(pipe: Option<impl std::io::Read + Send + 'static>, tx: mpsc::Sender<String>) {
    let Some(pipe) = pipe else { return };
    std::thread::spawn(move || {
        let reader = BufReader::new(pipe);
        for line in reader.lines() {
            match line {
                Ok(line) => {
                    if tx.send(line).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_round_trip_and_ignore_unknown_lines() {
        let text = encode_settings(1.25, true);
        assert_eq!(decode_settings(&text), (1.25, true));
        assert_eq!(
            decode_settings("ui_scale 3\nshow_advanced false\nextra noise\n"),
            (1.75, false)
        );
        assert_eq!(decode_settings("ui_scale 0.1\n"), (0.75, false));
        assert_eq!(decode_settings(""), (1.0, false));
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("stalign-plan-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn touch(path: &Path) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, b"x").unwrap();
    }

    fn pair(audio: &Path, epub: &Path) -> PairRow {
        PairRow {
            audiobook: audio.display().to_string(),
            epub: epub.display().to_string(),
        }
    }

    fn plan(mode: Mode, pairs: &[PairRow], emissions: &str) -> Result<Vec<Planned>, String> {
        plan_queue(
            mode,
            pairs,
            emissions,
            false,
            "3",
            "",
            DeviceChoice::Auto,
            "",
            false,
            false,
            "",
        )
    }

    #[test]
    fn one_book_align_uses_the_emissions_file_and_writes_beside_the_epub() {
        let dir = scratch("one");
        let audio = dir.join("book.m4b");
        let epub = dir.join("book.epub");
        let emissions = dir.join("cache.emissions");
        touch(&audio);
        touch(&epub);
        touch(&emissions);
        let planned = plan(
            Mode::Align,
            &[pair(&audio, &epub)],
            emissions.to_str().unwrap(),
        )
        .unwrap();
        assert_eq!(planned.len(), 1);
        let args = planned[0].args.join(" ");
        assert!(args.starts_with("align "), "{args}");
        assert!(
            args.contains(&format!(
                "--output {}",
                dir.join("book-aligned.epub").display()
            )),
            "{args}"
        );
        assert!(
            args.contains(&format!("--emissions {}", emissions.display())),
            "{args}"
        );
        assert!(args.contains("--granularity sentence"));
        assert!(args.contains("--gap-seconds 3"));
        assert!(!args.contains("--download"), "{args}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_queue_reads_one_emissions_file_per_audiobook_from_a_folder() {
        let dir = scratch("queue");
        let emissions = dir.join("emissions");
        std::fs::create_dir_all(&emissions).unwrap();
        let mut pairs = Vec::new();
        for name in ["a", "b"] {
            let audio = dir.join(format!("{name}.m4b"));
            let epub = dir.join(format!("{name}.epub"));
            touch(&audio);
            touch(&epub);
            touch(&emissions.join(format!("{name}.emissions")));
            pairs.push(pair(&audio, &epub));
        }
        let planned = plan(Mode::Align, &pairs, emissions.to_str().unwrap()).unwrap();
        assert_eq!(planned.len(), 2);
        assert!(planned[0]
            .args
            .iter()
            .any(|arg| arg.ends_with("a.emissions")));
        assert!(planned[1]
            .args
            .iter()
            .any(|arg| arg.ends_with("b.emissions")));
        assert!(planned[0]
            .args
            .iter()
            .any(|arg| arg.ends_with("a-aligned.epub")));
        assert!(planned[1]
            .args
            .iter()
            .any(|arg| arg.ends_with("b-aligned.epub")));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn one_emissions_file_does_not_cover_two_audiobooks() {
        let dir = scratch("onefile");
        let audio_a = dir.join("a.m4b");
        let audio_b = dir.join("b.m4b");
        let epub_a = dir.join("a.epub");
        let epub_b = dir.join("b.epub");
        let emissions = dir.join("only.emissions");
        for path in [&audio_a, &audio_b, &epub_a, &epub_b, &emissions] {
            touch(path);
        }
        let err = plan(
            Mode::Align,
            &[pair(&audio_a, &epub_a), pair(&audio_b, &epub_b)],
            emissions.to_str().unwrap(),
        )
        .unwrap_err();
        assert!(err.contains("folder"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn two_rows_of_the_same_epub_do_not_write_one_file() {
        let dir = scratch("dup");
        let audio_a = dir.join("a.m4b");
        let audio_b = dir.join("b.m4b");
        let epub = dir.join("same.epub");
        let emissions = dir.join("emissions");
        std::fs::create_dir_all(&emissions).unwrap();
        touch(&audio_a);
        touch(&audio_b);
        touch(&epub);
        touch(&emissions.join("a.emissions"));
        touch(&emissions.join("b.emissions"));
        let err = plan(
            Mode::Align,
            &[pair(&audio_a, &epub), pair(&audio_b, &epub)],
            emissions.to_str().unwrap(),
        )
        .unwrap_err();
        assert!(err.contains("same file"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_blank_row_is_skipped_and_a_missing_epub_names_its_row() {
        let dir = scratch("blank");
        let audio = dir.join("book.m4b");
        let epub = dir.join("book.epub");
        let emissions = dir.join("book.emissions");
        touch(&audio);
        touch(&epub);
        touch(&emissions);
        let pairs = vec![
            PairRow::default(),
            pair(&audio, &epub),
            PairRow {
                audiobook: audio.display().to_string(),
                epub: String::new(),
            },
        ];
        let err = plan(Mode::Align, &pairs, emissions.to_str().unwrap()).unwrap_err();
        assert!(err.starts_with("Row 3:"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn run_without_an_emissions_field_caches_beside_each_audiobook() {
        let dir = scratch("run");
        let mut pairs = Vec::new();
        for name in ["a", "b"] {
            let audio = dir.join(format!("{name}.m4b"));
            let epub = dir.join(format!("{name}.epub"));
            touch(&audio);
            touch(&epub);
            pairs.push(pair(&audio, &epub));
        }
        let planned = plan(Mode::Run, &pairs, "").unwrap();
        assert!(planned[0].args.first().unwrap() == "run");
        assert!(planned[0]
            .args
            .iter()
            .any(|arg| arg.ends_with("a.emissions")));
        assert!(planned[1]
            .args
            .iter()
            .any(|arg| arg.ends_with("b.emissions")));
        assert!(planned[0].args.iter().any(|arg| arg == "--device"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn emit_writes_emissions_and_does_not_pass_an_epub() {
        let dir = scratch("emit");
        let audio = dir.join("book.m4b");
        touch(&audio);
        let pairs = [PairRow {
            audiobook: audio.display().to_string(),
            epub: String::new(),
        }];
        let planned = plan(Mode::Emit, &pairs, "").unwrap();
        let args = planned[0].args.join(" ");
        assert!(args.starts_with("emit "), "{args}");
        assert!(!args.contains("--epub"), "{args}");
        assert!(
            args.contains(&format!(
                "--output {}",
                dir.join("book.emissions").display()
            )),
            "{args}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
