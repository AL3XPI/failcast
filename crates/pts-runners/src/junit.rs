use std::collections::HashMap;

use pts_core::{PtsError, TestExecutionOutcome, TestStatus};
use quick_xml::Reader;
use quick_xml::events::{BytesStart, Event};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum ActiveDetail {
    #[default]
    None,
    Failure,
    Error,
    Skipped,
}

#[derive(Debug, Default)]
struct PendingTestCase {
    classname: String,
    name: String,
    file: Option<String>,
    duration_ms: u64,
    status: Option<TestStatus>,
    detail_message_attr: Option<String>,
    detail_body: String,
    active_detail: ActiveDetail,
}

impl PendingTestCase {
    fn into_outcome(self, max_traceback_chars: usize) -> TestExecutionOutcome {
        let test_id = canonical_test_id(self.file.as_deref(), &self.classname, &self.name);
        let status = self.status.unwrap_or(TestStatus::Pass);

        let failure_message = match status {
            TestStatus::Fail | TestStatus::Error => {
                let raw = combine_message(self.detail_message_attr.as_deref(), &self.detail_body);
                if raw.is_empty() {
                    None
                } else {
                    Some(truncate_chars(&raw, max_traceback_chars))
                }
            }
            TestStatus::Pass | TestStatus::Skip => None,
        };

        TestExecutionOutcome {
            test_id,
            status,
            duration_ms: self.duration_ms,
            failure_message,
        }
    }
}

/// Parses JUnit XML produced by pytest (`xunit1`/`legacy` or `xunit2`) into
/// canonical `TestExecutionOutcome` entries.
pub fn parse_junit_xml(
    xml: &str,
    max_traceback_chars: usize,
) -> Result<Vec<TestExecutionOutcome>, PtsError> {
    let mut reader = Reader::from_str(xml);
    let mut current: Option<PendingTestCase> = None;
    let mut outcomes: Vec<TestExecutionOutcome> = Vec::new();
    let mut index_by_id: HashMap<String, usize> = HashMap::new();

    loop {
        match reader.read_event() {
            Ok(Event::Start(ref e)) => {
                let name = e.name();
                match name.as_ref() {
                    b"testcase" => {
                        current = Some(parse_testcase_attrs(e)?);
                    }
                    b"failure" => {
                        if let Some(tc) = current.as_mut() {
                            tc.status = Some(TestStatus::Fail);
                            tc.active_detail = ActiveDetail::Failure;
                            if let Some(msg) = extract_attr(e, b"message")? {
                                tc.detail_message_attr = Some(msg);
                            }
                        }
                    }
                    b"error" => {
                        if let Some(tc) = current.as_mut() {
                            // Prefer Fail if already marked Fail, otherwise Error.
                            if tc.status != Some(TestStatus::Fail) {
                                tc.status = Some(TestStatus::Error);
                            }
                            tc.active_detail = ActiveDetail::Error;
                            if tc.detail_message_attr.is_none()
                                && let Some(msg) = extract_attr(e, b"message")?
                            {
                                tc.detail_message_attr = Some(msg);
                            }
                        }
                    }
                    b"skipped" => {
                        if let Some(tc) = current.as_mut() {
                            if tc.status.is_none() {
                                tc.status = Some(TestStatus::Skip);
                            }
                            tc.active_detail = ActiveDetail::Skipped;
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::Empty(ref e)) => {
                let name = e.name();
                match name.as_ref() {
                    b"testcase" => {
                        let tc = parse_testcase_attrs(e)?;
                        push_or_merge(
                            &mut outcomes,
                            &mut index_by_id,
                            tc.into_outcome(max_traceback_chars),
                            max_traceback_chars,
                        );
                    }
                    b"failure" => {
                        if let Some(tc) = current.as_mut() {
                            tc.status = Some(TestStatus::Fail);
                            if let Some(msg) = extract_attr(e, b"message")? {
                                tc.detail_message_attr = Some(msg);
                            }
                        }
                    }
                    b"error" => {
                        if let Some(tc) = current.as_mut() {
                            if tc.status != Some(TestStatus::Fail) {
                                tc.status = Some(TestStatus::Error);
                            }
                            if tc.detail_message_attr.is_none()
                                && let Some(msg) = extract_attr(e, b"message")?
                            {
                                tc.detail_message_attr = Some(msg);
                            }
                        }
                    }
                    b"skipped" => {
                        if let Some(tc) = current.as_mut()
                            && tc.status.is_none()
                        {
                            tc.status = Some(TestStatus::Skip);
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::Text(ref e)) => {
                if let Some(tc) = current.as_mut()
                    && matches!(tc.active_detail, ActiveDetail::Failure | ActiveDetail::Error)
                {
                    let text = e
                        .unescape()
                        .map_err(|err| PtsError::Parse(format!("invalid XML text: {err}")))?;
                    tc.detail_body.push_str(&text);
                }
            }
            Ok(Event::CData(ref e)) => {
                if let Some(tc) = current.as_mut()
                    && matches!(tc.active_detail, ActiveDetail::Failure | ActiveDetail::Error)
                {
                    let text = String::from_utf8_lossy(e.as_ref());
                    tc.detail_body.push_str(&text);
                }
            }
            Ok(Event::End(ref e)) => match e.name().as_ref() {
                b"testcase" => {
                    if let Some(tc) = current.take() {
                        push_or_merge(
                            &mut outcomes,
                            &mut index_by_id,
                            tc.into_outcome(max_traceback_chars),
                            max_traceback_chars,
                        );
                    }
                }
                b"failure" | b"error" | b"skipped" => {
                    if let Some(tc) = current.as_mut() {
                        tc.active_detail = ActiveDetail::None;
                    }
                }
                _ => {}
            },
            Ok(Event::Eof) => break,
            Err(e) => {
                return Err(PtsError::Parse(format!(
                    "failed to parse JUnit XML at position {}: {e}",
                    reader.error_position()
                )));
            }
            _ => {}
        }
    }

    Ok(outcomes)
}

fn push_or_merge(
    outcomes: &mut Vec<TestExecutionOutcome>,
    index_by_id: &mut HashMap<String, usize>,
    outcome: TestExecutionOutcome,
    max_traceback_chars: usize,
) {
    if let Some(&idx) = index_by_id.get(&outcome.test_id) {
        let existing = &mut outcomes[idx];
        existing.duration_ms = existing.duration_ms.saturating_add(outcome.duration_ms);
        if status_severity(outcome.status) > status_severity(existing.status) {
            existing.status = outcome.status;
        }
        if let Some(new_msg) = outcome.failure_message {
            match &mut existing.failure_message {
                Some(prev) => {
                    prev.push('\n');
                    prev.push_str(&new_msg);
                    *prev = truncate_chars(prev, max_traceback_chars);
                }
                None => {
                    existing.failure_message = Some(new_msg);
                }
            }
        }
    } else {
        let idx = outcomes.len();
        index_by_id.insert(outcome.test_id.clone(), idx);
        outcomes.push(outcome);
    }
}

fn status_severity(status: TestStatus) -> u8 {
    match status {
        TestStatus::Pass => 0,
        TestStatus::Skip => 1,
        TestStatus::Fail => 2,
        TestStatus::Error => 3,
    }
}

fn parse_testcase_attrs(e: &BytesStart<'_>) -> Result<PendingTestCase, PtsError> {
    let mut tc = PendingTestCase::default();
    for attr in e.attributes() {
        let attr = attr.map_err(|err| PtsError::Parse(format!("invalid XML attr: {err}")))?;
        let val = attr
            .unescape_value()
            .map_err(|err| PtsError::Parse(format!("invalid XML attr value: {err}")))?
            .into_owned();
        match attr.key.as_ref() {
            b"classname" => tc.classname = val,
            b"name" => tc.name = val,
            b"file" => tc.file = Some(val),
            b"time" => {
                if let Ok(secs) = val.trim().parse::<f64>()
                    && secs.is_finite()
                {
                    tc.duration_ms = (secs.max(0.0) * 1000.0).round() as u64;
                }
            }
            _ => {}
        }
    }
    Ok(tc)
}

fn extract_attr(e: &BytesStart<'_>, key: &[u8]) -> Result<Option<String>, PtsError> {
    for attr in e.attributes() {
        let attr = attr.map_err(|err| PtsError::Parse(format!("invalid XML attr: {err}")))?;
        if attr.key.as_ref() == key {
            let val = attr
                .unescape_value()
                .map_err(|err| PtsError::Parse(format!("invalid XML attr value: {err}")))?
                .into_owned();
            return Ok(Some(val));
        }
    }
    Ok(None)
}

fn combine_message(attr_msg: Option<&str>, body: &str) -> String {
    let body_trimmed = body.trim();
    let attr_trimmed = attr_msg.map(str::trim).unwrap_or("");
    match (attr_trimmed.is_empty(), body_trimmed.is_empty()) {
        (true, true) => String::new(),
        (false, true) => attr_trimmed.to_string(),
        (true, false) => body_trimmed.to_string(),
        (false, false) => {
            if body_trimmed.contains(attr_trimmed) {
                body_trimmed.to_string()
            } else {
                format!("{attr_trimmed}\n{body_trimmed}")
            }
        }
    }
}

pub fn truncate_chars(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    s.chars().take(max_chars).collect()
}

/// Normalizes a repo-relative path to POSIX separators without leading `./`.
pub fn normalize_posix_path(path: &str) -> String {
    let replaced = path.replace('\\', "/");
    replaced
        .strip_prefix("./")
        .unwrap_or(&replaced)
        .to_string()
}

/// Converts JUnit `(file, classname, name)` attributes into a canonical pytest node ID,
/// e.g. `tests/test_calc.py::test_add_positive` or `tests/test_calc.py::TestMath::test_add`.
pub fn canonical_test_id(file: Option<&str>, classname: &str, name: &str) -> String {
    if let Some(raw_file) = file
        && !raw_file.trim().is_empty()
    {
        let posix_file = normalize_posix_path(raw_file.trim());
        let module_prefix = posix_file
            .strip_suffix(".py")
            .unwrap_or(&posix_file)
            .replace('/', ".");
        let module_prefix = module_prefix
            .strip_suffix(".__init__")
            .unwrap_or(&module_prefix);

        let class_suffix = if let Some(rest) = classname.strip_prefix(module_prefix) {
            rest.strip_prefix('.').unwrap_or("")
        } else {
            ""
        };

        if class_suffix.is_empty() {
            return format!("{posix_file}::{name}");
        }
        let class_path = class_suffix.replace('.', "::");
        return format!("{posix_file}::{class_path}::{name}");
    }

    let trimmed = classname.trim();
    if trimmed.is_empty() {
        return name.to_string();
    }

    if trimmed.contains('/') || trimmed.ends_with(".py") {
        let posix_file = normalize_posix_path(trimmed);
        return format!("{posix_file}::{name}");
    }

    let parts: Vec<&str> = trimmed.split('.').collect();
    // Heuristic for xunit2 (no `file` attr): segments starting with an uppercase ASCII
    // letter after the first module segment represent Python test classes (e.g. TestCalc).
    let class_idx = parts
        .iter()
        .enumerate()
        .skip(1)
        .find(|(_, seg)| seg.starts_with(|c: char| c.is_ascii_uppercase()))
        .map(|(i, _)| i);

    match class_idx {
        Some(idx) => {
            let file_path = format!("{}.py", parts[..idx].join("/"));
            let class_path = parts[idx..].join("::");
            format!("{file_path}::{class_path}::{name}")
        }
        None => {
            let file_path = format!("{}.py", parts.join("/"));
            format!("{file_path}::{name}")
        }
    }
}
