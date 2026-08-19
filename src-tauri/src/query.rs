use super::*;

pub(super) fn date_bounds(year: i32, month: Option<u32>, day: Option<u32>) -> Option<DateRange> {
    if !(1900..=2200).contains(&year) {
        return None;
    }
    let month = month.unwrap_or(1);
    if !(1..=12).contains(&month) {
        return None;
    }
    let max_day = days_in_month(year, month);
    let first_day = day.unwrap_or(1);
    if first_day == 0 || first_day > max_day {
        return None;
    }
    let last_day = day.unwrap_or(max_day);
    Some(DateRange {
        from: Some(format!("{year:04}-{month:02}-{first_day:02} 00:00:00")),
        to: Some(format!("{year:04}-{month:02}-{last_day:02} 23:59:59")),
    })
}

pub(super) fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        2 if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

pub(super) fn digits_before(value: &str, end: usize) -> Option<i32> {
    let prefix = &value[..end];
    let start = prefix
        .char_indices()
        .rev()
        .find(|(_, character)| !character.is_ascii_digit())
        .map(|(index, character)| index + character.len_utf8())
        .unwrap_or(0);
    prefix[start..].parse().ok()
}

pub(super) fn parse_date_range(query: &str) -> DateRange {
    let now_year = time::OffsetDateTime::now_utc().year();
    if query.contains("去年夏天") {
        return DateRange {
            from: Some(format!("{:04}-06-01 00:00:00", now_year - 1)),
            to: Some(format!("{:04}-08-31 23:59:59", now_year - 1)),
        };
    }
    if query.contains("今年夏天") {
        return DateRange {
            from: Some(format!("{now_year:04}-06-01 00:00:00")),
            to: Some(format!("{now_year:04}-08-31 23:59:59")),
        };
    }
    for (phrase, year_offset) in [("前年", -2), ("去年", -1), ("今年", 0), ("明年", 1)] {
        if query.contains(phrase) {
            return date_bounds(now_year + year_offset, None, None).unwrap_or_default();
        }
    }
    let year_index = query.find('年');
    let Some(year_index) = year_index else {
        return DateRange::default();
    };
    let Some(year) = digits_before(query, year_index) else {
        return DateRange::default();
    };
    let month = query
        .find('月')
        .and_then(|index| digits_before(query, index))
        .map(|month| month as u32);
    let day = query
        .find('日')
        .and_then(|index| digits_before(query, index))
        .map(|day| day as u32);
    date_bounds(year, month, day).unwrap_or_default()
}

pub(super) fn remove_date_words(value: &str) -> String {
    let mut result = value.to_owned();
    for phrase in [
        "去年夏天", "今年夏天", "前年", "去年", "今年", "明年", "夏天", "年", "月", "日",
    ] {
        result = result.replace(phrase, " ");
    }
    if result.chars().any(|character| character.is_ascii_digit()) {
        result = result
            .chars()
            .map(|character| if character.is_ascii_digit() { ' ' } else { character })
            .collect();
    }
    result
}

pub(super) fn parse_limit(query: &str) -> Option<usize> {
    for marker in ["最多", "前", "取"] {
        let Some(index) = query.find(marker) else { continue };
        let suffix = query[index + marker.len()..].trim_start();
        let digits: String = suffix
            .chars()
            .take_while(|character| character.is_ascii_digit())
            .collect();
        if let Ok(limit) = digits.parse::<usize>() {
            if limit > 0 {
                return Some(limit);
            }
        }
    }
    None
}

pub(super) fn tokenize_query(value: &str) -> Vec<String> {
    value
        .split(|character: char| {
            character.is_whitespace()
                || matches!(character, '、' | '，' | ',' | '。' | '.' | '！' | '!' | '?' | '？')
        })
        .flat_map(|term| term.split(|character: char| matches!(character, '的' | '和' | '與' | '在' | '有')))
        .map(str::trim)
        .filter(|term| !term.is_empty())
        .map(str::to_owned)
        .filter(|term| !QUERY_STOP_WORDS.contains(&term.as_str()))
        .collect()
}

pub(super) fn extract_negative_terms(value: &mut String) -> Vec<String> {
    let mut negative_terms = Vec::new();
    for marker in ["不要", "不含", "排除", "沒有"] {
        let mut search_from = 0;
        while let Some(relative_index) = value[search_from..].find(marker) {
            let index = search_from + relative_index;
            let start = index + marker.len();
            let tail = &value[start..];
            let candidate: String = tail
                .chars()
                .take_while(|character| {
                    !character.is_whitespace()
                        && !matches!(
                            character,
                            '的' | '和' | '與' | '在' | '有' | '或' | '、' | '，' | ',' | '。'
                        )
                })
                .collect();
            if !candidate.is_empty() && !QUERY_STOP_WORDS.contains(&candidate.as_str()) {
                negative_terms.push(candidate.clone());
                value.replace_range(index..start + candidate.len(), " ");
                search_from = index;
            } else {
                value.replace_range(index..start, " ");
                search_from = index;
            }
        }
    }
    negative_terms
}

pub(super) fn parse_query(query: &str, known_people: &[String]) -> QueryPlan {
    let normalized = query.trim().to_lowercase();
    let mut residual = normalized.clone();
    let date = parse_date_range(&normalized);
    residual = remove_date_words(&residual);
    let limit = parse_limit(&normalized);
    for marker in ["最多", "前", "取"] {
        if let Some(index) = residual.find(marker) {
            let end = index + marker.len();
            let whitespace = residual[end..]
                .chars()
                .take_while(|character| character.is_whitespace())
                .map(char::len_utf8)
                .sum::<usize>();
            let digits = residual[end + whitespace..]
                .chars()
                .take_while(|character| character.is_ascii_digit())
                .count();
            residual.replace_range(index..end + whitespace + digits, " ");
        }
    }

    let mut face_presence = FacePresence::Any;
    for phrase in ["不要有人", "沒有人的", "沒有臉", "無人臉", "無人物", "不要人"] {
        if normalized.contains(phrase) {
            face_presence = FacePresence::Forbidden;
            residual = residual.replace(phrase, " ");
        }
    }
    for phrase in ["有人臉", "有人", "人物", "人像", "合照"] {
        if face_presence == FacePresence::Any && normalized.contains(phrase) {
            face_presence = FacePresence::Required;
            residual = residual.replace(phrase, " ");
        }
    }

    let mut people_include = Vec::new();
    let mut people_exclude = Vec::new();
    for person in known_people {
        let person = person.trim();
        if person.is_empty() || !normalized.contains(&person.to_lowercase()) {
            continue;
        }
        let person_lower = person.to_lowercase();
        let negative = ["不要", "不含", "排除", "沒有", "無"]
            .iter()
            .any(|marker| normalized.contains(&format!("{marker}{person_lower}")));
        if negative {
            people_exclude.push(person.to_owned());
        } else {
            people_include.push(person.to_owned());
            if face_presence == FacePresence::Any {
                face_presence = FacePresence::Required;
            }
        }
        residual = residual.replace(&person_lower, " ");
    }

    let mut extensions = Vec::new();
    for extension in IMAGE_EXTENSIONS {
        if normalized.contains(extension) {
            extensions.push((*extension).to_owned());
            residual = residual.replace(extension, " ");
        }
    }
    let mut text = TextQueryPlan::default();
    text.must_not = extract_negative_terms(&mut residual);
    let mut positive_terms = Vec::new();
    for term in tokenize_query(&residual) {
        if ["不要", "不含", "排除", "沒有", "無"].contains(&term.as_str()) {
            continue;
        }
        positive_terms.push(term);
    }
    text.should = positive_terms.clone();
    let semantic_query = positive_terms.join(" ");
    QueryPlan {
        semantic_query,
        text,
        people_include,
        people_exclude,
        face_presence,
        date,
        extensions,
        limit,
    }
}

pub(super) fn record_date_key(record: &ImageRecord) -> Option<String> {
    if let Some(captured_at) = &record.captured_at {
        return Some(captured_at.clone());
    }
    let timestamp = record.modified_at.parse::<i64>().ok()?;
    let date = time::OffsetDateTime::from_unix_timestamp(timestamp).ok()?;
    Some(format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
        date.year(),
        date.month() as u8,
        date.day(),
        date.hour(),
        date.minute(),
        date.second()
    ))
}

pub(super) fn record_matches_plan(record: &ImageRecord, plan: &QueryPlan) -> bool {
    if !plan.extensions.is_empty() {
        let extension = Path::new(&record.path)
            .extension()
            .and_then(|extension| extension.to_str())
            .map(str::to_lowercase);
        if extension.is_none_or(|extension| !plan.extensions.contains(&extension)) {
            return false;
        }
    }
    if plan.face_presence == FacePresence::Required && record.face_group_ids.is_empty() {
        return false;
    }
    if plan.face_presence == FacePresence::Forbidden && !record.face_group_ids.is_empty() {
        return false;
    }
    let people = record.people.iter().map(|person| person.to_lowercase()).collect::<Vec<_>>();
    if plan.people_include.iter().any(|person| !people.contains(&person.to_lowercase())) {
        return false;
    }
    if plan.people_exclude.iter().any(|person| people.contains(&person.to_lowercase())) {
        return false;
    }
    if plan.date.from.is_some() || plan.date.to.is_some() {
        let Some(date) = record_date_key(record) else { return false };
        if plan.date.from.as_ref().is_some_and(|from| date < *from) {
            return false;
        }
        if plan.date.to.as_ref().is_some_and(|to| date > *to) {
            return false;
        }
    }
    true
}
