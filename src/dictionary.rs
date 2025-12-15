//! Dictionary-based text correction for OCR output
//!
//! Uses fuzzy matching against known terms to fix OCR errors.
//! Dictionary sources: shape labels, code symbols, Jira terms, user corrections.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Dictionary for fuzzy text correction
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Dictionary {
    /// Known terms with frequency (higher = prefer)
    terms: HashMap<String, u32>,
    /// User corrections: wrong → right
    corrections: HashMap<String, String>,
    /// Domain-specific terms (code symbols, Jira, etc)
    domains: HashMap<DomainSource, HashSet<String>>,
    /// Config settings
    min_length: usize,
    max_distance_short: usize,
    max_distance_long: usize,
}

/// Source of dictionary terms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DomainSource {
    /// Terms from current diagram shapes
    DiagramLabels,
    /// Code symbols from GitHub
    CodeSymbols,
    /// Jira ticket titles and terms
    Jira,
    /// Confluence page titles
    Confluence,
    /// User's personal dictionary
    Personal,
    /// Flow control terms (START, END, YES, NO)
    FlowControl,
    /// Action terms (PROCESS, INPUT, OUTPUT)
    Actions,
    /// Data terms (DATABASE, CACHE, FILE)
    Data,
    /// Status terms (ERROR, SUCCESS, PENDING)
    Status,
    /// Component terms (USER, SERVER, API)
    Components,
}

/// Correction result
#[derive(Debug, Clone)]
pub struct CorrectionResult {
    /// Original OCR text
    pub original: String,
    /// Corrected text
    pub corrected: String,
    /// Confidence (0.0 - 1.0)
    pub confidence: f32,
    /// Was this corrected or kept as-is?
    pub was_corrected: bool,
    /// Source of correction (if any)
    pub source: Option<DomainSource>,
}

impl Dictionary {
    /// Create empty dictionary
    pub fn new() -> Self {
        Self {
            terms: HashMap::new(),
            corrections: HashMap::new(),
            domains: HashMap::new(),
            min_length: 3,
            max_distance_short: 1,
            max_distance_long: 2,
        }
    }

    /// Load dictionary from file (native format)
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let dict: Dictionary = serde_json::from_str(&content)?;
        Ok(dict)
    }

    /// Load dictionary from JSON config file
    /// Supports both flat format and categorized format:
    ///
    /// Flat format:
    /// {
    ///   "corrections": { "usr": "User" },
    ///   "terms": ["Start", "End"]
    /// }
    ///
    /// Categorized format:
    /// {
    ///   "terms": {
    ///     "flow_control": ["START", "END"],
    ///     "actions": ["PROCESS", "INPUT"]
    ///   }
    /// }
    pub fn load_from_json(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: serde_json::Value = serde_json::from_str(&content)?;

        let mut dict = Dictionary::new();
        dict.seed_common_terms();

        // Load config settings
        if let Some(min_len) = config.get("min_length").and_then(|v| v.as_u64()) {
            dict.min_length = min_len as usize;
        }
        if let Some(max_short) = config.get("max_distance_short").and_then(|v| v.as_u64()) {
            dict.max_distance_short = max_short as usize;
        }
        if let Some(max_long) = config.get("max_distance_long").and_then(|v| v.as_u64()) {
            dict.max_distance_long = max_long as usize;
        }

        // Load corrections (flat format)
        if let Some(corrections) = config.get("corrections").and_then(|c| c.as_object()) {
            for (wrong, right) in corrections {
                if let Some(r) = right.as_str() {
                    dict.add_correction(wrong, r);
                }
            }
        }

        // Load terms - handle both flat array and categorized object
        if let Some(terms) = config.get("terms") {
            if let Some(terms_array) = terms.as_array() {
                // Flat format: "terms": ["Start", "End"]
                for term in terms_array {
                    if let Some(t) = term.as_str() {
                        dict.add_term(t, DomainSource::Personal);
                    }
                }
            } else if let Some(terms_obj) = terms.as_object() {
                // Categorized format: "terms": { "flow_control": [...], "actions": [...] }
                for (category, terms_array) in terms_obj {
                    let source = match category.as_str() {
                        "flow_control" => DomainSource::FlowControl,
                        "actions" => DomainSource::Actions,
                        "data" => DomainSource::Data,
                        "status" => DomainSource::Status,
                        "components" => DomainSource::Components,
                        _ => DomainSource::Personal,
                    };

                    if let Some(arr) = terms_array.as_array() {
                        for term in arr {
                            if let Some(t) = term.as_str() {
                                dict.add_term(t, source);
                            }
                        }
                    }
                }
            }
        }

        tracing::info!(
            "Loaded dictionary from {:?}: {} terms, {} corrections",
            path,
            dict.terms.len(),
            dict.corrections.len()
        );

        Ok(dict)
    }

    /// Save dictionary to file
    pub fn save(&self, path: &Path) -> Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Add a term to the dictionary
    pub fn add_term(&mut self, term: &str, source: DomainSource) {
        let normalized = term.trim().to_string();
        if normalized.is_empty() {
            return;
        }

        // Add to frequency map (both original case and uppercase for matching)
        *self.terms.entry(normalized.clone()).or_insert(0) += 1;
        *self.terms.entry(normalized.to_uppercase()).or_insert(0) += 1;

        // Add to domain set
        self.domains
            .entry(source)
            .or_insert_with(HashSet::new)
            .insert(normalized);
    }

    /// Add multiple terms from a source
    pub fn add_terms(&mut self, terms: &[&str], source: DomainSource) {
        for term in terms {
            self.add_term(term, source);
        }
    }

    /// Add a user correction (learns from mistakes)
    pub fn add_correction(&mut self, wrong: &str, right: &str) {
        self.corrections.insert(wrong.to_lowercase(), right.to_string());
        self.corrections.insert(wrong.to_uppercase(), right.to_string());
        self.add_term(right, DomainSource::Personal);
    }

    /// Correct OCR text using dictionary
    pub fn correct(&self, text: &str) -> CorrectionResult {
        let text = text.trim();

        if text.is_empty() {
            return CorrectionResult {
                original: text.to_string(),
                corrected: text.to_string(),
                confidence: 1.0,
                was_corrected: false,
                source: None,
            };
        }

        // Check exact user corrections first (case-insensitive)
        if let Some(corrected) = self.corrections.get(&text.to_lowercase()) {
            return CorrectionResult {
                original: text.to_string(),
                corrected: corrected.clone(),
                confidence: 1.0,
                was_corrected: true,
                source: Some(DomainSource::Personal),
            };
        }

        // Also check uppercase
        if let Some(corrected) = self.corrections.get(&text.to_uppercase()) {
            return CorrectionResult {
                original: text.to_string(),
                corrected: corrected.clone(),
                confidence: 1.0,
                was_corrected: true,
                source: Some(DomainSource::Personal),
            };
        }

        // Check if already a known term (case-insensitive)
        if self.terms.contains_key(text) || self.terms.contains_key(&text.to_uppercase()) {
            return CorrectionResult {
                original: text.to_string(),
                corrected: text.to_string(),
                confidence: 1.0,
                was_corrected: false,
                source: None,
            };
        }

        // Skip fuzzy matching for very short text
        if text.len() < self.min_length {
            return CorrectionResult {
                original: text.to_string(),
                corrected: text.to_string(),
                confidence: 0.5,
                was_corrected: false,
                source: None,
            };
        }

        // Fuzzy match against dictionary
        if let Some((best_match, distance, source)) = self.fuzzy_match(text) {
            let confidence = self.distance_to_confidence(distance, text.len());

            // Only correct if confidence is high enough
            if confidence >= 0.7 {
                return CorrectionResult {
                    original: text.to_string(),
                    corrected: best_match,
                    confidence,
                    was_corrected: true,
                    source: Some(source),
                };
            }
        }

        // No good match, return original
        CorrectionResult {
            original: text.to_string(),
            corrected: text.to_string(),
            confidence: 0.5,
            was_corrected: false,
            source: None,
        }
    }

    /// Correct multi-word text (split and correct each word)
    pub fn correct_phrase(&self, text: &str) -> CorrectionResult {
        let words: Vec<&str> = text.split_whitespace().collect();

        if words.len() <= 1 {
            return self.correct(text);
        }

        let mut corrected_words = Vec::new();
        let mut total_confidence = 0.0;
        let mut any_corrected = false;

        for word in &words {
            let result = self.correct(word);
            corrected_words.push(result.corrected);
            total_confidence += result.confidence;
            if result.was_corrected {
                any_corrected = true;
            }
        }

        CorrectionResult {
            original: text.to_string(),
            corrected: corrected_words.join(" "),
            confidence: total_confidence / words.len() as f32,
            was_corrected: any_corrected,
            source: Some(DomainSource::Personal),
        }
    }

    /// Find best fuzzy match in dictionary
    fn fuzzy_match(&self, text: &str) -> Option<(String, usize, DomainSource)> {
        let text_upper = text.to_uppercase();
        let max_distance = self.max_allowed_distance(text.len());

        let mut best: Option<(String, usize, DomainSource)> = None;

        // Search all domain sources, prioritizing certain ones
        let priority_order = [
            DomainSource::FlowControl,    // START, END, YES, NO
            DomainSource::Components,     // USER, SERVER, API
            DomainSource::Actions,        // PROCESS, INPUT, OUTPUT
            DomainSource::Status,         // ERROR, SUCCESS
            DomainSource::Data,           // DATABASE, CACHE
            DomainSource::DiagramLabels,  // Current context
            DomainSource::Personal,       // User's corrections
            DomainSource::CodeSymbols,    // Code terms
            DomainSource::Jira,
            DomainSource::Confluence,
        ];

        for source in &priority_order {
            if let Some(terms) = self.domains.get(source) {
                for term in terms {
                    // Compare uppercase to uppercase for case-insensitive matching
                    let distance = self.levenshtein(&text_upper, &term.to_uppercase());

                    if distance <= max_distance {
                        let dominated = best.as_ref()
                            .map(|(_, d, _)| distance < *d)
                            .unwrap_or(true);

                        if dominated {
                            best = Some((term.clone(), distance, *source));
                        }
                    }
                }
            }
        }

        best
    }

    /// Calculate Levenshtein edit distance
    fn levenshtein(&self, a: &str, b: &str) -> usize {
        let a_chars: Vec<char> = a.chars().collect();
        let b_chars: Vec<char> = b.chars().collect();

        let m = a_chars.len();
        let n = b_chars.len();

        if m == 0 { return n; }
        if n == 0 { return m; }

        let mut dp = vec![vec![0usize; n + 1]; m + 1];

        for i in 0..=m { dp[i][0] = i; }
        for j in 0..=n { dp[0][j] = j; }

        for i in 1..=m {
            for j in 1..=n {
                let cost = if a_chars[i-1] == b_chars[j-1] { 0 } else { 1 };
                dp[i][j] = (dp[i-1][j] + 1)
                    .min(dp[i][j-1] + 1)
                    .min(dp[i-1][j-1] + cost);
            }
        }

        dp[m][n]
    }

    /// Maximum allowed edit distance based on word length
    fn max_allowed_distance(&self, len: usize) -> usize {
        if len <= 4 {
            self.max_distance_short
        } else {
            self.max_distance_long
        }
    }

    /// Convert edit distance to confidence score
    fn distance_to_confidence(&self, distance: usize, word_len: usize) -> f32 {
        if word_len == 0 { return 0.0; }
        let ratio = distance as f32 / word_len as f32;
        (1.0 - ratio).max(0.0)
    }

    /// Get dictionary stats
    pub fn stats(&self) -> DictionaryStats {
        DictionaryStats {
            total_terms: self.terms.len(),
            user_corrections: self.corrections.len(),
            terms_by_source: self.domains.iter()
                .map(|(k, v)| (*k, v.len()))
                .collect(),
        }
    }

    /// Seed with common diagramming terms
    pub fn seed_common_terms(&mut self) {
        // Flow control
        let flow_control = [
            "START", "END", "STOP", "BEGIN", "FINISH",
            "YES", "NO", "TRUE", "FALSE", "OK", "FAIL",
            "IF", "ELSE", "THEN", "WHILE", "FOR", "LOOP",
        ];
        self.add_terms(&flow_control.map(|s| s as &str), DomainSource::FlowControl);

        // Actions
        let actions = [
            "PROCESS", "INPUT", "OUTPUT", "READ", "WRITE",
            "SAVE", "LOAD", "DELETE", "UPDATE", "CREATE",
            "SEND", "RECEIVE", "CALL", "RETURN", "EXIT",
            "FROM", "TO", "INIT", "VALIDATE", "GET", "SET",
        ];
        self.add_terms(&actions.map(|s| s as &str), DomainSource::Actions);

        // Data
        let data = [
            "DATA", "DATABASE", "DB", "CACHE", "STORAGE",
            "FILE", "LOG", "CONFIG", "QUEUE", "STACK",
            "TABLE", "ROW", "COLUMN", "INDEX", "KEY",
        ];
        self.add_terms(&data.map(|s| s as &str), DomainSource::Data);

        // Status
        let status = [
            "ERROR", "VALID", "INVALID", "CHECK", "VERIFY",
            "SUCCESS", "FAILURE", "RETRY", "TIMEOUT",
            "PENDING", "DONE", "ACTIVE", "IDLE",
        ];
        self.add_terms(&status.map(|s| s as &str), DomainSource::Status);

        // Components
        let components = [
            "USER", "ADMIN", "CLIENT", "SERVER", "API",
            "SERVICE", "HANDLER", "CONTROLLER", "MODEL",
            "VIEW", "ROUTER", "WORKER", "AGENT",
        ];
        self.add_terms(&components.map(|s| s as &str), DomainSource::Components);

        // Add common OCR corrections
        let ocr_corrections = [
            ("usr", "USER"),
            ("svr", "SERVER"),
            ("db", "DATABASE"),
            ("api", "API"),
            ("req", "REQUEST"),
            ("res", "RESPONSE"),
            ("auth", "AUTH"),
            ("svc", "SERVICE"),
            ("ctrl", "CONTROLLER"),
            ("msg", "MESSAGE"),
            ("err", "ERROR"),
            ("cfg", "CONFIG"),
            ("init", "INIT"),
            ("proc", "PROCESS"),
            ("val", "VALIDATE"),
        ];

        for (wrong, right) in ocr_corrections {
            self.add_correction(wrong, right);
        }
    }
}

/// Dictionary statistics
#[derive(Debug, Clone)]
pub struct DictionaryStats {
    pub total_terms: usize,
    pub user_corrections: usize,
    pub terms_by_source: HashMap<DomainSource, usize>,
}

/// Builder for creating dictionaries from various sources
pub struct DictionaryBuilder {
    dict: Dictionary,
}

impl DictionaryBuilder {
    pub fn new() -> Self {
        Self { dict: Dictionary::new() }
    }

    /// Add common diagramming terms
    pub fn with_common_terms(mut self) -> Self {
        self.dict.seed_common_terms();
        self
    }

    /// Add terms from shape labels in current diagram
    pub fn with_diagram_labels(mut self, labels: &[&str]) -> Self {
        self.dict.add_terms(labels, DomainSource::DiagramLabels);
        self
    }

    /// Add code symbols (class names, functions, etc)
    pub fn with_code_symbols(mut self, symbols: &[&str]) -> Self {
        self.dict.add_terms(symbols, DomainSource::CodeSymbols);
        self
    }

    /// Load existing dictionary and merge
    pub fn with_existing(mut self, path: &Path) -> Result<Self> {
        if path.exists() {
            let existing = Dictionary::load(path)?;
            for (term, freq) in existing.terms {
                *self.dict.terms.entry(term).or_insert(0) += freq;
            }
            self.dict.corrections.extend(existing.corrections);
            for (source, terms) in existing.domains {
                self.dict.domains.entry(source)
                    .or_insert_with(HashSet::new)
                    .extend(terms);
            }
        }
        Ok(self)
    }

    /// Load from JSON config file and merge
    pub fn with_json_config(mut self, path: &Path) -> Self {
        if path.exists() {
            if let Ok(loaded) = Dictionary::load_from_json(path) {
                for (term, freq) in loaded.terms {
                    *self.dict.terms.entry(term).or_insert(0) += freq;
                }
                self.dict.corrections.extend(loaded.corrections);
                for (source, terms) in loaded.domains {
                    self.dict.domains.entry(source)
                        .or_insert_with(HashSet::new)
                        .extend(terms);
                }
                // Copy config settings
                self.dict.min_length = loaded.min_length;
                self.dict.max_distance_short = loaded.max_distance_short;
                self.dict.max_distance_long = loaded.max_distance_long;
            }
        }
        self
    }

    pub fn build(self) -> Dictionary {
        self.dict
    }
}

impl Default for DictionaryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_match() {
        let mut dict = Dictionary::new();
        dict.add_term("Authentication", DomainSource::DiagramLabels);

        let result = dict.correct("Authentication");
        assert!(!result.was_corrected);
        assert_eq!(result.corrected, "Authentication");
    }

    #[test]
    fn test_fuzzy_match() {
        let mut dict = Dictionary::new();
        dict.add_term("PROCESS", DomainSource::Actions);

        let result = dict.correct("PROCSS"); // missing 'E'
        assert!(result.was_corrected);
        assert_eq!(result.corrected, "PROCESS");
    }

    #[test]
    fn test_user_correction() {
        let mut dict = Dictionary::new();
        dict.add_correction("usr", "USER");

        let result = dict.correct("usr");
        assert!(result.was_corrected);
        assert_eq!(result.corrected, "USER");
    }

    #[test]
    fn test_case_insensitive_correction() {
        let mut dict = Dictionary::new();
        dict.add_correction("usr", "USER");

        let result = dict.correct("USR");
        assert!(result.was_corrected);
        assert_eq!(result.corrected, "USER");
    }

    #[test]
    fn test_phrase_correction() {
        let mut dict = Dictionary::new();
        dict.add_term("USER", DomainSource::Components);
        dict.add_term("INPUT", DomainSource::Actions);

        let result = dict.correct_phrase("USR INPT");
        assert!(result.was_corrected);
        // Should correct both words
    }

    #[test]
    fn test_levenshtein() {
        let dict = Dictionary::new();
        assert_eq!(dict.levenshtein("kitten", "sitting"), 3);
        assert_eq!(dict.levenshtein("", "abc"), 3);
        assert_eq!(dict.levenshtein("same", "same"), 0);
    }

    #[test]
    fn test_builder() {
        let dict = DictionaryBuilder::new()
            .with_common_terms()
            .with_diagram_labels(&["MyService", "UserController"])
            .build();

        assert!(dict.terms.contains_key("USER"));
        assert!(dict.terms.contains_key("MyService"));
    }

    #[test]
    fn test_seeded_corrections() {
        let mut dict = Dictionary::new();
        dict.seed_common_terms();

        let result = dict.correct("usr");
        assert!(result.was_corrected);
        assert_eq!(result.corrected, "USER");

        let result = dict.correct("svr");
        assert!(result.was_corrected);
        assert_eq!(result.corrected, "SERVER");
    }

    #[test]
    fn test_flow_control_terms() {
        let mut dict = Dictionary::new();
        dict.seed_common_terms();

        // Exact match
        let result = dict.correct("START");
        assert!(!result.was_corrected);

        // Fuzzy match
        let result = dict.correct("STRT");
        assert!(result.was_corrected);
        assert_eq!(result.corrected, "START");
    }
}