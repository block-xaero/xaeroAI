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
        Self::default()
    }

    /// Load dictionary from file
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let dict: Dictionary = serde_json::from_str(&content)?;
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

        // Add to frequency map
        *self.terms.entry(normalized.clone()).or_insert(0) += 1;

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

        // Check exact user corrections first
        if let Some(corrected) = self.corrections.get(&text.to_lowercase()) {
            return CorrectionResult {
                original: text.to_string(),
                corrected: corrected.clone(),
                confidence: 1.0,
                was_corrected: true,
                source: Some(DomainSource::Personal),
            };
        }

        // Check if already a known term
        if self.terms.contains_key(text) {
            return CorrectionResult {
                original: text.to_string(),
                corrected: text.to_string(),
                confidence: 1.0,
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
            source: Some(DomainSource::Personal), // Simplified
        }
    }

    /// Find best fuzzy match in dictionary
    fn fuzzy_match(&self, text: &str) -> Option<(String, usize, DomainSource)> {
        let text_lower = text.to_lowercase();
        let max_distance = self.max_allowed_distance(text.len());

        let mut best: Option<(String, usize, DomainSource)> = None;

        // Search all domain sources, prioritizing certain ones
        let priority_order = [
            DomainSource::DiagramLabels,  // Current context first
            DomainSource::Personal,        // User's corrections
            DomainSource::CodeSymbols,     // Code terms
            DomainSource::Jira,
            DomainSource::Confluence,
        ];

        for source in &priority_order {
            if let Some(terms) = self.domains.get(source) {
                for term in terms {
                    let distance = self.levenshtein(&text_lower, &term.to_lowercase());
                    
                    if distance <= max_distance {
                        let dominated = best.as_ref()
                            .map(|(_, d, _)| distance < *d)
                            .unwrap_or(true);
                        
                        if dominated {
                            best = Some((term.clone(), distance, source.clone()));
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
        match len {
            0..=3 => 1,      // Short words: 1 error max
            4..=6 => 2,      // Medium: 2 errors
            7..=12 => 3,     // Long: 3 errors
            _ => 4,          // Very long: 4 errors
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
                .map(|(k, v)| (k.clone(), v.len()))
                .collect(),
        }
    }

    /// Seed with common diagramming terms
    pub fn seed_common_terms(&mut self) {
        let common = [
            // Flowchart
            "Start", "End", "Yes", "No", "True", "False",
            "Input", "Output", "Process", "Decision",
            // Architecture
            "User", "Client", "Server", "Database", "API",
            "Service", "Controller", "Model", "View",
            "Request", "Response", "Authentication", "Authorization",
            // Actions
            "Create", "Read", "Update", "Delete",
            "Send", "Receive", "Validate", "Transform",
            "Login", "Logout", "Register", "Submit",
            // Technical
            "HTTP", "REST", "GraphQL", "WebSocket",
            "JSON", "XML", "Token", "Session",
            "Cache", "Queue", "Event", "Message",
        ];

        self.add_terms(&common, DomainSource::DiagramLabels);
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
        dict.add_term("Authentication", DomainSource::DiagramLabels);
        
        let result = dict.correct("Authntication"); // missing 'e'
        assert!(result.was_corrected);
        assert_eq!(result.corrected, "Authentication");
    }

    #[test]
    fn test_user_correction() {
        let mut dict = Dictionary::new();
        dict.add_correction("usr", "User");
        
        let result = dict.correct("usr");
        assert!(result.was_corrected);
        assert_eq!(result.corrected, "User");
    }

    #[test]
    fn test_phrase_correction() {
        let mut dict = Dictionary::new();
        dict.add_term("User", DomainSource::DiagramLabels);
        dict.add_term("Authentication", DomainSource::DiagramLabels);
        
        let result = dict.correct_phrase("Usr Authntication");
        assert!(result.was_corrected);
        assert_eq!(result.corrected, "User Authentication");
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
        
        assert!(dict.terms.contains_key("User"));
        assert!(dict.terms.contains_key("MyService"));
    }
}
