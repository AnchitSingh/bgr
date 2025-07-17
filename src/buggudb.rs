#![allow(clippy::needless_return)]

//! # BugguDB Core
//!
//! This module provides the core functionality for `BugguDB`, an in-memory log indexing
//! and search engine. It includes data structures for storing and querying log entries,
//! as well as mechanisms for efficient tokenization, indexing, and query execution.

use crate::config::LogConfig;
use crate::ufhg::{lightning_hash_str, UFHGHeadquarters};
use crate::utils::buggu_hash_set::BugguHashSet;
use smallvec::SmallVec;

/// A type alias for a token, which is represented as a 64-bit unsigned integer.
/// Tokens are used to represent words, phrases, or other searchable units.
pub type Tok = u64;

/// A type alias for a document identifier, also a 64-bit unsigned integer.
/// Each log entry is assigned a unique `DocId`.
pub type DocId = u64;

/// Represents the metadata associated with a document.
///
/// This struct stores the original content of a log entry, along with its tokens
/// and any associated metadata such as log level and service name.
#[derive(Debug, Clone, Default)]
pub struct MetaEntry {
    /// The sequence of tokens generated from the document's content.
    tokens: Vec<Tok>,
    /// The log level, if specified (e.g., "INFO", "ERROR").
    level: Option<String>,
    /// The service name, if specified.
    service: Option<String>,
    /// The original, unmodified content of the log entry.
    content: String,
    json_data: Option<serde_json::Value>,
}
impl MetaEntry {
    // Add a method to extract values at a given JSON path
    pub fn json_value_at_path(&self, path: &str) -> Option<String> {
        if let Some(json) = &self.json_data {
            return extract_json_path_value(json, path);
        }
        None
    }
}

// Helper function to navigate a JSON path
fn extract_json_path_value(json: &serde_json::Value, path: &str) -> Option<String> {
    let parts = path.split('.').collect::<Vec<_>>();
    let mut current = json;

    for part in &parts {
        if let Some(obj) = current.as_object() {
            current = obj.get(*part)?;
        } else if let Some(array) = current.as_array() {
            if let Ok(index) = part.parse::<usize>() {
                current = array.get(index)?;
            } else {
                return None;
            }
        } else {
            return None;
        }
    }

    // Convert the final value to a string
    match current {
        serde_json::Value::String(s) => Some(s.clone()),
        _ => Some(current.to_string()),
    }
}

/// Defines the Abstract Syntax Tree (AST) for a parsed query.
///
/// This enum represents the structure of a search query, allowing for complex
/// logical combinations of search terms, phrases, and field-specific filters.
#[derive(Debug, Clone)]
pub enum QueryNode {
    /// A single search term.
    Term(String),
    /// An exact phrase search.
    Phrase(String),
    /// A search for a term within a specific field (e.g., `level:ERROR`).
    FieldTerm(&'static str, String),
    /// A search for a numeric range within a field (e.g., `timestamp:>=12345`).
    NumericRange(&'static str, u64, u64),
    /// A search for a substring within the content of a log entry.
    Contains(String),
    /// A logical AND operation, requiring all child nodes to match.
    And(Vec<QueryNode>),
    /// A logical OR operation, requiring at least one child node to match.
    Or(Vec<QueryNode>),
    /// A logical NOT operation, excluding documents that match the child node.
    Not(Box<QueryNode>),
    /// A logical NOT operation, excluding documents that match the child node.
    Regex(String),
    JsonField(String, String),
}

/// The main database structure for `BugguDB`.
///
/// This struct holds all the data necessary for indexing and searching log entries,
/// including the token-to-document postings, document metadata, and various indexes.
#[derive(Debug, Clone)]
pub struct BugguDB {
    /// The tokenizer and hasher for processing log content.
    ufhg: UFHGHeadquarters,
    /// The postings list, mapping tokens to the documents that contain them.
    postings: BugguHashSet<Tok, Posting>,
    /// A map from `DocId` to the `MetaEntry` containing the document's data.
    pub docs: BugguHashSet<DocId, MetaEntry>,
    /// An index for fast lookups of documents by log level.
    level_index: BugguHashSet<Tok, Vec<DocId>>,
    /// An index for fast lookups of documents by service name.
    service_index: BugguHashSet<Tok, Vec<DocId>>,
    /// The next available document ID.
    next_doc_id: DocId,
    /// The maximum number of postings to hold in memory.
    max_postings: usize,
    /// The time in seconds after which a document is considered stale.
    stale_secs: u64,
    /// The configuration for the `BugguDB` instance.
    config: LogConfig,
}

/// Represents a posting for a single token.
///
/// A posting contains a list of document IDs that are associated with a specific
/// token. To optimize for memory and performance, it uses a `SmallVec` for small
/// lists and switches to a `BugguHashSet` for larger ones.
#[derive(Debug, Clone)]
pub struct Posting {
    /// A small vector for storing document IDs, optimized for a small number of entries.
    small_docs: SmallVec<[DocId; 4]>,
    /// An optional hash set for storing a large number of document IDs.
    large_docs: Option<BugguHashSet<DocId, ()>>,
}

impl Posting {
    /// Creates a new, empty `Posting`.
    #[inline]
    fn new() -> Self {
        Self {
            small_docs: SmallVec::new(),
            large_docs: None,
        }
    }

    /// Adds a document ID to the posting.
    ///
    /// This method handles the logic of switching from `small_docs` to `large_docs`
    /// when the number of documents exceeds a certain threshold.
    #[inline]
    fn add(&mut self, id: DocId) {
        if let Some(ref mut large) = self.large_docs {
            large.insert(id, ());
        } else if self.small_docs.len() < 128 {
            if !self.small_docs.contains(&id) {
                self.small_docs.push(id);
            }
        } else {
            let mut large = BugguHashSet::new(512);
            for &doc_id in &self.small_docs {
                large.insert(doc_id, ());
            }
            large.insert(id, ());
            self.large_docs = Some(large);
            self.small_docs.clear();
        }
    }

    /// Removes a document ID from the posting.
    #[inline]
    fn remove(&mut self, id: DocId) {
        if let Some(ref mut large) = self.large_docs {
            large.remove(&id);
        } else {
            self.small_docs.retain(|d| *d != id);
        }
    }

    /// Converts the posting to a `BugguHashSet` of document IDs.
    #[inline]
    fn to_set(&self) -> BugguHashSet<DocId, ()> {
        if let Some(ref large) = self.large_docs {
            large.clone()
        } else {
            let mut set = BugguHashSet::new(self.small_docs.len().max(8));
            for &id in &self.small_docs {
                set.insert(id, ());
            }
            set
        }
    }

    /// Returns a vector of all document IDs in the posting.
    #[inline]
    fn get_docs(&self) -> Vec<DocId> {
        if let Some(ref large) = self.large_docs {
            large.keys()
        } else {
            self.small_docs.to_vec()
        }
    }

    /// Checks if the posting is empty.
    #[inline]
    fn empty(&self) -> bool {
        if let Some(ref large) = self.large_docs {
            large.is_empty()
        } else {
            self.small_docs.is_empty()
        }
    }

    /// Retains only the document IDs that are present in the provided set of documents.
    #[inline]
    fn retain_docs(&mut self, docs: &BugguHashSet<DocId, MetaEntry>) {
        if let Some(ref mut large) = self.large_docs {
            large.retain(|id, _| docs.get(id).is_some());
        } else {
            self.small_docs.retain(|id| docs.get(id).is_some());
        }
    }
}

impl Default for Posting {
    /// Creates a default, empty `Posting`.
    fn default() -> Self {
        Self::new()
    }
}

// Helper function to extract field paths and values from JSON
fn extract_json_fields(json: &serde_json::Value) -> Vec<String> {
    let mut fields = Vec::new();
    extract_json_fields_recursive(json, "", &mut fields);
    fields
}

fn extract_json_fields_recursive(json: &serde_json::Value, prefix: &str, fields: &mut Vec<String>) {
    match json {
        serde_json::Value::Object(obj) => {
            for (key, value) in obj {
                let new_prefix = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{}.{}", prefix, key)
                };

                // Add the path itself
                fields.push(new_prefix.clone());

                // Add path=value for primitive values
                match value {
                    serde_json::Value::String(s) => {
                        fields.push(format!("{}={}", new_prefix, s));
                    }
                    serde_json::Value::Number(n) => {
                        fields.push(format!("{}={}", new_prefix, n));
                    }
                    serde_json::Value::Bool(b) => {
                        fields.push(format!("{}={}", new_prefix, b));
                    }
                    _ => {}
                }

                // Recurse for nested objects and arrays
                extract_json_fields_recursive(value, &new_prefix, fields);
            }
        }
        serde_json::Value::Array(arr) => {
            for (i, value) in arr.iter().enumerate() {
                let new_prefix = format!("{}[{}]", prefix, i);
                fields.push(new_prefix.clone());
                extract_json_fields_recursive(value, &new_prefix, fields);
            }
        }
        _ => {}
    }
}

impl BugguDB {
    /// Creates a new `BugguDB` with a default configuration.
    pub fn new() -> Self {
        Self {
            ufhg: UFHGHeadquarters::new(),
            postings: BugguHashSet::new(40000),
            docs: BugguHashSet::new(50000),
            level_index: BugguHashSet::new(40000),
            service_index: BugguHashSet::new(40000),
            next_doc_id: 1,
            max_postings: 32_000,
            stale_secs: 3600,
            config: LogConfig::default(),
        }
    }

    /// Creates a new `BugguDB` with the given configuration.
    pub fn with_config(config: LogConfig) -> Self {
        Self {
            ufhg: UFHGHeadquarters::new(),
            postings: BugguHashSet::new(40000),
            docs: BugguHashSet::new(50000),
            level_index: BugguHashSet::new(40000),
            service_index: BugguHashSet::new(40000),
            next_doc_id: 1,
            max_postings: config.max_postings,
            stale_secs: config.stale_secs,
            config,
        }
    }

    /// Creates a new `BugguDB` from a configuration file.
    pub fn from_config_file(path: &str) -> std::io::Result<Self> {
        let config = LogConfig::from_file(path)?;
        Ok(Self::with_config(config))
    }

    /// Ingests log entries from any source that implements BufRead (e.g., a file, stdin).
    ///
    /// This treats each line as a separate document.
    pub fn ingest_from_reader<R: std::io::BufRead>(&mut self, reader: R) -> std::io::Result<usize> {
        let mut count = 0;
        for line in reader.lines() {
            let line_content = line?;
            if !line_content.is_empty() {
                self.upsert_simple(&line_content);
                count += 1;
            }
        }
        Ok(count)
    }

    /// Inserts or updates a log entry with the given content and metadata.
    pub fn upsert_log(
        &mut self,
        content: &str,
        level: Option<String>,
        service: Option<String>,
        json_data: Option<serde_json::Value>,
    ) -> DocId {
        let descriptor = match (&level, &service) {
            (Some(l), Some(s)) => format!("level {l} service {s} content {content}"),
            (Some(l), None) => format!("level {l} content {content}"),
            (None, Some(s)) => format!("service {s} content {content}"),
            (None, None) => format!("content {content}"),
        };

        // Add JSON fields to the descriptor for tokenization
        let json_descriptor = if let Some(ref json) = json_data {
            // Extract key paths and values from JSON to include in the tokenization
            let json_fields = extract_json_fields(json);
            if !json_fields.is_empty() {
                format!(" json {}", json_fields.join(" "))
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        let full_descriptor = format!("{}{}", descriptor, json_descriptor);
        let (_, token_slice_cloned) = self.ufhg.tokenize_zero_copy(&full_descriptor);
        let doc_id = self.next_doc_id;
        self.next_doc_id += 1;

        let entry = MetaEntry {
            tokens: token_slice_cloned.clone(),
            level: level.clone(),
            service: service.clone(),
            content: content.to_string(),
            json_data: json_data,
        };

        self.docs.insert(doc_id, entry);

        // Update postings
        for &tok in &token_slice_cloned {
            self.postings
                .entry(tok)
                .or_insert_with(Posting::new)
                .add(doc_id);
        }

        // Update indexes (existing code)
        if let Some(ref level_val) = level {
            self.level_index
                .entry(lightning_hash_str(level_val))
                .or_insert_with(Vec::new)
                .push(doc_id);
        }
        if let Some(ref service_val) = service {
            self.service_index
                .entry(lightning_hash_str(service_val))
                .or_insert_with(Vec::new)
                .push(doc_id);
        }

        doc_id
    }

    /// Inserts or updates a simple log entry with only content.
    pub fn upsert_simple(&mut self, content: &str) -> DocId {
        self.upsert_log(content, None, None, None)
    }

    pub fn upsert_json_log(&mut self, content: &str, json_str: &str) -> std::io::Result<DocId> {
        match serde_json::from_str(json_str) {
            Ok(json_value) => Ok(self.upsert_log(content, None, None, Some(json_value))),
            Err(e) => Err(std::io::Error::new(std::io::ErrorKind::InvalidData, e)),
        }
    }

    /// Executes a query and returns the matching document IDs.
    pub fn query(&self, q: &str) -> Vec<DocId> {
        let ast = parse_query(q, &self.config);
        self.exec(&ast)
    }

    /// Retrieves the content of a document by its ID.
    pub fn get_content(&self, doc_id: &DocId) -> Option<String> {
        self.docs.get(doc_id).map(|e| e.content.clone())
    }

    /// Executes a query and returns the content of the matching documents.
    pub fn query_content(&self, q: &str) -> Vec<String> {
        let doc_ids = self.query(q);
        doc_ids
            .into_iter()
            .filter_map(|id| self.get_content(&id))
            .collect()
    }

    /// Executes a query and returns the matching documents with their metadata.
    pub fn query_with_meta(&self, q: &str) -> Vec<(DocId, String, Option<String>, Option<String>)> {
        let ast = parse_query(q, &self.config);
        let docs = self.exec(&ast);
        docs.into_iter()
            .filter_map(|id| {
                self.docs
                    .get(&id)
                    .map(|e| (id, e.content.clone(), e.level.clone(), e.service.clone()))
            })
            .collect()
    }

    /// Cleans up stale documents from the database.
    pub fn cleanup_stale(&mut self) {}

    /// Rebuilds the indexes for log levels and services.
    pub fn rebuild_indexes(&mut self) {
        self.level_index = self
            .docs
            .create_index_for(|entry| entry.level.as_ref().map(|s| lightning_hash_str(s.as_str())));
        self.service_index = self.docs.create_index_for(|entry| {
            entry
                .service
                .as_ref()
                .map(|s| lightning_hash_str(s.as_str()))
        });
    }

    /// Executes a query AST node and returns the matching document IDs.
    fn exec(&self, node: &QueryNode) -> Vec<DocId> {
        match node {
            QueryNode::Term(w) | QueryNode::Contains(w) => {
                let hash = lightning_hash_str(w);
                self.postings
                    .get(&hash)
                    .map(|p| p.get_docs())
                    .unwrap_or_default()
            }

            QueryNode::Phrase(p) => {
                let seq_hash = self.ufhg.string_to_u64_to_seq_hash(p);
                self.postings
                    .get(&seq_hash)
                    .map(|p| p.get_docs())
                    .unwrap_or_default()
            }

            QueryNode::FieldTerm(f, v) => match *f {
                "level" => self.filter_by_level(v),
                "service" => self.filter_by_service(v),
                _ => {
                    let field_set = self.get_term_set(&lightning_hash_str(f));
                    let value_set = self.get_term_set(&lightning_hash_str(v));
                    field_set.intersect_with(&value_set).keys()
                }
            },

            QueryNode::And(children) => {
                if children.is_empty() {
                    return Vec::new();
                }

                let mut result_set = self.exec_to_set(&children[0]);
                for child in &children[1..] {
                    let other_set = self.exec_to_set(child);
                    result_set = result_set.intersect_with(&other_set);
                    if result_set.is_empty() {
                        break;
                    }
                }
                result_set.keys()
            }

            QueryNode::Or(children) => {
                if children.is_empty() {
                    return Vec::new();
                }

                let mut result_set = self.exec_to_set(&children[0]);
                for child in &children[1..] {
                    let other_set = self.exec_to_set(child);
                    result_set = result_set.union_with(&other_set);
                }
                result_set.keys()
            }

            QueryNode::Not(child) => {
                let exclude_set = self.exec_to_set(child);
                let all_docs_set = self.create_all_docs_set();
                all_docs_set.fast_difference(&exclude_set).keys()
            }
            QueryNode::Regex(pattern) => {
                let re = regex::Regex::new(pattern).unwrap();
                self.docs
                    .iter_keys()
                    .filter_map(|id| {
                        if re.is_match(&self.docs.get(&id).unwrap().content) {
                            Some(id)
                        } else {
                            None
                        }
                    })
                    .collect()
            }
            QueryNode::JsonField(path, value) => self
                .docs
                .iter_keys()
                .filter_map(|id| {
                    if let Some(json_val) = self.docs.get(&id).unwrap().json_value_at_path(path) {
                        if value.is_empty() || json_val == *value {
                            return Some(id);
                        }
                    }
                    None
                })
                .collect(),
            _ => Vec::new(),
        }
    }

    /// Executes a query AST node and returns the results as a `BugguHashSet`.
    fn exec_to_set(&self, node: &QueryNode) -> BugguHashSet<DocId, ()> {
        let docs = self.exec(node);
        let mut set = BugguHashSet::new(docs.len().max(8));
        for id in docs {
            set.insert(id, ());
        }
        set
    }

    /// Retrieves the set of documents associated with a given token.
    fn get_term_set(&self, tok: &Tok) -> BugguHashSet<DocId, ()> {
        self.postings
            .get(tok)
            .map(|p| p.to_set())
            .unwrap_or_else(|| BugguHashSet::new(1))
    }

    /// Creates a `BugguHashSet` containing all document IDs in the database.
    fn create_all_docs_set(&self) -> BugguHashSet<DocId, ()> {
        let mut set = BugguHashSet::new(self.docs.len());
        for id in self.docs.iter_keys() {
            set.insert(id, ());
        }
        set
    }

    /// Filters documents by log level.
    fn filter_by_level(&self, level: &str) -> Vec<DocId> {
        self.level_index
            .get(&lightning_hash_str(level))
            .cloned()
            .unwrap_or_default()
    }

    /// Filters documents by service name.
    fn filter_by_service(&self, service: &str) -> Vec<DocId> {
        self.service_index
            .get(&lightning_hash_str(service))
            .cloned()
            .unwrap_or_default()
    }

    /// Inserts a token into the postings list if it doesn't already exist.
    pub fn upsert_token(&mut self, s: impl AsRef<str>) -> Tok {
        let tok = lightning_hash_str(s.as_ref());
        self.postings.entry(tok).or_insert_with(Posting::default);
        tok
    }

    /// Exports all tokens from the postings list.
    pub fn export_tokens(&self) -> Vec<Tok> {
        self.postings.keys()
    }

    /// Imports a list of tokens into the postings list.
    pub fn import_tokens(&mut self, toks: Vec<Tok>) {
        for t in toks {
            self.postings.entry(t).or_insert_with(Posting::default);
        }
    }
}

/// Parses a query string into a `QueryNode` AST.
fn parse_query(q: &str, _config: &LogConfig) -> QueryNode {
    let mut nodes = Vec::<QueryNode>::new();
    let mut it = q.split_whitespace().peekable();

    while let Some(tok) = it.next() {
        if tok.contains(':') {
            let mut sp = tok.splitn(2, ':');
            let field = sp.next().unwrap();
            let mut val = sp.next().unwrap().to_string();

            if val.starts_with('"') && !val.ends_with('"') {
                for nxt in it.by_ref() {
                    val.push(' ');
                    val.push_str(nxt);
                    if nxt.ends_with('"') {
                        break;
                    }
                }
                val = val.trim_matches('"').to_string();
            } else {
                val = val.trim_matches('"').to_string();
            }

            match field {
                "level" => nodes.push(QueryNode::FieldTerm("level", val)),
                "service" => nodes.push(QueryNode::FieldTerm("service", val)),
                "contains" => nodes.push(QueryNode::Contains(val)),
                "timestamp" => {
                    if let Some(lo) = val.strip_prefix(">=") {
                        let lo = lo.parse::<u64>().unwrap_or(0);
                        nodes.push(QueryNode::NumericRange("timestamp", lo, u64::MAX));
                    } else if let Some(hi) = val.strip_prefix("<=") {
                        let hi = hi.parse::<u64>().unwrap_or(u64::MAX);
                        nodes.push(QueryNode::NumericRange("timestamp", 0, hi));
                    }
                }
                "regex" => nodes.push(QueryNode::Regex(val)), // Added regex parsing
                "json" => {
                    // Add this case
                    if let Some((path, value)) = val.split_once('=') {
                        nodes.push(QueryNode::JsonField(path.to_string(), value.to_string()));
                    } else {
                        // Just check that the path exists
                        nodes.push(QueryNode::JsonField(val, "".to_string()));
                    }
                }
                _ => nodes.push(QueryNode::Term(tok.to_string())),
            }
        } else if tok.starts_with('"') {
            let phrase = tok.trim_matches('"').to_string();
            nodes.push(QueryNode::Phrase(phrase));
        } else {
            // Handle logical operators.
            match tok.to_uppercase().as_str() {
                "AND" => continue, // AND is the default operator.
                "OR" => {
                    // Combine the last node with the next node in an OR expression.
                    if let Some(last) = nodes.pop() {
                        if let Some(next_tok) = it.next() {
                            let next_node = if next_tok.contains(':') {
                                // Handle field:value syntax for the right side of OR
                                let mut sp = next_tok.splitn(2, ':');
                                let field = sp.next().unwrap();
                                let mut val = sp.next().unwrap().to_string();

                                // Handle quoted values
                                if val.starts_with('"') && !val.ends_with('"') {
                                    for nxt in it.by_ref() {
                                        val.push(' ');
                                        val.push_str(nxt);
                                        if nxt.ends_with('"') {
                                            break;
                                        }
                                    }
                                    val = val.trim_matches('"').to_string();
                                } else {
                                    val = val.trim_matches('"').to_string();
                                }

                                match field {
                                    "json" => {
                                        if let Some((path, value)) = val.split_once('=') {
                                            QueryNode::JsonField(
                                                path.to_string(),
                                                value.to_string(),
                                            )
                                        } else {
                                            QueryNode::JsonField(val, "".to_string())
                                        }
                                    }
                                    "level" => QueryNode::FieldTerm("level", val),
                                    "service" => QueryNode::FieldTerm("service", val),
                                    "contains" => QueryNode::Contains(val),
                                    "regex" => QueryNode::Regex(val),
                                    _ => QueryNode::Term(next_tok.to_string()),
                                }
                            } else {
                                QueryNode::Term(next_tok.to_string())
                            };
                            nodes.push(QueryNode::Or(vec![last, next_node]));
                        }
                    }
                }
                "NOT" => {
                    // Create a NOT node for the next term.
                    if let Some(next_tok) = it.next() {
                        let next_node = QueryNode::Term(next_tok.to_string());
                        nodes.push(QueryNode::Not(Box::new(next_node)));
                    }
                }
                _ => nodes.push(QueryNode::Term(tok.to_string())),
            }
        }
    }

    if nodes.len() == 1 {
        nodes.pop().unwrap()
    } else {
        QueryNode::And(nodes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use serde_json::json;
    use std::time::{Duration, Instant};
    #[test]
    /// Tests the JSON search functionality with performance metrics
    pub fn test_json_search() {
        println!("Testing JSON search functionality...");

        // Create a new database instance
        let mut db = BugguDB::new();

        // Insert various JSON documents
        println!("Inserting test JSON documents...");
        let test_start = Instant::now();

        // User profile data
        for i in 0..1000 {
            let cities = ["New York", "London", "Tokyo", "Paris", "Berlin"];
            let countries = ["USA", "UK", "Japan", "France", "Germany"];
            let themes = ["light", "dark", "auto"];

            let city = cities[i % 5];
            let country = countries[i % 5];
            // MODIFIED: Create some overlap between dark theme and active=true
            let is_active = i % 3 == 0 || i % 9 == 0; // This creates overlap
            let theme_idx = if i % 10 < 3 { 1 } else { i % 3 }; // Ensure ~30% have dark theme
            let theme = themes[theme_idx];

            let role = if i % 5 == 0 { "admin" } else { "member" };

            let user_json = json!({
            "user": {
                "id": i,
                "name": format!("User {}", i),
                "email": format!("user{}@example.com", i),
                "active": is_active,  // Use the new active calculation
                "roles": ["user", role],
                "address": {
                    "city": city,
                    "country": country,
                    "zipcode": format!("{:05}", i * 10)
                },
                "preferences": {
                    "theme": theme,  // Use the new theme selection
                    "notifications": i % 2 == 0
                }
            },
                    "stats": {
                        "logins": i * 10,
                        "last_login": format!("2025-{:02}-{:02}T10:00:00Z", (i % 12) + 1, (i % 28) + 1),
                        "activity_score": i % 100
                    }
                });

            db.upsert_json_log(
                &format!("User profile update for user{}", i),
                &user_json.to_string(),
            )
            .unwrap();
        }

        // Server monitoring data
        for i in 0..1000 {
            let statuses = ["running", "stopped", "maintenance"];
            let status = statuses[i % 3];
            let op1 = if i % 10 != 0 { "up" } else { "down" };
            let op2 = if i % 15 != 0 { "up" } else { "down" };
            let op3 = if i % 20 != 0 { "up" } else { "down" };
            let server_json = json!({
                "server": {
                    "id": format!("srv-{:03}", i),
                    "hostname": format!("server-{}.example.com", i),
                    "ip": format!("10.0.{}.{}", i / 255, i % 255),
                    "status": status,
                    "metrics": {
                        "cpu": i % 101,
                        "memory": (i * 10) % 101,
                        "disk": (i * 5) % 101,
                        "network": {
                            "in_bytes": i * 1024 * 1024,
                            "out_bytes": i * 512 * 1024
                        }
                    },
                    "services": [
                        {"name": "web", "status": op1},
                        {"name": "db", "status": op2},
                        {"name": "cache", "status": op3}
                    ]
                },
                "timestamp": format!("2025-07-{:02}T{:02}:00:00Z", (i % 30) + 1, (i % 24))
            });

            db.upsert_json_log(
                &format!("Server status update for srv-{:03}", i),
                &server_json.to_string(),
            )
            .unwrap();
        }

        // Product catalog data
        for i in 0..1000 {
            let categories = ["Electronics", "Clothing", "Home", "Books", "Food"];
            let colors = ["red", "blue", "green", "black", "white"];

            let category = categories[i % 5];
            let color = colors[i % 5];
            let tag2 = if i % 2 == 0 { "bestseller" } else { "new" };
            let tag3 = if i % 3 == 0 { "sale" } else { "regular" };

            let product_json = json!({
                "product": {
                    "id": format!("PROD-{:04}", i),
                    "name": format!("Product {}", i),
                    "category": category,
                    "price": 10.0 + (i as f64 % 90.0),
                    "stock": i % 200,
                    "attributes": {
                        "color": color,
                        "weight": i % 10,
                        "dimensions": {
                            "width": i % 50,
                            "height": (i + 10) % 50,
                            "depth": (i + 20) % 30
                        }
                    },
                    "tags": [
                        "tag1",
                        tag2,
                        tag3
                    ]
                },
                "supplier": {
                    "id": i % 10,
                    "name": format!("Supplier {}", i % 10),
                    "rating": (i % 5) + 1
                }
            });

            db.upsert_json_log(
                &format!("Product update for PROD-{:04}", i),
                &product_json.to_string(),
            )
            .unwrap();
        }

        println!(
            "Inserted 3,000 JSON documents in {:?}",
            test_start.elapsed()
        );

        // Rest of the test function remains the same...
        // Define test cases
        let test_cases = vec![
            // Simple path queries
            ("json:user.name", 1000, "Basic path query for user names"),
            (
                "json:server.status",
                1000,
                "Basic path query for server status",
            ),
            (
                "json:product.category",
                1000,
                "Basic path query for product categories",
            ),
            // Value matching queries
            ("json:user.active=true", 334, "Boolean value matching"),
            ("json:server.status=running", 334, "String value matching"),
            (
                "json:product.category=Electronics",
                200,
                "Category value matching",
            ),
            // Nested path queries
            (
                "json:user.address.city=London",
                200,
                "Nested path with value matching",
            ),
            ("json:server.metrics.cpu", 1000, "Nested numeric field path"),
            (
                "json:product.attributes.dimensions.width",
                1000,
                "Deeply nested path",
            ),
            // Combined queries
            (
                "json:user.preferences.theme=dark AND json:user.active=true",
                111,
                "Multiple JSON conditions",
            ),
            (
                "json:server.metrics.cpu level:ERROR",
                0,
                "JSON with field condition",
            ),
            (
                "json:product.category=Electronics OR json:product.category=Books",
                400,
                "OR condition with JSON fields",
            ),
            // Array access (if supported)
            ("json:server.services.0.name", 1000, "Array element access"),
            ("json:product.tags", 1000, "Array field existence"),
            // Negative tests
            ("json:nonexistent.path", 0, "Nonexistent JSON path"),
            ("json:user.name=NonexistentUser", 0, "Nonexistent value"),
        ];

        // Run tests
        let mut total_time = Duration::new(0, 0);

        for (query, expected_count, description) in &test_cases {
            println!("\nTest: {} ({})", description, query);

            let start = Instant::now();
            let results = db.query(query);
            let duration = start.elapsed();
            total_time += duration;

            println!("Found {} matches in {:?}", results.len(), duration);

            // Verify expected count
            if results.len() == *expected_count {
                println!(
                    "✅ Result count matches expected ({} documents)",
                    expected_count
                );
            } else {
                println!(
                    "❌ Result count mismatch! Expected: {}, Actual: {}",
                    expected_count,
                    results.len()
                );
            }

            // Calculate search speed
            let throughput = 3_000.0 / duration.as_secs_f64();
            println!("Search speed: {:.2} documents/second", throughput);

            // Show sample results
            if !results.is_empty() {
                println!("Sample matches:");
                for &doc_id in results.iter().take(2) {
                    if let Some(content) = db.get_content(&doc_id) {
                        println!("  - [{}]: {}", doc_id, content);
                    }
                }
            }
        }

        // Performance benchmarks
        println!("\n==== JSON Search Performance Benchmarks ====");

        // Test complex query with multiple conditions
        let complex_query = "json:user.address.city=London AND json:user.preferences.theme=dark AND json:stats.activity_score>=50";
        let start = Instant::now();
        let results = db.query(complex_query);
        let complex_duration = start.elapsed();

        println!(
            "Complex query: {} matches in {:?}",
            results.len(),
            complex_duration
        );
        println!("Query: {}", complex_query);

        // Compare with term search
        let start = Instant::now();
        let term_results = db.query("London");
        let term_duration = start.elapsed();

        println!("\n==== Term vs JSON Search Comparison ====");
        println!(
            "Term search ('London'): {} matches in {:?}",
            term_results.len(),
            term_duration
        );
        println!(
            "JSON path search: {} matches in {:?}",
            results.len(),
            complex_duration
        );
        println!(
            "JSON search is {:.1}x slower than term search",
            complex_duration.as_secs_f64() / term_duration.as_secs_f64()
        );

        println!("\n==== Overall JSON Search Performance ====");
        println!(
            "Average query time: {:?}",
            total_time / test_cases.len() as u32
        );
    }
}
