#![allow(clippy::needless_return)]

//! # LogDB Core
//!
//! This module provides the core functionality for `LogDB`, an in-memory log indexing
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
}

/// The main database structure for `LogDB`.
///
/// This struct holds all the data necessary for indexing and searching log entries,
/// including the token-to-document postings, document metadata, and various indexes.
#[derive(Debug, Clone)]
pub struct LogDB {
    /// The tokenizer and hasher for processing log content.
    ufhg: UFHGHeadquarters,
    /// The postings list, mapping tokens to the documents that contain them.
    postings: BugguHashSet<Tok, Posting>,
    /// A map from `DocId` to the `MetaEntry` containing the document's data.
    docs: BugguHashSet<DocId, MetaEntry>,
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
    /// The configuration for the `LogDB` instance.
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

impl LogDB {
    /// Creates a new `LogDB` with a default configuration.
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

    /// Creates a new `LogDB` with the given configuration.
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

    /// Creates a new `LogDB` from a configuration file.
    pub fn from_config_file(path: &str) -> std::io::Result<Self> {
        let config = LogConfig::from_file(path)?;
        Ok(Self::with_config(config))
    }

    /// Inserts or updates a log entry with the given content and metadata.
    pub fn upsert_log(
        &mut self,
        content: &str,
        level: Option<String>,
        service: Option<String>,
    ) -> DocId {
        let descriptor = match (&level, &service) {
            (Some(l), Some(s)) => format!("level {l} service {s} content {content}"),
            (Some(l), None) => format!("level {l} content {content}"),
            (None, Some(s)) => format!("service {s} content {content}"),
            (None, None) => format!("content {content}"),
        };

        let (_, token_slice_cloned) = self.ufhg.tokenize_zero_copy(&descriptor);
        let doc_id = self.next_doc_id;
        self.next_doc_id += 1;

        let entry = MetaEntry {
            tokens: token_slice_cloned.clone(),
            level: level.clone(),
            service: service.clone(),
            content: content.to_string(),
        };

        self.docs.insert(doc_id, entry);

        // Update postings
        for &tok in &token_slice_cloned {
            self.postings
                .entry(tok)
                .or_insert_with(Posting::new)
                .add(doc_id);
        }

        // Update indexes
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
        self.upsert_log(content, None, None)
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
fn parse_query(q: &str, config: &LogConfig) -> QueryNode {
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
                _ => nodes.push(QueryNode::Term(tok.to_string())),
            }
        } else if tok.starts_with('"') {
            let phrase = tok.trim_matches('"').to_string();
            nodes.push(QueryNode::Phrase(phrase));
        } else {
            nodes.push(QueryNode::Term(tok.to_string()));
        }
    }

    if nodes.len() == 1 {
        nodes.pop().unwrap()
    } else {
        QueryNode::And(nodes)
    }
}
