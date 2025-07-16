//! # Core Data Types
//!
//! This module defines the fundamental data types and structures used throughout the
//! BugguDB system. These types provide a consistent and efficient representation for
//! key entities such as tokens, document IDs, and log entries.

use std::time::{SystemTime, UNIX_EPOCH};

/// Represents a token, which is a fundamental unit of information in the search index.
///
/// A token is typically a word, a phrase, or an attribute extracted from a log entry.
/// It is represented as a `u64` for efficient storage and lookup.
pub type Tok = u64;

/// A unique identifier for a document, which corresponds to a single log entry.
///
/// Each document in the database is assigned a unique `DocId` to allow for precise
/// retrieval and management.
pub type DocId = u64;

/// Defines the tokenization mode for a log entry.
///
/// This enum allows for different strategies when processing log content, enabling
/// a balance between structured data extraction and full-text indexing.
#[derive(Debug, Clone, Copy)]
pub enum TokenMode {
    /// Only parses structured `field:value` pairs, ignoring unstructured text.
    Structured,
    /// Performs full-text indexing, including N-grams and pattern extraction.
    FullText,
    /// Combines both structured parsing and full-text indexing.
    Mixed,
}

/// Represents a single log entry with its associated metadata.
///
/// This struct is the primary data structure for storing log information. It includes
/// the raw content, a timestamp, and optional fields for log level and service name.
#[derive(Debug, Clone)]
pub struct LogEntry {
    /// The unique identifier for the log entry, assigned by BugguDB.
    pub id: DocId,
    /// The raw, unprocessed content of the log entry.
    pub content: String,
    /// The timestamp of the log entry, in seconds since the Unix epoch.
    pub timestamp: u64,
    /// The log level (e.g., "INFO", "WARN"), if available.
    pub level: Option<String>,
    /// The name of the service that generated the log, if available.
    pub service: Option<String>,
    /// The tokenization mode to be used for this log entry.
    pub mode: TokenMode,
}

impl LogEntry {
    /// Creates a new `LogEntry` with the given content and tokenization mode.
    ///
    /// The `id` is initialized to 0 and will be assigned by BugguDB upon insertion.
    /// The timestamp is set to the current time.
    pub fn new(content: String, mode: TokenMode) -> Self {
        Self {
            id: 0, // The ID will be assigned by BugguDB during insertion.
            content,
            timestamp: now_secs(),
            level: None,
            service: None,
            mode,
        }
    }

    /// Creates a new `LogEntry` with full metadata.
    ///
    /// This constructor allows for the creation of a `LogEntry` with all fields
    /// specified, providing maximum flexibility for structured logging.
    pub fn with_metadata(
        content: String,
        level: Option<String>,
        service: Option<String>,
        mode: TokenMode,
    ) -> Self {
        Self {
            id: 0, // The ID will be assigned by BugguDB.
            content,
            timestamp: now_secs(),
            level,
            service,
            mode,
        }
    }
}

/// Returns the current time in seconds since the Unix epoch.
///
/// This is a convenience function for creating timestamps for log entries.
#[inline]
pub fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
