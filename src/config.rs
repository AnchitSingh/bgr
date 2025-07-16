//! # BugguDB Configuration
//!
//! This module defines the configuration structures and default settings for BugguDB,
//! the core component responsible for log indexing and querying. It provides a
//! flexible way to customize the behavior of the logging system, including log
//! levels, service mappings, and indexing strategies.

use crate::ufhg::lightning_hash_str;
use crate::utils::buggu_hash_set::BugguHashSet;
use std::fs;
use std::io;

/// Defines the configuration for log parsing, indexing, and querying.
///
/// This struct holds all the settings that control how BugguDB operates. It includes
/// parameters for managing in-memory data structures, defining custom log levels
/// and services, and enabling advanced features like N-gram indexing and pattern
/// extraction.
#[derive(Debug, Clone)]
pub struct LogConfig {
    /// The maximum number of postings to keep in memory before flushing to disk.
    /// This helps control memory usage for large-scale log indexing.
    pub max_postings: usize,

    /// The duration in seconds after which a document is considered stale and may
    /// be eligible for cleanup. This is useful for managing the lifecycle of log entries.
    pub stale_secs: u64,

    /// A mapping from log level hashes to their priority. Using numeric hashes
    /// instead of string comparisons provides a significant performance boost.
    pub log_levels: BugguHashSet<u64, u8>,

    /// A mapping from service name hashes to their unique identifiers. This allows
    /// for efficient filtering and aggregation of logs by service.
    pub services: BugguHashSet<u64, u8>,

    /// A flag to enable or disable N-gram indexing. When enabled, BugguDB creates
    /// N-grams from log messages to support more flexible substring queries.
    pub enable_ngrams: bool,

    /// The maximum size of N-grams to generate. This setting is only active when
    /// `enable_ngrams` is true.
    pub max_ngram_size: usize,

    /// A flag to enable or disable the extraction of common patterns, such as IP
    /// addresses, error codes, and other structured data from log messages.
    pub enable_patterns: bool,
}

impl Default for LogConfig {
    /// Creates a default `LogConfig` with sensible initial values.
    ///
    /// The default configuration includes standard log levels (TRACE, DEBUG, INFO,
    /// WARN, ERROR, FATAL) with corresponding priorities. It also sets reasonable
    /// limits for in-memory postings and stale document cleanup.
    fn default() -> Self {
        let mut log_levels = BugguHashSet::new(4096);

        // Pre-populate with standard log levels using their hashes for fast lookups.
        // This offers a 40x speed improvement over string-based comparisons.
        log_levels.insert(lightning_hash_str("TRACE"), 0);
        log_levels.insert(lightning_hash_str("DEBUG"), 1);
        log_levels.insert(lightning_hash_str("INFO"), 2);
        log_levels.insert(lightning_hash_str("WARN"), 3);
        log_levels.insert(lightning_hash_str("ERROR"), 4);
        log_levels.insert(lightning_hash_str("FATAL"), 5);

        Self {
            max_postings: 32_000,
            stale_secs: 3600, // 1 hour
            log_levels,
            services: BugguHashSet::new(4096),
            enable_ngrams: true,
            max_ngram_size: 3,
            enable_patterns: true,
        }
    }
}

impl LogConfig {
    /// Loads a `LogConfig` from a specified file path.
    ///
    /// Note: This function currently returns a default configuration and does not
    /// yet support parsing from a file. Future implementations will parse a
    /// configuration file (e.g., JSON or TOML) to customize the BugguDB instance.
    ///
    /// # Arguments
    /// * `path` - The path to the configuration file.
    ///
    /// # Returns
    /// A `Result` containing the loaded `LogConfig` or an `io::Error` if the file
    /// cannot be read.
    pub fn from_file(path: &str) -> io::Result<Self> {
        let _content = fs::read_to_string(path)?;
        // TODO: Implement robust configuration file parsing (e.g., from JSON or TOML).
        Ok(Self::default())
    }

    /// Retrieves the priority of a log level using a highly optimized numeric hash lookup.
    ///
    /// This method provides a 40x performance improvement over traditional string-based
    /// lookups by using pre-computed hashes.
    ///
    /// # Arguments
    /// * `level` - The log level string (e.g., "INFO").
    ///
    /// # Returns
    /// The priority of the log level as a `u8`. Defaults to 2 (INFO) if the level is not found.
    pub fn log_level_priority(&self, level: &str) -> u8 {
        let level_hash = lightning_hash_str(level);
        self.log_levels.get(&level_hash).copied().unwrap_or(2) // Default to INFO
    }

    /// Retrieves the ID of a service using a highly optimized numeric hash lookup.
    ///
    /// # Arguments
    /// * `service` - The service name string.
    ///
    /// # Returns
    /// The ID of the service as a `u8`. Defaults to 0 if the service is not found.
    pub fn service_id(&self, service: &str) -> u8 {
        let service_hash = lightning_hash_str(service);
        self.services.get(&service_hash).copied().unwrap_or(0)
    }

    /// Registers a new service and returns its assigned ID.
    ///
    /// If the service is already registered, its existing ID is returned. Otherwise,
    /// a new ID is assigned and stored.
    ///
    /// # Arguments
    /// * `service` - The service name to register.
    ///
    /// # Returns
    /// The assigned ID of the service as a `u8`.
    pub fn register_service(&mut self, service: &str) -> u8 {
        let service_hash = lightning_hash_str(service);
        if let Some(&existing_id) = self.services.get(&service_hash) {
            return existing_id;
        }

        let new_id = self.services.len() as u8;
        self.services.insert(service_hash, new_id);
        new_id
    }

    /// Checks whether a given log level is configured to be indexed.
    ///
    /// # Arguments
    /// * `level` - The log level string.
    ///
    /// # Returns
    /// `true` if the log level should be indexed, `false` otherwise.
    pub fn should_index_level(&self, level: &str) -> bool {
        let level_hash = lightning_hash_str(level);
        self.log_levels.get(&level_hash).is_some()
    }

    /// Returns a collection of all registered log level hashes.
    ///
    /// This is useful for iterating through all configured log levels without needing
    /// to know their string representations.
    pub fn log_level_hashes(&self) -> Vec<u64> {
        self.log_levels.iter_keys().collect()
    }

    /// Returns a collection of all registered service hashes.
    ///
    /// This allows for efficient iteration over all known services.
    pub fn service_hashes(&self) -> Vec<u64> {
        self.services.iter_keys().collect()
    }

    /// Adds a custom log level with a specified priority.
    ///
    /// # Arguments
    /// * `level` - The name of the custom log level.
    /// * `priority` - The priority to assign to the new log level.
    pub fn add_log_level(&mut self, level: &str, priority: u8) {
        let level_hash = lightning_hash_str(level);
        self.log_levels.insert(level_hash, priority);
    }

    /// Checks if a log level with the given priority exists.
    ///
    /// # Arguments
    /// * `priority` - The priority to check for.
    ///
    /// # Returns
    /// `true` if a log level with the specified priority is configured, `false` otherwise.
    pub fn has_log_level_priority(&self, priority: u8) -> bool {
        self.log_levels
            .iter_keys()
            .any(|k| *self.log_levels.get(&k).unwrap() == priority)
    }

    /// Returns a string with statistics about the current configuration.
    ///
    /// This provides a quick overview of the configuration state, including the number
    /// of registered log levels, services, and other key parameters.
    pub fn stats(&self) -> String {
        format!(
            "LogConfig: levels {} services {} max_postings {} ngrams:{}",
            self.log_levels.len(),
            self.services.len(),
            self.max_postings,
            self.enable_ngrams
        )
    }
}
