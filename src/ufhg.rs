//! # Ultra-Fast Hash Generator (UFHG)
//!
//! This module provides the `UFHGHeadquarters`, a highly optimized component for
//! tokenizing and hashing strings. It is designed for maximum performance in log
//! processing, where efficient string-to-hash conversion is critical. The UFHG
//! employs a zero-copy tokenization strategy and a specialized hashing algorithm
//! to minimize overhead and accelerate indexing.

use crate::utils::buggu_ultra_fast_hash::buggu_hash_u64_minimal;
use crate::utils::buggu_ultra_fast_hash::lightning_hash_str_64;

/// A specialized string hashing function optimized for speed.
///
/// This function is designed to be extremely fast for short strings, particularly
/// those containing only alphanumeric characters. It uses a custom algorithm that
/// avoids more complex hashing logic when possible, falling back to a more robust
/// hash function for strings with special characters.
///
/// # Arguments
/// * `s` - The string to hash.
///
/// # Returns
/// A `u64` hash value.
#[inline(always)]
pub fn lightning_hash_str(s: &str) -> u64 {
    if s.is_empty() {
        return 0;
    }
    let mut result = 0u64;
    let mut has_special = false;
    for &byte in s.as_bytes() {
        let pos = match byte {
            b'a'..=b'z' => byte - b'a' + 1,
            b'A'..=b'Z' => byte - b'A' + 1,
            _ => {
                has_special = true;
                break;
            }
        };
        result = if pos < 10 {
            result * 10 + pos as u64
        } else {
            result * 100 + pos as u64
        };
    }
    if has_special {
        return lightning_hash_str_64(s);
    }
    result
}

/// The central component for tokenization and hashing.
///
/// The `UFHGHeadquarters` is responsible for converting raw strings into sequences
/// of hash-based tokens. It uses a pre-allocated buffer to avoid repeated memory
/// allocations during tokenization, making it highly efficient for processing large
/// volumes of log data.
#[derive(Debug, Clone)]
pub struct UFHGHeadquarters {
    /// A reusable vector for storing word hashes during tokenization.
    word_hashes: Vec<u64>,
}

impl UFHGHeadquarters {
    /// Creates a new `UFHGHeadquarters` with an initial capacity.
    pub fn new() -> Self {
        Self {
            word_hashes: Vec::with_capacity(64),
        }
    }

    /// A highly optimized string hashing function.
    ///
    /// This is an instance method version of the `lightning_hash_str` function,
    /// providing the same performance benefits within the context of the
    /// `UFHGHeadquarters`.
    #[inline(always)]
    pub fn lightning_hash_str(&mut self, s: &str) -> u64 {
        if s.is_empty() {
            return 0;
        }
        let mut result = 0u64;
        let mut has_special = false;
        for &byte in s.as_bytes() {
            let pos = match byte {
                b'a'..=b'z' => byte - b'a' + 1,
                b'A'..=b'Z' => byte - b'A' + 1,
                _ => {
                    has_special = true;
                    break;
                }
            };
            result = if pos < 10 {
                result * 10 + pos as u64
            } else {
                result * 100 + pos as u64
            };
        }
        if has_special {
            return lightning_hash_str_64(s);
        }
        result
    }

    /// Converts a string into a sequence hash.
    ///
    /// This function processes a string, word by word, and computes a rolling hash
    /// of the sequence of word hashes. This is useful for creating a single hash
    /// value that represents an entire phrase or sentence.
    #[inline(always)]
    pub fn string_to_u64_to_seq_hash(&self, s: &str) -> u64 {
        let bytes = s.as_bytes();
        let mut i = 0;
        let mut seq_hash: u64 = 0;
        while i < bytes.len() {
            let byte = bytes[i];
            let is_whitespace = byte == b' ' || byte == b'\t' || byte == b'\n' || byte == b'\r';
            if is_whitespace {
                // Skip whitespace
                while i < bytes.len() {
                    let b = bytes[i];
                    if b == b' ' || b == b'\t' || b == b'\n' || b == b'\r' {
                        i += 1;
                    } else {
                        break;
                    }
                }
            } else {
                let start = i;
                while i < bytes.len()
                    && bytes[i] != b' '
                    && bytes[i] != b'\t'
                    && bytes[i] != b'\n'
                    && bytes[i] != b'\r'
                {
                    i += 1;
                }
                let word_slice = unsafe { std::str::from_utf8_unchecked(&bytes[start..i]) };
                let trimmed_slice = word_slice.trim_matches(|c: char| !c.is_alphanumeric());

                if !trimmed_slice.is_empty() {
                    seq_hash = (lightning_hash_str(trimmed_slice))
                        .wrapping_mul(31)
                        .wrapping_add(seq_hash);
                }
            }
        }
        seq_hash
    }

    /// Tokenizes a message using a zero-copy approach.
    ///
    /// This method processes a string message, breaking it into words and whitespace,
    /// and converting each component into a hash. It avoids unnecessary memory
    /// allocations by writing the hashes directly into a pre-allocated buffer.
    ///
    /// # Returns
    /// A tuple containing two copies of the hash sequence. This is done to allow
    /// the caller to consume one copy while retaining the other.
    #[inline(always)]
    pub fn tokenize_zero_copy(&mut self, message: &str) -> (Vec<u64>, Vec<u64>) {
        self.word_hashes.clear();
        if message.is_empty() {
            return (vec![], vec![]);
        }
        let bytes = message.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            let byte = bytes[i];
            let is_whitespace = byte == b' ' || byte == b'\t' || byte == b'\n' || byte == b'\r';
            if is_whitespace {
                let mut whitespace_count = 0u64;
                while i < bytes.len() {
                    let b = bytes[i];
                    if b == b' ' || b == b'\t' || b == b'\n' || b == b'\r' {
                        whitespace_count += 1;
                        i += 1;
                    } else {
                        break;
                    }
                }
                let hash = process_whitespace_len(whitespace_count);
                self.word_hashes.push(hash);
            } else {
                let start = i;
                while i < bytes.len()
                    && bytes[i] != b' '
                    && bytes[i] != b'\t'
                    && bytes[i] != b'\n'
                    && bytes[i] != b'\r'
                {
                    i += 1;
                }
                let word_slice = unsafe { std::str::from_utf8_unchecked(&bytes[start..i]) };
                let trimmed_slice = word_slice.trim_matches(|c: char| !c.is_alphanumeric());

                if !trimmed_slice.is_empty() {
                    let hash = self.lightning_hash_str(trimmed_slice);
                    self.word_hashes.push(hash);
                }
            }
        }
        // Take ownership of the computed hashes, leaving an empty Vec in its place.
        // This avoids cloning twice – we only clone once to create the second copy.
        let hashes = std::mem::take(&mut self.word_hashes);
        let hashes_clone = hashes.clone();
        (hashes_clone, hashes)
    }
}

/// Processes the length of a whitespace sequence to generate a hash.
///
/// This function takes the length of a sequence of whitespace characters and
/// converts it into a deterministic hash value. This allows whitespace to be
/// treated as a token, which can be useful in certain search scenarios.
fn process_whitespace_len(len: u64) -> u64 {
    let count = len % 8;
    let mut x = 0_u64;
    for _ in 0..count {
        x = x * 100 + 32;
    }
    x = x * 1000 + len;
    buggu_hash_u64_minimal(x)
}
