//! # Buggu Ultra-Fast Hash Functions
//!
//! This module provides a collection of highly optimized, non-cryptographic hash
//! functions designed for maximum speed. These functions are ideal for performance-critical
//! applications such as hash tables, in-memory databases, and indexing engines where
//! the primary goal is to distribute keys quickly and efficiently. The implementations
//! prioritize speed over collision resistance, making them unsuitable for security-related
//! purposes.

/// A primary multiplier constant used in the hash functions.
///
/// This constant is derived from a large prime number and is chosen for its ability
/// to produce a good distribution of hash values, which helps to minimize collisions
/// in hash-based data structures.
pub const FAST_K1: u64 = 0x517cc1b727220a95;

/// Computes an extremely fast 64-bit hash for a string slice.
///
/// This function is optimized for speed, particularly for short strings. It uses
/// `unsafe` memory operations to read bytes in chunks, which significantly reduces
/// the overhead of bounds checking. The final hash is produced by mixing the read
/// data with a minimal hash function.
///
/// # Arguments
/// * `s` - The string slice (`&str`) to be hashed.
///
/// # Returns
/// A 64-bit hash value (`u64`).
///
/// # Safety
/// This function contains `unsafe` code that performs direct memory access. It assumes
/// that the pointer `bytes.as_ptr()` is valid and that reading chunks of memory will
/// not go out of bounds. This is safe for string slices, which are guaranteed to be
/// valid UTF-8 and have a known length.
#[inline(always)]
pub fn lightning_hash_str(s: &str) -> u64 {
    let bytes = s.as_bytes();
    let len = bytes.len();

    if len == 0 {
        return FAST_K1;
    }

    let data: u64 = unsafe {
        match len {
            1 => bytes[0] as u64,
            2..=3 => (bytes.as_ptr() as *const u16).read_unaligned() as u64,
            4..=5 => {
                let chunk1 = (bytes.as_ptr() as *const u16).read_unaligned() as u64;
                let chunk2 = (bytes.as_ptr().add(2) as *const u16).read_unaligned() as u64;
                chunk1 | (chunk2 << 16)
            }
            _ => {
                // This case handles lengths >= 6.
                let chunk1 = (bytes.as_ptr() as *const u16).read_unaligned() as u64;
                let chunk2 = (bytes.as_ptr().add(2) as *const u16).read_unaligned() as u64;
                let chunk3 = (bytes.as_ptr().add(4) as *const u16).read_unaligned() as u64;
                chunk1 | (chunk2 << 16) | (chunk3 << 32)
            }
        }
    };
    buggu_hash_u64_minimal(data)
}

/// A minimal, single-instruction hash function for `u64` values.
///
/// This function performs only a single multiplication, making it one of the fastest
/// possible ways to hash a `u64`. It offers basic collision resistance and is suitable
/// for use cases where speed is the absolute priority.
///
/// # Arguments
/// * `value` - The `u64` integer to be hashed.
///
/// # Returns
/// A 64-bit hash value.
#[inline(always)]
pub fn buggu_hash_u64_minimal(value: u64) -> u64 {
    value.wrapping_mul(FAST_K1)
}

/// A branchless, zero-optimized hash function for `u64` values.
///
/// This version is designed to avoid conditional branches by using bitwise operations
/// to handle the case where the input is zero. This can lead to improved performance
/// on certain CPU architectures by avoiding branch prediction penalties.
///
/// # Arguments
/// * `value` - The `u64` integer to be hashed.
///
/// # Returns
/// A 64-bit hash value.
#[inline(always)]
pub fn buggu_hash_u64_branchless(value: u64) -> u64 {
    // Create a mask that is all 1s if value is 0, and all 0s otherwise.
    let mask = ((value == 0) as u64).wrapping_neg();
    // If value was 0, make it 1. Otherwise, keep the original value.
    let adjusted = value | (mask & 1);
    adjusted.wrapping_mul(FAST_K1) ^ (adjusted >> 32)
}

/// A 64-bit version of the `lightning_hash_str` function.
///
/// This function is identical in implementation to `lightning_hash_str` and is provided
/// for consistency or for use in contexts where a more explicit name is desired.
#[inline(always)]
pub fn lightning_hash_str_64(s: &str) -> u64 {
    let bytes = s.as_bytes();
    let len = bytes.len();

    if len == 0 {
        return FAST_K1;
    }

    let data: u64 = unsafe {
        match len {
            1 => bytes[0] as u64,
            2..=3 => (bytes.as_ptr() as *const u16).read_unaligned() as u64,
            4..=5 => {
                let chunk1 = (bytes.as_ptr() as *const u16).read_unaligned() as u64;
                let chunk2 = (bytes.as_ptr().add(2) as *const u16).read_unaligned() as u64;
                chunk1 | (chunk2 << 16)
            }
            _ => {
                // This case handles lengths >= 6.
                let chunk1 = (bytes.as_ptr() as *const u16).read_unaligned() as u64;
                let chunk2 = (bytes.as_ptr().add(2) as *const u16).read_unaligned() as u64;
                let chunk3 = (bytes.as_ptr().add(4) as *const u16).read_unaligned() as u64;
                chunk1 | (chunk2 << 16) | (chunk3 << 32)
            }
        }
    };
    buggu_hash_u64_minimal(data)
}
