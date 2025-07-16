//! # Utility Modules
//!
//! This module serves as the central hub for a collection of utility modules that
//! provide core data structures and algorithms used throughout the application.
//! By organizing these utilities into a single module, we can ensure consistent
//! access and reuse of fundamental components, such as high-performance hash sets,
//! ultra-fast hashing functions, and a statistically sound random number generator.

/// A high-performance, cache-friendly hash set implementation.
///
/// This module provides `BugguHashSet`, a custom hash set optimized for speed and
/// low memory overhead. It is designed to be highly efficient for in-memory
/// indexing and other performance-critical tasks.
pub mod buggu_hash_set;

/// A collection of ultra-fast, non-cryptographic hash functions.
///
/// This module offers a set of hashing algorithms that prioritize speed over
/// collision resistance, making them ideal for use in hash tables and other
/// data structures where performance is paramount.
pub mod buggu_ultra_fast_hash;

/// A high-performance, statistically sound random number generator.
///
/// This module contains `BugguRng`, a random number generator that combines the
/// XOROSHIRO128+ algorithm with Lemire's method for unbiased range generation,
/// ensuring both speed and statistical quality.
pub mod buggu_random_generator;
