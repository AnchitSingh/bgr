//! # BugguHashSet: A High-Performance, Cache-Friendly Hash Set
//!
//! This module provides `BugguHashSet`, a custom hash set implementation designed
//! for extreme performance and low memory overhead. It is optimized for scenarios
//! where cache efficiency is critical, such as in-memory indexing and high-speed
//! data processing. The implementation uses a combination of inline buckets for
//! small collections and overflow buckets for larger ones, minimizing pointer
//! chasing and improving data locality.

use crate::utils::buggu_random_generator::BugguRng;
use crate::utils::buggu_ultra_fast_hash::{buggu_hash_u64_minimal, lightning_hash_str};

/// The number of entries that can be stored directly within a bucket before
/// it transitions to an overflow structure. This is a key parameter for tuning
/// the cache performance of the hash set.
const INLINE_BUCKET_SIZE: usize = 4;

/// The initial size of an overflow bucket. When a bucket exceeds `INLINE_BUCKET_SIZE`,
/// it allocates an overflow vector with this capacity.
const OVERFLOW_BUCKET_SIZE: usize = 8;

// =============================================================================
// HASHABLE TRAIT
// =============================================================================

/// A trait for types that can be hashed by `BugguHashSet`.
///
/// This trait provides a custom hashing method, `buggu_hash`, which allows for
/// specialized hashing logic tailored to the performance characteristics of the
/// hash set.
pub trait BugguHashable: Eq + PartialEq {
    /// Computes the hash of a value.
    fn buggu_hash(&self) -> u64;
}

// =============================================================================
// HASHABLE IMPLEMENTATIONS FOR STRING TYPES
// =============================================================================

impl BugguHashable for &str {
    /// Hashes a string slice using a high-speed hashing algorithm.
    fn buggu_hash(&self) -> u64 {
        lightning_hash_str(self)
    }
}

impl BugguHashable for String {
    /// Hashes a `String` by converting it to a string slice.
    fn buggu_hash(&self) -> u64 {
        lightning_hash_str(self.as_str())
    }
}

// =============================================================================
// HASHABLE IMPLEMENTATIONS FOR NUMERIC TYPES
// =============================================================================

/// A macro to implement `BugguHashable` for numeric types.
///
/// This macro simplifies the implementation of `BugguHashable` for various integer
/// types by converting them to `u64` and then applying a minimal hash function.
macro_rules! impl_buggu_hashable_numeric {
    ($($t:ty),*) => {
        $(
            impl BugguHashable for $t {
                /// Hashes a numeric value using a minimal, high-speed hash function.
                fn buggu_hash(&self) -> u64 {
                    buggu_hash_u64_minimal(*self as u64)
                }
            }
        )*
    };
}

impl_buggu_hashable_numeric!(u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, isize);

/// A macro to implement `BugguHashable` for byte arrays and slices.
///
/// This macro packs the first few bytes of an array or slice into a `u64` and
/// then applies a minimal hash function.
macro_rules! impl_buggu_hashable_bytes {
    ($($t:ty, $element:ty),*) => {
        $(
            impl BugguHashable for $t {
                /// Hashes a byte array or slice.
                fn buggu_hash(&self) -> u64 {
                    let mut num = 0u64;
                    let bytes_to_take = std::cmp::min(8, self.len());
                    for (i, &byte) in self.iter().enumerate().take(bytes_to_take) {
                        num |= (byte as u64) << (i * 8);
                    }
                    buggu_hash_u64_minimal(num)
                }
            }
        )*
    };
}

impl_buggu_hashable_bytes!([u8], u8, [u16], u16, Vec<u8>, u8, Vec<u16>, u16);

// Implementations for fixed-size arrays.
impl<const N: usize> BugguHashable for [u8; N] {
    /// Hashes a fixed-size byte array.
    fn buggu_hash(&self) -> u64 {
        let mut num = 0u64;
        let bytes_to_take = std::cmp::min(8, N);
        for (i, &byte) in self.iter().enumerate().take(bytes_to_take) {
            num |= (byte as u64) << (i * 8);
        }
        buggu_hash_u64_minimal(num)
    }
}

impl<const N: usize> BugguHashable for [u16; N] {
    /// Hashes a fixed-size array of `u16` values.
    fn buggu_hash(&self) -> u64 {
        let mut num = 0u64;
        for (i, &byte) in self.iter().enumerate().take(8) {
            num |= (byte as u64) << (i * 8);
        }
        buggu_hash_u64_minimal(num)
    }
}

// Implementations for tuples.
impl BugguHashable for (u32, u32) {
    /// Hashes a tuple of two `u32` values.
    fn buggu_hash(&self) -> u64 {
        let mut num = self.1;
        num |= self.0 << 8;
        buggu_hash_u64_minimal(num as u64)
    }
}

impl BugguHashable for (i32, i32) {
    /// Hashes a tuple of two `i32` values.
    fn buggu_hash(&self) -> u64 {
        let mut num = self.1;
        num |= self.0 << 8;
        buggu_hash_u64_minimal(num as u64)
    }
}

// =============================================================================
// BUCKET STRUCTURE
// =============================================================================

/// Represents a bucket in the `BugguHashSet`.
///
/// A bucket can be in one of three states:
/// - `Empty`: The bucket contains no entries.
/// - `Inline`: The bucket stores a small number of entries directly in an array,
///   avoiding heap allocations and improving cache performance.
/// - `Overflow`: The bucket has exceeded its inline capacity and now stores its
///   entries in a heap-allocated vector.
#[derive(Debug, Clone, Default)]
pub enum BugguBucket<K, V> {
    #[default]
    Empty,
    Inline {
        entries: [(K, V); INLINE_BUCKET_SIZE],
        len: u8,
    },
    Overflow {
        entries: Vec<(K, V)>,
    },
}

// =============================================================================
// ITERATORS
// =============================================================================

/// An iterator over the keys of a `BugguHashSet`.
#[derive(Debug, Clone)]
pub struct BugguKeyIterator<'a, K, V> {
    storage: &'a [BugguBucket<K, V>],
    bucket_idx: usize,
    entry_idx: usize,
    remaining: usize,
}

impl<'a, K, V> Iterator for BugguKeyIterator<'a, K, V>
where
    K: BugguHashable + Eq + PartialEq + Clone + Default,
    V: Clone + Default,
{
    type Item = K;

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        while self.bucket_idx < self.storage.len() {
            let bucket = unsafe { self.storage.get_unchecked(self.bucket_idx) };
            match bucket {
                BugguBucket::Empty => {
                    self.bucket_idx += 1;
                    self.entry_idx = 0;
                }
                BugguBucket::Inline { entries, len } => {
                    if self.entry_idx < *len as usize {
                        let key = unsafe { entries.get_unchecked(self.entry_idx).0.clone() };
                        self.entry_idx += 1;
                        self.remaining -= 1;
                        return Some(key);
                    } else {
                        self.bucket_idx += 1;
                        self.entry_idx = 0;
                    }
                }
                BugguBucket::Overflow { entries } => {
                    if self.entry_idx < entries.len() {
                        let key = unsafe { entries.get_unchecked(self.entry_idx).0.clone() };
                        self.entry_idx += 1;
                        self.remaining -= 1;
                        return Some(key);
                    } else {
                        self.bucket_idx += 1;
                        self.entry_idx = 0;
                    }
                }
            }
        }
        None
    }
}

/// A mutable iterator over the entries of a `BugguHashSet`.
pub struct BugguIterMut<'a, K, V> {
    storage: std::slice::IterMut<'a, BugguBucket<K, V>>,
    current_bucket: Option<&'a mut BugguBucket<K, V>>,
    entry_idx: usize,
    remaining: usize,
}

impl<'a, K, V> BugguIterMut<'a, K, V>
where
    K: BugguHashable + Eq + PartialEq + Clone + Default,
    V: Clone + Default,
{
    /// Creates a new mutable iterator.
    fn new(storage: &'a mut [BugguBucket<K, V>], remaining: usize) -> Self {
        let mut iter = storage.iter_mut();
        let current_bucket = iter.next();
        Self {
            storage: iter,
            current_bucket,
            entry_idx: 0,
            remaining,
        }
    }
}

impl<'a, K, V> Iterator for BugguIterMut<'a, K, V>
where
    K: BugguHashable + Eq + PartialEq + Clone + Default,
    V: Clone + Default,
{
    type Item = (&'a K, &'a mut V);

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        loop {
            match self.current_bucket.as_mut() {
                Some(bucket) => match bucket {
                    BugguBucket::Empty => {
                        self.current_bucket = self.storage.next();
                        self.entry_idx = 0;
                    }
                    BugguBucket::Inline { entries, len } => {
                        if self.entry_idx < *len as usize {
                            let entry_ptr = entries.as_mut_ptr();
                            unsafe {
                                let entry = &mut *entry_ptr.add(self.entry_idx);
                                let key_ptr = &entry.0 as *const K;
                                let value_ptr = &mut entry.1 as *mut V;
                                self.entry_idx += 1;
                                self.remaining -= 1;
                                return Some((&*key_ptr, &mut *value_ptr));
                            }
                        } else {
                            self.current_bucket = self.storage.next();
                            self.entry_idx = 0;
                        }
                    }
                    BugguBucket::Overflow { entries } => {
                        if self.entry_idx < entries.len() {
                            let entry_ptr = entries.as_mut_ptr();
                            unsafe {
                                let entry = &mut *entry_ptr.add(self.entry_idx);
                                let key_ptr = &entry.0 as *const K;
                                let value_ptr = &mut entry.1 as *mut V;
                                self.entry_idx += 1;
                                self.remaining -= 1;
                                return Some((&*key_ptr, &mut *value_ptr));
                            }
                        } else {
                            self.current_bucket = self.storage.next();
                            self.entry_idx = 0;
                        }
                    }
                },
                None => return None,
            }
        }
    }
}

/// Represents an entry in the `BugguHashSet`, which can be either occupied or vacant.
pub enum BugguEntry<'a, K, V>
where
    K: BugguHashable + Eq + PartialEq + Clone + Default,
    V: Clone + Default,
{
    Occupied(BugguOccupiedEntry<'a, K, V>),
    Vacant(BugguVacantEntry<'a, K, V>),
}

/// An occupied entry in the `BugguHashSet`.
pub struct BugguOccupiedEntry<'a, K, V>
where
    K: BugguHashable + Eq + PartialEq + Clone + Default,
    V: Clone + Default,
{
    key: K,
    hashset: &'a mut BugguHashSet<K, V>,
    bucket_idx: usize,
    entry_idx: usize,
}

/// A vacant entry in the `BugguHashSet`.
pub struct BugguVacantEntry<'a, K, V>
where
    K: BugguHashable + Eq + PartialEq + Clone + Default,
    V: Clone + Default,
{
    key: K,
    hashset: &'a mut BugguHashSet<K, V>,
    bucket_idx: usize,
}

impl<'a, K, V> BugguEntry<'a, K, V>
where
    K: BugguHashable + Eq + PartialEq + Clone + Default,
    V: Clone + Default,
{
    /// Inserts a default value if the entry is vacant.
    #[inline(always)]
    pub fn or_insert(self, default: V) -> &'a mut V {
        match self {
            BugguEntry::Occupied(entry) => entry.into_mut(),
            BugguEntry::Vacant(entry) => entry.insert(default),
        }
    }

    /// Inserts a value computed from a closure if the entry is vacant.
    #[inline(always)]
    pub fn or_insert_with<F>(self, default: F) -> &'a mut V
    where
        F: FnOnce() -> V,
    {
        match self {
            BugguEntry::Occupied(entry) => entry.into_mut(),
            BugguEntry::Vacant(entry) => entry.insert(default()),
        }
    }

    /// Returns the key of the entry.
    #[inline(always)]
    pub fn key(&self) -> &K {
        match self {
            BugguEntry::Occupied(entry) => entry.key(),
            BugguEntry::Vacant(entry) => entry.key(),
        }
    }

    /// Modifies the entry if it is occupied.
    #[inline(always)]
    pub fn and_modify<F>(self, f: F) -> Self
    where
        F: FnOnce(&mut V),
    {
        match self {
            BugguEntry::Occupied(mut entry) => {
                f(entry.get_mut());
                BugguEntry::Occupied(entry)
            }
            BugguEntry::Vacant(entry) => BugguEntry::Vacant(entry),
        }
    }
}

impl<'a, K, V> BugguOccupiedEntry<'a, K, V>
where
    K: BugguHashable + Eq + PartialEq + Clone + Default,
    V: Clone + Default,
{
    /// Returns the key of the occupied entry.
    #[inline(always)]
    pub fn key(&self) -> &K {
        &self.key
    }

    /// Returns a reference to the value of the occupied entry.
    #[inline(always)]
    pub fn get(&self) -> &V {
        let bucket = unsafe { self.hashset.storage.get_unchecked(self.bucket_idx) };
        match bucket {
            BugguBucket::Inline { entries, .. } => unsafe {
                &entries.get_unchecked(self.entry_idx).1
            },
            BugguBucket::Overflow { entries } => unsafe {
                &entries.get_unchecked(self.entry_idx).1
            },
            _ => unreachable!(),
        }
    }

    /// Returns a mutable reference to the value of the occupied entry.
    #[inline(always)]
    pub fn get_mut(&mut self) -> &mut V {
        let bucket = unsafe { self.hashset.storage.get_unchecked_mut(self.bucket_idx) };
        match bucket {
            BugguBucket::Inline { entries, .. } => unsafe {
                &mut entries.get_unchecked_mut(self.entry_idx).1
            },
            BugguBucket::Overflow { entries } => unsafe {
                &mut entries.get_unchecked_mut(self.entry_idx).1
            },
            _ => unreachable!(),
        }
    }

    /// Converts the occupied entry into a mutable reference to its value.
    #[inline(always)]
    pub fn into_mut(self) -> &'a mut V {
        let bucket = unsafe { self.hashset.storage.get_unchecked_mut(self.bucket_idx) };
        match bucket {
            BugguBucket::Inline { entries, .. } => unsafe {
                &mut entries.get_unchecked_mut(self.entry_idx).1
            },
            BugguBucket::Overflow { entries } => unsafe {
                &mut entries.get_unchecked_mut(self.entry_idx).1
            },
            _ => unreachable!(),
        }
    }

    /// Inserts a new value, returning the old value.
    #[inline(always)]
    pub fn insert(&mut self, value: V) -> V {
        std::mem::replace(self.get_mut(), value)
    }
}

impl<'a, K, V> BugguVacantEntry<'a, K, V>
where
    K: BugguHashable + Eq + PartialEq + Clone + Default,
    V: Clone + Default,
{
    /// Returns the key of the vacant entry.
    #[inline(always)]
    pub fn key(&self) -> &K {
        &self.key
    }

    /// Inserts a value into the vacant entry.
    #[inline(always)]
    pub fn insert(self, value: V) -> &'a mut V {
        let bucket_idx = self.bucket_idx;
        let bucket = unsafe { self.hashset.storage.get_unchecked_mut(bucket_idx) };

        match bucket {
            BugguBucket::Empty => {
                let mut entries = core::array::from_fn(|_| (K::default(), V::default()));
                entries[0] = (self.key, value);
                *bucket = BugguBucket::Inline { entries, len: 1 };
                self.hashset.count += 1;
            }
            BugguBucket::Inline { entries, len } => {
                let current_len = *len as usize;
                if current_len < INLINE_BUCKET_SIZE {
                    entries[current_len] = (self.key, value);
                    *len += 1;
                    self.hashset.count += 1;
                } else {
                    let mut overflow_vec = Vec::with_capacity(OVERFLOW_BUCKET_SIZE);
                    for item in entries.iter_mut().take(INLINE_BUCKET_SIZE) {
                        overflow_vec.push(std::mem::take(item));
                    }
                    overflow_vec.push((self.key, value));
                    *bucket = BugguBucket::Overflow {
                        entries: overflow_vec,
                    };
                    self.hashset.count += 1;
                }
            }
            BugguBucket::Overflow { entries } => {
                if entries.len() == entries.capacity() {
                    entries.reserve(entries.capacity());
                }
                entries.push((self.key, value));
                self.hashset.count += 1;
            }
        }

        // Get the reference after all modifications are done
        let bucket = unsafe { self.hashset.storage.get_unchecked_mut(bucket_idx) };
        match bucket {
            BugguBucket::Inline { entries, len } => unsafe {
                &mut entries.get_unchecked_mut((*len as usize) - 1).1
            },
            BugguBucket::Overflow { entries } => {
                let len = entries.len();
                unsafe { &mut entries.get_unchecked_mut(len - 1).1 }
            }
            _ => unreachable!(),
        }
    }
}

// =============================================================================
// HASHSET IMPLEMENTATION
// =============================================================================

/// A high-performance, cache-friendly hash set.
#[derive(Debug, Clone, Default)]
pub struct BugguHashSet<K, V = ()>
where
    K: BugguHashable + Eq + PartialEq + Clone + Default,
    V: Clone + Default,
{
    pub storage: Vec<BugguBucket<K, V>>,
    count: usize,
}

impl<K, V> BugguHashSet<K, V>
where
    K: BugguHashable + Eq + PartialEq + Clone + Default,
    V: Clone + Default,
{
    /// Creates a new `BugguHashSet` with a specified table size.
    pub fn new(table_size: usize) -> Self {
        BugguHashSet {
            storage: vec![BugguBucket::Empty; table_size],
            count: 0,
        }
    }

    /// Returns the number of entries in the hash set.
    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns `true` if the hash set is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Retains only the elements specified by the predicate.
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        let mut total_removed = 0;

        for bucket in self.storage.iter_mut() {
            match bucket {
                BugguBucket::Empty => continue,
                BugguBucket::Inline { entries, len } => {
                    let current_len = *len as usize;
                    let mut write_idx = 0;

                    for read_idx in 0..current_len {
                        let should_keep = {
                            let entry = unsafe { entries.get_unchecked_mut(read_idx) };
                            f(&entry.0, &mut entry.1)
                        };

                        if should_keep {
                            if write_idx != read_idx {
                                let temp = std::mem::take(&mut entries[read_idx]);
                                entries[write_idx] = temp;
                            }
                            write_idx += 1;
                        } else {
                            total_removed += 1;
                        }
                    }

                    for item in entries.iter_mut().take(current_len).skip(write_idx) {
                        *item = (K::default(), V::default());
                    }
                    *len = write_idx as u8;

                    if write_idx == 0 {
                        *bucket = BugguBucket::Empty;
                    }
                }
                BugguBucket::Overflow { entries } => {
                    let original_len = entries.len();
                    entries.retain_mut(|entry| f(&entry.0, &mut entry.1));
                    let new_len = entries.len();
                    total_removed += original_len - new_len;

                    if new_len == 0 {
                        *bucket = BugguBucket::Empty;
                    } else if new_len <= INLINE_BUCKET_SIZE {
                        let mut inline_entries =
                            core::array::from_fn(|_| (K::default(), V::default()));
                        for (i, entry) in entries.drain(..).enumerate() {
                            inline_entries[i] = entry;
                        }
                        *bucket = BugguBucket::Inline {
                            entries: inline_entries,
                            len: new_len as u8,
                        };
                    }
                }
            }
        }

        self.count -= total_removed;
    }

    /// Computes the rank (bucket index) for a given key.
    #[inline(always)]
    fn get_rank_for_key(&self, key: &K) -> usize {
        let seed = key.buggu_hash();
        let mut rng = BugguRng::new(seed);
        rng.range(0, self.storage.len() as u64 - 1) as usize
    }

    /// Performs a fast intersection with a slice of keys.
    pub fn fast_intersect_slice(&self, keys: &[K]) -> Vec<K> {
        let mut result = Vec::new();

        for key in keys {
            let rank_idx = self.get_rank_for_key(key);
            let bucket = unsafe { self.storage.get_unchecked(rank_idx) };

            let found = match bucket {
                BugguBucket::Empty => false,
                BugguBucket::Inline { entries, len } => {
                    let current_len = *len as usize;
                    (0..current_len).any(|i| unsafe { &entries.get_unchecked(i).0 == key })
                }
                BugguBucket::Overflow { entries } => entries.iter().any(|(k, _)| k == key),
            };

            if found {
                result.push(key.clone());
            }
        }
        result
    }

    /// Creates an index from the hash set based on a field extractor function.
    pub fn create_index_for<F, V2>(&self, field_extractor: F) -> BugguHashSet<V2, Vec<K>>
    where
        F: Fn(&V) -> Option<V2>,
        V2: BugguHashable + Eq + PartialEq + Clone + Default,
    {
        let mut index = BugguHashSet::new(128);

        for bucket in &self.storage {
            match bucket {
                BugguBucket::Empty => continue,
                BugguBucket::Inline { entries, len } => {
                    for i in 0..*len as usize {
                        let (k, v) = unsafe { entries.get_unchecked(i) };
                        if let Some(field_value) = field_extractor(v) {
                            index
                                .entry(field_value)
                                .or_insert_with(Vec::new)
                                .push(k.clone());
                        }
                    }
                }
                BugguBucket::Overflow { entries } => {
                    for (k, v) in entries {
                        if let Some(field_value) = field_extractor(v) {
                            index
                                .entry(field_value)
                                .or_insert_with(Vec::new)
                                .push(k.clone());
                        }
                    }
                }
            }
        }

        index
    }

    /// Computes the intersection of two hash sets.
    pub fn intersect_to_set(&self, other: &BugguHashSet<K, V>) -> BugguHashSet<K, ()>
    where
        V: Default + Clone,
    {
        let mut result = BugguHashSet::new(self.len().min(other.len()));

        let (smaller, larger) = if self.len() < other.len() {
            (self, other)
        } else {
            (other, self)
        };

        for k in smaller.iter_keys() {
            if larger.get(&k).is_some() {
                result.insert(k, ());
            }
        }

        result
    }

    /// Computes the difference between two hash sets.
    pub fn fast_difference(&self, exclude: &BugguHashSet<K, V>) -> BugguHashSet<K, ()>
    where
        V: Clone + Default,
    {
        let mut result = BugguHashSet::new(self.len());
        for key in self.iter_keys() {
            if exclude.get(&key).is_none() {
                result.insert(key, ());
            }
        }
        result
    }

    /// Computes the union of two hash sets.
    pub fn union_with(&self, other: &BugguHashSet<K, V>) -> BugguHashSet<K, ()>
    where
        V: Clone + Default,
    {
        let mut result = BugguHashSet::new(self.len() + other.len());

        for k in self.iter_keys() {
            result.insert(k, ());
        }

        for k in other.iter_keys() {
            result.insert(k, ());
        }

        result
    }

    /// Computes the intersection of two hash sets.
    pub fn intersect_with(&self, other: &BugguHashSet<K, V>) -> BugguHashSet<K, ()>
    where
        V: Clone + Default,
    {
        let (smaller, larger) = if self.len() < other.len() {
            (self, other)
        } else {
            (other, self)
        };

        let mut result = BugguHashSet::new(smaller.len());
        for k in smaller.iter_keys() {
            if larger.get(&k).is_some() {
                result.insert(k, ());
            }
        }

        result
    }

    /// Inserts a key-value pair into the hash set.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let rank_idx = self.get_rank_for_key(&key);
        let bucket = unsafe { self.storage.get_unchecked_mut(rank_idx) };

        match bucket {
            BugguBucket::Empty => {
                let mut entries = core::array::from_fn(|_| (K::default(), V::default()));
                entries[0] = (key, value);
                *bucket = BugguBucket::Inline { entries, len: 1 };
                self.count += 1;
                None
            }
            BugguBucket::Inline { entries, len } => {
                let current_len = *len as usize;
                for i in 0..current_len {
                    let entry = unsafe { entries.get_unchecked_mut(i) };
                    if entry.0 == key {
                        return Some(std::mem::replace(&mut entry.1, value));
                    }
                }
                if current_len < INLINE_BUCKET_SIZE {
                    entries[current_len] = (key, value);
                    *len += 1;
                    self.count += 1;
                    None
                } else {
                    let mut overflow_vec = Vec::with_capacity(OVERFLOW_BUCKET_SIZE);
                    for item in entries.iter_mut().take(INLINE_BUCKET_SIZE) {
                        overflow_vec.push(std::mem::take(item));
                    }
                    overflow_vec.push((key, value));
                    *bucket = BugguBucket::Overflow {
                        entries: overflow_vec,
                    };
                    self.count += 1;
                    None
                }
            }
            BugguBucket::Overflow { entries } => {
                for entry in entries.iter_mut() {
                    if entry.0 == key {
                        return Some(std::mem::replace(&mut entry.1, value));
                    }
                }
                if entries.len() == entries.capacity() {
                    entries.reserve(entries.capacity());
                }
                entries.push((key, value));
                self.count += 1;
                None
            }
        }
    }

    /// Gets an entry for the given key, allowing for insertion or modification.
    pub fn entry(&mut self, key: K) -> BugguEntry<K, V> {
        let bucket_idx = self.get_rank_for_key(&key);

        let _entry_info: Option<usize> = {
            let bucket = unsafe { self.storage.get_unchecked(bucket_idx) };
            match bucket {
                BugguBucket::Empty => None,
                BugguBucket::Inline { entries, len } => {
                    let current_len = *len as usize;
                    for i in 0..current_len {
                        if unsafe { &entries.get_unchecked(i).0 } == &key {
                            return BugguEntry::Occupied(BugguOccupiedEntry {
                                key,
                                hashset: self,
                                bucket_idx,
                                entry_idx: i,
                            });
                        }
                    }
                    None
                }
                BugguBucket::Overflow { entries } => {
                    for (i, (k, _)) in entries.iter().enumerate() {
                        if k == &key {
                            return BugguEntry::Occupied(BugguOccupiedEntry {
                                key,
                                hashset: self,
                                bucket_idx,
                                entry_idx: i,
                            });
                        }
                    }
                    None
                }
            }
        };

        BugguEntry::Vacant(BugguVacantEntry {
            key,
            hashset: self,
            bucket_idx,
        })
    }

    /// Retrieves a reference to the value associated with the given key.
    #[inline(always)]
    pub fn get(&self, key: &K) -> Option<&V> {
        let rank_idx = self.get_rank_for_key(key);
        let bucket = unsafe { self.storage.get_unchecked(rank_idx) };

        match bucket {
            BugguBucket::Empty => None,
            BugguBucket::Inline { entries, len } => {
                let current_len = *len as usize;
                for i in 0..current_len {
                    let (k, v) = unsafe { entries.get_unchecked(i) };
                    if k == key {
                        return Some(v);
                    }
                }
                None
            }
            BugguBucket::Overflow { entries } => {
                for (k, v) in entries.iter() {
                    if k == key {
                        return Some(v);
                    }
                }
                None
            }
        }
    }

    /// Removes a key-value pair from the hash set.
    #[inline(always)]
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let rank_idx = self.get_rank_for_key(key);
        let bucket = unsafe { self.storage.get_unchecked_mut(rank_idx) };

        match bucket {
            BugguBucket::Empty => None,
            BugguBucket::Inline { entries, len } => {
                let current_len = *len as usize;
                for i in 0..current_len {
                    let entry = unsafe { entries.get_unchecked_mut(i) };
                    if entry.0 == *key {
                        let old_value = std::mem::take(&mut entry.1);
                        unsafe {
                            let ptr = entries.as_mut_ptr();
                            for j in i..(current_len - 1) {
                                let src_ptr = ptr.add(j + 1);
                                let dst_ptr = ptr.add(j);
                                std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, 1);
                            }
                        }
                        unsafe {
                            let last = entries.get_unchecked_mut(current_len - 1);
                            *last = (K::default(), V::default());
                        }
                        *len -= 1;
                        if *len == 0 {
                            *bucket = BugguBucket::Empty;
                        }
                        self.count -= 1;
                        return Some(old_value);
                    }
                }
                None
            }
            BugguBucket::Overflow { entries } => {
                for i in 0..entries.len() {
                    if unsafe { &entries.get_unchecked(i).0 } == key {
                        let (_, old_value) = entries.swap_remove(i);
                        if entries.len() <= INLINE_BUCKET_SIZE {
                            let entries_len = entries.len();
                            let mut inline_entries =
                                core::array::from_fn(|_| (K::default(), V::default()));
                            for (i, entry) in entries.drain(..).enumerate() {
                                inline_entries[i] = entry;
                            }
                            *bucket = if entries_len == 0 {
                                BugguBucket::Empty
                            } else {
                                BugguBucket::Inline {
                                    entries: inline_entries,
                                    len: entries_len as u8,
                                }
                            };
                        }
                        self.count -= 1;
                        return Some(old_value);
                    }
                }
                None
            }
        }
    }

    /// Updates the value associated with a key.
    #[inline(always)]
    pub fn update(&mut self, key: &K, value: V) -> Option<V> {
        let rank_idx = self.get_rank_for_key(key);
        let bucket = unsafe { self.storage.get_unchecked_mut(rank_idx) };

        match bucket {
            BugguBucket::Empty => None,
            BugguBucket::Inline { entries, len } => {
                let current_len = *len as usize;
                for i in 0..current_len {
                    let entry = unsafe { entries.get_unchecked_mut(i) };
                    if entry.0 == *key {
                        return Some(std::mem::replace(&mut entry.1, value));
                    }
                }
                None
            }
            BugguBucket::Overflow { entries } => {
                for entry in entries.iter_mut() {
                    if entry.0 == *key {
                        return Some(std::mem::replace(&mut entry.1, value));
                    }
                }
                None
            }
        }
    }

    /// Retrieves a mutable reference to the value associated with the given key.
    #[inline(always)]
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        let rank_idx = self.get_rank_for_key(key);
        let bucket = unsafe { self.storage.get_unchecked_mut(rank_idx) };

        match bucket {
            BugguBucket::Empty => None,
            BugguBucket::Inline { entries, len } => {
                let current_len = *len as usize;
                let entries_ptr = entries.as_mut_ptr();
                for i in 0..current_len {
                    unsafe {
                        let entry_ptr = entries_ptr.add(i);
                        if (*entry_ptr).0 == *key {
                            return Some(&mut (*entry_ptr).1);
                        }
                    }
                }
                None
            }
            BugguBucket::Overflow { entries } => {
                for entry in entries.iter_mut() {
                    if entry.0 == *key {
                        return Some(&mut entry.1);
                    }
                }
                None
            }
        }
    }

    /// Returns a vector of all keys in the hash set.
    pub fn keys(&self) -> Vec<K> {
        if self.count == 0 {
            return Vec::new();
        }
        let mut keys = Vec::with_capacity(self.count);
        for bucket in self.storage.iter() {
            match bucket {
                BugguBucket::Empty => continue,
                BugguBucket::Inline { entries, len } => {
                    let current_len = *len as usize;
                    for i in 0..current_len {
                        keys.push(unsafe { entries.get_unchecked(i).0.clone() });
                    }
                }
                BugguBucket::Overflow { entries } => {
                    keys.extend(entries.iter().map(|(k, _)| k.clone()));
                }
            }
        }
        keys
    }

    /// Returns an iterator over the keys of the hash set.
    pub fn iter_keys(&self) -> BugguKeyIterator<K, V> {
        BugguKeyIterator {
            storage: &self.storage,
            bucket_idx: 0,
            entry_idx: 0,
            remaining: self.count,
        }
    }

    /// Returns a mutable iterator over the entries of the hash set.
    pub fn iter_mut(&mut self) -> BugguIterMut<K, V> {
        BugguIterMut::new(&mut self.storage, self.count)
    }

    /// Inserts a batch of key-value pairs into the hash set.
    pub fn insert_batch(&mut self, items: Vec<(K, V)>) {
        for (key, value) in items.into_iter() {
            self.insert(key, value);
        }
    }

    /// Returns statistics about the bucket distribution.
    pub fn bucket_stats(&self) -> (usize, usize, usize) {
        let mut empty = 0;
        let mut inline = 0;
        let mut overflow = 0;
        for bucket in &self.storage {
            match bucket {
                BugguBucket::Empty => empty += 1,
                BugguBucket::Inline { .. } => inline += 1,
                BugguBucket::Overflow { .. } => overflow += 1,
            }
        }
        (empty, inline, overflow)
    }
}
