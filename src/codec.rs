//! # Delta Encoder/Decoder
//!
//! This module provides functionality for delta encoding and decoding, a technique
//! used to efficiently transmit log data by sending only the differences between
//! consecutive versions of a document. This reduces the amount of data that needs
//! to be sent over the network, improving performance in log transmission scenarios.

use crate::types::{DocId, Tok};
use std::io;

/// Tag for a full frame, indicating a complete snapshot of a document.
pub const TAG_FULL: u8 = 0;

/// Tag for a differential frame, representing the changes since the last version.
pub const TAG_DIFF: u8 = 1;

/// Represents a data frame, which can be either a full snapshot or a differential update.
#[derive(Debug, PartialEq)]
pub enum Frame {
    /// A full snapshot of a document, containing all its tokens.
    Full {
        doc_id: DocId,
        tokens: Vec<Tok>,
    },
    /// A differential update, containing tokens to be removed and added.
    Diff {
        doc_id: DocId,
        remove: Vec<Tok>,
        add: Vec<Tok>,
    },
}

/// Encodes a full token set into a byte vector.
///
/// The resulting byte vector is structured as follows:
/// - `TAG_FULL` (1 byte)
/// - `doc_id` (variable-length u64)
/// - `tokens.len()` (variable-length u64)
/// - `tokens` (a sequence of variable-length u64 values)
///
/// # Arguments
/// * `doc` - The document ID.
/// * `tokens` - A slice of tokens representing the full document content.
///
/// # Returns
/// A `Vec<u8>` containing the encoded full frame.
pub fn encode_full(doc: DocId, tokens: &[Tok]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(tokens.len() * 9 + 10);
    buf.push(TAG_FULL);
    write_uvar(doc, &mut buf);
    write_uvar(tokens.len() as u64, &mut buf);
    for &t in tokens {
        write_uvar(t, &mut buf);
    }
    buf
}

/// Encodes a differential update into a byte vector.
///
/// The resulting byte vector is structured as follows:
/// - `TAG_DIFF` (1 byte)
/// - `doc_id` (variable-length u64)
/// - `remove.len()` (variable-length u64)
/// - `remove` tokens (a sequence of variable-length u64 values)
/// - `add.len()` (variable-length u64)
/// - `add` tokens (a sequence of variable-length u64 values)
///
/// # Arguments
/// * `doc` - The document ID.
/// * `remove` - A slice of tokens to be removed from the document.
/// * `add` - A slice of tokens to be added to the document.
///
/// # Returns
/// A `Vec<u8>` containing the encoded differential frame.
pub fn encode_diff(doc: DocId, remove: &[Tok], add: &[Tok]) -> Vec<u8> {
    let mut buf = Vec::with_capacity((remove.len() + add.len()) * 9 + 10);
    buf.push(TAG_DIFF);
    write_uvar(doc, &mut buf);
    write_uvar(remove.len() as u64, &mut buf);
    for &t in remove {
        write_uvar(t, &mut buf);
    }
    write_uvar(add.len() as u64, &mut buf);
    for &t in add {
        write_uvar(t, &mut buf);
    }
    buf
}

/// Decodes a byte slice into a `Frame`.
///
/// This function reads the tag from the first byte to determine whether the frame
/// is a full snapshot or a differential update, then decodes the rest of the bytes
/// accordingly.
///
/// # Arguments
/// * `bytes` - The byte slice to decode.
///
/// # Returns
/// A `Result` containing the decoded `Frame` or an `io::Error` if decoding fails.
pub fn decode(mut bytes: &[u8]) -> io::Result<Frame> {
    if bytes.is_empty() {
        return Err(io::ErrorKind::UnexpectedEof.into());
    }

    let tag = bytes[0];
    bytes = &bytes[1..];
    let doc_id = read_uvar(&mut bytes)?;

    match tag {
        TAG_FULL => {
            let len = read_uvar(&mut bytes)? as usize;
            let mut tokens = Vec::with_capacity(len);
            for _ in 0..len {
                tokens.push(read_uvar(&mut bytes)?);
            }
            Ok(Frame::Full { doc_id, tokens })
        }
        TAG_DIFF => {
            let rlen = read_uvar(&mut bytes)? as usize;
            let mut remove = Vec::with_capacity(rlen);
            for _ in 0..rlen {
                remove.push(read_uvar(&mut bytes)?);
            }
            let alen = read_uvar(&mut bytes)? as usize;
            let mut add = Vec::with_capacity(alen);
            for _ in 0..alen {
                add.push(read_uvar(&mut bytes)?);
            }
            Ok(Frame::Diff {
                doc_id,
                remove,
                add,
            })
        }
        _ => Err(io::Error::new(io::ErrorKind::InvalidData, "bad tag")),
    }
}

/// Writes a `u64` as a variable-length integer to a byte vector.
///
/// This encoding scheme uses the most significant bit of each byte to indicate
/// whether more bytes follow. This allows for efficient storage of integers with
/// varying magnitudes.
///
/// # Arguments
/// * `n` - The `u64` value to write.
/// * `out` - The mutable `Vec<u8>` to write the encoded bytes to.
#[inline]
fn write_uvar(mut n: u64, out: &mut Vec<u8>) {
    loop {
        let byte = (n & 0x7F) as u8;
        n >>= 7;
        if n == 0 {
            out.push(byte);
            break;
        } else {
            out.push(byte | 0x80);
        }
    }
}

/// Reads a variable-length integer from a byte slice.
///
/// This function decodes a `u64` that was previously written with `write_uvar`.
/// It reads bytes until it finds one without the most significant bit set.
///
/// # Arguments
/// * `src` - A mutable reference to the byte slice to read from. The slice is
///           advanced past the bytes that are read.
///
/// # Returns
/// A `Result` containing the decoded `u64` or an `io::Error` if decoding fails.
#[inline]
fn read_uvar(src: &mut &[u8]) -> io::Result<u64> {
    let mut shift = 0;
    let mut acc = 0u64;
    for _ in 0..10 {
        if src.is_empty() {
            return Err(io::ErrorKind::UnexpectedEof.into());
        }
        let b = src[0];
        *src = &src[1..];
        acc |= ((b & 0x7F) as u64) << shift;
        if b & 0x80 == 0 {
            return Ok(acc);
        }
        shift += 7;
    }
    Err(io::Error::new(
        io::ErrorKind::InvalidData,
        "varint too long",
    ))
}