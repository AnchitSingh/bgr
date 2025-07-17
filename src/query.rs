//! # Query Parsing and Execution
//!
//! This module is responsible for parsing user-provided query strings into a structured
//! Abstract Syntax Tree (AST) and providing the building blocks for query execution.
//! It defines the `QueryNode` enum, which represents the different types of query
//! operations, such as term searches, phrase searches, and logical combinations.

use crate::config::LogConfig;

/// Represents a node in the Abstract Syntax Tree (AST) of a parsed query.
///
/// Each variant of this enum corresponds to a specific type of search operation,
/// allowing for a structured representation of complex user queries. This AST is
/// then used by the query execution engine to retrieve matching documents.
#[derive(Debug, Clone)]
pub enum QueryNode {
    /// A basic search for a single term.
    Term(String),
    /// A search for an exact sequence of terms.
    Phrase(String),
    /// A search for a term within a specific field, such as `level:ERROR`.
    FieldTerm(&'static str, String),
    /// A search for a numeric value within a given range, such as `timestamp:>=12345`.
    NumericRange(&'static str, u64, u64),
    /// A search for content that contains a specific substring.
    Contains(String),
    /// A search using N-grams, which allows for partial word matching.
    NGram(Vec<String>),
    /// A search that allows for approximate matching of a term.
    Fuzzy(String, u8),
    /// A search using regular expressions for advanced pattern matching.
    Regex(String),
    /// A logical AND, requiring all sub-queries to match.
    And(Vec<QueryNode>),
    /// A logical OR, requiring at least one sub-query to match.
    Or(Vec<QueryNode>),
    /// A logical NOT, excluding documents that match the sub-query.
    Not(Box<QueryNode>),
    /// A search for a term within a specific field, such as `level:ERROR`.
    JsonField(String, String),
}

/// Parses a raw query string into a `QueryNode` AST.
///
/// This function tokenizes the input string and constructs a query tree based on
/// special characters and keywords. It supports field-based searches (e.g., `field:value`),
/// quoted phrases, and logical operators (AND, OR, NOT).
///
/// # Arguments
/// * `q` - The raw query string to parse.
/// * `config` - The `LogConfig` to use for parsing, which may contain information
///              about available fields and other settings.
///
/// # Returns
/// A `QueryNode` representing the root of the parsed query AST.
pub fn parse_query(q: &str, _config: &LogConfig) -> QueryNode {
    let mut nodes = Vec::new();
    let mut it = q.split_whitespace().peekable();

    while let Some(tok) = it.next() {
        if tok.contains(':') {
            let mut sp = tok.splitn(2, ':');
            let field = sp.next().unwrap();
            let mut val = sp.next().unwrap().to_string();

            // Handle quoted values that may contain spaces.
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
                "phrase" => nodes.push(QueryNode::Phrase(val)),
                "fuzzy" => {
                    if let Some((word, dist)) = val.split_once('~') {
                        let distance = dist.parse::<u8>().unwrap_or(1);
                        nodes.push(QueryNode::Fuzzy(word.to_string(), distance));
                    } else {
                        nodes.push(QueryNode::Fuzzy(val, 1));
                    }
                }
                "regex" => nodes.push(QueryNode::Regex(val)),
                "json" => {
                    if let Some((path, value)) = val.split_once('=') {
                        nodes.push(QueryNode::JsonField(path.to_string(), value.to_string()));
                    } else {
                        // Just check that the path exists
                        nodes.push(QueryNode::JsonField(val, "".to_string()));
                    }
                }
                "timestamp" => {
                    if let Some(lo) = val.strip_prefix(">=") {
                        let lo = lo.parse::<u64>().unwrap_or(0);
                        nodes.push(QueryNode::NumericRange("timestamp", lo, u64::MAX));
                    } else if let Some(hi) = val.strip_prefix("<=") {
                        let hi = hi.parse::<u64>().unwrap_or(u64::MAX);
                        nodes.push(QueryNode::NumericRange("timestamp", 0, hi));
                    } else {
                        let ts = val.parse::<u64>().unwrap_or(0);
                        nodes.push(QueryNode::NumericRange("timestamp", ts, ts));
                    }
                }
                _ => nodes.push(QueryNode::FieldTerm("unknown", val)),
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

    // Combine multiple nodes with a default AND operator.
    if nodes.len() == 1 {
        nodes.pop().unwrap()
    } else if nodes.is_empty() {
        QueryNode::Term("".to_string()) // Return an empty term if the query is empty.
    } else {
        QueryNode::And(nodes)
    }
}
