// src/main.rs

use anyhow::{Context, Result};
use bgr::BugguDB; // Use our library
use clap::Parser;
use std::fs::File;
use std::io::{self, BufReader};
use std::path::PathBuf;
use std::time::Instant;

/// BugguDB: A blazing-fast in-memory search engine for logs and structured data.
///
/// Features:
///   - Ultra-fast text search with microsecond-level response times
///   - Support for field-specific queries (level:ERROR, service:api)
///   - Regular expression searching (regex:pattern\d+)
///   - JSON field querying (json:user.name=value)
///   - Boolean operators (AND, OR, NOT)
///
/// Examples:
///   bgr logs.txt                              # Interactive mode with logs.txt
///   bgr --query="error"                       # Search for "error" in stdin
///   bgr logs1.txt logs2.txt                   # Load multiple files
///   bgr --query="level:ERROR service:auth"    # Field-specific search
///   bgr --query="regex:failed.*timeout"       # Regular expression search
///   bgr --query="json:user.id=1234"           # JSON field search
///   cat logs.txt | bgr                        # Process from stdin
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The file(s) to load. If not provided, reads from stdin.
    #[arg(name = "FILE")]
    files: Vec<PathBuf>,

    /// The query to run in non-interactive mode.
    #[arg(short, long)]
    query: Option<String>,

    /// Maximum number of results to display
    #[arg(short, long, default_value = "20")]
    limit: usize,

    /// Show detailed timing information
    #[arg(short, long)]
    timing: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut db = BugguDB::new();

    let start = Instant::now();
    let mut total_lines = 0;

    if args.files.is_empty() {
        // Read from stdin
        println!("Loading from stdin...");
        let stdin = io::stdin();
        let reader = stdin.lock();
        total_lines += db.ingest_from_reader(reader)
            .context("Failed to read from stdin")?;
    } else {
        // Read from files
        for path in &args.files {
            println!("Loading from {}...", path.display());
            let file = File::open(path)
                .with_context(|| format!("Failed to open file: {}", path.display()))?;
            let reader = BufReader::new(file);
            total_lines += db.ingest_from_reader(reader)
                .with_context(|| format!("Failed to read from file: {}", path.display()))?;
        }
    }

    let load_duration = start.elapsed();
    println!(
        "Loaded {} lines in {:.2?} ({:.2} lines/sec)",
        total_lines,
        load_duration,
        total_lines as f64 / load_duration.as_secs_f64()
    );

    if let Some(query) = args.query {
        // Non-interactive mode
        run_query(&db, &query, args.limit, args.timing);
    } else {
        // Interactive mode
        run_interactive_session(db, args.limit, args.timing)?;
    }

    Ok(())
}

/// Runs a single query and prints the results.
fn run_query(db: &BugguDB, query: &str, limit: usize, show_timing: bool) {
    let start = Instant::now();
    
    // First, check if it's a help request
    if query.trim() == "help" || query.trim() == "?" {
        print_help();
        return;
    }

    // Determine if this is a JSON query for better formatting
    let is_json_query = query.contains("json:");
    
    if is_json_query {
        // For JSON queries, use query_with_meta to get more context
        let results = db.query_with_meta(query);
        let duration = start.elapsed();
        
        println!("Query: {}", query);
        println!("Found {} results in {:.2?}", results.len(), duration);
        
        if show_timing && !results.is_empty() {
            println!("Average: {:.2} µs per document", 
                duration.as_micros() as f64 / db.docs.len() as f64);
        }
        
        for (i, (id, content, level, service)) in results.iter().take(limit).enumerate() {
            print!("{}. [ID:{}]", i + 1, id);
            
            if let Some(lvl) = level {
                print!(" [{}]", lvl);
            }
            
            if let Some(svc) = service {
                print!(" [{}]", svc);
            }
            
            println!(" {}", content);
        }
        
        if results.len() > limit {
            println!("... and {} more matches", results.len() - limit);
        }
    } else {
        // Original behavior for non-JSON queries
        let results = db.query_content(query);
        let duration = start.elapsed();
        
        println!("Query: {}", query);
        println!("Found {} results in {:.2?}", results.len(), duration);
        
        if show_timing && !results.is_empty() {
            println!("Average: {:.2} µs per document", 
                duration.as_micros() as f64 / db.docs.len() as f64);
        }
        
        for (i, result) in results.iter().take(limit).enumerate() {
            println!("{}. {}", i + 1, result);
        }
        
        if results.len() > limit {
            println!("... and {} more matches", results.len() - limit);
        }
    }
}

/// Print help information for interactive mode
fn print_help() {
    println!("\nBugguDB Search Commands:");
    println!("------------------------");
    println!("  term                      Simple text search");
    println!("  \"exact phrase\"           Phrase search (quoted)");
    println!("  level:ERROR               Field search");
    println!("  term1 AND term2           Logical AND (default between terms)");
    println!("  term1 OR term2            Logical OR");
    println!("  term NOT other            Exclude matches containing 'other'");
    println!("  regex:pattern\\d+          Regular expression search");
    println!("  json:user.name            JSON field existence");
    println!("  json:user.active=true     JSON field value matching");
    println!("  json:metrics.cpu>=90      JSON numeric comparison");
    println!("\nInteractive Commands:");
    println!("--------------------");
    println!("  help, ?                   Show this help message");
    println!("  limit N                   Set result limit to N");
    println!("  timing on|off             Toggle detailed timing info");
    println!("  quit, exit, q             Exit the program");
    println!("");
}

/// Starts an interactive REPL session.
fn run_interactive_session(db: BugguDB, mut limit: usize, mut show_timing: bool) -> Result<()> {
    let mut rl = rustyline::Editor::<()>::new()?;
    
    println!("BugguDB CLI v{} - Ultra-fast search engine", env!("CARGO_PKG_VERSION"));
    println!("Features: term search, field filters, regex, JSON queries, boolean operators");
    println!("Type 'help' to see available commands or 'quit' to exit.");
    println!("");

    loop {
        let readline = rl.readline("> ");
        match readline {
            Ok(line) => {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                rl.add_history_entry(line);

                // Handle special commands
                match line.to_lowercase().as_str() {
                    "quit" | "exit" | "q" => break,
                    "help" | "?" => {
                        print_help();
                        continue;
                    }
                    _ if line.starts_with("limit ") => {
                        if let Some(num_str) = line.strip_prefix("limit ") {
                            if let Ok(num) = num_str.trim().parse::<usize>() {
                                limit = num;
                                println!("Result limit set to {}", limit);
                            } else {
                                println!("Invalid limit. Usage: limit <number>");
                            }
                        }
                        continue;
                    }
                    "timing on" => {
                        show_timing = true;
                        println!("Detailed timing enabled");
                        continue;
                    }
                    "timing off" => {
                        show_timing = false;
                        println!("Detailed timing disabled");
                        continue;
                    }
                    _ => {} // Regular query, handled below
                }

                run_query(&db, line, limit, show_timing);
                println!(); // Add a newline for spacing
            }
            Err(_) => break, // Ctrl-C or Ctrl-D
        }
    }
    println!("Goodbye!");
    Ok(())
}