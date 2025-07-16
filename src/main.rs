// src/main.rs

use anyhow::Result;
use bgr::BugguDB; // Use our library
use clap::Parser;
use std::fs::File;
use std::io::{self, BufReader};
use std::path::PathBuf;
use std::time::Instant;

/// A blazing-fast, interactive command-line tool for exploring text files.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The file(s) to load. If not provided, reads from stdin.
    #[arg(name = "FILE")]
    files: Vec<PathBuf>,

    /// The query to run in non-interactive mode.
    #[arg(short, long)]
    query: Option<String>,
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
        total_lines += db.ingest_from_reader(reader)?;
    } else {
        // Read from files
        for path in &args.files {
            println!("Loading from {}...", path.display());
            let file = File::open(path)?;
            let reader = BufReader::new(file);
            total_lines += db.ingest_from_reader(reader)?;
        }
    }

    println!(
        "Loaded {} lines in {:.2?}
",
        total_lines,
        start.elapsed()
    );

    if let Some(query) = args.query {
        // Non-interactive mode
        run_query(&db, &query);
    } else {
        // Interactive mode
        run_interactive_session(db)?;
    }

    Ok(())
}

/// Runs a single query and prints the results.
fn run_query(db: &BugguDB, query: &str) {
    let start = Instant::now();
    let results = db.query_content(query);
    let duration = start.elapsed();

    println!("Query: {}", query);
    println!("Found {} results in {:.2?}", results.len(), duration);

    for (i, result) in results.iter().enumerate() {
        println!("{}: {}", i + 1, result);
    }
}

/// Starts an interactive REPL session.
fn run_interactive_session(db: BugguDB) -> Result<()> {
    let mut rl = rustyline::Editor::<()>::new()?;
    println!("Welcome to bgr. Type your query or \"quit\" to exit.");

    loop {
        let readline = rl.readline("> ");
        match readline {
            Ok(line) => {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                rl.add_history_entry(line);

                if line == "quit" || line == "q" {
                    break;
                }

                run_query(&db, line);
                println!(); // Add a newline for spacing
            }
            Err(_) => break, // Ctrl-C or Ctrl-D
        }
    }
    Ok(())
}
