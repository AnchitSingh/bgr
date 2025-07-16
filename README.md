# bgr (Buggu-GREP)

**bgr** is an ultra-fast, in-memory log search engine and command-line tool for exploring text files with microsecond query performance. It loads data from files or stdin and provides an interactive REPL for immediate exploration.

The name "Buggu" is inspired by a term of endearment, reflecting its role as a helpful assistant in debugging and log analysis.

## Overview

`bgr` is designed for speed and efficiency. It uses custom-built, cache-friendly data structures and a high-speed hashing mechanism (`UFHGHeadquarters`) to index log data in memory, allowing for near-instantaneous querying even on large files. It's a powerful alternative to traditional tools like `grep` when you need to perform structured and repeated queries on log data.

## Features

- **Blazing-Fast Search**: In-memory indexing provides query results in microseconds.
- **Interactive REPL**: An interactive shell for iteratively querying your data without re-loading it.
- **Non-Interactive Mode**: Run a single query directly from the command line for scripting.
- **Piped Input**: Works seamlessly with `stdin`, allowing you to pipe data from other commands (e.g., `cat`, `tail`, `docker logs`).
- **Powerful Query Syntax**:
    - **Term Search**: `error`
    - **Phrase Search**: `"database connection failed"`
    - **Field Search**: `level:error`, `service:api`
    - **Substring Search**: `contains:192.168.1`
    - **Logical Operators**: `AND` (default), `OR`, `NOT`

## Installation

You can install `bgr` directly from crates.io using Cargo:

```sh
cargo install bgr
```

## Usage

### Loading Data

Load one or more files:
```sh
bgr /var/log/syslog /var/log/auth.log
```

Load from stdin:
```sh
cat my-app.log | bgr
```

Tail a log file and pipe it into `bgr`:
```sh
tail -f my-app.log | bgr
```

### Interactive Mode

Running `bgr` with a file or piped data will drop you into an interactive session where you can run multiple queries.

```
$ cat app.log | bgr
Loading from stdin...
Loaded 10000 lines in 50.1ms

Welcome to bgr. Type your query or "quit" to exit.
> level:error
Query: level:error
Found 15 results in 12.3µs
1: [2025-07-17T10:00:00Z ERROR api] Failed to process request
...

> "user authentication"
Query: "user authentication"
Found 42 results in 15.8µs
1: [2025-07-17T10:01:12Z INFO auth] user authentication successful for user 'admin'
...

> quit
```

### Non-Interactive Mode

Use the `-q` or `--query` flag to run a single query and exit. This is useful for scripting.

```sh
bgr my-app.log --query "level:warn OR level:error"
```

## Building from Source

1. **Clone the repository:**
   ```sh
   git clone https://github.com/AnchitSingh/bgr.git
   cd bgr
   ```

2. **Build the project:**
   ```sh
   cargo build --release
   ```

3. **Run the executable:**
   The optimized binary will be located at `target/release/bgr`.
   ```sh
   ./target/release/bgr --help
   ```

## License

This project is licensed under the MIT License.
