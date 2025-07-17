# bgr (Buggu-GREP)

**bgr(v0.1.3)** is an ultra-fast, in-memory search engine and command-line tool that delivers microsecond-level query performance for logs and structured data. It indexes data from files or stdin and provides an interactive interface for immediate exploration and analysis.

The name "Buggu" is inspired by a term of endearment, reflecting its role as a trusted companion in debugging, log analysis, and data exploration.

## Overview

`bgr` is engineered for exceptional performance and efficiency. It leverages custom-built, cache-friendly data structures and an optimized hashing algorithm (`BugguHashSet` and `UFHGHeadquarters`) to index data in memory up to 40x faster than standard implementations. This architecture enables near-instantaneous querying even across large datasets, making it a powerful alternative to traditional tools like `grep` when you need structured, complex, and repeated queries.

## Features

- **Microsecond Query Performance**: In-memory indexing with specialized data structures delivers results in microseconds
- **Interactive Command Shell**: Explore data iteratively with a REPL interface and helpful commands
- **Advanced Query Capabilities**:
  - **Term & Phrase Search**: `error` or `"database connection failed"`
  - **Field-Specific Queries**: `level:ERROR`, `service:api`
  - **Regular Expression Search**: `regex:error.*timeout`
  - **JSON Field Queries**: `json:user.address.city=London`
  - **Logical Operators**: `AND` (default), `OR`, `NOT`
  - **Range Queries**: `timestamp:>=1627776000`
  - **JSONPath Syntax**: `json:user.address.city=London`
  - **Array Access**: Direct index `services[0]`, wildcard `services[?]`, slices `services[0:2]`, multiple indices `services[0,2,4]`
  - **Recursive Descent**: `json:..name` to find all name fields at any level
  - **Combined Queries**: `json:user.role=admin AND json:metrics.cpu>=90`
- **Flexible Input Sources**: Process files, stdin, piped data, or streaming logs
- **Scriptable**: Non-interactive mode for automation and integration with other tools

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

Running `bgr` with a file or piped data will drop you into an interactive session where you can run multiple queries:

```
$ cat app.log | bgr
Loading from stdin...
Loaded 10000 lines in 50.1ms (199,600 lines/sec)

BugguDB CLI v0.1.0 - Ultra-fast search engine
Features: term search, field filters, regex, JSON queries, boolean operators
Type 'help' to see available commands or 'quit' to exit.

> level:error
Query: level:error
Found 15 results in 12.3µs
1. [2025-07-17T10:00:00Z ERROR api] Failed to process request
...

> "user authentication"
Query: "user authentication"
Found 42 results in 15.8µs
1. [2025-07-17T10:01:12Z INFO auth] user authentication successful for user 'admin'
...

> regex:failed.*timeout
Query: regex:failed.*timeout
Found 8 results in 1.2ms
1. [2025-07-17T10:05:23Z ERROR api] Request failed due to connection timeout
...

> json:user.location.city=Seattle
Query: json:user.location.city=Seattle
Found 17 results in 1.5ms
1. [ID:1042] [INFO] User profile updated for user 'jsmith' in Seattle
...

> help
BugguDB Search Commands:
------------------------
  term                      Simple text search
  "exact phrase"           Phrase search (quoted)
  level:ERROR               Field search
  term1 AND term2           Logical AND (default between terms)
  term1 OR term2            Logical OR
  term NOT other            Exclude matches containing 'other'
  regex:pattern\d+          Regular expression search
  json:user.name            JSON field existence
  json:user.active=true     JSON field value matching
  json:metrics.cpu>=90      JSON numeric comparison
...

> quit
Goodbye!
```

### Non-Interactive Mode

Use the `-q` or `--query` flag to run a single query and exit. This is useful for scripting:

```sh
# Search for error logs
bgr my-app.log --query "level:error"

# Find all occurrences of a regex pattern
bgr my-app.log --query "regex:exception.*memory"

# Search within JSON fields
bgr data.log --query "json:server.metrics.cpu>=90"

# Complex queries with logical operators
bgr logs.txt --query "json:user.role=admin AND level:ERROR"
```

### Advanced Query Examples

#### Regular Expression Search
Find all lines matching a specific pattern:
```
> regex:user\d+\.login
Query: regex:user\d+\.login
Found 27 results in 1.4ms
```
### Advanced JSON Query Examples

`bgr` v0.1.3 introduces powerful JSONPath support for structured data queries:

```sh
# Basic JSON path queries
> json:user.name
> json:server.status

# Array access with multiple methods
> json:services[0].name            # Direct index access
> json:services[?].status=running  # Wildcard - any array element with status "running"
> json:metrics[0:2]                # Array slice - first two elements
> json:services[0,2,4].name        # Multiple specific indices

# Deep traversal with recursive descent
> json:..country                   # Find "country" field at any nesting level

# Complex conditions
> json:user.role=admin AND json:server.status=error
> json:server.metrics.cpu>=90 OR json:server.metrics.memory>=80
```

#### JSON Field Search
Search within nested JSON structures:
```
> json:user.address.country=Germany
Query: json:user.address.country=Germany
Found 12 results in 1.2ms
```

Combined with logical operators:
```
> json:server.status=error AND json:server.metrics.memory>=90
Query: json:server.status=error AND json:server.metrics.memory>=90
Found 3 results in 1.8ms
```

#### Array and Object Access
Access specific elements in arrays:
```
> json:server.services.0.name=database
Query: json:server.services.0.name=database
Found 5 results in 1.3ms
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

## Performance

`bgr` utilizes the custom `BugguHashSet` implementation that delivers up to 40x faster performance than the standard Rust HashMap. This optimization enables:

- Loading and indexing at 150,000+ lines per second
- Query response times in the microsecond range
- Efficient memory utilization even with large datasets
- **Blazing-Fast Indexing**: Processes 150,000+ lines per second
- **Microsecond Queries**: Most queries complete in 1-20μs
- **Efficient JSON Processing**: JSONPath queries process 400,000+ documents per second
- **Optimized Memory Usage**: Custom BugguHashSet implementation is 20-40x faster than standard hashmaps

The "collision loving" architecture prioritizes real-world performance over theoretical perfection, making it ideal for interactive log analysis and data exploration.

## New in v0.1.3

- **Enhanced JSONPath Support**: Advanced JSON querying with array wildcards, slices, and recursive descent
- **Performance Optimizations**: Improved memory usage and query performance
- **CLI Improvements**: Better help system and result formatting

## License

This project is licensed under the MIT License.