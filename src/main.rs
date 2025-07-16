use std::time::Instant;

mod config;
mod logdb;
mod ufhg;
mod utils;
use logdb::LogDB;

fn main() {
    println!("=== LogDB Demo ===\n");

    let mut db = LogDB::new();

    // Ingest variety of logs
    println!("Ingesting logs...");
    let logs = vec![
        (
            "User authentication successful",
            Some("INFO".to_string()),
            Some("auth-service".to_string()),
        ),
        (
            "Failed login attempt for user john",
            Some("ERROR".to_string()),
            Some("auth-service".to_string()),
        ),
        (
            "Database connection established",
            Some("INFO".to_string()),
            Some("db-service".to_string()),
        ),
        (
            "Payment processing started",
            Some("INFO".to_string()),
            Some("payment-service".to_string()),
        ),
        (
            "Credit card validation failed",
            Some("ERROR".to_string()),
            Some("payment-service".to_string()),
        ),
        (
            "API rate limit exceeded",
            Some("WARN".to_string()),
            Some("api-gateway".to_string()),
        ),
        (
            "Server startup complete",
            Some("INFO".to_string()),
            Some("web-server".to_string()),
        ),
        (
            "Memory usage high",
            Some("WARN".to_string()),
            Some("monitoring".to_string()),
        ),
        (
            "Backup process completed successfully",
            Some("INFO".to_string()),
            Some("backup-service".to_string()),
        ),
        (
            "SSL certificate expiring soon",
            Some("WARN".to_string()),
            Some("security".to_string()),
        ),
        (
            "User session timeout",
            Some("INFO".to_string()),
            Some("session-manager".to_string()),
        ),
        (
            "Database query took 5.2 seconds",
            Some("WARN".to_string()),
            Some("db-service".to_string()),
        ),
        (
            "Cache miss for user profile",
            Some("DEBUG".to_string()),
            Some("cache-service".to_string()),
        ),
        (
            "Email notification sent",
            Some("INFO".to_string()),
            Some("notification-service".to_string()),
        ),
        (
            "Disk space low on server",
            Some("ERROR".to_string()),
            Some("monitoring".to_string()),
        ),
        (
            "User john logged out",
            Some("INFO".to_string()),
            Some("auth-service".to_string()),
        ),
        (
            "Payment transaction completed",
            Some("INFO".to_string()),
            Some("payment-service".to_string()),
        ),
        (
            "API response time degraded",
            Some("WARN".to_string()),
            Some("api-gateway".to_string()),
        ),
        (
            "Configuration file reloaded",
            Some("INFO".to_string()),
            Some("config-manager".to_string()),
        ),
        (
            "Health check failed",
            Some("ERROR".to_string()),
            Some("health-service".to_string()),
        ),
    ];

    for (content, level, service) in logs {
        db.upsert_log(content, level, service);
    }

    println!("Ingested {} log entries\n", 20);

    // Test various queries
    let queries = vec![
        "authentication",
        "level:ERROR",
        "service:payment-service",
        "level:INFO service:auth-service",
        "failed",
        "user john",
        "level:WARN",
        "service:db-service",
        "contains:timeout",
        "payment",
        "level:ERROR service:monitoring",
        "database",
        "server",
        "level:INFO contains:completed",
    ];

    println!("=== Query Results ===\n");

    for query in queries {
        let start = Instant::now();
        let results = db.query_content(query);
        let duration = start.elapsed();

        println!("Query: \"{}\"", query);
        println!("Time taken: {:?}", duration);
        println!("Results found: {}", results.len());

        if results.is_empty() {
            println!("No results found");
        } else {
            for (i, result) in results.iter().enumerate() {
                println!("  {}. {}", i + 1, result);
            }
        }
        println!("{}", "-".repeat(50));
    }

    // Test with metadata
    println!("\n=== Query with Metadata ===\n");
    let meta_query = "level:ERROR";
    let start = Instant::now();
    let meta_results = db.query_with_meta(meta_query);
    let duration = start.elapsed();

    println!("Query: \"{}\"", meta_query);
    println!("Time taken: {:?}", duration);
    println!("Results with metadata:");

    for (doc_id, content, level, service) in meta_results {
        println!(
            "  ID: {}, Content: {}, Level: {:?}, Service: {:?}",
            doc_id, content, level, service
        );
    }

    // Test compound queries
    println!("\n=== Compound Query Tests ===\n");

    let compound_queries = vec![
        "level:INFO service:auth-service",
        "level:ERROR service:payment-service",
        "level:WARN contains:server",
        "user authentication",
        "service:db-service level:WARN",
    ];

    for query in compound_queries {
        let start = Instant::now();
        let results = db.query_content(query);
        let duration = start.elapsed();

        println!("Compound Query: \"{}\"", query);
        println!("Time taken: {:?}", duration);
        println!("Results: {}", results.len());

        for (i, result) in results.iter().enumerate() {
            println!("  {}. {}", i + 1, result);
        }
        println!("{}", "-".repeat(50));
    }

    // Performance summary
    println!("\n=== Performance Summary ===");
    let start = Instant::now();
    let _total_docs = db.query_content("level:INFO").len()
        + db.query_content("level:ERROR").len()
        + db.query_content("level:WARN").len()
        + db.query_content("level:DEBUG").len();
    let total_query_time = start.elapsed();

    println!("Total time for 4 level queries: {:?}", total_query_time);
    println!("Average query time: {:?}", total_query_time / 4);

    println!("\n=== Demo Complete ===");
}
