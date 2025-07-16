use bgr::buggudb::BugguDB;
use std::time::Instant;

fn main() {
    println!("🚀 BugguDB Stress Test Demo");
    println!("========================");

    let mut db = BugguDB::new();

    // Rapid fire insert test
    println!("\n📝 Testing rapid inserts...");
    let entries: Vec<String> = (0..10000)
        .map(|i| {
            format!(
                "ERROR [service-{}] Database timeout - user:{} duration:{}ms",
                i % 10,
                i % 1000,
                (i % 5000) + 100
            )
        })
        .collect();

    let insert_start = Instant::now();
    for (i, entry) in entries.iter().enumerate() {
        db.upsert_simple(entry);
        if i % 1000 == 0 && i > 0 {
            let elapsed = insert_start.elapsed();
            let rate = i as f64 / elapsed.as_secs_f64();
            println!("  {} entries inserted, rate: {:.0} inserts/sec", i, rate);
        }
    }
    let total_insert_time = insert_start.elapsed();

    println!(
        "✅ Inserted {} entries in {:.2}ms",
        entries.len(),
        total_insert_time.as_millis()
    );
    println!(
        "   Average: {:.2}μs per insert",
        total_insert_time.as_nanos() as f64 / entries.len() as f64 / 1000.0
    );

    // Rapid fire query test
    println!("\n🔍 Testing rapid queries...");
    let queries = [
        "ERROR",
        "contains:database",
        "contains:timeout",
        "\"Database timeout\"",
        "level:ERROR",
        "service:auth",
    ];

    for query in &queries {
        let mut total_time = 0u128;
        let iterations = 1000;

        for _ in 0..iterations {
            let start = Instant::now();
            let results = db.query(query);
            total_time += start.elapsed().as_nanos();

            // Verify we got results
            if results.is_empty() {
                println!("  ⚠️  Query '{}' returned no results", query);
            }
        }

        let avg_time_us = total_time as f64 / iterations as f64 / 1000.0;
        println!(
            "  Query '{}': {:.2}μs average ({} iterations)",
            query, avg_time_us, iterations
        );
    }

    // Memory and performance stats
    println!("\n📊 Final Statistics:");

    // Cleanup test
    println!("\n🧹 Testing cleanup...");
    let cleanup_start = Instant::now();
    db.cleanup_stale();
    let cleanup_time = cleanup_start.elapsed();
    println!("Cleanup took: {:.2}ms", cleanup_time.as_millis());

    println!("\n✨ Stress test completed!");
}
