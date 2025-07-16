```
=== LogDB Demo ===

Ingesting logs...
Ingested 20 log entries

=== Query Results ===

Query: "authentication"
Time taken: 1.784µs
Results found: 1
  1. User authentication successful
--------------------------------------------------
Query: "level:ERROR"
Time taken: 1.662µs
Results found: 4
  1. Failed login attempt for user john
  2. Credit card validation failed
  3. Disk space low on server
  4. Health check failed
--------------------------------------------------
Query: "service:payment-service"
Time taken: 754ns
Results found: 3
  1. Payment processing started
  2. Credit card validation failed
  3. Payment transaction completed
--------------------------------------------------
Query: "level:INFO service:auth-service"
Time taken: 2.69µs
Results found: 2
  1. User authentication successful
  2. User john logged out
--------------------------------------------------
Query: "failed"
Time taken: 458ns
Results found: 3
  1. Failed login attempt for user john
  2. Credit card validation failed
  3. Health check failed
--------------------------------------------------
Query: "user john"
Time taken: 1.113µs
Results found: 2
  1. Failed login attempt for user john
  2. User john logged out
--------------------------------------------------
Query: "level:WARN"
Time taken: 708ns
Results found: 5
  1. API rate limit exceeded
  2. Memory usage high
  3. SSL certificate expiring soon
  4. Database query took 5.2 seconds
  5. API response time degraded
--------------------------------------------------
Query: "service:db-service"
Time taken: 458ns
Results found: 2
  1. Database connection established
  2. Database query took 5.2 seconds
--------------------------------------------------
Query: "contains:timeout"
Time taken: 468ns
Results found: 1
  1. User session timeout
--------------------------------------------------
Query: "payment"
Time taken: 288ns
Results found: 2
  1. Payment processing started
  2. Payment transaction completed
--------------------------------------------------
Query: "level:ERROR service:monitoring"
Time taken: 1.144µs
Results found: 1
  1. Disk space low on server
--------------------------------------------------
Query: "database"
Time taken: 273ns
Results found: 2
  1. Database connection established
  2. Database query took 5.2 seconds
--------------------------------------------------
Query: "server"
Time taken: 330ns
Results found: 2
  1. Server startup complete
  2. Disk space low on server
--------------------------------------------------
Query: "level:INFO contains:completed"
Time taken: 1.332µs
Results found: 2
  1. Payment transaction completed
  2. Backup process completed successfully
--------------------------------------------------

=== Query with Metadata ===

Query: "level:ERROR"
Time taken: 1.13µs
Results with metadata:
  ID: 2, Content: Failed login attempt for user john, Level: Some("ERROR"), Service: Some("auth-service")
  ID: 5, Content: Credit card validation failed, Level: Some("ERROR"), Service: Some("payment-service")
  ID: 15, Content: Disk space low on server, Level: Some("ERROR"), Service: Some("monitoring")
  ID: 20, Content: Health check failed, Level: Some("ERROR"), Service: Some("health-service")

=== Compound Query Tests ===

Compound Query: "level:INFO service:auth-service"
Time taken: 1.254µs
Results: 2
  1. User authentication successful
  2. User john logged out
--------------------------------------------------
Compound Query: "level:ERROR service:payment-service"
Time taken: 927ns
Results: 1
  1. Credit card validation failed
--------------------------------------------------
Compound Query: "level:WARN contains:server"
Time taken: 1.001µs
Results: 0
--------------------------------------------------
Compound Query: "user authentication"
Time taken: 819ns
Results: 1
  1. User authentication successful
--------------------------------------------------
Compound Query: "service:db-service level:WARN"
Time taken: 990ns
Results: 1
  1. Database query took 5.2 seconds
--------------------------------------------------

=== Performance Summary ===
Total time for 4 level queries: 2.479µs
Average query time: 619ns

=== Demo Complete ===

```