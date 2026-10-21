# Parallel Testing Setup

This project is configured to run RSpec tests in parallel using all available CPU cores for faster test execution.

## Available Commands

### Basic Parallel Testing
```bash
# Run all tests in parallel (automatically uses optimal number of processes)
bundle exec parallel_rspec spec/

# Run with specific number of processes
bundle exec parallel_rspec spec/ -n 8

# Use the custom script
./bin/parallel_test 8
```

### Performance Comparison
```bash
# Sequential execution (original)
time bundle exec rspec --format progress

# Parallel execution 
time bundle exec parallel_rspec spec/ -n 6
```

## Results on Your System (24 cores)

- **Sequential**: ~4.0 seconds
- **Parallel (6 processes)**: ~3.0-4.0 seconds  
- **Test count**: 162 examples, 0 failures

## Why Parallel Testing?

1. **Faster Feedback**: Reduced test execution time
2. **Better CPU Utilization**: Uses multiple cores instead of just one
3. **Scalable**: Automatically adjusts to available spec files
4. **Database Isolation**: Each process gets its own test database

## Technical Details

- **Gem**: `parallel_tests` 
- **Execution Model**: **MULTIPLE PROCESSES** (not threads)
- **True Parallelism**: Bypasses Ruby's Global Interpreter Lock (GIL)
- **Database**: PostgreSQL with separate test databases per process
- **Process Count**: Automatically limited by number of spec files (6 in this project)
- **Silent Mode**: All tests run with progress indicators suppressed

### Processes vs Threads (Key Difference!)

**✅ PROCESSES (what parallel_tests uses):**
- True parallelism - bypasses Ruby's GIL
- Each process has its own memory space
- Can utilize ALL CPU cores simultaneously
- Separate database connections per process
- Higher memory usage but true parallel execution

**❌ THREADS (Ruby's default threading):**
- Limited by Ruby's GIL - only one thread executes Ruby code at a time
- Shared memory space between threads
- Good for I/O-bound operations, poor for CPU-bound work
- Lower memory usage but not truly parallel for Ruby code

## Tips

- For your 24-core system, 6-8 processes is optimal for this test suite
- The parallel_tests gem automatically creates separate databases for each process
- All existing RSpec features work (factories, shoulda-matchers, etc.)
- Test isolation is maintained - each process runs independently

## Troubleshooting

If you encounter database issues:
```bash
bundle exec rake db:test:prepare
bundle exec parallel_rspec spec/ -n 4  # Try fewer processes
```