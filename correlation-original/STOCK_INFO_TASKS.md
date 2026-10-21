# Stock Information Rake Tasks

This document describes the available rake tasks for managing stock information data.

## Available Tasks

### Summary and Status

#### `rails stock_info:summary`
Shows coverage statistics for stock information.

**Example output:**
```
========================================
STOCK INFO COVERAGE SUMMARY
========================================
Total symbols in database: 11431
Symbols with stock info: 94
Missing stock info: 11337
Coverage: 0.8%
```

### Updating Missing Information

#### `rails stock_info:test_update`
Process first 20 missing symbols (for testing).

#### `rails stock_info:batch_update`
Update stock info in controlled batches.

**Usage:**
```bash
# Process 100 symbols (default)
rails stock_info:batch_update

# Process 500 symbols
LIMIT=500 rails stock_info:batch_update

# Process 50 symbols
LIMIT=50 rails stock_info:batch_update
```

#### `rails stock_info:update_missing`
Process ALL missing symbols (use with caution - 11,000+ symbols).

### Refreshing Existing Data

#### `rails stock_info:refresh_existing`
Refresh all existing stock info records with updated data.

## Current Data Source

The system currently creates placeholder records with:
- **Company Name**: Generated from symbol patterns
- **Sector**: Educated guesses based on symbol patterns
- **Exchange**: Guessed from symbol length and market data
- **Country**: From existing stock_prices data
- **Description**: Placeholder text

## Future Enhancements

The `StockInfoLookupService` is designed to be easily extended with external API integrations:

### Potential APIs to integrate:
- **Alpha Vantage** - Company fundamentals
- **Yahoo Finance API** - Basic company info
- **Financial Modeling Prep** - Comprehensive data
- **IEX Cloud** - Real-time and historical data
- **Polygon.io** - Market data
- **Quandl** - Financial datasets

### To add API integration:
1. Uncomment and modify the API methods in `StockInfoLookupService`
2. Add API keys to environment variables
3. Add error handling and rate limiting
4. Test with small batches first

## Usage Recommendations

### For Development/Testing:
```bash
# Check current status
rails stock_info:summary

# Test with small batch
rails stock_info:test_update

# Process moderate batch
LIMIT=100 rails stock_info:batch_update
```

### For Production:
```bash
# Process in manageable chunks
LIMIT=1000 rails stock_info:batch_update

# Monitor progress
rails stock_info:summary

# Continue until complete
LIMIT=1000 rails stock_info:batch_update
```

## Safety Features

- **Rate limiting**: Built-in delays between requests
- **Batch processing**: Configurable limits to prevent overwhelming
- **Error handling**: Continues processing if individual symbols fail
- **Progress tracking**: Shows completion status
- **Duplicate prevention**: Won't create duplicate records

## Database Schema

The `stock_infos` table includes:
```sql
- symbol (string, unique index)
- name (string)
- description (text)
- sector (string, indexed)
- industry (string, indexed)
- market_cap (decimal)
- exchange (string)
- country (string)
- website (string)
- employees (integer)
- founded (integer)
```

## Performance Notes

- Processing 100 symbols takes ~1-2 minutes with current delays
- Each symbol has a 0.1 second delay (can be adjusted)
- Progress updates every 25 symbols with 1 second pause
- Memory usage is minimal as records are processed individually