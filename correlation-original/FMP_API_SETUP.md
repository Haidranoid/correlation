# Financial Modeling Prep (FMP) API Setup

## Step 1: Get Your Free API Key

1. Visit [https://financialmodelingprep.com/developer](https://financialmodelingprep.com/developer)
2. Click "Get Free API Key" 
3. Sign up with your email
4. Verify your email address
5. Copy your API key from the dashboard

## Step 2: Add API Key to Environment

### For Development (.env file):
```bash
# Add to your .env file (create if it doesn't exist)
FMP_API_KEY=your_api_key_here
```

### For Production:
```bash
# Set environment variable
export FMP_API_KEY=your_api_key_here

# Or add to your deployment configuration
# Heroku: heroku config:set FMP_API_KEY=your_key
# Railway: Add to environment variables
# Docker: Add to docker-compose.yml environment section
```

## Step 3: Test the Integration

```bash
# Test with a single known symbol
rails runner "puts StockInfoLookupService.new('AAPL').call.inspect"

# Test the rake task
rails stock_info:test_update
```

## API Details

### Free Tier Limits:
- **500MB bandwidth per month**
- **No daily request limits**
- **No symbol limits** 
- **Access to all endpoints**

### Data Available:
- Company name and description
- Sector and industry
- Market capitalization
- Exchange and country
- Website URL
- Employee count
- Financial fundamentals

### API Endpoints Used:
```
GET https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={your_key}
```

### Example Response:
```json
[
  {
    "symbol": "AAPL",
    "companyName": "Apple Inc.",
    "sector": "Technology",
    "industry": "Consumer Electronics",
    "description": "Apple Inc. designs, manufactures, and markets smartphones...",
    "exchangeShortName": "NASDAQ",
    "country": "US",
    "website": "https://www.apple.com",
    "mktCap": 3000000000000,
    "fullTimeEmployees": 164000
  }
]
```

## Usage Examples

### Update 10 symbols with real API data:
```bash
# Make sure FMP_API_KEY is set
echo $FMP_API_KEY

# Test with small batch
LIMIT=10 rails stock_info:batch_update
```

### Check results:
```bash
rails stock_info:summary
```

### View updated record:
```bash
rails runner "puts StockInfo.find_by(symbol: 'AAPL')&.name"
```

## Error Handling

The service includes comprehensive error handling:
- **API failures** → Falls back to placeholder data
- **Invalid responses** → Logs errors and continues
- **Rate limiting** → Built-in delays in rake tasks
- **Missing data** → Graceful handling with nil checks

## Monitoring Usage

### Check API usage:
- Login to [FMP Dashboard](https://financialmodelingprep.com/developer)
- View "Usage" section
- Monitor bandwidth consumption

### Optimize usage:
- Process in batches of 100-500 symbols
- Add delays between requests if needed
- Cache results to avoid duplicate calls

## Troubleshooting

### Common Issues:

1. **"FMP API failed" messages**
   - Check if FMP_API_KEY is set: `echo $FMP_API_KEY`
   - Verify API key is valid in FMP dashboard
   - Check bandwidth usage hasn't exceeded 500MB

2. **Empty responses**
   - Some symbols may not exist in FMP database
   - International symbols may have different formatting
   - Falls back to placeholder data automatically

3. **Rate limiting**
   - FMP free tier has no rate limits
   - If seeing errors, check API key validity
   - Monitor bandwidth usage in dashboard

### Test API key:
```bash
curl "https://financialmodelingprep.com/api/v3/profile/AAPL?apikey=YOUR_KEY"
```

## Alternative APIs (Backup Options)

If FMP doesn't work for your needs:

1. **Finnhub** (500 requests/day)
2. **Alpha Vantage** (25 requests/day) 
3. **Polygon.io** (5 requests/minute)

The service is designed to be easily switched between APIs by modifying the `fetch_from_fmp_api` method.