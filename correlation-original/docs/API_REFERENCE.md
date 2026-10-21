# API Reference

This document provides a comprehensive guide to all API endpoints available in the Stock Correlation Analysis Platform.

## Table of Contents

- [Authentication](#authentication)
- [Response Format](#response-format)
- [Chart Data Endpoints](#chart-data-endpoints)
- [Stock Data Endpoints](#stock-data-endpoints)
- [Correlation Analysis Endpoints](#correlation-analysis-endpoints)
- [Error Handling](#error-handling)

## Authentication

Currently, all API endpoints are publicly accessible. No authentication is required.

## Response Format

All API responses are returned in JSON format with the following structure:

### Success Response
```json
{
  "data": [...],
  "count": 123,
  "symbol": "AAPL"
}
```

### Error Response
```json
{
  "error": "Error message",
  "status": 400
}
```

## Chart Data Endpoints

### GET `/api/chart_data`

Returns aggregated data for the main dashboard charts.

![Platform Overview](../screenshots/home-page-full.png)
*Example: Complete platform interface showing charts powered by `/api/chart_data` and table powered by `/api/stocks`*

![Stock Detail Interface](../screenshots/stock-detail-full.png)
*Example: Individual stock detail page showing price data from `/api/price_data/:symbol` endpoint*

**Response:**
```json
{
  "stocks": [
    [2020, 5000],
    [2021, 5500],
    [2022, 6000]
  ],
  "correlations": [
    [0, 150],
    [7, 200],
    [14, 180]
  ]
}
```

**Data Structure:**
- `stocks`: Array of `[year, cumulative_stock_count]` pairs
- `correlations`: Array of `[offset_days, correlation_count]` pairs

### GET `/api/price_data/:symbol`

Returns historical price data for a specific stock symbol.

**Parameters:**
- `symbol` (required): Stock symbol (e.g., "AAPL")

**Response:**
```json
{
  "data": [
    [1640995200, 150.25],
    [1641081600, 152.10],
    [1641168000, 148.75]
  ],
  "symbol": "AAPL",
  "count": 500
}
```

**Data Structure:**
- `data`: Array of `[timestamp, price]` pairs
- `symbol`: Requested stock symbol
- `count`: Number of data points returned

**Notes:**
- Returns last 2 years of data by default
- If less than 500 points in 2 years, returns all available data
- Timestamps are Unix timestamps (seconds since epoch)

### GET `/api/recent_price_data/:symbol`

Returns recent price data (last 30 days) for a specific stock symbol.

**Parameters:**
- `symbol` (required): Stock symbol (e.g., "AAPL")

**Response:**
```json
{
  "data": [
    [1703980800, 185.64],
    [1704067200, 186.19],
    [1704153600, 184.40]
  ],
  "symbol": "AAPL",
  "count": 30
}
```

**Notes:**
- Returns last 30 days by default
- If less than 10 points in 30 days, returns last 30 records
- Same format as `/api/price_data/:symbol`

## Stock Data Endpoints

### GET `/api/stocks`

Returns paginated data for the interactive stocks table.

*Note: The DataTables interface shown in the platform overview above is powered by this endpoint*

**Query Parameters:**
- `draw`: DataTables draw counter
- `start`: Starting record number
- `length`: Number of records to return
- `search[value]`: Search term for filtering

**Response:**
```json
{
  "data": [
    {
      "symbol": "AAPL",
      "market": "NASDAQ",
      "country": "United States",
      "records": 1500,
      "date_range": "2020-01-01 to 2024-01-01",
      "price_range": "120.50 - 198.23"
    }
  ]
}
```

**Data Structure:**
- `symbol`: Stock symbol
- `market`: Exchange/market name
- `country`: Country of origin
- `records`: Number of price records available
- `date_range`: Date range of available data
- `price_range`: Min and max closing prices

## Correlation Analysis Endpoints

### Stock Correlation Model

The correlation analysis is primarily handled through the Rails models and rake tasks, but the results are accessible through the web interface. Here's the data structure:

```ruby
# StockCorrelation attributes
{
  symbol1: "AAPL",
  symbol2: "MSFT", 
  window1_days: 7,
  window2_days: 7,
  offset_days: 0,
  correlation_coefficient: 0.7234,
  r_squared: 0.5233,
  data_points: 156,
  start_date: "2023-01-01",
  end_date: "2023-12-31",
  created_at: "2024-01-01T12:00:00Z"
}
```

## Error Handling

### Common HTTP Status Codes

- `200`: Success
- `400`: Bad Request (invalid parameters)
- `404`: Not Found (symbol or resource not found)
- `500`: Internal Server Error

### Error Examples

**Symbol Not Found:**
```json
{
  "error": "Stock symbol INVALID not found",
  "status": 404
}
```

**Invalid Parameters:**
```json
{
  "error": "Symbol parameter is required",
  "status": 400
}
```

## Usage Examples

### JavaScript/Fetch

```javascript
// Get chart data for dashboard
fetch('/api/chart_data')
  .then(response => response.json())
  .then(data => {
    console.log('Stocks over time:', data.stocks);
    console.log('Correlations by offset:', data.correlations);
  });

// Get price data for AAPL
fetch('/api/price_data/AAPL')
  .then(response => response.json())
  .then(data => {
    console.log(`${data.count} price points for ${data.symbol}`);
    data.data.forEach(([timestamp, price]) => {
      console.log(new Date(timestamp * 1000), price);
    });
  });
```

### cURL

```bash
# Get chart data
curl -X GET "http://localhost:3000/api/chart_data"

# Get AAPL price data
curl -X GET "http://localhost:3000/api/price_data/AAPL"

# Get stocks table data
curl -X GET "http://localhost:3000/api/stocks?draw=1&start=0&length=10"
```

## Rate Limiting

Currently, there are no rate limits imposed on the API endpoints. However, for production use, consider implementing rate limiting to prevent abuse.

## Data Freshness

- **Stock Prices**: Data freshness depends on your data import schedule
- **Stock Information**: Updated via rake tasks with FMP API integration
- **Correlations**: Generated on-demand via correlation analysis rake tasks

## Integration Notes

### For Chart Libraries

The API endpoints are designed to work seamlessly with popular charting libraries:

- **uPlot**: `/api/price_data/:symbol` returns data in uPlot-compatible format
- **Chart.js**: Data can be easily transformed for Chart.js consumption
- **D3.js**: Raw data format is suitable for D3 visualizations

### For DataTables

The `/api/stocks` endpoint is specifically designed for DataTables integration and includes:
- Server-side processing support
- Search functionality
- Pagination support
- Proper response format for DataTables

## Future Enhancements

Planned API improvements include:

1. **Authentication System**: JWT-based authentication for secure access
2. **Real-time Updates**: WebSocket endpoints for live data streaming
3. **Bulk Operations**: Endpoints for batch processing and analysis
4. **Advanced Filtering**: More sophisticated query parameters and filters
5. **API Versioning**: Proper versioning scheme for backward compatibility