# Known Shortcomings

This document outlines the current limitations, known issues, and areas for improvement in the Stock Correlation Analysis Platform.

## Table of Contents

- [Statistical Limitations](#statistical-limitations)
- [Data Quality Issues](#data-quality-issues)
- [Performance Limitations](#performance-limitations)
- [API and Integration Issues](#api-and-integration-issues)
- [User Interface Limitations](#user-interface-limitations)
- [Technical Debt](#technical-debt)
- [Scalability Concerns](#scalability-concerns)
- [Security Considerations](#security-considerations)

## Statistical Limitations

### 1. Linear Correlation Only

**Issue:** The platform only implements Pearson correlation, which only captures linear relationships.

**Impact:**
- Misses non-linear but meaningful relationships
- May underestimate correlation strength for curved relationships
- Limited applicability to complex market dynamics

**Example:**
```
Stock A and Stock B might have a strong exponential relationship,
but Pearson correlation would show weak linear correlation.
```

**Potential Solutions:**
- Implement Spearman rank correlation for monotonic relationships
- Add Kendall's tau for non-parametric correlation
- Consider polynomial regression analysis

### 2. Static Time Windows

**Issue:** Analysis uses fixed time windows without considering market volatility or regime changes.

**Impact:**
- May miss short-term correlations during high volatility
- Could include regime changes that distort correlation calculations
- Doesn't adapt to different market conditions

**Potential Solutions:**
- Implement adaptive window sizing based on volatility
- Add regime detection algorithms
- Support for multiple simultaneous window sizes

### 3. No Causality Testing

**Issue:** Correlation doesn't establish causation or directional relationships.

**Impact:**
- Cannot determine which stock leads the other
- Missing information for trading strategies
- Limited insight into market dynamics

**Potential Solutions:**
- Implement Granger causality testing
- Add lead-lag analysis
- Vector autoregression (VAR) models

### 4. Autocorrelation Ignorance

**Issue:** Doesn't account for time series autocorrelation in the data.

**Impact:**
- May overestimate correlation significance
- Doesn't adjust for serial correlation in residuals
- Statistical tests may be invalid

**Potential Solutions:**
- Implement Durbin-Watson test for autocorrelation
- Add Newey-West standard error corrections
- Use autoregressive models for preprocessing

## Data Quality Issues

### 1. Inactive Data Contamination

**Issue:** Historical correlations include data from now-inactive symbols.

**Current Status:** ⚠️ **Partially Fixed**
- Moved inactive flag to stock_infos table
- Updated queries to use only active symbols
- Legacy correlations still include inactive symbol data

**Remaining Impact:**
- Historical correlation analysis may be skewed
- Inconsistent correlation counts between old and new data
- Need to recompute correlations for active symbols only

**Solution:** Recompute all correlations using only active symbols

### 2. Missing Company Information

**Issue:** Many symbols lack comprehensive company information.

**Statistics:**
- 11,197 symbols have placeholder data
- Only ~224 symbols are currently active
- API rate limits restrict data enrichment speed

**Impact:**
- Limited sector/industry analysis capabilities
- Reduced filtering and grouping options
- Incomplete fundamental analysis

### 3. Data Synchronization

**Issue:** Stock prices and company information may be out of sync.

**Impact:**
- Inconsistent data freshness
- Potential analysis errors
- User confusion about data currency

**Potential Solutions:**
- Implement data synchronization checks
- Add last updated timestamps
- Automated data refresh schedules

## Performance Limitations

### 1. Large Dataset Memory Usage

**Issue:** Processing large datasets can consume significant memory.

**Specific Problems:**
- Loading entire price histories into memory
- Multiple symbol analysis creates large intermediate arrays
- Garbage collection delays during heavy processing

**Impact:**
- Potential memory exhaustion on large analyses
- Slower performance with increasing data size
- Limited concurrent analysis capability

**Potential Solutions:**
- Implement streaming data processing
- Add memory usage monitoring
- Optimize data structures and algorithms

### 2. Database Query Optimization

**Issue:** Some queries could be further optimized.

**Areas for Improvement:**
- Complex joins in correlation queries
- Repeated calculations in aggregation queries
- Missing indexes on some frequently accessed columns

**Potential Solutions:**
- Add database query profiling
- Implement query result caching
- Optimize frequently used query patterns

### 3. Single-Threaded API Calls

**Issue:** API calls to external services are sequential.

**Impact:**
- Slow company information updates
- Inefficient use of API rate limits
- Extended processing times for large batches

**Potential Solutions:**
- Implement concurrent API calls with rate limiting
- Add request queuing and batching
- Use multiple API keys for increased throughput

## API and Integration Issues

### 1. API Rate Limiting

**Issue:** Financial Modeling Prep API has strict rate limits.

**Current Limits:**
- Free tier: 500MB/month
- Rate limiting: ~3-5 requests per second
- Daily quotas may be reached quickly

**Impact:**
- Slow data enrichment process
- Frequent task interruptions
- Limited scalability for large datasets

**Workarounds:**
- Intelligent batching and resumable tasks
- Fallback to placeholder data
- Rate limit respecting delays

### 2. Single API Dependency

**Issue:** Relies on single external API for company data.

**Risk:**
- API service outages affect functionality
- Vendor lock-in concerns
- Limited data source diversity

**Potential Solutions:**
- Implement multiple API provider support
- Add local data caching strategies
- Create fallback data sources

### 3. Error Handling Coverage

**Issue:** Some edge cases in API integration aren't fully handled.

**Examples:**
- Temporary network failures
- Malformed API responses
- Service maintenance windows

**Potential Solutions:**
- Expand error handling scenarios
- Implement retry mechanisms with exponential backoff
- Add comprehensive logging and monitoring

## User Interface Limitations

### 1. Limited Real-time Updates

**Issue:** Charts and data don't update in real-time.

**Impact:**
- Users must manually refresh to see new data
- No live correlation monitoring
- Static user experience

**Potential Solutions:**
- Implement WebSocket connections
- Add automatic data refresh
- Real-time correlation alerts

### 2. Basic Search and Filtering

**Issue:** Limited search capabilities across the platform.

**Current Limitations:**
- Basic text search only
- No advanced filtering options
- Limited sorting capabilities

**Potential Solutions:**
- Add advanced search with filters
- Implement faceted search
- Support for saved searches and alerts

### 3. Mobile Responsiveness

**Issue:** Some components aren't fully optimized for mobile devices.

**Specific Issues:**
- Chart interactions on touch devices
- Table scrolling on small screens
- Navigation menu on mobile

**Potential Solutions:**
- Improve touch interaction support
- Optimize table layouts for mobile
- Enhance responsive design patterns

## Technical Debt

### 1. Test Coverage

**Issue:** Limited automated test coverage.

**Current Status:**
- Basic model tests exist
- Missing controller and integration tests
- No automated regression testing

**Impact:**
- Risk of introducing bugs during development
- Difficult to refactor with confidence
- Limited continuous integration capabilities

### 2. Code Documentation

**Issue:** Inconsistent code documentation and comments.

**Areas Needing Improvement:**
- Complex statistical algorithms
- API integration services
- Database migration strategies

### 3. Configuration Management

**Issue:** Some configuration is hardcoded rather than externalized.

**Examples:**
- Statistical significance thresholds
- Time window ranges
- Batch processing sizes

**Potential Solutions:**
- Move configuration to external files
- Add runtime configuration updates
- Implement feature flags

## Scalability Concerns

### 1. Database Growth

**Issue:** Database will grow significantly with more data and correlations.

**Projections:**
- Correlation table could reach millions of records
- Price data continues to accumulate daily
- Index maintenance overhead will increase

**Potential Solutions:**
- Implement data archiving strategies
- Add database partitioning
- Consider NoSQL alternatives for specific use cases

### 2. Concurrent User Support

**Issue:** Platform not designed for high concurrent usage.

**Limitations:**
- Shared database connection pool
- No user session management
- No load balancing considerations

**Potential Solutions:**
- Add user authentication and session management
- Implement caching layers
- Design for horizontal scaling

### 3. Background Job Processing

**Issue:** Long-running tasks block the main application.

**Current State:**
- Rake tasks run synchronously
- No job queue implementation
- Limited progress monitoring

**Potential Solutions:**
- Implement background job processing (Sidekiq, etc.)
- Add job progress tracking
- Support for job cancellation and restart

## Security Considerations

### 1. No Authentication System

**Issue:** Platform is completely open with no user management.

**Security Implications:**
- No access control
- No audit trails
- Potential for misuse

**Note:** This may be intentional for a research platform, but limits production deployment options.

### 2. API Key Management

**Issue:** API keys are stored in environment variables without encryption.

**Potential Improvements:**
- Encrypted key storage
- Key rotation capabilities
- Multiple key management

### 3. Input Validation

**Issue:** Limited input validation in some areas.

**Potential Vulnerabilities:**
- SQL injection (mitigated by ActiveRecord)
- Parameter manipulation
- File upload vulnerabilities (if added)

## Priority Assessment

### High Priority (Should Fix Soon)

1. **Recompute correlations for active symbols only**
   - Affects core functionality accuracy
   - Relatively straightforward to implement

2. **Improve test coverage**
   - Critical for maintenance and development
   - Prevents regression bugs

3. **Memory optimization for large datasets**
   - Affects platform usability
   - Important for scalability

### Medium Priority (Nice to Have)

1. **Add Spearman correlation**
   - Expands analysis capabilities
   - Moderate implementation complexity

2. **Implement background job processing**
   - Improves user experience
   - Enables better scalability

3. **Enhanced error handling**
   - Improves reliability
   - Better user experience

### Low Priority (Future Enhancements)

1. **Real-time updates**
   - Significant implementation effort
   - Limited immediate business value

2. **Multiple API providers**
   - Complex integration work
   - Current single provider is sufficient

3. **Advanced UI features**
   - Polish improvements
   - Lower impact on core functionality

## Contributing to Improvements

If you're interested in addressing any of these shortcomings:

1. **Review the issue** thoroughly to understand the scope
2. **Check existing implementations** for patterns and conventions
3. **Start with tests** to define expected behavior
4. **Implement incrementally** with backward compatibility
5. **Document changes** thoroughly

For major architectural changes, please discuss the approach in issues before implementation.