# Data Service Integration - API Contract

**Project:** Gravity Technical Analysis Microservice  
**Document Version:** 1.0  
**Last Updated:** November 14, 2025  
**Author:** Dr. Chen Wei (CTO Software) + Dmitry Volkov (Backend Architect)

---

## 🎯 Overview

This document defines the **API Contract** between:
- **Data Ingestion Service** (upstream - provides clean, adjusted data)
- **Technical Analysis Service** (this service - consumes data for analysis)

### Architecture Principle: Separation of Concerns

```
┌─────────────────────────────────────────────────────────┐
│         MICROSERVICE RESPONSIBILITY MATRIX              │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────────────┐       ┌───────────────────┐  │
│  │  Data Ingestion      │       │  Technical        │  │
│  │  Service             │──────▶│  Analysis Service │  │
│  │                      │ JSON  │  (THIS SERVICE)   │  │
│  └──────────────────────┘       └───────────────────┘  │
│                                                          │
│  RESPONSIBLE FOR:          │    RESPONSIBLE FOR:        │
│  • API integrations        │    • Technical indicators  │
│    (Alpha Vantage,         │    • Pattern detection     │
│     CODAL, Yahoo)          │    • Scenario analysis     │
│  • Web scraping            │    • Fundamental scoring   │
│  • Data validation         │    • Signal generation     │
│  • Data cleaning           │    • Risk analysis         │
│  • Price adjustments       │    • Multi-horizon         │
│    (splits, dividends)     │      analysis              │
│  • Volume adjustments      │                            │
│  • Data storage            │    NOT RESPONSIBLE FOR:    │
│  • Data quality scoring    │    • Data fetching         │
│                            │    • Data cleaning         │
│  NOT RESPONSIBLE FOR:      │    • API keys management   │
│  • Technical analysis      │    • Data storage          │
│  • Indicators              │                            │
│  • Pattern detection       │                            │
│                            │                            │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 API Contract Specification

### Endpoint: Get Adjusted Candle Data

**URL:** `GET /api/v1/candles/{symbol}`

**Description:** Retrieve adjusted OHLCV (Open, High, Low, Close, Volume) data for a symbol.

### Request Parameters

| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `symbol` | string | ✅ Yes | Stock/crypto symbol | `AAPL`, `BTC-USD`, `فولاد` |
| `timeframe` | string | ✅ Yes | Candle timeframe | `1m`, `5m`, `15m`, `1h`, `4h`, `1d`, `1w` |
| `start_date` | ISO 8601 | ✅ Yes | Start date (UTC) | `2024-01-01T00:00:00Z` |
| `end_date` | ISO 8601 | ✅ Yes | End date (UTC) | `2024-11-14T23:59:59Z` |
| `fields` | string | ❌ No | Comma-separated fields | `adjusted_close,adjusted_volume` |

**Default behavior:**
- If `fields` not specified, return all OHLCV fields
- If `start_date` not specified, default to 1 year ago
- If `end_date` not specified, default to current UTC time

---

### Request Example

```http
GET /api/v1/candles/AAPL?timeframe=1d&start_date=2024-01-01T00:00:00Z&end_date=2024-11-14T23:59:59Z&fields=adjusted_open,adjusted_high,adjusted_low,adjusted_close,adjusted_volume
Host: data-service.gravity.local
Authorization: Bearer <JWT_TOKEN>
Accept: application/json
```

---

### Response Schema

```json
{
  "symbol": "AAPL",
  "timeframe": "1d",
  "candles": [
    {
      "timestamp": "2024-01-01T00:00:00Z",
      "adjusted_open": 150.0,
      "adjusted_high": 152.5,
      "adjusted_low": 149.25,
      "adjusted_close": 151.75,
      "adjusted_volume": 45678900
    },
    {
      "timestamp": "2024-01-02T00:00:00Z",
      "adjusted_open": 151.80,
      "adjusted_high": 153.20,
      "adjusted_low": 151.00,
      "adjusted_close": 152.50,
      "adjusted_volume": 38900500
    }
  ],
  "metadata": {
    "total_candles": 220,
    "adjustments_applied": ["splits", "dividends"],
    "data_quality_score": 0.98,
    "last_split_date": "2023-06-15",
    "last_dividend_date": "2024-05-10",
    "data_source": "alpha_vantage",
    "last_updated": "2024-11-14T10:30:00Z"
  }
}
```

---

### Response Fields

#### Candle Object

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `timestamp` | ISO 8601 | Candle timestamp (UTC) | Required, chronological order |
| `adjusted_open` | float | Open price (split/dividend adjusted) | > 0 |
| `adjusted_high` | float | High price (split/dividend adjusted) | > 0, >= low |
| `adjusted_low` | float | Low price (split/dividend adjusted) | > 0, <= high |
| `adjusted_close` | float | Close price (split/dividend adjusted) | > 0, between low and high |
| `adjusted_volume` | integer | Volume (split adjusted) | >= 0 |

#### Metadata Object

| Field | Type | Description |
|-------|------|-------------|
| `total_candles` | integer | Total number of candles returned |
| `adjustments_applied` | array[string] | Types of adjustments applied |
| `data_quality_score` | float | Quality score (0.0 - 1.0) |
| `last_split_date` | ISO 8601 | Date of last split (if any) |
| `last_dividend_date` | ISO 8601 | Date of last dividend (if any) |
| `data_source` | string | Original data source |
| `last_updated` | ISO 8601 | When data was last updated |

---

### Error Responses

#### 400 Bad Request
```json
{
  "error": "invalid_request",
  "message": "Invalid timeframe. Allowed: 1m, 5m, 15m, 1h, 4h, 1d, 1w",
  "details": {
    "field": "timeframe",
    "value": "3d",
    "allowed_values": ["1m", "5m", "15m", "1h", "4h", "1d", "1w"]
  }
}
```

#### 404 Not Found
```json
{
  "error": "symbol_not_found",
  "message": "Symbol 'XYZ123' not found in database",
  "details": {
    "symbol": "XYZ123",
    "suggestion": "Check symbol spelling or try another exchange"
  }
}
```

#### 429 Too Many Requests
```json
{
  "error": "rate_limit_exceeded",
  "message": "Rate limit exceeded: 100 requests per minute",
  "details": {
    "limit": 100,
    "window": "60s",
    "retry_after": 45
  }
}
```

#### 503 Service Unavailable
```json
{
  "error": "upstream_service_unavailable",
  "message": "Data source temporarily unavailable",
  "details": {
    "upstream_service": "alpha_vantage",
    "retry_after": 300,
    "fallback_available": false
  }
}
```

---

## 🔧 Technical Analysis Service - Client Implementation

### Python Client (Async)

```python
from gravity_tech.clients import DataServiceClient
from datetime import datetime, timedelta

# Initialize client
client = DataServiceClient(
    base_url="http://data-service:8080",
    timeout=30.0,
    max_retries=3,
    redis_url="redis://redis:6379/0",
    cache_ttl=21600  # 6 hours
)

# Get candles
candles = await client.get_candles(
    symbol="AAPL",
    timeframe="1d",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 11, 14),
    use_cache=True
)

# Use in indicators
from gravity_tech.indicators.trend import TrendIndicators

trend = TrendIndicators()
closes = [c.adjusted_close for c in candles]
sma_20 = trend.sma(closes, period=20)
```

---

## 🔒 Authentication & Security

### JWT Token Authentication

All requests must include a valid JWT token:

```http
Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Token Requirements:**
- Algorithm: RS256
- Expiration: 1 hour
- Refresh token available
- Scopes: `candles:read`

### Rate Limiting

- **Per IP:** 1000 requests / minute
- **Per API Key:** 10,000 requests / minute
- **Premium tier:** 100,000 requests / minute

### HTTPS Only

All communication must use TLS 1.3+

---

## 📈 Service Discovery

### Kubernetes Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: data-service
  namespace: gravity
spec:
  selector:
    app: data-service
  ports:
    - protocol: TCP
      port: 8080
      targetPort: 8080
  type: ClusterIP
```

### DNS Resolution

Internal services can access via:
- `http://data-service:8080` (same namespace)
- `http://data-service.gravity.svc.cluster.local:8080` (full FQDN)

---

## 📊 Performance Requirements

### Latency Targets

| Metric | Target | Maximum |
|--------|--------|---------|
| P50 (median) | < 50ms | 100ms |
| P95 | < 100ms | 200ms |
| P99 | < 200ms | 500ms |

### Throughput

- **Minimum:** 10,000 requests/second
- **Target:** 50,000 requests/second
- **Burst:** 100,000 requests/second

### Availability

- **SLA:** 99.9% uptime
- **Downtime budget:** 43 minutes/month

---

## 🧪 Testing Contract

### Contract Testing with Pact

Technical Analysis Service (consumer) defines expected contract:

```python
# tests/contract/test_data_service_contract.py
import pytest
from pact import Consumer, Provider

pact = Consumer('technical-analysis-service').has_pact_with(
    Provider('data-service')
)

pact.given('symbol AAPL exists with daily candles').upon_receiving(
    'a request for AAPL daily candles'
).with_request(
    method='GET',
    path='/api/v1/candles/AAPL',
    query={'timeframe': '1d', 'start_date': '2024-01-01T00:00:00Z', 'end_date': '2024-01-31T23:59:59Z'}
).will_respond_with(
    status=200,
    body={
        'symbol': 'AAPL',
        'timeframe': '1d',
        'candles': [
            {
                'timestamp': '2024-01-01T00:00:00Z',
                'adjusted_close': 150.0,
                'adjusted_volume': 1000000
            }
        ]
    }
)
```

---

## 📋 Versioning

### API Version

Current version: **v1**

Breaking changes will increment version:
- `/api/v1/candles/...` (current)
- `/api/v2/candles/...` (future)

### Backward Compatibility

- v1 supported until December 31, 2026
- 6 months notice before deprecation
- Migration guide provided

---

## 🔄 Caching Strategy

### Technical Analysis Service (Consumer) Caching

**Redis Cache:**
- **TTL:** 6 hours for daily candles
- **TTL:** 1 hour for intraday candles
- **Key format:** `candles:{symbol}:{timeframe}:{start}:{end}`
- **Invalidation:** Manual or TTL expiration

**Cache Hit Rate Target:** 85%+

---

## 📞 Support & SLA

### Data Service Team Contact

- **Team Lead:** [Data Team Lead Name]
- **Slack Channel:** `#data-service-support`
- **Email:** data-service@gravity.tech
- **On-call:** PagerDuty rotation

### Issue Escalation

1. **P0 (Critical):** Service down - 15 min response
2. **P1 (High):** Degraded performance - 1 hour response
3. **P2 (Medium):** Minor issues - 4 hours response
4. **P3 (Low):** Questions - 24 hours response

---

## 📝 Changelog

### v1.0 - November 14, 2025
- Initial API contract definition
- Added authentication requirements
- Defined error responses
- Specified performance targets

---

**Approved By:**
- ✅ Dr. Chen Wei (CTO Software)
- ✅ Dmitry Volkov (Backend Architect)
- ✅ Shakour Alishahi (Product Owner)
- ✅ Lars Andersson (DevOps Lead)
- ✅ Marco Rossi (Security Expert)
