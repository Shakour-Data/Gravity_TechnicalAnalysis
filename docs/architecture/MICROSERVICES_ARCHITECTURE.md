# Microservices Architecture - Service Responsibilities

**Project:** Gravity Financial Analysis Platform  
**Document Version:** 1.0  
**Last Updated:** November 14, 2025  
**Author:** Dr. Chen Wei (CTO Software) + Shakour Alishahi (Product Owner)

---

## 🎯 Overview

Gravity platform از معماری Microservices استفاده می‌کند. هر سرویس یک مسئولیت مشخص دارد (Single Responsibility Principle).

```
┌─────────────────────────────────────────────────────────────┐
│           GRAVITY MICROSERVICES ECOSYSTEM                    │
└─────────────────────────────────────────────────────────────┘

    ┌──────────────────────┐
    │  Data Ingestion      │  External APIs
    │  Service             ├──────────────┐
    └──────────┬───────────┘              │
               │                          │
               │ REST/gRPC                ├─ Alpha Vantage API
               │                          ├─ CODAL API  
               ↓                          ├─ Yahoo Finance API
    ┌──────────────────────┐              ├─ TSETMC Scraper
    │  Technical Analysis  │              └─ Binance API
    │  Service (THIS)      │
    └──────────┬───────────┘
               │
               │ REST API
               │
               ↓
    ┌──────────────────────┐
    │  Fundamental         │  Financial Data
    │  Analysis Service    ├──────────────┐
    └──────────┬───────────┘              │
               │                          ├─ Financial Ratios
               │                          ├─ Company Metrics
               ↓                          └─ Sector Analysis
    ┌──────────────────────┐
    │  Signal Aggregation  │  Combined Signals
    │  Service             │
    └──────────┬───────────┘
               │
               ↓
    ┌──────────────────────┐
    │  Frontend / Apps     │
    └──────────────────────┘
```

---

## 📊 Service Responsibilities Matrix

| Service | مسئولیت‌های اصلی | مسئولیت ندارد | Status |
|---------|------------------|----------------|---------|
| **Data Ingestion Service** | • دریافت raw data از API ها<br>• Web scraping<br>• Data validation<br>• Data cleaning<br>• Price/volume adjustments (splits, dividends)<br>• Data storage<br>• Data quality scoring | • Technical analysis<br>• Fundamental analysis<br>• Signal generation | 🔴 Not Started |
| **Technical Analysis Service** (این پروژه) | • محاسبه 60+ اندیکاتور تکنیکال<br>• Pattern detection (candlestick, classical)<br>• Elliott Wave analysis<br>• Support/Resistance zones<br>• Scenario analysis (optimistic/neutral/pessimistic)<br>• Multi-timeframe analysis<br>• Signal generation (technical) | • Data fetching from external APIs<br>• Data cleaning<br>• Fundamental analysis<br>• Financial ratios | 🟢 In Progress |
| **Fundamental Analysis Service** | • محاسبه نسبت‌های مالی (P/E, P/B, ROE, etc.)<br>• تحلیل صورت‌های مالی<br>• رشد درآمد و سود<br>• مقایسه با صنعت<br>• رتبه‌بندی بنیادی<br>• Financial health scoring | • Technical indicators<br>• Price patterns<br>• Chart analysis<br>• Data ingestion | 🔴 Not Started |
| **Signal Aggregation Service** | • ترکیب سیگنال‌های technical + fundamental<br>• وزن‌دهی هوشمند<br>• تصمیم‌گیری نهایی (Buy/Sell/Hold)<br>• Risk management<br>• Portfolio optimization | • داده‌های خام<br>• محاسبات پایه | 🔴 Not Started |

---

## 🔧 Technical Analysis Service - این پروژه فعلی

### ✅ چه کارهایی انجام می‌دهد:

#### 1. Indicator Calculation (60+ indicators)
```python
# Trend Indicators
- SMA, EMA, WMA, DEMA, TEMA
- MACD, Signal, Histogram
- ADX, +DI, -DI
- Parabolic SAR
- Supertrend
- Ichimoku Cloud (Tenkan, Kijun, Senkou A/B, Chikou)

# Momentum Indicators
- RSI, Stochastic RSI
- Stochastic Oscillator (K, D)
- CCI, Williams %R
- ROC, MFI
- TSI, UO

# Volatility Indicators
- Bollinger Bands (Upper, Middle, Lower, %B, Width)
- ATR, Keltner Channels
- Standard Deviation, Historical Volatility
- Donchian Channels

# Volume Indicators
- OBV, VWAP, Volume Profile
- Accumulation/Distribution Line
- Chaikin Money Flow
- Money Flow Index
- Volume Rate of Change

# Cycle Indicators
- Dominant Cycle Period
- Trend vs Cycle Decomposition
```

#### 2. Pattern Detection
```python
# Candlestick Patterns (40+)
- Doji, Hammer, Shooting Star
- Engulfing (Bullish/Bearish)
- Morning/Evening Star
- Three White Soldiers/Black Crows
- Harami, Piercing, Dark Cloud Cover

# Classical Chart Patterns
- Head & Shoulders
- Double/Triple Top/Bottom
- Triangles (Ascending, Descending, Symmetrical)
- Flags & Pennants
- Cup & Handle
- Wedges (Rising, Falling)

# Elliott Wave Analysis
- Wave counting (1-2-3-4-5, A-B-C)
- Fibonacci retracements
- Extension levels
- Wave validation
```

#### 3. Scenario Analysis ✅ (NEW)
```python
# Three-Scenario Analysis
- Optimistic (65-75% probability)
  • Target: +3×ATR
  • Stop: -0.5×ATR
  • Risk/Reward: 1:6

- Neutral (45-55% probability)
  • Target: +1.5×ATR
  • Stop: -1×ATR
  • Risk/Reward: 1:1.5

- Pessimistic (25-35% probability)
  • Target: +0.5×ATR
  • Stop: -1.5×ATR
  • Risk/Reward: 1:0.33

# Expected Value Calculation
E(Return) = P(opt)×R(opt) + P(neu)×R(neu) + P(pes)×R(pes)
Sharpe Ratio = E(Return) / σ(Risk)
```

#### 4. Support/Resistance Detection
```python
# Automatic S/R Zones
- Historical price pivots
- Volume profile POC (Point of Control)
- Fibonacci levels
- Psychological levels (round numbers)
- Dynamic S/R (moving averages)
```

#### 5. Multi-Timeframe Analysis
```python
# Timeframe Correlation
- 1min, 5min, 15min, 1h, 4h, 1d, 1w
- Trend alignment across timeframes
- Timeframe-specific weights
- Higher timeframe dominance
```

---

### ❌ چه کارهایی انجام نمی‌دهد:

#### 1. Data Fetching
```python
# این کارها مسئولیت Data Ingestion Service است:
❌ Alpha Vantage API calls
❌ CODAL API integration
❌ Yahoo Finance scraping
❌ TSETMC data extraction
❌ Binance WebSocket connections
❌ API key management
❌ Rate limiting external APIs
```

#### 2. Data Cleaning
```python
# این کارها مسئولیت Data Ingestion Service است:
❌ Missing data interpolation
❌ Outlier detection/removal
❌ Data validation
❌ Split/dividend adjustments
❌ Currency conversion
❌ Data normalization
```

#### 3. Fundamental Analysis
```python
# این کارها مسئولیت Fundamental Analysis Service است:
❌ P/E ratio calculation
❌ EPS analysis
❌ Revenue growth metrics
❌ Profit margin analysis
❌ ROE, ROA calculation
❌ Debt-to-Equity ratio
❌ Financial statement parsing
❌ Sector comparison
❌ Industry ranking
❌ Dividend analysis
```

---

## 📡 API Contracts

### Data Ingestion Service → Technical Analysis Service

**Endpoint:** `GET /api/v1/candles/{symbol}`

**Request:**
```json
{
  "symbol": "AAPL",
  "timeframe": "1d",
  "start_date": "2024-01-01T00:00:00Z",
  "end_date": "2024-11-14T23:59:59Z"
}
```

**Response:**
```json
{
  "symbol": "AAPL",
  "candles": [
    {
      "timestamp": "2024-01-01T00:00:00Z",
      "adjusted_open": 150.0,
      "adjusted_high": 152.5,
      "adjusted_low": 149.25,
      "adjusted_close": 151.75,
      "adjusted_volume": 45678900
    }
  ],
  "metadata": {
    "adjustments_applied": ["splits", "dividends"],
    "data_quality_score": 0.98
  }
}
```

---

### Technical Analysis Service → Frontend/Apps

**Endpoint:** `GET /api/v1/scenarios/{symbol}`

**Response:**
```json
{
  "symbol": "AAPL",
  "current_price": 180.5,
  "optimistic": {
    "score": 78.5,
    "probability": 70.0,
    "target_price": 195.0,
    "stop_loss": 178.0,
    "risk_reward_ratio": 3.0,
    "recommendation": "BUY"
  },
  "neutral": {...},
  "pessimistic": {...},
  "expected_return": 5.8,
  "sharpe_ratio": 1.81
}
```

---

### Fundamental Analysis Service → Signal Aggregation

**Endpoint:** `GET /api/v1/fundamental/{symbol}` (آینده)

**Response:**
```json
{
  "symbol": "AAPL",
  "financial_health_score": 85.0,
  "growth_score": 78.0,
  "valuation_score": 65.0,
  "pe_ratio": 28.5,
  "eps_growth": 12.3,
  "roe": 45.6,
  "sector_rank": 5,
  "recommendation": "BUY"
}
```

---

## 🔐 Authentication & Security

همه سرویس‌ها از JWT authentication استفاده می‌کنند:

```http
Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Rate Limiting:**
- Per IP: 1000 req/min
- Per API Key: 10,000 req/min
- Premium: 100,000 req/min

---

## 📊 Service Discovery

Kubernetes Service Mesh:

```yaml
# Technical Analysis Service
apiVersion: v1
kind: Service
metadata:
  name: technical-analysis-service
  namespace: gravity
spec:
  selector:
    app: technical-analysis
  ports:
    - port: 8000
  type: ClusterIP

# Internal DNS:
http://technical-analysis-service:8000
http://technical-analysis-service.gravity.svc.cluster.local:8000
```

---

## 📅 Development Roadmap

### ✅ Phase 1: Technical Analysis Service (In Progress)
- [x] 60+ indicators
- [x] Pattern detection
- [x] Scenario analysis
- [x] Data Service integration
- [ ] API endpoints complete
- [ ] Testing (95%+ coverage)
- [ ] Performance optimization (10000x)

### 🔴 Phase 2: Data Ingestion Service (Not Started)
- [ ] Alpha Vantage integration
- [ ] CODAL API integration
- [ ] TSETMC scraper
- [ ] Yahoo Finance integration
- [ ] Data cleaning pipeline
- [ ] Adjustment calculations
- [ ] Data storage (PostgreSQL)

### 🔴 Phase 3: Fundamental Analysis Service (Not Started)
- [ ] Financial ratio calculations
- [ ] Income statement parsing
- [ ] Balance sheet analysis
- [ ] Cash flow analysis
- [ ] Sector comparison
- [ ] Industry ranking
- [ ] Fundamental scoring (0-100)

### 🔴 Phase 4: Signal Aggregation Service (Not Started)
- [ ] Technical + Fundamental combination
- [ ] ML-based weight optimization
- [ ] Risk-adjusted scoring
- [ ] Portfolio optimization
- [ ] Real-time signal generation

---

## 🎯 Team Assignments

### Technical Analysis Service (این پروژه)
**Team Lead:** Shakour Alishahi  
**Members:**
- Dr. James Richardson (Quantitative Analysis)
- Prof. Alexandre Dubois (Technical Analysis)
- Dr. Rajesh Patel (ML & Algo Trading)
- Maria Gonzalez (Volume Analysis)
- Emily Watson (Performance)
- Dmitry Volkov (Backend)

### Data Ingestion Service (آینده)
**Team Lead:** TBD  
**Focus:** Data engineering, ETL, API integration

### Fundamental Analysis Service (آینده)
**Team Lead:** TBD  
**Focus:** Financial analysis, accounting, valuation

### Signal Aggregation Service (آینده)
**Team Lead:** TBD  
**Focus:** ML, portfolio optimization, risk management

---

## 📝 Decision Log

### November 14, 2025 - Fundamental Analysis Removal
**Decision:** حذف Fundamental Analysis از Technical Analysis Service

**Rationale:**
1. **Separation of Concerns:** هر microservice یک کار مشخص
2. **Team Specialization:** تیم‌های مختلف expertise های متفاوت
3. **Independent Scaling:** هر سرویس مستقل scale می‌شود
4. **Maintenance:** easier to maintain smaller services
5. **Testing:** easier to test single responsibility

**Approved By:**
- ✅ Shakour Alishahi (Product Owner)
- ✅ Dr. Chen Wei (CTO Software)
- ✅ Dr. James Richardson (Chief Quant)

---

**Document Owner:** Dr. Chen Wei  
**Approved By:** Shakour Alishahi  
**Version:** 1.0  
**Last Updated:** November 14, 2025
