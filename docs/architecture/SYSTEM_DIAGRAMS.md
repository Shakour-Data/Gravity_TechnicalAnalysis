# System Diagrams

This document contains the complete and accurate UML, ER, and DFD diagrams for the Gravity Technical Analysis system.

## Table of Contents
1. [Entity-Relationship Diagram (ERD)](#entity-relationship-diagram-erd)
2. [Data Flow Diagram (DFD)](#data-flow-diagram-dfd)
3. [Unified Modeling Language (UML) Diagrams](#unified-modeling-language-uml-diagrams)

---

## Entity-Relationship Diagram (ERD)

```plantuml
@startuml ERD
entity "Market Data" as MD {
  * symbol : VARCHAR(10)
  * timestamp : DATETIME
  * open : DECIMAL(10,4)
  * high : DECIMAL(10,4)
  * low : DECIMAL(10,4)
  * close : DECIMAL(10,4)
  * volume : BIGINT
  --
  * primary key (symbol, timestamp)
}

entity "Analysis Results" as AR {
  * id : BIGINT
  * symbol : VARCHAR(10)
  * timestamp : DATETIME
  * dimension : ENUM
  * indicator : VARCHAR(50)
  * value : DECIMAL(10,6)
  * signal : ENUM
  --
  * primary key (id)
  * index (symbol, timestamp, dimension)
}

entity "ML Weights" as MW {
  * id : BIGINT
  * dimension : ENUM
  * indicator : VARCHAR(50)
  * weight : DECIMAL(5,4)
  * accuracy_score : DECIMAL(5,4)
  * last_updated : DATETIME
  --
  * primary key (id)
  * unique (dimension, indicator)
}

entity "Historical Scores" as HS {
  * id : BIGINT
  * symbol : VARCHAR(10)
  * date : DATE
  * trend_score : DECIMAL(5,4)
  * momentum_score : DECIMAL(5,4)
  * volatility_score : DECIMAL(5,4)
  * cycle_score : DECIMAL(5,4)
  * support_resistance_score : DECIMAL(5,4)
  * final_signal : ENUM
  --
  * primary key (id)
  * index (symbol, date)
}

MD ||--o{ AR : generates
AR ||--o{ MW : uses
MW ||--o{ HS : produces
HS ||--o{ AR : references

note right of MD : Raw OHLCV data from exchanges
note right of AR : Calculated indicator values and signals
note right of MW : ML-optimized weights for each indicator
note right of HS : Historical analysis results for backtesting
@enduml
```

---

## Data Flow Diagram (DFD)

### Level 0 - Context Diagram

```plantuml
@startuml DFD_Level0
actor "User" as U
actor "Data Provider" as DP
rectangle "Gravity Technical Analysis System" as GTAS {
}

U --> GTAS : Analysis Requests
GTAS --> U : Analysis Results
DP --> GTAS : Market Data
GTAS --> DP : Data Requests (optional)
@enduml
```

### Level 1 - Main Processes

```plantuml
@startuml DFD_Level1
actor "User" as U
actor "Data Provider" as DP
database "Database" as DB

rectangle "Data Ingestion" as DI {
  process "Validate Data" as VD
  process "Store Raw Data" as SRD
}

rectangle "Analysis Engine" as AE {
  process "Calculate Indicators" as CI
  process "Generate Signals" as GS
  process "Apply ML Weights" as AMW
}

rectangle "API Layer" as AL {
  process "Handle Requests" as HR
  process "Format Responses" as FR
}

rectangle "ML Training" as MLT {
  process "Train Models" as TM
  process "Update Weights" as UW
}

U --> HR : API Requests
HR --> FR : Process Requests
FR --> U : Formatted Results

DP --> VD : Raw Market Data
VD --> SRD : Validated Data
SRD --> DB : Store Data

DB --> CI : Historical Data
CI --> GS : Indicator Values
GS --> AMW : Raw Signals
AMW --> DB : Weighted Signals

DB --> TM : Training Data
TM --> UW : New Weights
UW --> DB : Updated Weights

DB --> FR : Analysis Results
@enduml
```

### Level 2 - Analysis Engine Detail

```plantuml
@startuml DFD_Level2_Analysis
database "Market Data DB" as DB
database "Weights DB" as WDB

process "Trend Analysis" as TA {
  process "SMA/EMA Calculation" as SMA
  process "MACD Calculation" as MACD
  process "ADX Calculation" as ADX
}

process "Momentum Analysis" as MA {
  process "RSI Calculation" as RSI
  process "Stochastic Calc" as STO
  process "MFI Calculation" as MFI
}

process "Volatility Analysis" as VA {
  process "Bollinger Bands" as BB
  process "ATR Calculation" as ATR
  process "Keltner Channels" as KC
}

process "Cycle Analysis" as CA {
  process "Phase Detection" as PD
  process "Cycle Length Calc" as CLC
}

process "Support/Resistance" as SR {
  process "Pivot Points" as PP
  process "Fibonacci Levels" as FL
}

process "Signal Aggregation" as SA
process "ML Weight Application" as MWA
process "Final Signal Generation" as FSG

DB --> TA : OHLC Data
DB --> MA : OHLC Data
DB --> VA : OHLC Data
DB --> CA : OHLC Data
DB --> SR : OHLC Data

TA --> SA : Trend Signals
MA --> SA : Momentum Signals
VA --> SA : Volatility Signals
CA --> SA : Cycle Signals
SR --> SA : S/R Signals

SA --> MWA : Aggregated Signals
WDB --> MWA : ML Weights
MWA --> FSG : Weighted Signals
FSG --> DB : Final Signals
@enduml
```

---

## Unified Modeling Language (UML) Diagrams

### Class Diagram

```plantuml
@startuml Class_Diagram
class MarketData {
  - symbol: String
  - timestamp: DateTime
  - open: BigDecimal
  - high: BigDecimal
  - low: BigDecimal
  - close: BigDecimal
  - volume: Long
  + validate(): boolean
  + toJson(): String
}

class Indicator {
  - name: String
  - dimension: Dimension
  - parameters: Map<String, Object>
  + calculate(data: MarketData[]): BigDecimal
  + getSignal(): Signal
}

enum Dimension {
  TREND
  MOMENTUM
  VOLATILITY
  CYCLE
  SUPPORT_RESISTANCE
}

enum Signal {
  STRONG_BUY
  BUY
  NEUTRAL
  SELL
  STRONG_SELL
}

class AnalysisEngine {
  - indicators: List<Indicator>
  - mlWeights: MLWeights
  + analyze(data: MarketData[]): AnalysisResult
  + getWeightedSignal(): Signal
}

class MLWeights {
  - weights: Map<String, BigDecimal>
  - accuracy: BigDecimal
  + getWeight(indicator: String): BigDecimal
  + updateWeights(newWeights: Map<String, BigDecimal>)
}

class AnalysisResult {
  - symbol: String
  - timestamp: DateTime
  - signals: Map<Dimension, Signal>
  - finalSignal: Signal
  - confidence: BigDecimal
  + toJson(): String
}

class APIService {
  - analysisEngine: AnalysisEngine
  - dataService: DataService
  + analyzeSymbol(symbol: String): AnalysisResult
  + getHistoricalData(symbol: String, days: int): MarketData[]
}

class DataService {
  - database: Database
  + getMarketData(symbol: String, start: Date, end: Date): MarketData[]
  + saveAnalysisResult(result: AnalysisResult)
}

interface Database {
  + save(entity: Object)
  + find(query: Query): List<Object>
  + update(entity: Object)
}

MarketData --> Indicator : uses
Indicator --> Signal : produces
AnalysisEngine --> Indicator : contains
AnalysisEngine --> MLWeights : uses
AnalysisEngine --> AnalysisResult : produces
APIService --> AnalysisEngine : uses
APIService --> DataService : uses
DataService --> Database : implements
@enduml
```

### Sequence Diagram - Analysis Request

```plantuml
@startuml Sequence_Analysis
actor User
participant APIService
participant AnalysisEngine
participant Indicator
participant MLWeights
participant DataService
participant Database

User -> APIService: analyzeSymbol("BTCUSDT")
APIService -> DataService: getMarketData("BTCUSDT", 30)
DataService -> Database: query historical data
Database --> DataService: return MarketData[]
DataService --> APIService: return data

APIService -> AnalysisEngine: analyze(data)
AnalysisEngine -> Indicator: calculate(data) [loop for each indicator]
Indicator --> AnalysisEngine: return signals
AnalysisEngine -> MLWeights: getWeights()
MLWeights --> AnalysisEngine: return weights
AnalysisEngine -> AnalysisEngine: applyWeights(signals)
AnalysisEngine -> DataService: saveResult(result)
DataService -> Database: save(result)
Database --> DataService: confirm save
DataService --> AnalysisEngine: save confirmed
AnalysisEngine --> APIService: return AnalysisResult
APIService --> User: return formatted result
@enduml
```

### Use Case Diagram

```plantuml
@startuml Use_Case
left to right direction

actor "Data Analyst" as DA
actor "Trader" as T
actor "Developer" as D
actor "System Admin" as SA

rectangle "Gravity Technical Analysis System" {
  usecase "Perform Technical Analysis" as PTA
  usecase "Generate Trading Signals" as GTS
  usecase "Backtest Strategies" as BS
  usecase "Train ML Models" as TMM
  usecase "Configure Indicators" as CI
  usecase "Monitor System Health" as MSH
  usecase "Manage Data Sources" as MDS
  usecase "View Analysis Reports" as VAR
}

DA --> PTA
DA --> BS
DA --> VAR

T --> PTA
T --> GTS
T --> BS

D --> TMM
D --> CI
D --> MSH

SA --> MDS
SA --> MSH
SA --> TMM

PTA --> GTS : extends
BS --> PTA : includes
TMM --> PTA : includes
CI --> PTA : includes
@enduml
```

### Component Diagram

```plantuml
@startuml Component_Diagram
component [API Gateway] as AG
component [Analysis Service] as AS
component [Data Service] as DS
component [ML Service] as MLS
component [Database] as DB
component [Cache] as C
component [Message Queue] as MQ

AG --> AS : REST API calls
AG --> DS : Data requests
AS --> MLS : ML predictions
AS --> DS : Historical data
AS --> C : Cached results
DS --> DB : CRUD operations
MLS --> DB : Training data
MLS --> MQ : Async training jobs
MQ --> MLS : Job results

note right of AG : Handles external requests
note right of AS : Core analysis logic
note right of DS : Data persistence layer
note right of MLS : Machine learning components
note right of DB : PostgreSQL/MySQL
note right of C : Redis/Memcached
note right of MQ : RabbitMQ/Kafka
@enduml
```

### Deployment Diagram

```plantuml
@startuml Deployment_Diagram
node "Load Balancer" as LB {
  component [Nginx] as NG
}

node "Application Server 1" as AS1 {
  component [API Service] as API1
  component [Analysis Engine] as AE1
}

node "Application Server 2" as AS2 {
  component [API Service] as API2
  component [Analysis Engine] as AE2
}

node "Database Server" as DBS {
  database [PostgreSQL] as PG
  database [Redis Cache] as RC
}

node "ML Server" as MLS {
  component [ML Training Service] as MLTS
  component [Model Storage] as MS
}

node "Data Ingestion Server" as DIS {
  component [Data Collector] as DC
  component [Data Processor] as DP
}

LB --> AS1 : routes requests
LB --> AS2 : routes requests
AS1 --> DBS : database queries
AS2 --> DBS : database queries
AS1 --> MLS : ML requests
AS2 --> MLS : ML requests
DIS --> DBS : insert market data
DC --> DP : raw data
DP --> DBS : processed data

note right of LB : Distributes load across servers
note right of DBS : High availability database cluster
note right of MLS : GPU-enabled for ML training
note right of DIS : Collects data from exchanges
@enduml
```

---

## Diagram Rendering

To render these diagrams:
1. Install PlantUML plugin for your IDE/editor
2. Or use online PlantUML servers
3. Or use the PlantUML command-line tool

Example command:
```bash
plantuml SYSTEM_DIAGRAMS.md
```

This will generate PNG/SVG files for each diagram.

---

**Last Updated**: April 2024
**Version**: 1.0.0
