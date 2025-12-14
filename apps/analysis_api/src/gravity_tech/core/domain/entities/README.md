# Domain Entities Package

**Package:** `src.core.domain.entities`  
**Layer:** Domain Layer (Clean Architecture)  
**Purpose:** Core business entities for Technical Analysis  
**Created:** Phase 2 (2025-11-07)  
**Updated:** Phase 2.1 - Task 1.3 (2025-11-07)

---

## Overview

This package contains all core domain entities for the Gravity Technical Analysis system. All entities follow Clean Architecture principles:

- **Immutable:** All entities are frozen dataclasses
- **Framework-Independent:** No external dependencies (FastAPI, Pydantic, etc.)
- **Self-Validating:** Validation in `__post_init__`
- **Type-Safe:** Full type hints on all fields
- **Well-Documented:** 17-field identity cards

---

## Entity Files

### Phase 2 Entities (Original)

#### 1. candle.py ✅
- **Type:** Dataclass (frozen)
- **Purpose:** OHLCV price data
- **Lines:** 150
- **Properties:** `typical_price`, `is_bullish`, `is_bearish`, `body_size`, `upper_shadow`, `lower_shadow`
- **Methods:** `true_range(previous_candle)`
- **Usage:** Foundation for all technical analysis

#### 2. signal.py ✅
- **Type:** Dataclass + Enum
- **Purpose:** Trading signal entity
- **Enums:** SignalType, SignalStrength (old version)
- **Usage:** Signal generation

#### 3. decision.py ✅
- **Type:** Dataclass + Enum
- **Purpose:** Trading decision entity
- **Enums:** DecisionType, ConfidenceLevel
- **Usage:** Final trading decisions

---

### Phase 2.1 Entities (NEW - Task 1.3)

#### 4. signal_strength.py 🆕
- **Type:** Enum (str, Enum)
- **Purpose:** Signal strength with Persian labels
- **Levels:** 7 (VERY_BULLISH to VERY_BEARISH)
- **Methods:**
  - `from_value(float)` - Convert -1 to 1 → SignalStrength
  - `get_score()` - Convert SignalStrength → -2.0 to 2.0
- **Usage:** All indicators and patterns

**Example:**
```python
from src.core.domain.entities import CoreSignalStrength

# Convert numeric value to enum
signal = CoreSignalStrength.from_value(0.85)  # VERY_BULLISH

# Get numeric score
score = signal.get_score()  # 2.0

# Display Persian label
print(signal.value)  # "بسیار صعودی"
```

---

#### 5. indicator_category.py 🆕
- **Type:** Enum (str, Enum)
- **Purpose:** 6 indicator categories with Persian labels
- **Categories:**
  - TREND = "روند"
  - MOMENTUM = "مومنتوم"
  - CYCLE = "سیکل"
  - VOLUME = "حجم"
  - VOLATILITY = "نوسان"
  - SUPPORT_RESISTANCE = "حمایت و مقاومت"
- **Usage:** Categorize all 60+ indicators

**Example:**
```python
from src.core.domain.entities import IndicatorCategory

category = IndicatorCategory.MOMENTUM
print(category.value)  # "مومنتوم"
```

---

#### 6. indicator_result.py 🆕
- **Type:** Dataclass (frozen)
- **Purpose:** Result from single indicator calculation
- **Fields:**
  - `indicator_name: str` - e.g., "RSI", "MACD"
  - `category: IndicatorCategory` - TREND, MOMENTUM, etc.
  - `signal: SignalStrength` - Signal strength
  - `value: float` - Primary value (e.g., RSI 65.0)
  - `additional_values: Dict[str, float]` - For multi-line indicators
  - `confidence: float` - 0.0 to 1.0
  - `description: str` - Human-readable
  - `timestamp: datetime` - Calculation time
- **Validation:** confidence 0.0-1.0, non-empty indicator_name
- **Usage:** All 60+ indicators

**Example:**
```python
from src.core.domain.entities import (
    IndicatorResult,
    IndicatorCategory,
    CoreSignalStrength
)

result = IndicatorResult(
    indicator_name="RSI",
    category=IndicatorCategory.MOMENTUM,
    signal=CoreSignalStrength.BULLISH,
    value=65.0,
    confidence=0.85,
    description="RSI above 50, bullish momentum"
)
```

---

#### 7. pattern_type.py 🆕
- **Type:** Enum (str, Enum)
- **Purpose:** Pattern type classification
- **Types:**
  - CLASSICAL = "کلاسیک" (e.g., Head & Shoulders)
  - CANDLESTICK = "کندل استیک" (e.g., Doji, Hammer)
- **Usage:** Pattern recognition

**Example:**
```python
from src.core.domain.entities import PatternType

pattern_type = PatternType.CANDLESTICK
print(pattern_type.value)  # "کندل استیک"
```

---

#### 8. pattern_result.py 🆕
- **Type:** Dataclass (frozen)
- **Purpose:** Chart pattern recognition result
- **Fields:**
  - `pattern_name: str` - e.g., "Head and Shoulders", "Doji"
  - `pattern_type: PatternType` - CLASSICAL or CANDLESTICK
  - `signal: SignalStrength` - Pattern signal
  - `confidence: float` - 0.0 to 1.0
  - `start_time: datetime` - Pattern start
  - `end_time: datetime` - Pattern completion
  - `description: str` - Human-readable
  - `price_target: float` - Optional target
  - `stop_loss: float` - Optional SL
- **Validation:** confidence 0.0-1.0, end_time >= start_time, positive price_target/stop_loss
- **Usage:** Classical and candlestick pattern detection

**Example:**
```python
from src.core.domain.entities import (
    PatternResult,
    PatternType,
    CoreSignalStrength
)
from datetime import datetime

pattern = PatternResult(
    pattern_name="Doji",
    pattern_type=PatternType.CANDLESTICK,
    signal=CoreSignalStrength.NEUTRAL,
    confidence=0.90,
    start_time=datetime(2025, 11, 7, 10, 0),
    end_time=datetime(2025, 11, 7, 11, 0),
    description="Doji formed at resistance level",
    price_target=50000.0,
    stop_loss=48000.0
)
```

---

#### 9. wave_point.py 🆕
- **Type:** Dataclass (frozen)
- **Purpose:** Single Elliott Wave point (peak or trough)
- **Fields:**
  - `wave_number: int` - 1-5 for impulse, A-C for correction
  - `price: float` - Price at wave point
  - `timestamp: datetime` - When wave occurred
  - `wave_type: str` - "PEAK" or "TROUGH"
- **Validation:** wave_type must be PEAK or TROUGH, positive price
- **Usage:** Elliott Wave analysis

**Example:**
```python
from src.core.domain.entities import WavePoint
from datetime import datetime

wave = WavePoint(
    wave_number=1,
    price=49500.0,
    timestamp=datetime(2025, 11, 7, 10, 0),
    wave_type="PEAK"
)
```

---

#### 10. elliott_wave_result.py 🆕
- **Type:** Dataclass (frozen)
- **Purpose:** Complete Elliott Wave analysis
- **Fields:**
  - `wave_pattern: str` - "IMPULSIVE" or "CORRECTIVE"
  - `current_wave: int` - Current wave being formed
  - `waves: List[WavePoint]` - All identified wave points
  - `signal: SignalStrength` - Overall signal
  - `confidence: float` - 0.0 to 1.0
  - `description: str` - Wave analysis
  - `projected_target: float` - Optional target
- **Validation:**
  - confidence 0.0-1.0
  - wave_pattern must be IMPULSIVE or CORRECTIVE
  - IMPULSIVE max 5 waves
  - CORRECTIVE max 3 waves
  - positive projected_target
- **Usage:** Elliott Wave pattern recognition

**Example:**
```python
from src.core.domain.entities import (
    ElliottWaveResult,
    WavePoint,
    CoreSignalStrength
)
from datetime import datetime

waves = [
    WavePoint(1, 49000.0, datetime(2025, 11, 7, 10, 0), "PEAK"),
    WavePoint(2, 48500.0, datetime(2025, 11, 7, 11, 0), "TROUGH"),
    # ... more waves
]

result = ElliottWaveResult(
    wave_pattern="IMPULSIVE",
    current_wave=3,
    waves=waves,
    signal=CoreSignalStrength.BULLISH,
    confidence=0.75,
    description="Impulse wave forming, wave 3 in progress",
    projected_target=52000.0
)
```

---

## Import Guide

### Recommended Imports (Phase 2.1)

```python
# All new entities
from src.core.domain.entities import (
    # Phase 2 (existing)
    Candle,
    CandleType,
    Signal,
    Decision,
    
    # Phase 2.1 (NEW)
    CoreSignalStrength,      # NEW SignalStrength with Persian
    IndicatorCategory,
    IndicatorResult,
    PatternType,
    PatternResult,
    WavePoint,
    ElliottWaveResult,
)
```

### Backward Compatibility

```python
# Old import (Phase 2)
from src.core.domain.entities import SignalStrength  # From signal.py

# New import (Phase 2.1)
from src.core.domain.entities import CoreSignalStrength  # From signal_strength.py
```

**Note:** `SignalStrength` from `signal.py` is deprecated. Use `CoreSignalStrength` for new code.

---

## Migration from models.schemas

### Old Import (DEPRECATED)

```python
# ❌ DEPRECATED - Do NOT use
from models.schemas import (
    Candle,
    SignalStrength,
    IndicatorResult,
    IndicatorCategory,
    PatternResult,
    PatternType,
    ElliottWaveResult,
    WavePoint,
)
```

### New Import (RECOMMENDED)

```python
# ✅ RECOMMENDED - Use this
from src.core.domain.entities import (
    Candle,
    CoreSignalStrength,      # Note: CoreSignalStrength, not SignalStrength
    IndicatorResult,
    IndicatorCategory,
    PatternResult,
    PatternType,
    ElliottWaveResult,
    WavePoint,
)
```

---

## Design Principles

### 1. Immutability
All entities use `@dataclass(frozen=True)` to ensure immutability:

```python
@dataclass(frozen=True)
class IndicatorResult:
    indicator_name: str
    # ... other fields
```

**Why?** Immutable entities prevent bugs from unexpected modifications.

### 2. Self-Validation
All entities validate in `__post_init__`:

```python
def __post_init__(self):
    if not 0.0 <= self.confidence <= 1.0:
        raise ValueError(f"confidence must be 0.0-1.0, got {self.confidence}")
```

**Why?** Ensures invalid entities cannot exist.

### 3. Framework Independence
No external dependencies (FastAPI, Pydantic, SQLAlchemy):

```python
# ✅ Good - stdlib only
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

# ❌ Bad - external dependency
from pydantic import BaseModel
```

**Why?** Clean Architecture - domain layer should be pure business logic.

### 4. Type Safety
Full type hints on all fields:

```python
@dataclass(frozen=True)
class IndicatorResult:
    indicator_name: str
    category: IndicatorCategory
    signal: CoreSignalStrength
    value: float
    additional_values: Optional[Dict[str, float]] = None
```

**Why?** IDE autocomplete, static type checking, better documentation.

---

## Testing

### Unit Test Example

```python
import pytest
from src.core.domain.entities import (
    IndicatorResult,
    IndicatorCategory,
    CoreSignalStrength
)

def test_indicator_result_creation():
    result = IndicatorResult(
        indicator_name="RSI",
        category=IndicatorCategory.MOMENTUM,
        signal=CoreSignalStrength.BULLISH,
        value=65.0,
        confidence=0.85
    )
    assert result.indicator_name == "RSI"
    assert result.value == 65.0

def test_indicator_result_validation():
    with pytest.raises(ValueError, match="confidence must be 0.0-1.0"):
        IndicatorResult(
            indicator_name="RSI",
            category=IndicatorCategory.MOMENTUM,
            signal=CoreSignalStrength.BULLISH,
            value=65.0,
            confidence=1.5  # Invalid!
        )
```

---

## Performance Considerations

### Dataclass vs Pydantic

**Dataclass Benefits:**
- 5x faster instantiation
- No external dependencies
- Clean Architecture compliance
- Native Python (stdlib)

**Pydantic Benefits:**
- Automatic JSON serialization
- OpenAPI schema generation
- Rich validation DSL

**Decision:** Use dataclasses in domain layer (this package), convert to Pydantic in API layer (presentation layer).

---

## File Structure

```
src/core/domain/entities/
├── __init__.py                    # Central exports
├── README.md                      # This file
│
├── candle.py                      # ✅ Phase 2
├── signal.py                      # ✅ Phase 2
├── decision.py                    # ✅ Phase 2
│
├── signal_strength.py             # 🆕 Phase 2.1
├── indicator_category.py          # 🆕 Phase 2.1
├── indicator_result.py            # 🆕 Phase 2.1
├── pattern_type.py                # 🆕 Phase 2.1
├── pattern_result.py              # 🆕 Phase 2.1
├── wave_point.py                  # 🆕 Phase 2.1
└── elliott_wave_result.py         # 🆕 Phase 2.1
```

---

## Next Steps (Phase 2.1 - Day 2)

1. **Update all imports** in `src/core/indicators/` (6 files)
2. **Update all imports** in `src/core/patterns/` (4 files)
3. **Update all imports** in `src/core/analysis/` (1 file)
4. **Create backward compatibility** layer in `models/schemas.py`
5. **Update tests** to use new imports
6. **Delete old models** from `models/schemas.py` (Day 3)

---

## Contributors

- **Dr. Chen Wei** (SW-001) - Chief Technology Officer - Architecture & Implementation
- **Prof. Alexandre Dubois** (TAA-005) - Technical Analysis Authority - Validation

---

**Last Updated:** 2025-11-07 (Phase 2.1 - Task 1.3)  
**Status:** COMPLETE ✅  
**Next:** Day 2 - Update core imports
