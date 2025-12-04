"""
================================================================================
Market Data Entity

Clean Architecture - Domain Layer
Defines market data structure for real-time price and volume information.

Last Updated: 2025-11-07 (Phase 2.1 - Task 1.4)
================================================================================
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional


@dataclass(frozen=True)
class MarketData:
    """
    Market data entity for real-time price and volume information.

    Immutable dataclass representing current market conditions.
    Used by real-time handlers to broadcast live market data.
    """

    # Symbol and timestamp
    symbol: str
    """Trading symbol (e.g., 'BTCUSDT', 'AAPL')"""

    timestamp: datetime
    """Data timestamp (UTC)"""

    # Price data
    price: Decimal
    """Current price"""

    open_price: Optional[Decimal] = None
    """Opening price for current period"""

    high_price: Optional[Decimal] = None
    """Highest price for current period"""

    low_price: Optional[Decimal] = None
    """Lowest price for current period"""

    # Volume data
    volume: Optional[Decimal] = None
    """Trading volume"""

    quote_volume: Optional[Decimal] = None
    """Quote asset volume"""

    # Market statistics
    price_change: Optional[Decimal] = None
    """Price change from previous period"""

    price_change_percent: Optional[Decimal] = None
    """Price change percentage"""

    weighted_avg_price: Optional[Decimal] = None
    """Volume weighted average price"""

    # Order book data (optional)
    bid_price: Optional[Decimal] = None
    """Best bid price"""

    bid_quantity: Optional[Decimal] = None
    """Best bid quantity"""

    ask_price: Optional[Decimal] = None
    """Best ask price"""

    ask_quantity: Optional[Decimal] = None
    """Best ask quantity"""

    # Additional metadata
    source: Optional[str] = None
    """Data source identifier"""

    sequence_number: Optional[int] = None
    """Sequence number for ordering"""

    @classmethod
    def from_candle_data(
        cls,
        symbol: str,
        timestamp: datetime,
        open_price: Decimal,
        high_price: Decimal,
        low_price: Decimal,
        close_price: Decimal,
        volume: Decimal,
        source: Optional[str] = None
    ) -> 'MarketData':
        """
        Create MarketData from candle/OHLCV data.

        Args:
            symbol: Trading symbol
            timestamp: Candle timestamp
            open_price: Opening price
            high_price: Highest price
            low_price: Lowest price
            close_price: Closing price (current price)
            volume: Trading volume
            source: Optional data source

        Returns:
            MarketData instance
        """
        return cls(
            symbol=symbol,
            timestamp=timestamp,
            price=close_price,
            open_price=open_price,
            high_price=high_price,
            low_price=low_price,
            volume=volume,
            source=source
        )

    @classmethod
    def from_tick_data(
        cls,
        symbol: str,
        timestamp: datetime,
        price: Decimal,
        volume: Optional[Decimal] = None,
        source: Optional[str] = None
    ) -> 'MarketData':
        """
        Create MarketData from tick price data.

        Args:
            symbol: Trading symbol
            timestamp: Tick timestamp
            price: Current price
            volume: Optional volume
            source: Optional data source

        Returns:
            MarketData instance
        """
        return cls(
            symbol=symbol,
            timestamp=timestamp,
            price=price,
            volume=volume,
            source=source
        )

    def to_dict(self) -> dict:
        """
        Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation
        """
        result = {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "price": str(self.price)
        }

        # Add optional fields if present
        optional_fields = [
            "open_price", "high_price", "low_price", "volume", "quote_volume",
            "price_change", "price_change_percent", "weighted_avg_price",
            "bid_price", "bid_quantity", "ask_price", "ask_quantity",
            "source", "sequence_number"
        ]

        for field in optional_fields:
            value = getattr(self, field)
            if value is not None:
                result[field] = str(value) if isinstance(value, Decimal) else value

        return result

    @classmethod
    def from_dict(cls, data: dict) -> 'MarketData':
        """
        Create from dictionary (JSON deserialization).

        Args:
            data: Dictionary representation

        Returns:
            MarketData instance

        Raises:
            ValueError: If required fields are missing
        """
        try:
            # Required fields
            symbol = data["symbol"]
            timestamp = datetime.fromisoformat(data["timestamp"])
            price = Decimal(data["price"])

            # Optional fields
            kwargs = {}
            decimal_fields = [
                "open_price", "high_price", "low_price", "volume", "quote_volume",
                "price_change", "price_change_percent", "weighted_avg_price",
                "bid_price", "bid_quantity", "ask_price", "ask_quantity"
            ]

            for field in decimal_fields:
                if field in data:
                    kwargs[field] = Decimal(data[field])

            # Non-decimal optional fields
            if "source" in data:
                kwargs["source"] = data["source"]
            if "sequence_number" in data:
                kwargs["sequence_number"] = data["sequence_number"]

            return cls(
                symbol=symbol,
                timestamp=timestamp,
                price=price,
                **kwargs
            )

        except KeyError as e:
            raise ValueError(f"Missing required field: {e}")
        except Exception as e:
            raise ValueError(f"Invalid market data format: {e}")

    @property
    def is_complete_candle(self) -> bool:
        """Check if this represents complete OHLC data."""
        return all([
            self.open_price is not None,
            self.high_price is not None,
            self.low_price is not None
        ])

    @property
    def price_range(self) -> Optional[Decimal]:
        """Calculate price range (high - low)."""
        if self.high_price is not None and self.low_price is not None:
            return self.high_price - self.low_price
        return None

    @property
    def midpoint_price(self) -> Optional[Decimal]:
        """Calculate midpoint price ((high + low) / 2)."""
        if self.high_price is not None and self.low_price is not None:
            return (self.high_price + self.low_price) / 2
        return None

    @property
    def spread(self) -> Optional[Decimal]:
        """Calculate bid-ask spread."""
        if self.ask_price is not None and self.bid_price is not None:
            return self.ask_price - self.bid_price
        return None

    def __post_init__(self):
        """Validate data after initialization."""
        if self.price <= 0:
            raise ValueError("Price must be positive")

        if self.volume is not None and self.volume < 0:
            raise ValueError("Volume cannot be negative")

        if self.price_change_percent is not None and abs(self.price_change_percent) > 1000:
            raise ValueError("Price change percent seems unreasonable")

        # Validate OHLC relationship
        if self.is_complete_candle:
            # All values are guaranteed to be non-None here due to is_complete_candle check
            if not (self.low_price is not None and  # type: ignore
                    self.open_price is not None and  # type: ignore
                    self.high_price is not None and  # type: ignore
                    self.low_price <= self.open_price <= self.high_price and
                    self.low_price <= self.price <= self.high_price):
                raise ValueError("OHLC prices are inconsistent")