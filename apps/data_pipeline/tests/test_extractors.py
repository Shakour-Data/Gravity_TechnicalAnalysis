"""
Tests for data extractors
"""

from unittest.mock import AsyncMock

import pytest
from gravity_pipeline.extractors import TSEExtractor, TSEExtractorConfig


@pytest.fixture
def tse_config():
    """Create TSE extractor config"""
    return TSEExtractorConfig(
        api_base_url="https://api.test.ir",
        use_local_db=False,
    )


@pytest.fixture
def tse_extractor(tse_config):
    """Create TSE extractor instance"""
    return TSEExtractor(config=tse_config)


@pytest.mark.asyncio
async def test_validate_connection(tse_extractor):
    """Test connection validation"""
    # Mock the session
    mock_session = AsyncMock()
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_session.get.return_value.__aenter__.return_value = mock_response

    tse_extractor.session = mock_session

    result = await tse_extractor.validate_connection()
    assert result is True


@pytest.mark.asyncio
async def test_get_available_symbols(tse_extractor):
    """Test fetching available symbols"""
    # Mock response
    mock_session = AsyncMock()
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(
        return_value={
            "tickers": [
                {"symbol": "BTCUSDT"},
                {"symbol": "ETHUSDT"},
                {"symbol": "BNBUSDT"},
            ]
        }
    )
    mock_session.get.return_value.__aenter__.return_value = mock_response

    tse_extractor.session = mock_session

    symbols = await tse_extractor.get_available_symbols()

    assert len(symbols) == 3
    assert "BTCUSDT" in symbols
    assert "ETHUSDT" in symbols


@pytest.mark.asyncio
async def test_extract_single_symbol(tse_extractor):
    """Test extracting data for single symbol"""
    # Mock response
    mock_session = AsyncMock()
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(
        return_value={
            "candles": [
                {
                    "timestamp": "2024-01-01",
                    "o": "100",
                    "h": "110",
                    "l": "90",
                    "c": "105",
                    "v": "1000",
                },
                {
                    "timestamp": "2024-01-02",
                    "o": "105",
                    "h": "115",
                    "l": "95",
                    "c": "110",
                    "v": "1200",
                },
            ]
        }
    )
    mock_session.get.return_value.__aenter__.return_value = mock_response

    tse_extractor.session = mock_session

    candles = await tse_extractor._extract_symbol("BTCUSDT", None, None)

    assert len(candles) == 2
    assert candles[0]["symbol"] == "BTCUSDT"
    assert candles[0]["open"] == 100.0
    assert candles[0]["close"] == 105.0


@pytest.mark.asyncio
async def test_extract_multiple_symbols(tse_extractor):
    """Test extracting data for multiple symbols"""
    # Mock
    mock_session = AsyncMock()
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"candles": []})
    mock_session.get.return_value.__aenter__.return_value = mock_response

    tse_extractor.session = mock_session
    tse_extractor.available_symbols = ["BTCUSDT", "ETHUSDT"]

    candles = await tse_extractor.extract(symbols=["BTCUSDT", "ETHUSDT"])

    # Should call get for each symbol
    assert mock_session.get.call_count >= 2


@pytest.mark.asyncio
async def test_close_session(tse_extractor):
    """Test closing HTTP session"""
    mock_session = AsyncMock()
    tse_extractor.session = mock_session

    await tse_extractor.close()

    mock_session.close.assert_called_once()


@pytest.mark.asyncio
async def test_stats(tse_extractor):
    """Test statistics tracking"""
    tse_extractor.extracted_count = 100
    tse_extractor.error_count = 5

    stats = tse_extractor.get_stats()

    assert stats["extracted"] == 100
    assert stats["errors"] == 5
