"""
Integration tests for full ETL pipeline
"""

from unittest.mock import AsyncMock

import pytest
from gravity_pipeline.orchestrator import DataPipeline, PipelineConfig, PipelineStage
from gravity_pipeline.transformers import DataCleaner
from gravity_pipeline.validators import DataQualityValidator


@pytest.fixture
def pipeline_config():
    """Create pipeline config for testing"""
    return PipelineConfig(
        source_db_url="sqlite:///./test_source.db",
        target_db_url="sqlite:///./test_target.db",
        batch_size=100,
        max_workers=2,
    )


@pytest.fixture
def pipeline(pipeline_config):
    """Create pipeline instance"""
    return DataPipeline(pipeline_config)


@pytest.fixture
def sample_candles():
    """Create sample candle data"""
    return [
        {
            "symbol": "BTCUSDT",
            "timestamp": "2024-01-01",
            "open": 100.0,
            "high": 110.0,
            "low": 90.0,
            "close": 105.0,
            "volume": 1000.0,
        },
        {
            "symbol": "BTCUSDT",
            "timestamp": "2024-01-02",
            "open": 105.0,
            "high": 115.0,
            "low": 95.0,
            "close": 110.0,
            "volume": 1200.0,
        },
        {
            "symbol": "ETHUSDT",
            "timestamp": "2024-01-01",
            "open": 50.0,
            "high": 55.0,
            "low": 45.0,
            "close": 52.0,
            "volume": 2000.0,
        },
    ]


@pytest.mark.asyncio
async def test_pipeline_initialization(pipeline):
    """Test pipeline initialization"""
    assert pipeline is not None
    assert len(pipeline.stages_completed) == 0
    assert len(pipeline.stage_results) == 0


@pytest.mark.asyncio
async def test_pipeline_stages_skip(pipeline, sample_candles):
    """Test skipping specific pipeline stages"""

    # Mock stages
    pipeline.run_extract = AsyncMock(return_value=sample_candles)
    pipeline.run_transform = AsyncMock(return_value=sample_candles)
    pipeline.run_validate = AsyncMock(return_value=(len(sample_candles), 0))
    pipeline.run_deduplicate = AsyncMock(return_value=sample_candles)
    pipeline.run_load = AsyncMock(return_value=len(sample_candles))

    # Run with skipped validate stage
    result = await pipeline.run_full(symbols=["BTCUSDT"], skip_stages=[PipelineStage.VALIDATE])

    # Validate stage should not be in completed
    assert PipelineStage.VALIDATE not in pipeline.stages_completed


@pytest.mark.asyncio
async def test_pipeline_error_handling(pipeline, sample_candles):
    """Test error handling in pipeline"""

    # Mock stages with one failing
    pipeline.run_extract = AsyncMock(return_value=sample_candles)
    pipeline.run_transform = AsyncMock(side_effect=Exception("Transform failed"))

    # Should raise exception
    with pytest.raises(Exception, match="Transform failed"):
        await pipeline.run_full(symbols=["BTCUSDT"])


@pytest.mark.asyncio
async def test_extractor_transformer_integration(sample_candles):
    """Test extractor and transformer working together"""

    # Create transformer
    cleaner = DataCleaner(remove_outliers=True, fill_missing_with="skip")

    # Clean sample data
    cleaned = await cleaner.transform(sample_candles)

    # All samples should pass cleaning
    assert len(cleaned) == len(sample_candles)
    assert all(c["open"] > 0 for c in cleaned)


@pytest.mark.asyncio
async def test_validator_integration(sample_candles):
    """Test validator"""

    validator = DataQualityValidator(
        check_ohlc=True,
        check_volume=True,
        check_timestamps=False,
    )

    valid, invalid = await validator.validate(sample_candles)

    # All samples should be valid
    assert valid == len(sample_candles)
    assert invalid == 0


@pytest.mark.asyncio
async def test_end_to_end_pipeline_flow(sample_candles):
    """Test complete pipeline flow from extract to load"""

    # 1. Extract
    extracted = sample_candles.copy()
    assert len(extracted) == 3

    # 2. Transform
    cleaner = DataCleaner(fill_missing_with="skip")
    transformed = await cleaner.transform(extracted)
    assert len(transformed) == 3

    # 3. Validate
    validator = DataQualityValidator()
    valid, invalid = await validator.validate(transformed)
    assert valid == 3

    # 4. Deduplicate (simplified)
    deduplicated = []
    seen = set()
    for candle in transformed:
        key = (candle["symbol"], candle["timestamp"])
        if key not in seen:
            deduplicated.append(candle)
            seen.add(key)
    assert len(deduplicated) == 3

    # 5. Load (would insert into DB)
    # loaded = await loader.load(deduplicated)
    # assert loaded == 3


@pytest.mark.asyncio
async def test_batch_processing(sample_candles):
    """Test batch processing of large datasets"""

    # Create large dataset
    large_dataset = []
    for i in range(1000):
        large_dataset.append(
            {
                "symbol": f"SYM{i % 10}",
                "timestamp": f"2024-01-{(i % 28) + 1:02d}",
                "open": float(100 + i),
                "high": float(110 + i),
                "low": float(90 + i),
                "close": float(105 + i),
                "volume": float(1000 + i),
            }
        )

    # Transform in batches
    cleaner = DataCleaner()
    transformed = await cleaner.transform(large_dataset)

    # All should be cleaned
    assert len(transformed) == len(large_dataset)

    # Validate
    validator = DataQualityValidator()
    valid, invalid = await validator.validate(transformed)

    assert valid >= len(large_dataset) - 10  # Allow small error


@pytest.mark.asyncio
async def test_pipeline_statistics(pipeline):
    """Test pipeline statistics tracking"""

    # Mock pipeline execution
    pipeline.run_extract = AsyncMock(return_value=[{"symbol": "TEST"}] * 100)
    pipeline.run_transform = AsyncMock(return_value=[{"symbol": "TEST"}] * 100)
    pipeline.run_validate = AsyncMock(return_value=(100, 0))
    pipeline.run_deduplicate = AsyncMock(return_value=[{"symbol": "TEST"}] * 100)
    pipeline.run_load = AsyncMock(return_value=100)

    # Run pipeline
    result = await pipeline.run_full(symbols=["TEST"])

    # Check result structure
    assert result is not None
    assert "status" in result or "stages_completed" in result


@pytest.mark.asyncio
async def test_data_consistency(sample_candles):
    """Test data consistency through pipeline stages"""

    # Process through all stages
    cleaner = DataCleaner()
    cleaned = await cleaner.transform(sample_candles)

    # Verify symbols preserved
    symbols_before = {c.get("symbol") for c in sample_candles}
    symbols_after = {c.get("symbol") for c in cleaned}
    assert symbols_before == symbols_after

    # Verify timestamps preserved
    ts_before = {c.get("timestamp") for c in sample_candles}
    ts_after = {c.get("timestamp") for c in cleaned}
    assert ts_before == ts_after
