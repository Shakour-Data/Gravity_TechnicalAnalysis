"""
Unit tests for scenarios API endpoints.

Tests the three-scenario analysis endpoints using simplified testing approach
to avoid FastAPI TestClient compatibility issues.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from gravity_tech.analysis.scenario_analysis import ScenarioResult, ThreeScenarioAnalysis
from gravity_tech.api.v1.scenarios import (
    analyze_scenarios,
    get_data_client,
    get_neutral_scenario,
    get_optimistic_scenario,
    get_pessimistic_scenario,
    get_scenario_analyzer,
    router,
)


class TestScenariosAPI:
    """Test scenarios API endpoints."""

    def test_router_setup(self):
        """Test that router is properly configured."""
        assert router.prefix == "/api/v1/scenarios"
        assert "Scenario Analysis" in router.tags
        assert len(router.routes) == 4  # 4 endpoints total

    @patch('gravity_tech.api.v1.scenarios.get_settings')
    def test_get_data_client(self, mock_get_settings):
        """Test data client dependency injection."""
        mock_settings = MagicMock()
        mock_settings.DATA_SERVICE_URL = "http://test:8000"
        mock_settings.REDIS_URL = "redis://test:6379"
        mock_get_settings.return_value = mock_settings

        client = get_data_client()

        assert client is not None
        mock_get_settings.assert_called_once()

    @patch('gravity_tech.api.v1.scenarios.get_data_client')
    def test_get_scenario_analyzer(self, mock_get_data_client):
        """Test scenario analyzer dependency injection."""
        mock_client = MagicMock()
        mock_get_data_client.return_value = mock_client

        analyzer = get_scenario_analyzer(mock_client)

        assert analyzer is not None
        assert analyzer.data_client == mock_client

    @patch('gravity_tech.api.v1.scenarios.get_scenario_analyzer')
    @pytest.mark.asyncio
    async def test_analyze_scenarios_success(self, mock_get_analyzer):
        """Test successful scenario analysis."""
        # Mock analyzer
        mock_analyzer = AsyncMock()
        mock_analysis = ThreeScenarioAnalysis(
            symbol="AAPL",
            timestamp=datetime.now(),
            current_price=180.5,
            optimistic=ScenarioResult(
                scenario_type="optimistic",
                score=78.5,
                probability=70.0,
                target_price=195.0,
                stop_loss=178.0,
                risk_reward_ratio=3.0,
                key_signals=["Strong uptrend", "High momentum"],
                recommendation="BUY",
                confidence="HIGH",
                timeframe_days=30
            ),
            neutral=ScenarioResult(
                scenario_type="neutral",
                score=65.0,
                probability=50.0,
                target_price=187.0,
                stop_loss=175.0,
                risk_reward_ratio=1.5,
                key_signals=["Sideways movement", "Average volume"],
                recommendation="HOLD",
                confidence="MEDIUM",
                timeframe_days=30
            ),
            pessimistic=ScenarioResult(
                scenario_type="pessimistic",
                score=45.0,
                probability=30.0,
                target_price=182.0,
                stop_loss=172.0,
                risk_reward_ratio=0.5,
                key_signals=["Weak support", "Declining volume"],
                recommendation="SELL",
                confidence="MEDIUM",
                timeframe_days=30
            ),
            expected_return=5.8,
            expected_risk=3.2,
            sharpe_ratio=1.81,
            recommended_scenario="optimistic",
            overall_confidence="HIGH"
        )
        mock_analyzer.analyze_from_service.return_value = mock_analysis
        mock_get_analyzer.return_value = mock_analyzer

        # Call function
        result = await analyze_scenarios(
            symbol="AAPL",
            timeframe="1d",
            lookback_days=365,
            analyzer=mock_analyzer
        )

        # Assertions
        assert result == mock_analysis
        assert result.symbol == "AAPL"
        assert result.current_price == 180.5
        assert result.expected_return == 5.8
        assert result.sharpe_ratio == 1.81
        mock_analyzer.analyze_from_service.assert_called_once_with(
            symbol="AAPL",
            timeframe="1d",
            lookback_days=365
        )

    @patch('gravity_tech.api.v1.scenarios.get_scenario_analyzer')
    @pytest.mark.asyncio
    async def test_analyze_scenarios_validation_error(self, mock_get_analyzer):
        """Test scenario analysis with validation error."""
        mock_analyzer = AsyncMock()
        mock_analyzer.analyze_from_service.side_effect = ValueError("Invalid symbol")
        mock_get_analyzer.return_value = mock_analyzer

        with pytest.raises(HTTPException) as exc_info:
            await analyze_scenarios(
                symbol="INVALID",
                timeframe="1d",
                lookback_days=365,
                analyzer=mock_analyzer
            )

        assert exc_info.value.status_code == 400
        assert "Invalid symbol" in exc_info.value.detail

    @patch('gravity_tech.api.v1.scenarios.get_scenario_analyzer')
    @pytest.mark.asyncio
    async def test_analyze_scenarios_service_error(self, mock_get_analyzer):
        """Test scenario analysis with service error."""
        mock_analyzer = AsyncMock()
        mock_analyzer.analyze_from_service.side_effect = Exception("Service unavailable")
        mock_get_analyzer.return_value = mock_analyzer

        with pytest.raises(HTTPException) as exc_info:
            await analyze_scenarios(
                symbol="AAPL",
                timeframe="1d",
                lookback_days=365,
                analyzer=mock_analyzer
            )

        assert exc_info.value.status_code == 503
        assert "Failed to analyze scenarios" in exc_info.value.detail

    @patch('gravity_tech.api.v1.scenarios.analyze_scenarios')
    @pytest.mark.asyncio
    async def test_get_optimistic_scenario(self, mock_analyze_scenarios):
        """Test getting optimistic scenario."""
        mock_analysis = ThreeScenarioAnalysis(
            symbol="AAPL",
            timestamp=datetime.now(),
            current_price=180.5,
            optimistic=ScenarioResult(
                scenario_type="optimistic",
                score=78.5,
                probability=70.0,
                target_price=195.0,
                stop_loss=178.0,
                risk_reward_ratio=3.0,
                key_signals=["Strong uptrend", "High momentum"],
                recommendation="BUY",
                confidence="HIGH",
                timeframe_days=30
            ),
            neutral=ScenarioResult(
                scenario_type="neutral",
                score=65.0,
                probability=50.0,
                target_price=187.0,
                stop_loss=175.0,
                risk_reward_ratio=1.5,
                key_signals=["Sideways movement", "Average volume"],
                recommendation="HOLD",
                confidence="MEDIUM",
                timeframe_days=30
            ),
            pessimistic=ScenarioResult(
                scenario_type="pessimistic",
                score=45.0,
                probability=30.0,
                target_price=182.0,
                stop_loss=172.0,
                risk_reward_ratio=0.5,
                key_signals=["Weak support", "Declining volume"],
                recommendation="SELL",
                confidence="MEDIUM",
                timeframe_days=30
            ),
            expected_return=5.8,
            expected_risk=3.2,
            sharpe_ratio=1.81,
            recommended_scenario="optimistic",
            overall_confidence="HIGH"
        )
        mock_analyze_scenarios.return_value = mock_analysis

        result = await get_optimistic_scenario(
            symbol="AAPL",
            timeframe="1d",
            lookback_days=365,
            analyzer=MagicMock()
        )

        assert result == mock_analysis.optimistic
        assert result.score == 78.5
        assert result.recommendation == "BUY"
        mock_analyze_scenarios.assert_called_once()

    @patch('gravity_tech.api.v1.scenarios.analyze_scenarios')
    @pytest.mark.asyncio
    async def test_get_neutral_scenario(self, mock_analyze_scenarios):
        """Test getting neutral scenario."""
        mock_analysis = ThreeScenarioAnalysis(
            symbol="AAPL",
            timestamp=datetime.now(),
            current_price=180.5,
            optimistic=ScenarioResult(
                scenario_type="optimistic",
                score=78.5,
                probability=70.0,
                target_price=195.0,
                stop_loss=178.0,
                risk_reward_ratio=3.0,
                key_signals=["Strong uptrend", "High momentum"],
                recommendation="BUY",
                confidence="HIGH",
                timeframe_days=30
            ),
            neutral=ScenarioResult(
                scenario_type="neutral",
                score=65.0,
                probability=50.0,
                target_price=187.0,
                stop_loss=175.0,
                risk_reward_ratio=1.5,
                key_signals=["Sideways movement", "Average volume"],
                recommendation="HOLD",
                confidence="MEDIUM",
                timeframe_days=30
            ),
            pessimistic=ScenarioResult(
                scenario_type="pessimistic",
                score=45.0,
                probability=30.0,
                target_price=182.0,
                stop_loss=172.0,
                risk_reward_ratio=0.5,
                key_signals=["Weak support", "Declining volume"],
                recommendation="SELL",
                confidence="MEDIUM",
                timeframe_days=30
            ),
            expected_return=5.8,
            expected_risk=3.2,
            sharpe_ratio=1.81,
            recommended_scenario="optimistic",
            overall_confidence="HIGH"
        )
        mock_analyze_scenarios.return_value = mock_analysis

        result = await get_neutral_scenario(
            symbol="AAPL",
            timeframe="1d",
            lookback_days=365,
            analyzer=MagicMock()
        )

        assert result == mock_analysis.neutral
        assert result.score == 65.0
        assert result.recommendation == "HOLD"
        mock_analyze_scenarios.assert_called_once()

    @patch('gravity_tech.api.v1.scenarios.analyze_scenarios')
    @pytest.mark.asyncio
    async def test_get_pessimistic_scenario(self, mock_analyze_scenarios):
        """Test getting pessimistic scenario."""
        mock_analysis = ThreeScenarioAnalysis(
            symbol="AAPL",
            timestamp=datetime.now(),
            current_price=180.5,
            optimistic=ScenarioResult(
                scenario_type="optimistic",
                score=78.5,
                probability=70.0,
                target_price=195.0,
                stop_loss=178.0,
                risk_reward_ratio=3.0,
                key_signals=["Strong uptrend", "High momentum"],
                recommendation="BUY",
                confidence="HIGH",
                timeframe_days=30
            ),
            neutral=ScenarioResult(
                scenario_type="neutral",
                score=65.0,
                probability=50.0,
                target_price=187.0,
                stop_loss=175.0,
                risk_reward_ratio=1.5,
                key_signals=["Sideways movement", "Average volume"],
                recommendation="HOLD",
                confidence="MEDIUM",
                timeframe_days=30
            ),
            pessimistic=ScenarioResult(
                scenario_type="pessimistic",
                score=45.0,
                probability=30.0,
                target_price=182.0,
                stop_loss=172.0,
                risk_reward_ratio=0.5,
                key_signals=["Weak support", "Declining volume"],
                recommendation="SELL",
                confidence="MEDIUM",
                timeframe_days=30
            ),
            expected_return=5.8,
            expected_risk=3.2,
            sharpe_ratio=1.81,
            recommended_scenario="optimistic",
            overall_confidence="HIGH"
        )
        mock_analyze_scenarios.return_value = mock_analysis

        result = await get_pessimistic_scenario(
            symbol="AAPL",
            timeframe="1d",
            lookback_days=365,
            analyzer=MagicMock()
        )

        assert result == mock_analysis.pessimistic
        assert result.score == 45.0
        assert result.recommendation == "SELL"
        mock_analyze_scenarios.assert_called_once()
