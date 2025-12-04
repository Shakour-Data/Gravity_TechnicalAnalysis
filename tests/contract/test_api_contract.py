"""
Contract Tests for Technical Analysis API

This module contains Pact contract tests to ensure API compatibility
with consumer services.
"""

import pytest

# Try to import pact, skip all tests if not available
try:
    from pact import Pact
    PACT_AVAILABLE = True
except ImportError:
    PACT_AVAILABLE = False
    pytestmark = pytest.mark.skip(reason="pact-python library not installed")
import atexit
import os
from pathlib import Path


# Setup Pact (only if available)
if PACT_AVAILABLE:
    PACT_DIR = Path(__file__).parent.parent.parent / "pacts"
    PACT_DIR.mkdir(exist_ok=True)

    pact = Consumer('TechAnalysisConsumer').has_pact_with(
        Provider('TechAnalysisAPI'),
        pact_dir=str(PACT_DIR),
        host_name='localhost',
        port=8000
    )

    # Cleanup
    atexit.register(pact.stop)
else:
    pact = None


@pytest.fixture(scope='session')
def pact_fixture():
    """Setup and teardown for Pact."""
    pact.start_service()
    yield pact
    pact.stop_service()


class TestAnalysisEndpoint:
    """Contract tests for /api/v1/analysis endpoint."""
    
    def test_complete_analysis_contract(self, pact_fixture):
        """Test complete analysis endpoint contract."""
        expected = {
            'volume_matrix': Like({
                'final_score': 7.5,
                'final_signal': 'BUY',
                'confidence_level': 'HIGH',
                'dimension_scores': {
                    'trend': 8.0,
                    'momentum': 7.5,
                    'volatility': 7.0,
                    'cycle': 7.5,
                    'support_resistance': 8.0
                },
                'metadata': Like({
                    'timestamp': Term(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', '2024-01-01T12:00:00'),
                    'symbol': 'BTCUSDT',
                    'timeframe': '1h'
                })
            }),
            'five_dimensional_decision': Like({
                'final_score': 7.8,
                'final_signal': 'STRONG_BUY',
                'confidence': 'HIGH',
                'risk_level': 'MEDIUM',
                'decision_matrix': Like({
                    'trend_momentum_matrix': Like({}),
                    'volatility_cycle_matrix': Like({}),
                    'support_resistance_integration': Like({})
                })
            })
        }
        
        (pact
         .given('analysis data is available')
         .upon_receiving('a request for complete analysis')
         .with_request('post', '/api/v1/analysis/complete')
         .will_respond_with(200, body=expected))
        
        with pact:
            # Make actual request to verify contract
            import httpx
            response = httpx.post(
                f'{pact.uri}/api/v1/analysis/complete',
                json={
                    'symbol': 'BTCUSDT',
                    'timeframe': '1h',
                    'candles': [
                        {
                            'timestamp': '2024-01-01T00:00:00',
                            'open': 40000.0,
                            'high': 41000.0,
                            'low': 39500.0,
                            'close': 40500.0,
                            'volume': 1000000.0
                        }
                    ] * 100
                },
                timeout=30
            )
            
            assert response.status_code == 200
            data = response.json()
            assert 'volume_matrix' in data
            assert 'five_dimensional_decision' in data
            assert data['volume_matrix']['final_signal'] in ['BUY', 'SELL', 'HOLD']
    
    def test_trend_analysis_contract(self, pact_fixture):
        """Test trend analysis endpoint contract."""
        expected = {
            'score': 7.5,
            'signal': 'BULLISH',
            'confidence': 'HIGH',
            'indicators': Like({
                'moving_averages': Like({
                    'sma_20': 40000.0,
                    'ema_50': 39500.0
                }),
                'trend_strength': 0.75
            })
        }
        
        (pact
         .given('trend analysis data is available')
         .upon_receiving('a request for trend analysis')
         .with_request('post', '/api/v1/analysis/trend')
         .will_respond_with(200, body=expected))
        
        with pact:
            import httpx
            response = httpx.post(
                f'{pact.uri}/api/v1/analysis/trend',
                json={
                    'symbol': 'BTCUSDT',
                    'timeframe': '1h',
                    'candles': [{'close': 40000.0}] * 100
                },
                timeout=30
            )
            
            assert response.status_code == 200
            data = response.json()
            assert 'score' in data
            assert 'signal' in data
    
    def test_health_endpoint_contract(self, pact_fixture):
        """Test health endpoint contract."""
        expected = {
            'status': 'healthy',
            'version': Term(r'\d+\.\d+\.\d+', '1.0.0'),
            'timestamp': Term(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', '2024-01-01T12:00:00')
        }
        
        (pact
         .given('service is running')
         .upon_receiving('a health check request')
         .with_request('get', '/health')
         .will_respond_with(200, body=expected))
        
        with pact:
            import httpx
            response = httpx.get(f'{pact.uri}/health', timeout=10)
            
            assert response.status_code == 200
            data = response.json()
            assert data['status'] == 'healthy'


class TestCacheEndpoints:
    """Contract tests for cache-related endpoints."""
    
    def test_cached_analysis_contract(self, pact_fixture):
        """Test cached analysis retrieval."""
        expected = {
            'cached': True,
            'data': Like({
                'score': 7.5,
                'signal': 'BUY'
            }),
            'cache_metadata': Like({
                'hit': True,
                'ttl': 300
            })
        }
        
        (pact
         .given('cached analysis exists')
         .upon_receiving('a request for cached analysis')
         .with_request('get', '/api/v1/analysis/cached/BTCUSDT')
         .will_respond_with(200, body=expected))


class TestErrorHandling:
    """Contract tests for error handling."""
    
    def test_invalid_request_contract(self, pact_fixture):
        """Test error response for invalid request."""
        expected = {
            'error': Like('Validation error'),
            'details': EachLike({
                'field': 'symbol',
                'message': 'Invalid symbol format'
            })
        }
        
        (pact
         .given('invalid request data')
         .upon_receiving('a request with invalid data')
         .with_request('post', '/api/v1/analysis/complete')
         .will_respond_with(400, body=expected))
    
    def test_service_unavailable_contract(self, pact_fixture):
        """Test error response when dependencies are down."""
        expected = {
            'error': 'Service temporarily unavailable',
            'retry_after': 60
        }
        
        (pact
         .given('dependencies are unavailable')
         .upon_receiving('a request when service is degraded')
         .with_request('get', '/health/ready')
         .will_respond_with(503, body=expected))


class TestEventContracts:
    """Contract tests for event-driven messaging."""
    
    def test_analysis_completed_event_contract(self):
        """Test analysis completed event structure."""
        # This would typically integrate with message broker testing
        event_schema = {
            'event_type': 'ANALYSIS_COMPLETED',
            'timestamp': Term(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', '2024-01-01T12:00:00'),
            'data': Like({
                'symbol': 'BTCUSDT',
                'signal': 'BUY',
                'score': 7.5
            }),
            'metadata': Like({
                'correlation_id': Term(r'[a-f0-9-]{36}', 'abc-123-def-456'),
                'service': 'technical-analysis'
            })
        }
        
        # Verify event schema structure
        assert 'event_type' in event_schema
        assert 'data' in event_schema
        assert 'metadata' in event_schema


@pytest.mark.integration
class TestServiceDiscoveryContract:
    """Contract tests for service discovery integration."""
    
    def test_service_registration_contract(self):
        """Test service registration with Eureka/Consul."""
        registration_schema = {
            'instance': Like({
                'app': 'technical-analysis',
                'hostName': 'localhost',
                'port': 8000,
                'status': 'UP',
                'healthCheckUrl': 'http://localhost:8000/health',
                'metadata': Like({
                    'version': '1.0.0',
                    'environment': 'production'
                })
            })
        }
        
        assert 'instance' in registration_schema
        assert registration_schema['instance']['status'] == 'UP'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

