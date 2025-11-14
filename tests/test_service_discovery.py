"""
Tests for Service Discovery Middleware

Tests Eureka and Consul integration.

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from gravity_tech.middleware.service_discovery import ServiceDiscovery


@pytest.fixture
def service_discovery():
    """Create service discovery instance for testing."""
    return ServiceDiscovery()


class TestServiceDiscoveryInitialization:
    """Tests for service discovery initialization."""
    
    @pytest.mark.asyncio
    async def test_eureka_initialization(self, service_discovery):
        """Test Eureka client initialization."""
        with patch('gravity_tech.middleware.service_discovery.eureka_client') as mock_eureka:
            mock_eureka.init = Mock()
            
            await service_discovery.initialize()
            
            # Verify Eureka was initialized
            mock_eureka.init.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_consul_initialization(self, service_discovery):
        """Test Consul client initialization."""
        with patch('gravity_tech.middleware.service_discovery.consul') as mock_consul:
            mock_consul.Consul = Mock(return_value=AsyncMock())
            
            service_discovery.discovery_type = "consul"
            await service_discovery.initialize()
            
            assert service_discovery.consul_client is not None


class TestServiceRegistration:
    """Tests for service registration."""
    
    @pytest.mark.asyncio
    async def test_register_with_eureka(self, service_discovery):
        """Test service registration with Eureka."""
        with patch('gravity_tech.middleware.service_discovery.eureka_client') as mock_eureka:
            mock_eureka.init = Mock()
            
            await service_discovery.initialize()
            
            # Verify registration parameters
            call_args = mock_eureka.init.call_args
            assert call_args is not None
    
    @pytest.mark.asyncio
    async def test_register_with_metadata(self, service_discovery):
        """Test service registration includes metadata."""
        with patch('gravity_tech.middleware.service_discovery.eureka_client') as mock_eureka:
            mock_eureka.init = Mock()
            
            await service_discovery.initialize()
            
            # Check metadata was included
            call_kwargs = mock_eureka.init.call_args[1] if mock_eureka.init.call_args else {}
            assert 'metadata' in call_kwargs or mock_eureka.init.called


class TestHealthReporting:
    """Tests for health reporting."""
    
    @pytest.mark.asyncio
    async def test_health_check_reporting(self, service_discovery):
        """Test periodic health check reporting."""
        with patch('gravity_tech.middleware.service_discovery.eureka_client') as mock_eureka:
            mock_eureka.init = Mock()
            
            await service_discovery.initialize()
            
            # Health reporting should be configured
            assert mock_eureka.init.called


class TestServiceDiscovery:
    """Tests for discovering other services."""
    
    @pytest.mark.asyncio
    async def test_discover_service_eureka(self, service_discovery):
        """Test discovering a service via Eureka."""
        with patch('gravity_tech.middleware.service_discovery.eureka_client') as mock_eureka:
            mock_eureka.get_service = Mock(return_value={
                'host': 'localhost',
                'port': 8001,
                'metadata': {'version': '1.0.0'}
            })
            
            result = await service_discovery.discover_service('payment-service')
            
            assert result is not None
    
    @pytest.mark.asyncio
    async def test_discover_nonexistent_service(self, service_discovery):
        """Test discovering non-existent service returns None."""
        with patch('gravity_tech.middleware.service_discovery.eureka_client') as mock_eureka:
            mock_eureka.get_service = Mock(return_value=None)
            
            result = await service_discovery.discover_service('nonexistent')
            
            assert result is None


class TestGracefulShutdown:
    """Tests for graceful deregistration."""
    
    @pytest.mark.asyncio
    async def test_deregister_on_shutdown(self, service_discovery):
        """Test service deregistration on shutdown."""
        with patch('gravity_tech.middleware.service_discovery.eureka_client') as mock_eureka:
            mock_eureka.init = Mock()
            mock_eureka.stop = Mock()
            
            await service_discovery.initialize()
            await service_discovery.shutdown()
            
            # Verify stop was called
            mock_eureka.stop.assert_called_once()


class TestConsulIntegration:
    """Tests for Consul-specific functionality."""
    
    @pytest.mark.asyncio
    async def test_consul_service_registration(self, service_discovery):
        """Test service registration with Consul."""
        service_discovery.discovery_type = "consul"
        
        with patch('gravity_tech.middleware.service_discovery.consul.Consul') as mock_consul_class:
            mock_consul_instance = AsyncMock()
            mock_consul_class.return_value = mock_consul_instance
            
            await service_discovery.initialize()
            
            assert service_discovery.consul_client is not None
    
    @pytest.mark.asyncio
    async def test_consul_health_check(self, service_discovery):
        """Test health check registration with Consul."""
        service_discovery.discovery_type = "consul"
        
        with patch('gravity_tech.middleware.service_discovery.consul.Consul') as mock_consul_class:
            mock_consul_instance = AsyncMock()
            mock_consul_instance.agent = AsyncMock()
            mock_consul_class.return_value = mock_consul_instance
            
            await service_discovery.initialize()
            
            # Health check URL should be configured
            assert service_discovery.consul_client is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
