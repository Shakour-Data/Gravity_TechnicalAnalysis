"""
================================================================================
WebSocket Message Entity

Clean Architecture - Domain Layer
Defines WebSocket message structure for real-time communication.

Last Updated: 2025-11-07 (Phase 2.1 - Task 1.4)
================================================================================
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional
from .subscription_type import SubscriptionType


@dataclass(frozen=True)
class WebSocketMessage:
    """
    WebSocket message entity for real-time data streaming.

    Immutable dataclass representing messages sent through WebSocket connections.
    Used by WebSocket handlers to structure data for client communication.
    """

    # Message metadata
    message_type: str
    """Type of message (e.g., 'data', 'error', 'ping', 'pong')"""

    subscription_type: Optional[SubscriptionType]
    """Type of subscription this message relates to (None for system messages)"""

    client_id: Optional[str]
    """Client identifier (None for broadcast messages)"""

    timestamp: datetime
    """Message creation timestamp"""

    # Message content
    data: Dict[str, Any]
    """Message payload data"""

    # Optional fields
    sequence_number: Optional[int] = None
    """Sequence number for ordered message delivery"""

    correlation_id: Optional[str] = None
    """Correlation ID for request-response pairing"""

    error_message: Optional[str] = None
    """Error message if message_type is 'error'"""

    @classmethod
    def create_data_message(
        cls,
        subscription_type: SubscriptionType,
        data: Dict[str, Any],
        client_id: Optional[str] = None,
        sequence_number: Optional[int] = None,
        correlation_id: Optional[str] = None
    ) -> 'WebSocketMessage':
        """
        Create a data message.

        Args:
            subscription_type: Type of subscription
            data: Message payload
            client_id: Optional client identifier
            sequence_number: Optional sequence number
            correlation_id: Optional correlation ID

        Returns:
            WebSocketMessage instance
        """
        return cls(
            message_type="data",
            subscription_type=subscription_type,
            client_id=client_id,
            timestamp=datetime.utcnow(),
            data=data,
            sequence_number=sequence_number,
            correlation_id=correlation_id
        )

    @classmethod
    def create_error_message(
        cls,
        error_message: str,
        client_id: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> 'WebSocketMessage':
        """
        Create an error message.

        Args:
            error_message: Error description
            client_id: Optional client identifier
            correlation_id: Optional correlation ID

        Returns:
            WebSocketMessage instance
        """
        return cls(
            message_type="error",
            subscription_type=None,
            client_id=client_id,
            timestamp=datetime.utcnow(),
            data={},
            correlation_id=correlation_id,
            error_message=error_message
        )

    @classmethod
    def create_ping_message(cls) -> 'WebSocketMessage':
        """
        Create a ping message for connection health check.

        Returns:
            WebSocketMessage instance
        """
        return cls(
            message_type="ping",
            subscription_type=None,
            client_id=None,
            timestamp=datetime.utcnow(),
            data={}
        )

    @classmethod
    def create_pong_message(cls) -> 'WebSocketMessage':
        """
        Create a pong message in response to ping.

        Returns:
            WebSocketMessage instance
        """
        return cls(
            message_type="pong",
            subscription_type=None,
            client_id=None,
            timestamp=datetime.utcnow(),
            data={}
        )

    @classmethod
    def create_system_message(
        cls,
        system_event: str,
        data: Dict[str, Any]
    ) -> 'WebSocketMessage':
        """
        Create a system message (status, alerts, etc.).

        Args:
            system_event: Type of system event
            data: Event data

        Returns:
            WebSocketMessage instance
        """
        return cls(
            message_type="system",
            subscription_type=None,
            client_id=None,
            timestamp=datetime.utcnow(),
            data={"event": system_event, **data}
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert message to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the message
        """
        result = {
            "message_type": self.message_type,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data
        }

        if self.subscription_type:
            result["subscription_type"] = self.subscription_type.value

        if self.client_id:
            result["client_id"] = self.client_id

        if self.sequence_number is not None:
            result["sequence_number"] = self.sequence_number

        if self.correlation_id:
            result["correlation_id"] = self.correlation_id

        if self.error_message:
            result["error_message"] = self.error_message

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WebSocketMessage':
        """
        Create message from dictionary (JSON deserialization).

        Args:
            data: Dictionary representation of message

        Returns:
            WebSocketMessage instance

        Raises:
            ValueError: If required fields are missing or invalid
        """
        try:
            subscription_type = None
            if "subscription_type" in data:
                subscription_type = SubscriptionType.from_string(data["subscription_type"])

            return cls(
                message_type=data["message_type"],
                subscription_type=subscription_type,
                client_id=data.get("client_id"),
                timestamp=datetime.fromisoformat(data["timestamp"]),
                data=data["data"],
                sequence_number=data.get("sequence_number"),
                correlation_id=data.get("correlation_id"),
                error_message=data.get("error_message")
            )
        except KeyError as e:
            raise ValueError(f"Missing required field: {e}")
        except ValueError as e:
            raise ValueError(f"Invalid message format: {e}")

    @property
    def is_data_message(self) -> bool:
        """Check if this is a data message."""
        return self.message_type == "data"

    @property
    def is_error_message(self) -> bool:
        """Check if this is an error message."""
        return self.message_type == "error"

    @property
    def is_system_message(self) -> bool:
        """Check if this is a system message."""
        return self.message_type == "system"

    @property
    def is_ping_pong(self) -> bool:
        """Check if this is a ping or pong message."""
        return self.message_type in ("ping", "pong")