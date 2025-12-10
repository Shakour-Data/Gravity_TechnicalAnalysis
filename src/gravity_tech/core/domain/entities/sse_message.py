"""
================================================================================
SSE Message Entity

Clean Architecture - Domain Layer
Defines Server-Sent Events message structure for real-time communication.

Last Updated: 2025-11-07 (Phase 2.1 - Task 1.4)
================================================================================
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .subscription_type import SubscriptionType
from datetime import timezone


@dataclass(frozen=True)
class SSEMessage:
    """
    Server-Sent Events message entity for real-time data streaming.

    Immutable dataclass representing messages sent through SSE connections.
    Used by SSE handlers to structure data for client communication.
    """

    # Message metadata
    event_type: str
    """SSE event type (e.g., 'data', 'error', 'ping', 'system')"""

    subscription_type: SubscriptionType | None
    """Type of subscription this message relates to (None for system messages)"""

    client_id: str | None
    """Client identifier (None for broadcast messages)"""

    timestamp: datetime
    """Message creation timestamp"""

    # Message content
    data: dict[str, Any]
    """Message payload data"""

    # Optional fields
    event_id: str | None = None
    """SSE event ID for client-side tracking"""

    retry: int | None = None
    """SSE retry delay in milliseconds"""

    error_message: str | None = None
    """Error message if event_type is 'error'"""

    @classmethod
    def create_data_event(
        cls,
        subscription_type: SubscriptionType,
        data: dict[str, Any],
        client_id: str | None = None,
        event_id: str | None = None
    ) -> 'SSEMessage':
        """
        Create a data event.

        Args:
            subscription_type: Type of subscription
            data: Event payload
            client_id: Optional client identifier
            event_id: Optional event ID

        Returns:
            SSEMessage instance
        """
        return cls(
            event_type="data",
            subscription_type=subscription_type,
            client_id=client_id,
            timestamp=datetime.now(timezone.utc),
            data=data,
            event_id=event_id
        )

    @classmethod
    def create_error_event(
        cls,
        error_message: str,
        client_id: str | None = None,
        event_id: str | None = None
    ) -> 'SSEMessage':
        """
        Create an error event.

        Args:
            error_message: Error description
            client_id: Optional client identifier
            event_id: Optional event ID

        Returns:
            SSEMessage instance
        """
        return cls(
            event_type="error",
            subscription_type=None,
            client_id=client_id,
            timestamp=datetime.now(timezone.utc),
            data={},
            event_id=event_id,
            error_message=error_message
        )

    @classmethod
    def create_ping_event(cls) -> 'SSEMessage':
        """
        Create a ping event for connection health check.

        Returns:
            SSEMessage instance
        """
        return cls(
            event_type="ping",
            subscription_type=None,
            client_id=None,
            timestamp=datetime.now(timezone.utc),
            data={}
        )

    @classmethod
    def create_system_event(
        cls,
        system_event: str,
        data: dict[str, Any],
        event_id: str | None = None
    ) -> 'SSEMessage':
        """
        Create a system event (status, alerts, etc.).

        Args:
            system_event: Type of system event
            data: Event data
            event_id: Optional event ID

        Returns:
            SSEMessage instance
        """
        return cls(
            event_type="system",
            subscription_type=None,
            client_id=None,
            timestamp=datetime.now(timezone.utc),
            data={"event": system_event, **data},
            event_id=event_id
        )

    def to_sse_format(self) -> str:
        """
        Convert message to Server-Sent Events format.

        Returns:
            SSE formatted string
        """
        lines = []

        # Event type
        lines.append(f"event: {self.event_type}")

        # Event ID
        if self.event_id:
            lines.append(f"id: {self.event_id}")

        # Retry
        if self.retry is not None:
            lines.append(f"retry: {self.retry}")

        # Data (JSON)
        import json
        data_str = json.dumps(self.to_dict(), ensure_ascii=False)
        lines.append(f"data: {data_str}")

        # Empty line to end the event
        lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert message to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the message
        """
        result = {
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data
        }

        if self.subscription_type:
            result["subscription_type"] = self.subscription_type.value

        if self.client_id:
            result["client_id"] = self.client_id

        if self.event_id:
            result["event_id"] = self.event_id

        if self.retry is not None:
            result["retry"] = self.retry

        if self.error_message:
            result["error_message"] = self.error_message

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'SSEMessage':
        """
        Create message from dictionary (JSON deserialization).

        Args:
            data: Dictionary representation of message

        Returns:
            SSEMessage instance

        Raises:
            ValueError: If required fields are missing or invalid
        """
        try:
            subscription_type = None
            if "subscription_type" in data:
                subscription_type = SubscriptionType.from_string(data["subscription_type"])

            return cls(
                event_type=data["event_type"],
                subscription_type=subscription_type,
                client_id=data.get("client_id"),
                timestamp=datetime.fromisoformat(data["timestamp"]),
                data=data["data"],
                event_id=data.get("event_id"),
                retry=data.get("retry"),
                error_message=data.get("error_message")
            )
        except KeyError as e:
            raise ValueError(f"Missing required field: {e}") from e
        except ValueError as e:
            raise ValueError(f"Invalid message format: {e}") from e

    @property
    def is_data_event(self) -> bool:
        """Check if this is a data event."""
        return self.event_type == "data"

    @property
    def is_error_event(self) -> bool:
        """Check if this is an error event."""
        return self.event_type == "error"

    @property
    def is_system_event(self) -> bool:
        """Check if this is a system event."""
        return self.event_type == "system"

    @property
    def is_ping_event(self) -> bool:
        """Check if this is a ping event."""
        return self.event_type == "ping"
