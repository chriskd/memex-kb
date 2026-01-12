"""Tests for live reload SSE functionality."""

import asyncio

import pytest

from memex.webapp.events import Event, EventBroadcaster, EventType, get_broadcaster


class TestEventBroadcaster:
    """Tests for the EventBroadcaster class."""

    @pytest.mark.asyncio
    async def test_broadcast_to_subscribers(self):
        """Test that events are broadcast to all subscribers."""
        broadcaster = EventBroadcaster()

        # Collect events in a list
        received_events = []

        async def collect_events():
            async for event in broadcaster.subscribe():
                received_events.append(event)
                if len(received_events) >= 2:
                    break

        # Start collecting in background
        task = asyncio.create_task(collect_events())

        # Give subscriber time to register
        await asyncio.sleep(0.01)

        # Broadcast some events
        await broadcaster.broadcast(Event(type=EventType.FILE_CHANGED, data={"path": "test.md"}))
        await broadcaster.broadcast(Event(type=EventType.HEARTBEAT))

        # Wait for collection to complete
        await asyncio.wait_for(task, timeout=1.0)

        assert len(received_events) == 2
        assert received_events[0].type == EventType.FILE_CHANGED
        assert received_events[0].data["path"] == "test.md"
        assert received_events[1].type == EventType.HEARTBEAT

    @pytest.mark.asyncio
    async def test_subscriber_count(self):
        """Test subscriber count tracking."""
        broadcaster = EventBroadcaster()

        assert broadcaster.subscriber_count == 0

        # Create a subscriber
        events = []

        async def subscriber():
            async for event in broadcaster.subscribe():
                events.append(event)
                break  # Exit after first event

        task = asyncio.create_task(subscriber())
        await asyncio.sleep(0.01)

        assert broadcaster.subscriber_count == 1

        # Send event to unblock subscriber
        await broadcaster.broadcast(Event(type=EventType.HEARTBEAT))
        await asyncio.wait_for(task, timeout=1.0)

        # Subscriber should be removed after exiting
        await asyncio.sleep(0.01)
        assert broadcaster.subscriber_count == 0


class TestEvent:
    """Tests for the Event class."""

    def test_to_sse_format(self):
        """Test SSE message formatting."""
        event = Event(
            type=EventType.FILE_CHANGED,
            data={"path": "tooling/beads.md"},
        )

        sse = event.to_sse()

        assert "event: file_changed\n" in sse
        assert "data: " in sse
        assert '"type": "file_changed"' in sse
        assert '"path": "tooling/beads.md"' in sse
        assert sse.endswith("\n\n")

    def test_heartbeat_event(self):
        """Test heartbeat event formatting."""
        event = Event(type=EventType.HEARTBEAT)

        sse = event.to_sse()

        assert "event: heartbeat\n" in sse


class TestGetBroadcaster:
    """Tests for the global broadcaster singleton."""

    def test_returns_same_instance(self):
        """Test that get_broadcaster returns the same instance."""
        b1 = get_broadcaster()
        b2 = get_broadcaster()

        assert b1 is b2
