"""Async message queue for decoupled channel-agent communication."""

import asyncio
from collections.abc import Callable

from nanobot.bus.events import InboundMessage, OutboundMessage


class MessageBus:
    """
    Async message bus that decouples chat channels from the agent core.

    Channels push messages to the inbound queue, and the agent processes
    them and pushes responses to the outbound queue.
    """

    def __init__(self):
        self.inbound: asyncio.Queue[InboundMessage] = asyncio.Queue()
        self.outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue()

    async def publish_inbound(self, msg: InboundMessage) -> None:
        """Publish a message from a channel to the agent."""
        await self.inbound.put(msg)

    async def consume_inbound(self) -> InboundMessage:
        """Consume the next inbound message (blocks until available)."""
        return await self.inbound.get()

    def discard_inbound(self, predicate: Callable[[InboundMessage], bool]) -> int:
        """Atomically discard queued inbound messages matching *predicate*.

        The method contains no await points, so callers on the owning event loop can close
        enqueue-to-dispatch cancellation races without reordering retained messages.
        """
        retained: list[InboundMessage] = []
        discarded = 0
        while True:
            try:
                msg = self.inbound.get_nowait()
            except asyncio.QueueEmpty:
                break
            if predicate(msg):
                discarded += 1
            else:
                retained.append(msg)
        for msg in retained:
            self.inbound.put_nowait(msg)
        return discarded

    async def publish_outbound(self, msg: OutboundMessage) -> None:
        """Publish a response from the agent to channels."""
        await self.outbound.put(msg)

    async def consume_outbound(self) -> OutboundMessage:
        """Consume the next outbound message (blocks until available)."""
        return await self.outbound.get()

    @property
    def inbound_size(self) -> int:
        """Number of pending inbound messages."""
        return self.inbound.qsize()

    @property
    def outbound_size(self) -> int:
        """Number of pending outbound messages."""
        return self.outbound.qsize()
