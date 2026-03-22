from abc import ABC, abstractmethod
from typing import Awaitable, Callable, List


class Base(ABC):
    def __init__(self):
        self._message_handlers: List[Callable[[str], Awaitable[None]]] = []

    def on_message(self, handler: Callable[[str], Awaitable[None]]) -> None:
        self._message_handlers.append(handler)

    async def _dispatch_message(self, message: str) -> None:
        for handler in self._message_handlers:
            await handler(message)

    @abstractmethod
    def get_session_id(self) -> str: pass

    @abstractmethod
    def receive_message(self, message): pass

    @abstractmethod
    def stream_start(self): pass

    @abstractmethod
    def stream_message(self, message): pass

    @abstractmethod
    def stream_stop(self): pass
