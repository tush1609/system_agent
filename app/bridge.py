import threading


class Bridge:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._stream_message_handler = None
        self._stream_start_handler = None
        self._stream_end_handler = None

    def set_stream_message_handler(self, stream_start_handler, stream_message_handler, stream_end_handler):
        self._stream_start_handler = stream_start_handler
        self._stream_message_handler = stream_message_handler
        self._stream_end_handler = stream_end_handler

    async def stream_start(self):
        if self._stream_start_handler is not None:
            await self._stream_start_handler()

    async def stream_message(self, data):
        if self._stream_message_handler is not None:
            await self._stream_message_handler(data)

    async def stream_end(self):
        if self._stream_end_handler is not None:
            await self._stream_end_handler()
