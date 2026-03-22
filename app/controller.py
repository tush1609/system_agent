from multi_agents.graph import Graph
from app.bridge import Bridge

class Controller:
    def __init__(self, ui):
        self.ui = ui
        self.ui.on_message(self.process_user_query)
        self.graph = Graph()
        Bridge().set_stream_message_handler(self.stream_start, self.stream_message, self.stream_stop)

    async def process_user_query(self, query):
        await self.graph.get_result(query, self.ui.get_session_id())

    async def stream_start(self):
        if self.ui and self.ui.stream_start:
            await self.ui.stream_start()

    async def stream_message(self, data):
        if self.ui and self.ui.stream_message:
            await self.ui.stream_message(data)

    async def stream_stop(self):
        if self.ui and self.ui.stream_stop:
            await self.ui.stream_stop()
