import chainlit as cl
from app.controller import Controller
from base import Base


@cl.on_message
async def receive_message(user_msg: cl.Message):
    await ch.receive_message(user_msg.content)

class Chainlit(Base):
    def __init__(self):
        self.msg = None
        super().__init__()

    async def receive_message(self, user_msg):
        await self._dispatch_message(user_msg)

    async def stream_start(self):
        self.msg = await cl.Message(content='').send()

    async def stream_message(self, data):
        await self.msg.stream_token(data)

    async def stream_stop(self):
        await self.msg.update()

ch = Chainlit()
Controller(ch)
