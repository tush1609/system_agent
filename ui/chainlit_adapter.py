import sys
from pathlib import Path

import chainlit as cl

# Ensure repo root is on sys.path when Chainlit runs this file directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.controller import Controller
from runtime_args import RuntimeArgs
from ui.base import Base

RuntimeArgs.apply_runtime_logging_args()


@cl.on_message
async def receive_message(user_msg: cl.Message):
    await ch.receive_message(user_msg.content)

@cl.on_chat_start
async def on_start():
    # Initialize empty history for this session
    cl.user_session.set("id", [])
    cl.user_session.set("chainlit_instance", ch)


class Chainlit(Base):
    def __init__(self):
        self.msg = None
        super().__init__()

    def get_session_id(self): return cl.user_session.get("id")

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
