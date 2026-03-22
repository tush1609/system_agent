import asyncio
import sys
import uuid
import streamlit as st
from app.controller import Controller
from runtime_args import RuntimeArgs
from ui.base import Base

from pathlib import Path
# Ensure repo root is on sys.path when Streamlit runs this file directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


RuntimeArgs.apply_runtime_logging_args()

def run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(coro)
        return

    import nest_asyncio
    nest_asyncio.apply()
    loop.run_until_complete(coro)


class StreamlitUI(Base):
    def __init__(self):
        super().__init__()
        self._current_assistant_text = ""
        self._stream_placeholder = None
        self.controller = None

    def set_controller(self, controller):
        self.controller = controller

    def render(self):
        st.set_page_config(page_title="System Agent", page_icon="S")

        if "user_session_id" not in st.session_state:
            st.session_state.user_session_id = str(uuid.uuid4())
        if "messages" not in st.session_state:
            st.session_state["messages"] = []
        if "graph_messages" not in st.session_state:
            st.session_state["graph_messages"] = []
        if self.controller is not None:
            self.controller.messages = st.session_state["graph_messages"]

        for msg in st.session_state["messages"]:
            st.chat_message(msg["role"]).markdown(msg["content"])

        user_input = st.chat_input("Ask something...")
        if user_input:
            st.session_state["messages"].append({"role": "user", "content": user_input})
            st.chat_message("user").markdown(user_input)
            run_async(self.receive_message(user_input))

    def get_session_id(self) -> str:
        return st.session_state.user_session_id

    async def receive_message(self, user_msg: str):
        await self._dispatch_message(user_msg)

    async def stream_start(self):
        self._current_assistant_text = ""
        container = st.chat_message("assistant")
        self._stream_placeholder = container.empty()

    async def stream_message(self, message: str):
        self._current_assistant_text += message
        if self._stream_placeholder is not None:
            self._stream_placeholder.markdown(self._current_assistant_text)

    async def stream_stop(self):
        if "messages" not in st.session_state:
            st.session_state["messages"] = []
        st.session_state["messages"].append(
            {"role": "assistant", "content": self._current_assistant_text}
        )
        if self.controller is not None:
            st.session_state["graph_messages"] = self.controller.messages


if "ui_instance" not in st.session_state:
    st.session_state["ui_instance"] = StreamlitUI()

ui = st.session_state["ui_instance"]

if "controller_instance" not in st.session_state:
    st.session_state["controller_instance"] = Controller(ui)

controller = st.session_state["controller_instance"]
ui.set_controller(controller)
ui.render()
