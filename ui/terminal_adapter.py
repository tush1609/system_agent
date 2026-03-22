import asyncio
import sys
from pathlib import Path

# Ensure repo root is on sys.path when running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.controller import Controller
from runtime_args import RuntimeArgs
from ui.base import Base


class TerminalUI(Base):
    def get_session_id(self) -> str:
        return ''

    def __init__(self):
        super().__init__()
        self._streaming = False

    async def receive_message(self, user_msg: str):
        await self._dispatch_message(user_msg)

    async def stream_start(self):
        self._streaming = True
        sys.stdout.write("Assistant: ")
        sys.stdout.flush()

    async def stream_message(self, message: str):
        sys.stdout.write(message)
        sys.stdout.flush()

    async def stream_stop(self):
        if self._streaming:
            sys.stdout.write("\n")
            sys.stdout.flush()
        self._streaming = False


async def prompt_loop(terminal: TerminalUI):
    loop = asyncio.get_running_loop()
    while True:
        try:
            user_msg = await loop.run_in_executor(None, input, "You: ")
        except (EOFError, KeyboardInterrupt):
            sys.stdout.write("\nExiting...\n")
            return

        user_msg = (user_msg or "").strip()
        if user_msg == "exit" or user_msg == "quit" or user_msg == "bye":
            sys.stdout.write("\nExiting...\n")
            return

        if not user_msg:
            continue
        await terminal.receive_message(user_msg)


if __name__ == "__main__":
    RuntimeArgs.apply_runtime_logging_args()
    ui = TerminalUI()
    Controller(ui)
    asyncio.run(prompt_loop(ui))
