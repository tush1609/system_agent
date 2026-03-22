import subprocess
import sys
from pathlib import Path
from logger import logger
from runtime_args import RuntimeArgs


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    chainlit_path = base_dir / "ui" / "chainlit_adapter.py"
    streamlit_path = base_dir / "ui" / "streamlit_adapter.py"
    terminal_path = base_dir / "ui" / "terminal_adapter.py"

    args, _ = RuntimeArgs.get_args()
    child_args = RuntimeArgs.build_subprocess_args(sys.argv[1:])
    RuntimeArgs.apply_runtime_logging_args(child_args)

    logger.info("APPLICATION_STARTED", ui=args.ui)
    logger.debug("APPLICATION_STARTED", ui=args.ui)
    if args.ui == "chainlit":
        subprocess.run(
            [
                sys.executable,
                "-m",
                "chainlit",
                "run",
                str(chainlit_path),
                "-w",
                "--",
                *child_args,
            ],
        )
    elif args.ui == "streamlit":
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(streamlit_path), "--", *child_args],
        )
    elif args.ui == "terminal":
        subprocess.run([sys.executable, str(terminal_path), *child_args])
    else:
        logger.error(f"UNKNOWN_UI: {args.ui}")
