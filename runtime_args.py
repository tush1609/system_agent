import argparse
import sys

from logger import logger


class RuntimeArgs:
    @classmethod
    def get_args(cls) -> tuple[argparse.Namespace, list[str]]:
        parser = argparse.ArgumentParser(description="Launch AI Chat UI")
        parser.add_argument(
            "--ui",
            choices=["chainlit", "streamlit", "terminal"],
            required=True,
            help="Choose UI: chainlit or streamlit",
        )
        cls.add_runtime_args(parser)
        return parser.parse_known_args()

    @classmethod
    def add_runtime_args(cls, parser: argparse.ArgumentParser) -> None:
        file_log_group = parser.add_mutually_exclusive_group()
        file_log_group.add_argument(
            "--enable-file-logs",
            metavar="PATH",
            help="Enable file logging at runtime with the given log file path",
        )

    @classmethod
    def build_subprocess_args(cls, argv: list[str]) -> list[str]:
        subprocess_args: list[str] = []
        skip_next = False

        for arg in argv:
            if skip_next:
                skip_next = False
                continue

            if arg == "--ui":
                skip_next = True
                continue

            subprocess_args.append(arg)

        return subprocess_args

    @classmethod
    def apply_runtime_logging_args(cls, argv: list[str] | None = None) -> argparse.Namespace:
        parser = argparse.ArgumentParser(add_help=False)
        cls.add_runtime_args(parser)
        args, _ = parser.parse_known_args(argv if argv is not None else sys.argv[1:])

        if args.enable_file_logs:
            logger.configure(
                file_logs_enabled=True,
                log_file_path=args.enable_file_logs,
            )

        return args
