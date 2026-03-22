import logging
import os
import sys
import structlog
from dataclasses import dataclass, field
from typing import Any
from configuration.config_loader import config


APP_LOGGER_NAME = "system_agent"
DEFAULT_LOG_FILE_PATH = os.path.expanduser("~/Library/Logs/system_agent.log")
FILE_LOGS_ENABLED_ENV_VAR = "SYSTEM_AGENT_FILE_LOGS_ENABLED"
LOG_FILE_PATH_ENV_VAR = "SYSTEM_AGENT_LOG_FILE_PATH"


def _env_bool(name: str) -> bool | None:
    raw_value = os.getenv(name)
    if raw_value is None:
        return None

    normalized_value = raw_value.strip().lower()
    if normalized_value in {"1", "true", "yes", "on"}:
        return True
    if normalized_value in {"0", "false", "no", "off"}:
        return False
    return None


def _default_file_logs_enabled() -> bool:
    env_value = _env_bool(FILE_LOGS_ENABLED_ENV_VAR)
    if env_value is not None:
        return env_value
    return config.file_logs_enabled


def _default_log_file_path() -> str:
    return os.getenv(LOG_FILE_PATH_ENV_VAR, DEFAULT_LOG_FILE_PATH)

@dataclass
class LoggerConfig:
    info_enabled: bool = config.logger_info_enabled
    debug_enabled: bool = config.logger_debug_enabled
    error_enabled: bool = config.logger_error_enabled
    warn_enabled: bool = config.logger_warn_enabled
    third_party_logs_enabled: bool = config.third_party_logs_enabled
    file_logs_enabled: bool = field(default_factory=_default_file_logs_enabled)
    terminal_logs_enabled: bool = config.terminal_logs_enabled
    log_file_path: str = field(default_factory=_default_log_file_path)


def _logger_level(cfg: LoggerConfig) -> int:
    if cfg.debug_enabled:
        return logging.DEBUG
    if cfg.info_enabled:
        return logging.INFO
    if cfg.warn_enabled:
        return logging.WARNING
    if cfg.error_enabled:
        return logging.ERROR
    return logging.CRITICAL + 1


def _ensure_log_directory(log_file_path: str) -> None:
    if log_file_path:
        log_directory = os.path.dirname(log_file_path)
        if log_directory:
            os.makedirs(log_directory, exist_ok=True)


def _clear_all_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass


def _build_shared_formatter() -> structlog.stdlib.ProcessorFormatter:
    timestamper = structlog.processors.TimeStamper(fmt="iso")
    shared_processors = [
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        timestamper,
    ]

    if not getattr(_build_shared_formatter, "_configured", False):
        structlog.configure(
            processors=shared_processors + [
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        _build_shared_formatter._configured = True

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.KeyValueRenderer(
                key_order=["timestamp", "logger", "level", "event", "request_id"],
                sort_keys=False,
            ),
        ],
    )
    return formatter


def _build_file_handler(formatter: logging.Formatter, log_file_path: str) -> logging.Handler:
    _ensure_log_directory(log_file_path)
    handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    return handler


def _build_terminal_handler(formatter: logging.Formatter) -> logging.Handler:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    return handler


def _configure_logging(cfg: LoggerConfig) -> None:
    formatter = _build_shared_formatter()
    level = _logger_level(cfg)

    root_logger = logging.getLogger()
    app_logger = logging.getLogger(APP_LOGGER_NAME)

    _clear_all_handlers(root_logger)
    _clear_all_handlers(app_logger)

    root_logger.setLevel(level if cfg.third_party_logs_enabled else logging.CRITICAL + 1)
    app_logger.setLevel(level)
    app_logger.propagate = cfg.third_party_logs_enabled

    handlers: list[logging.Handler] = []

    if cfg.file_logs_enabled:
        handlers.append(_build_file_handler(formatter, cfg.log_file_path))

    if cfg.terminal_logs_enabled:
        handlers.append(_build_terminal_handler(formatter))

    target_logger = root_logger if cfg.third_party_logs_enabled else app_logger

    for handler in handlers:
        target_logger.addHandler(handler)


def get_logger(
    name: str = APP_LOGGER_NAME,
    logger_config: LoggerConfig | None = None,
) -> structlog.stdlib.BoundLogger:
    cfg = logger_config or LoggerConfig()
    _configure_logging(cfg)
    return structlog.get_logger(name)


class Logger:
    def __init__(self, name: str = APP_LOGGER_NAME, config: LoggerConfig | None = None) -> None:
        self._name = name
        self._config = config or LoggerConfig()
        self._logger = get_logger(name, self._config)

    def _refresh_level(self) -> None:
        _configure_logging(self._config)

    def configure(
        self,
        *,
        file_logs_enabled: bool | None = None,
        log_file_path: str | None = None,
    ) -> None:
        if file_logs_enabled is not None:
            self._config.info_enabled = True
            self._config.debug_enabled = True
            self._config.error_enabled = True
            self._config.warn_enabled = True
            self._config.third_party_logs_enabled = True
            self._config.file_logs_enabled = True
            self._config.terminal_logs_enabled = False
            self._config.log_file_path = log_file_path
        self._refresh_level()

    def info(self, message: str, **kwargs: Any) -> None:
        if self._config.info_enabled:
            self._logger.info(message, **kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:
        if self._config.debug_enabled:
            self._logger.debug(message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        if self._config.error_enabled:
            self._logger.error(message, **kwargs)

    def warn(self, message: str, **kwargs: Any) -> None:
        if self._config.warn_enabled:
            self._logger.warning(message, **kwargs)


logger = Logger()
