"""Logging configuration for the emle-bespoke package."""
import logging
import logging.config
import os

from . import __version__

logger = logging.getLogger(__name__)


def log_banner() -> None:
    """Print a banner with the emle-bespoke logo."""
    banner = r"""
╔══════════════════════════════════════════════════════╗
║                       _                              ║
║                      | |                             ║
║       ___  _ __ ___  | |  ___                        ║
║      / _ \| '_ ` _ \ | | / _ \                       ║
║     |  __/| | | | | || ||  __/                       ║
║      \___||_| |_| |_||_| \___|                       ║
║      _                               _               ║
║     | |                             | |              ║
║     | |__    ___  ___  _ __    ___  | | __ ___       ║
║     | '_ \  / _ \/ __|| '_ \  / _ \ | |/ // _ \      ║
║     | |_) ||  __/\__ \| |_) || (_) ||   <|  __/      ║
║     |_.__/  \___||___/| .__/  \___/ |_|\_\\___|      ║
║                       | |                            ║
║                       |_|                            ║
║                                                      ║
║                  EMLE Bespoke v{}                 ║
╚══════════════════════════════════════════════════════╝

""".format(
        __version__
    )

    lines = banner.split("\n")

    # Iterate over each line and log them
    for line in lines[1:-1]:
        logger.info(line)


class BlockFilter(logging.Filter):
    """Filter to block loggers not starting with emle_bespoke."""

    def filter(self, record):
        """Filter loggers not starting with emle_bespoke."""
        return record.name.startswith("emle_bespoke")


def config_logger() -> None:
    """Configure the logger for the emle_bespoke package."""
    # Define log level
    log_level = os.environ.get("EMLE_BESPOKE_LOG_LEVEL", default="INFO").upper()
    silence_loggers = int(os.environ.get("EMLE_BESPOKE_FILTER_LOGGERS", default=1))

    fmt = "%(asctime)s %(levelname)-8s %(name)-40s %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # Set up basicConfig
    LOGGING = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"default": {"format": fmt, "datefmt": datefmt}},
        "handlers": {
            "stdout": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "formatter": "default",
            }
        },
        "loggers": {
            "": {"handlers": ["stdout"], "level": log_level},
        },
    }

    logging.config.dictConfig(LOGGING)

    try:
        import coloredlogs

        coloredlogs.install(level=getattr(logging, log_level), fmt=fmt, datefmt=datefmt)
    except ImportError:
        pass

    if silence_loggers:
        for handler in logging.getLogger().handlers:
            handler.addFilter(BlockFilter())


config_logger()
