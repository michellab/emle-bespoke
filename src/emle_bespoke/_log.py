import os as _os
import sys as _sys

from loguru import logger as _logger

from ._version import get_versions


# Print the banner with the emle-bespoke logo
def log_banner() -> None:
    """Print a banner with the emle-bespoke logo."""
    version = get_versions()["version"]
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
╚══════════════════════════════════════════════════════╝
version: {}
""".format(
        version
    )

    # Log each line of the banner
    for line in banner.split("\n")[1:-1]:
        _logger.info(line)


def log_cli_args(args):
    """Log the CLI arguments."""
    msg = r"""
╔════════════════════════════════════════════════════════════╗
║                      Input CLI Arguments                   ║
╚════════════════════════════════════════════════════════════╝
"""
    for line in msg.split("\n"):
        _logger.info(line)
    for arg in vars(args):
        _logger.info(f"{arg}: {getattr(args, arg)}")
    _logger.info("══════════════════════════════════════════════════════════════\n")


def log_termination():
    msg = r"""
╔════════════════════════════════════════════════════════════╗
║              emle-bespoke terminated normally!             ║
╚════════════════════════════════════════════════════════════╝"""
    for line in msg.split("\n"):
        _logger.info(line)


# Filter to block loggers not starting with "emle_bespoke"
class BlockFilter:
    def __call__(self, record):
        """Filter loggers not starting with 'emle_bespoke' or 'emle'."""
        return record["name"].startswith(("emle_bespoke", "emle"))


# Configure the logger for the emle_bespoke package
def config_logger() -> None:
    """Configure the logger for the emle_bespoke package."""
    log_level = _os.environ.get("EMLE_BESPOKE_LOG_LEVEL", default="INFO").upper()
    silence_loggers = int(_os.environ.get("EMLE_BESPOKE_FILTER_LOGGERS", default=1))

    fmt = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name:40}</cyan> | {message}"

    _logger.remove()  # Remove the default handler
    _logger.add(
        _sys.stdout,
        format=fmt,
        level=log_level,
        filter=BlockFilter() if silence_loggers else None,
    )


config_logger()
