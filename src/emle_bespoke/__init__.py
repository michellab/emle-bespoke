"""emle-bespoke base package."""

from . import _version
from ._log import config_logger, log_banner

__author__ = "Joao Morado"
__version__ = _version.get_versions()["version"]
__all__ = ["__version__"]
