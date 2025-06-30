import sys
from loguru import logger

# Optional: configure sink, formatting, level
logger.remove()  # Remove default handler
logger.add(
    sink=sys.stderr,
    level="INFO",  # Or "DEBUG" / "WARNING" / "ERROR"
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{module}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)
