from loguru import logger

from kabosu_plus.sbv2.utils.stdout_wrapper import SAFE_STDOUT


# Remove all default handlers
logger.remove()

# Add a new handler
logger.add(
    SAFE_STDOUT,
    format="<g>{time:MM-DD HH:mm:ss}</g> |<lvl>{level:^8}</lvl>| {file}:{line} | {message}",
    backtrace=True,
    diagnose=True,
)
