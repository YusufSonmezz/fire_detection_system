import logging
import os
from datetime import datetime

# Log file NAME has been created.
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
# Log file PATH has been created.
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
# Log FILE has been created.
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

def setup_logging(log_level=logging.INFO):
    """
    Configures logging with the specified log file path and log level.
    """
    logging.basicConfig(
        filename=LOG_FILE_PATH,
        format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
        level=log_level,
    )

if __name__ == "__main__":
    log_level = logging.WARNING
    setup_logging(log_level=log_level)
    logging.warning("Logging has been started!")