# ==============================================================================
# Module: logging_utils
#
# Description:
#   A centralized and reusable logging utility designed for bioinformatics
#   pipelines. It intelligently handles logging for both integrated pipeline runs
#   (e.g., from a Jupyter Notebook) and standalone script executions.
#
# Key Features:
#   - Centralized Logger: Uses a single, consistent logger name across an
#     entire project to unify log messages.
#   - Pipeline Integration via Environment Variables: A parent process
#     (like a Jupyter Notebook) can set an environment variable
#     ('PIPELINE_LOG_FILE') to dictate a single log file for all child scripts.
#   - Standalone Fallback: If a script is run by itself without the
#     environment variable, it automatically creates its own timestamped
#     log file in a default directory.
#   - Idempotent Setup: The `setup_logging` function is safe to call multiple
#     times; it configures the logger only once, preventing duplicate log entries.
#
# Author: [Your Name/Alias]
# Version: 1.0.0
# ==============================================================================
import logging
import os
import sys
from datetime import datetime
from typing import Tuple

# A unique name for the project's logger. Using a specific name prevents
# interference with loggers from other libraries (e.g., matplotlib, gseapy).
PROJECT_LOGGER_NAME = 'MyAnalysisLogger'


def setup_logging(log_dir="logs", level=logging.INFO) -> Tuple[logging.Logger, str]:
    """
    Initializes the project-wide logger for file and console output.

    This function serves as the single entry point for all logging configuration.
    It intelligently adapts based on the presence of an environment variable,
    making it seamless to use in any context.

    How it works:
    1.  **Pipeline Mode (Environment variable detected):**
        If the 'PIPELINE_LOG_FILE' environment variable is set, this function
        assumes it's part of a larger pipeline. It will direct all log messages
        to the file path specified in that variable. This is the key to unifying
        logs from multiple scripts into a single file.

    2.  **Standalone Mode (No environment variable):**
        If the environment variable is not found, the function assumes the script
        is being run on its own. It will create a new, timestamped log file inside
        the provided `log_dir` to capture the output of that single run.

    The function is idempotent, meaning it can be safely called multiple times
    without creating duplicate log handlers.

    Args:
        log_dir (str, optional): The default directory to create log files in
                                 when running in standalone mode.
                                 Defaults to "logs".
        level (int, optional): The logging level (e.g., logging.INFO,
                               logging.DEBUG). Defaults to logging.INFO.

    Returns:
        Tuple[logging.Logger, str]: A tuple containing:
            - The configured logger instance.
            - The absolute path to the log file being used.

    -----------------------------------------------------------------------------
    **HOW TO USE**
    -----------------------------------------------------------------------------

    **Scenario 1: In a controlling script (e.g., Jupyter Notebook, batch_runner.py)**

    In the first cell of your notebook, initialize the logger and set the
    environment variable for all subsequent script calls.

    .. code-block:: python

        import os
        from logging_utils import setup_logging

        # 1. Define where logs for this entire pipeline run should go.
        PIPELINE_LOG_DIR = "notebooks/logs/my_gsea_run"

        # 2. Set up the logger. It creates a timestamped log file inside the dir.
        logger, log_file_path = setup_logging(log_dir=PIPELINE_LOG_DIR)

        # 3. CRITICAL STEP: Set the environment variable for child processes.
        os.environ['PIPELINE_LOG_FILE'] = log_file_path

        # Now, any script called with !python will use this log file.
        logger.info("Pipeline started.")
        !python src/analysis/data_loading.py --config ...


    **Scenario 2: In any individual analysis script (e.g., data_loading.py)**

    At the top of each script, simply call `setup_logging()`. The function will
    automatically detect the environment variable (if set by a notebook) or
    create its own log file (if run alone).

    .. code-block:: python

        from logging_utils import setup_logging

        # This single line is all you need.
        # It gets the pipeline logger or creates a new one.
        log, _ = setup_logging()

        log.info("Data loading process started.")
        # ... rest of the script ...
    -----------------------------------------------------------------------------
    """
    logger = logging.getLogger(PROJECT_LOGGER_NAME)

    # If handlers are already configured, it means logging was set up
    # earlier in the process. Return the existing logger to avoid duplication.
    if logger.handlers:
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                # Return the existing logger and its file path
                return logger, handler.baseFilename
        return logger, ""

    logger.setLevel(level)
    # Prevent log messages from being passed up to the root logger, which might
    # have its own (unwanted) default handlers.
    logger.propagate = False

    # Check for the environment variable to see if we're in a pipeline.
    pipeline_log_file = os.environ.get('PIPELINE_LOG_FILE')

    if pipeline_log_file:
        # PIPELINE MODE: Use the file path provided by the parent process.
        log_path = pipeline_log_file
        # Ensure the directory exists, just in case.
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        init_message = f"Attached to pipeline log file: {log_path}"
    else:
        # STANDALONE MODE: Create a new, unique log file for this run.
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_path = os.path.join(log_dir, f"standalone_run_{timestamp}.log")
        init_message = f"No pipeline detected. Created new log file: {log_path}"

    # Define a consistent format for all log messages.
    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')

    # Handler 1: Writes log messages to the file.
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Handler 2: Writes log messages to the console (or notebook output).
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    logger.info(init_message)
    return logger, log_path

import sys

class StreamToLogger:
    """
    print() 문과 같은 스트림 출력을 logging 모듈로 리디렉션하는 클래스입니다.
    'with' 구문과 함께 사용하여 특정 코드 블록의 출력을 캡처할 수 있습니다.
    """
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass