import logging
import os
from datetime import datetime
from typing import Tuple

# 프로젝트 전체에서 사용할 고유한 로거 이름을 정의합니다.
# 이렇게 하면 다른 라이브러리의 로거와 충돌하는 것을 방지할 수 있습니다.
PROJECT_LOGGER_NAME = 'MyAnalysisLogger'

def setup_logging(log_dir="logs", level=logging.INFO):
    """
    프로젝트 전반에 걸쳐 사용할 로거를 설정합니다.

    이 함수는 지정된 이름의 로거를 구성하여 타임스탬프가 찍힌 파일에 로그를 기록합니다.
    애플리케이션이나 노트북 시작 시 한 번만 호출하도록 설계되었습니다.
    이미 핸들러가 설정된 경우 다시 구성하지 않습니다.

    Args:
        log_dir (str): 로그 파일을 저장할 디렉토리.
        level (int): 로깅 레벨 (예: logging.INFO).

    Returns:
        logging.Logger: 설정된 로거 인스턴스.
    """
    logger = logging.getLogger(PROJECT_LOGGER_NAME)

    # 핸들러가 이미 설정되어 있다면, 추가 작업을 하지 않고 로거를 바로 반환합니다.
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False  # 로그가 루트 로거로 전파되는 것을 방지합니다.

    os.makedirs(log_dir, exist_ok=True)

    # 파일 핸들러: 타임스탬프 기반의 로그 파일에 기록합니다.
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f"analysis_run_{timestamp}.log")
    file_handler = logging.FileHandler(log_file, encoding='utf-8')

    # 콘솔 핸들러: 화면(콘솔)에도 로그를 출력합니다.
    console_handler = logging.StreamHandler()

    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized. log file: {log_file}")
    return logger

def setup_named_logger(
    logger_name: str, log_dir: str = "logs", level=logging.INFO
) -> Tuple[logging.Logger, str]:
    """
    Configures and returns a named logger that writes to a timestamped file.

    Also adds a stream handler to show logs in the console/notebook.

    Args:
        logger_name (str): The name of the logger.
        log_dir (str): The parent directory to store log files.
        level: The logging level.

    Returns:
        A tuple of (logging.Logger, str) for the configured logger
        and the path to its log file.
    """
    # Create a specific directory for this logger's logs
    logger_log_dir = os.path.join(log_dir, logger_name)
    os.makedirs(logger_log_dir, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    # Avoid adding handlers multiple times in interactive environments
    if logger.hasHandlers():
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                return logger, handler.baseFilename
        logger.handlers.clear()

    # Create a timestamped log file path
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(logger_log_dir, f"{timestamp}.log")

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Add file and stream handlers
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger, log_path


# ---------------------------------------------------------------------------
# Environment-aware logging (merged from logging_utils_environ.py)
# ---------------------------------------------------------------------------
import sys
from typing import Tuple as _Tuple


class StreamToLogger:
    """
    Redirects stream output (e.g., print() statements) to the logging module.
    Use with a 'with' statement to capture output from a specific code block.
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


def setup_logging_environ(log_dir="logs", level=logging.INFO) -> _Tuple[logging.Logger, str]:
    """
    Initializes the project-wide logger with pipeline/standalone mode detection.

    If the 'PIPELINE_LOG_FILE' environment variable is set, attaches to that
    log file (pipeline mode). Otherwise creates a timestamped log file in
    log_dir (standalone mode). Idempotent: safe to call multiple times.

    Args:
        log_dir (str): Default directory for standalone log files.
        level (int): Logging level. Defaults to logging.INFO.

    Returns:
        Tuple[logging.Logger, str]: Configured logger and path to log file.
    """
    import os as _os
    logger = logging.getLogger(PROJECT_LOGGER_NAME)

    if logger.handlers:
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                return logger, handler.baseFilename
        return logger, ""

    logger.setLevel(level)
    logger.propagate = False

    pipeline_log_file = _os.environ.get('PIPELINE_LOG_FILE')

    if pipeline_log_file:
        log_path = pipeline_log_file
        _os.makedirs(_os.path.dirname(log_path), exist_ok=True)
        init_message = f"Attached to pipeline log file: {log_path}"
    else:
        from datetime import datetime as _dt
        _os.makedirs(log_dir, exist_ok=True)
        timestamp = _dt.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_path = _os.path.join(log_dir, f"standalone_run_{timestamp}.log")
        init_message = f"No pipeline detected. Created new log file: {log_path}"

    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')

    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.info(init_message)
    return logger, log_path
