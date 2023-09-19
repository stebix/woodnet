import datetime

DEFAULT_TIMESTAMP_FORMAT: str = '%Y-%m-%d_%H-%M-%S'

def create_timestamp(fmt: str | None = None) -> str:
    """Curent timestamp with second-wise accuracy."""
    fmt = fmt or DEFAULT_TIMESTAMP_FORMAT
    return datetime.datetime.now().strftime(fmt)
