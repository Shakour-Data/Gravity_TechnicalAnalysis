import io
import sys


def ensure_utf8_console():
    """
    Force stdout/stderr to use UTF-8 so Persian log messages do not break on
    Windows consoles that default to cp1252. Falls back to replacing
    unrepresentable characters if reconfiguration fails.
    """
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue

        encoding = getattr(stream, "encoding", None)
        if encoding and encoding.lower().startswith("utf-8"):
            continue

        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
            elif hasattr(stream, "buffer"):
                wrapper = io.TextIOWrapper(stream.buffer, encoding="utf-8", errors="replace")
                setattr(sys, stream_name, wrapper)
        except Exception:
            # If the console cannot be reconfigured, keep going to avoid failing
            # user commands entirely—subsequent prints will keep working with
            # replacement characters instead of crashing.
            continue
