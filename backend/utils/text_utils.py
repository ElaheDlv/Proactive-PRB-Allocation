def get_first_paragraph(text: str) -> str:
    return text.split("\n\n")[0] if text else ""


def bytes_pretty_printer(service_disk_size_bytes: int) -> str:
    # Handle negative or None input gracefully
    if service_disk_size_bytes is None or service_disk_size_bytes < 0:
        return "N/A"
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = float(service_disk_size_bytes)
    for unit in units:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def parse_memory_usage_string(memory_usage: str) -> float:
    """
    Parse a memory usage string and return the value in GB.
    Supports formats like '1.5 GB', '1536 MB', etc.
    """
    if not memory_usage:
        return 0.0
    memory_usage = memory_usage.strip().upper()
    if "GB" in memory_usage:
        return float(memory_usage.replace("GB", "").strip())
    elif "MB" in memory_usage:
        return float(memory_usage.replace("MB", "").strip()) / 1000.0
    else:
        raise ValueError(f"Unknown memory usage format: {memory_usage}")
