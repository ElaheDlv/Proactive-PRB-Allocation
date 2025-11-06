import csv
import os
import logging
from typing import List, Tuple, Dict, Optional, Any
import math
from statistics import median


def estimate_trace_period(
    samples: List[Tuple[float, int, int]],
    default_step: Optional[float] = None,
) -> float:
    """Infer the replay period (seconds) for a trace from sample timestamps."""

    if not samples:
        return 0.0

    try:
        fallback = float(default_step) if default_step is not None else None
    except Exception:
        fallback = None
    if fallback is None or fallback <= 0:
        fallback = 1.0

    try:
        times = sorted(float(s[0]) for s in samples if s is not None)
    except Exception:
        times = []
    if not times:
        return fallback

    last_t = times[-1]
    if len(times) == 1:
        return last_t + fallback

    deltas = []
    prev = times[0]
    for t in times[1:]:
        try:
            delta = float(t) - float(prev)
        except Exception:
            delta = 0.0
        if delta > 1e-9:
            deltas.append(delta)
        prev = t

    if deltas:
        step = median(deltas)
    else:
        step = fallback

    if step <= 0:
        step = fallback

    return last_t + step


def load_csv_trace(path: str) -> List[Tuple[float, int, int]]:
    """
    Disabled: aggregated CSV traces are not supported in this deployment.
    Use load_raw_packet_csv and the --trace-raw-map CLI instead.
    """
    raise NotImplementedError(
        "load_csv_trace is disabled. Use load_raw_packet_csv / --trace-raw-map."
    )
    # """
    # Load a traffic trace CSV into a list of (t_seconds, dl_bytes, ul_bytes).

    # Expected headers (case-sensitive any of):
    #   - time columns: one of ["t_s", "time", "timestamp"]
    #   - data columns: "dl_bytes" and optionally "ul_bytes"

    # If ul_bytes is missing, it defaults to 0.
    # Timestamps are normalized to start at 0.0.
    # """
    # # Accumulator for parsed samples as tuples (t_s, dl_bytes, ul_bytes)
    # samples: List[Tuple[float, int, int]] = []
    # # Open the CSV file for reading
    # with open(path, "r", newline="") as f:
    #     # DictReader yields rows as dicts keyed by header names
    #     reader = csv.DictReader(f)
    #     # If no header is present, DictReader.fieldnames is None
    #     if reader.fieldnames is None:
    #         raise ValueError("CSV has no header")
    #     # Normalize header names to a lookup map (preserving original case as values)
    #     fields = {name.strip(): name for name in reader.fieldnames}
    #     # Try to find a time column using supported names (case sensitive first)
    #     t_keys = [k for k in ("t_s", "time", "timestamp") if k in fields]
    #     if not t_keys:
    #         # Fall back to case-insensitive search by lowering all header names
    #         lowered = {k.lower(): k for k in fields}
    #         t_keys = [lowered.get(k) for k in ("t_s", "time", "timestamp") if lowered.get(k)]
    #     if not t_keys:
    #         # Give a clear error if no time column was found
    #         raise KeyError("Trace CSV must contain a time column: t_s, time, or timestamp")
    #     # Use the first matching time column
    #     t_key = t_keys[0]
    #     # Data columns: dl_bytes is required; ul_bytes is optional
    #     dl_key = fields.get("dl_bytes")
    #     ul_key = fields.get("ul_bytes")
    #     if dl_key is None:
    #         raise KeyError("Trace CSV must contain 'dl_bytes' column")
    #     # Parse each row, skipping lines that are empty or unparsable
    #     for row in reader:
    #         if not row:
    #             continue  # skip empty rows
    #         try:
    #             t = float(row[t_key])  # parse timestamp
    #         except Exception:
    #             continue  # skip if timestamp is not numeric
    #         try:
    #             dl = int(float(row[dl_key]))  # dl_bytes as int
    #         except Exception:
    #             dl = 0  # default to 0 if missing or non-numeric
    #         ul = 0
    #         if ul_key is not None:
    #             try:
    #                 ul = int(float(row[ul_key]))  # ul_bytes if present
    #             except Exception:
    #                 ul = 0
    #         samples.append((t, dl, ul))
    # if not samples:
    #     return []
    # # Normalize timestamps so the first sample starts at 0s
    # t0 = samples[0][0]
    # norm = [(s[0] - t0, s[1], s[2]) for s in samples]
    # return norm


def _find_col(name_map: Dict[str, int], candidates: List[str]) -> int:
    """
    Locate a column index in a case-insensitive header map using a
    tolerant matching strategy: exact match first, then prefix, then substring.
    """
    # 1) Exact match: prefer exact header names if available
    for n in candidates:
        if n in name_map:
            return name_map[n]
    # 2) Prefix match: accept headers that start with the candidate
    for n in candidates:
        for k, idx in name_map.items():
            if k.startswith(n):
                return idx
    # 3) Substring match: as a last resort, accept any header containing the candidate
    for n in candidates:
        for k, idx in name_map.items():
            if n in k:
                return idx
    # If nothing matched, raise with a helpful message listing the available headers
    raise KeyError(f"Columns {candidates} not found; got: {list(name_map.keys())}")


# Note: IP auto-detection removed by design. The raw loader now requires an
# explicit `ue_ip` and will raise an error if it is missing.


def load_raw_packet_csv(
    path: str,
    ue_ip: Optional[str] = None,
    bin_s: float = 1.0,
    overhead_sub_bytes: int = 0,
) -> List[Tuple[float, int, int]]:
    """
    Parse a raw packet capture exported as CSV and aggregate it into
    evenly binned traffic samples suitable for replay.

    Input CSV is expected to have time and IP endpoints (case-insensitive
    column names are supported). Common headers include:
      - Time: one of [t_s, time, timestamp, frame.time_epoch]
      - Source/Destination: [source/src/ip.src] and [destination/dst/ip.dst]
      - Length: one of [length, len, frame.len, bytes, size]

    Aggregation rules:
      - Each packet contributes (length - overhead_sub_bytes) bytes.
      - Direction is classified by `ue_ip`:
          DL if Destination == ue_ip; UL if Source == ue_ip.
      - Time is normalized to start at 0 using the first observed timestamp.
      - Packets are grouped into bins of width `bin_s` seconds using
        floor((t - t0)/bin_s) * bin_s; the output is a list of
        (bin_time_s, dl_bytes, ul_bytes) sorted by time.
    """
    # Read header row to build a name->index map (case-insensitive)
    if not ue_ip:
        raise ValueError("ue_ip is required to classify DL/UL; pass ue_ip explicitly.")

    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return []
        # Lowercase keys for case-insensitive lookups
        name_map: Dict[str, int] = {h.strip().lower(): i for i, h in enumerate(header)}

        # Locate required columns with tolerant matching
        idx_time = _find_col(name_map, ["t_s", "time", "timestamp", "frame.time_epoch"]) if name_map else 0
        idx_src = _find_col(name_map, ["source", "src", "ip.src"])  # type: ignore[arg-type]
        idx_dst = _find_col(name_map, ["destination", "dst", "ip.dst"])  # type: ignore[arg-type]
        idx_len = _find_col(name_map, ["length", "len", "frame.len", "bytes", "size"])  # type: ignore[arg-type]

        # Aggregate packet lengths into (DL, UL) buckets keyed by bin timestamp
        bins: Dict[float, Tuple[int, int]] = {}
        first_t = None
        for row in reader:
            if not row or len(row) <= max(idx_time, idx_src, idx_dst, idx_len):
                continue
            try:
                t = float(row[idx_time])
                length = int(float(row[idx_len]))
            except Exception:
                continue
            src = row[idx_src].strip()
            dst = row[idx_dst].strip()
            if first_t is None:
                first_t = t
            # Normalize time to start at zero (relative to first packet)
            t0 = t - (first_t or 0.0)
            # Compute bin start time with numeric stability guards
            b = math.floor(t0 / max(1e-6, float(bin_s))) * float(bin_s)
            dl, ul = bins.get(b, (0, 0))
            # Subtract protocol overhead to approximate payload bytes
            adj = max(0, int(length) - max(0, int(overhead_sub_bytes)))
            if ue_ip and dst == ue_ip:
                dl += adj
            if ue_ip and src == ue_ip:
                ul += adj
            bins[b] = (dl, ul)

        if not bins:
            return []
        # Produce ordered list of (t, dl, ul) sorted by bin time
        ts = sorted(bins.keys())
        samples: List[Tuple[float, int, int]] = [(t, bins[t][0], bins[t][1]) for t in ts]
        return samples


# -------------------------------------------------------------
# Trace validation helpers
# -------------------------------------------------------------

def _first_n(reader, n: int) -> List[Any]:
    """Consume and return the first n rows from an iterator (e.g., CSV reader)."""
    out = []
    for i, row in enumerate(reader):
        if i >= n:
            break  # stop after collecting n rows
        out.append(row)
    return out


def validate_preaggregated_trace_csv(path: str, sample_rows: int = 10) -> Dict[str, Any]:
    """
    Validate that a pre-aggregated CSV exists, has required columns, and
    that a small sample of rows is parsable as numbers.

    Returns a dict with keys: exists, valid, error, columns, sample_count.
    """
    # Prepare the result envelope with defaults
    res: Dict[str, Any] = {
        "kind": "agg",
        "path": path,
        "exists": os.path.exists(path),  # does the file exist on disk?
        "valid": False,                  # will be set true only after checks pass
        "error": None,                   # carries a human-readable error message
        "columns": {},                   # maps canonical names -> actual header used
        "sample_count": 0,               # number of rows we could parse in the sample
    }
    if not res["exists"]:
        res["error"] = "File not found"   # short-circuit when path is missing
        return res

    try:
        with open(path, "r", newline="") as f:
            # Use DictReader to access values by column name
            reader = csv.DictReader(f)
            # If the CSV has no header row, DictReader.fieldnames is None
            if reader.fieldnames is None:
                res["error"] = "CSV has no header"
                return res
            # Build a map of header name -> original header (trim whitespace)
            fields = {name.strip(): name for name in reader.fieldnames}
            # Accept any of these time headers (case-sensitive at this point)
            t_keys = [k for k in ("t_s", "time", "timestamp") if k in fields]
            # Data columns: dl_bytes required; ul_bytes optional
            dl_key = fields.get("dl_bytes")
            ul_key = fields.get("ul_bytes")
            # Fail fast when time column is missing
            if not t_keys:
                res["error"] = "Missing time column (t_s/time/timestamp)"
                return res
            # Fail fast when dl_bytes is missing
            if dl_key is None:
                res["error"] = "Missing 'dl_bytes' column"
                return res
            # Record which concrete header names we will use
            res["columns"] = {"time": t_keys[0], "dl_bytes": dl_key, "ul_bytes": ul_key}

            # Read a small sample to confirm numeric parsing works
            rows = _first_n(reader, sample_rows)
            ok_rows = 0
            for row in rows:
                try:
                    # Must be able to parse time and dl_bytes as numbers
                    float(row[t_keys[0]])
                    float(row[dl_key])
                    ok_rows += 1
                except Exception:
                    # Ignore individual bad lines in the sample
                    continue
            res["sample_count"] = ok_rows
            # If none of the sampled rows were valid, treat as invalid trace
            if ok_rows == 0:
                res["error"] = "No parsable rows in first sample"
                return res
            # All checks passed for the sample; mark as valid
            res["valid"] = True
            return res
    except Exception as e:
        # Any unexpected error (e.g., I/O) becomes the error message
        res["error"] = str(e)
        return res


def validate_raw_packet_trace_csv(path: str, sample_rows: int = 100) -> Dict[str, Any]:
    """
    Validate that a raw packet CSV exists, contains required columns, and
    that a small sample of rows is numeric. Also reports an auto-detected
    UE IP (if any) for reference.

    Returns a dict with keys: exists, valid, error, columns, sample_count.
    """
    # Result envelope with default values
    res: Dict[str, Any] = {
        "kind": "raw",                  # validator kind
        "path": path,                    # original path we attempted to validate
        "exists": os.path.exists(path),  # quick existence check
        "valid": False,                  # set to True only after checks pass
        "error": None,                   # human‑readable failure reason
        "columns": {},                   # resolved header names used
        "sample_count": 0,               # number of rows in the sample that parsed OK
    }
    if not res["exists"]:
        res["error"] = "File not found"  # short‑circuit if the file is missing
        return res

    try:
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                res["error"] = "Empty CSV"
                return res
            # Build a lower‑cased header name -> index map for tolerant lookups
            name_map: Dict[str, int] = {h.strip().lower(): i for i, h in enumerate(header)}
            try:
                # Locate the required columns (accept common variants)
                idx_time = _find_col(name_map, ["t_s", "time", "timestamp", "frame.time_epoch"])  # type: ignore[arg-type]
                idx_src  = _find_col(name_map, ["source", "src", "ip.src"])                      # type: ignore[arg-type]
                idx_dst  = _find_col(name_map, ["destination", "dst", "ip.dst"])                  # type: ignore[arg-type]
                idx_len  = _find_col(name_map, ["length", "len", "frame.len", "bytes", "size"])  # type: ignore[arg-type]
            except KeyError as e:
                res["error"] = str(e)
                return res
            # Record the concrete header names we resolved for logs/debug
            res["columns"] = {
                "time": list(name_map.keys())[idx_time] if idx_time is not None else None,
                "src": list(name_map.keys())[idx_src] if idx_src is not None else None,
                "dst": list(name_map.keys())[idx_dst] if idx_dst is not None else None,
                "len": list(name_map.keys())[idx_len] if idx_len is not None else None,
            }

            # Sample a few rows to ensure fields are numeric
            rows = _first_n(reader, sample_rows)
            ok_rows = 0
            for row in rows:
                # Skip rows that are too short to contain all indices
                if not row or len(row) <= max(idx_time, idx_src, idx_dst, idx_len):
                    continue
                try:
                    float(row[idx_time])  # time must be numeric
                    float(row[idx_len])   # length must be numeric
                except Exception:
                    continue  # tolerate a few malformed lines in the sample
                ok_rows += 1
            res["sample_count"] = ok_rows
            if ok_rows == 0:
                res["error"] = "No parsable rows in first sample"
                return res
            res["valid"] = True
            return res
    except Exception as e:
        res["error"] = str(e)
        return res


def validate_traces_configuration(
    trace_map: Optional[Dict[str, str]] = None,
    raw_map: Optional[List[Dict[str, Optional[str]]]] = None,
    bin_s: float = 1.0,
    overhead_bytes: int = 70,
    logger_name: str = __name__,
) -> Dict[str, Any]:
    """
    Validate both pre-aggregated and raw trace configurations and log a
    one-line status per entry so misconfigurations are visible early.

    Returns a dict with 'all_valid' and per-kind results.
    """
    logger = logging.getLogger(logger_name)
    results: Dict[str, Any] = {"all_valid": True, "agg": {}, "raw": {}}

    # ANSI colors for visibility (safe to display in most terminals)
    C_RESET = "\033[0m"
    C_GREEN = "\033[92m"
    C_RED = "\033[91m"
    C_CYAN = "\033[96m"
    C_YELLOW = "\033[93m"
    C_BOLD = "\033[1m"

    # Visible banner to make validation easy to spot in logs
    logger.info(f"{C_CYAN}{C_BOLD}=== TRACE VALIDATION START ==={C_RESET}")

    trace_map = trace_map or {}
    for imsi, path in trace_map.items():
        # Check each (IMSI -> aggregated csv) mapping
        r = validate_preaggregated_trace_csv(path)
        results["agg"][imsi] = r
        if not r.get("valid"):
            results["all_valid"] = False
            logger.error(f"{C_RED}Trace (agg) for {imsi}: {path} -> INVALID: {r.get('error')}{C_RESET}")
        else:
            logger.info(f"{C_GREEN}Trace (agg) for {imsi}: {path} -> OK{C_RESET} "
                        f"(cols={r['columns']}, sample_rows={r['sample_count']})")

    raw_map = raw_map or []
    for item in raw_map:
        if not isinstance(item, dict):
            continue
        imsi = item.get("imsi") or "?"
        path = item.get("file") or ""
        # Validate the packet CSV exists and is parsable
        r = validate_raw_packet_trace_csv(path)
        results["raw"][imsi] = r
        if not r.get("valid"):
            results["all_valid"] = False
            logger.error(f"{C_RED}Trace (raw) for {imsi}: {path} -> INVALID: {r.get('error')}{C_RESET}")
        else:
            logger.info(f"{C_GREEN}Trace (raw) for {imsi}: {path} -> OK{C_RESET} "
                        f"(cols={r['columns']}, sample_rows={r['sample_count']})")

    # Final summary line to make failures obvious but non-fatal
    if not results["all_valid"]:
        logger.warning(f"{C_YELLOW}Some traces invalid. Simulation will continue but affected UEs may have no replayed traffic.{C_RESET}")
    else:
        logger.info(f"{C_GREEN}All configured traces look valid.{C_RESET}")
    logger.info(f"{C_CYAN}{C_BOLD}=== TRACE VALIDATION END ==={C_RESET}")
    return results
