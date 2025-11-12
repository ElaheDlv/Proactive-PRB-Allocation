import logging
from typing import Dict

import settings
from .xapp_base import xAppBase


class xAppConstantPRBAllocator(xAppBase):
    """xApp that enforces fixed slice PRB quotas and logs throughput/latency stats.

    Useful for baselining whether the configured radio resources can meet the
    desired latency targets without any dynamic control logic.
    """

    def __init__(self, ric=None):
        super().__init__(ric=ric)
        self.enabled = bool(getattr(settings, "PRB_CONST_ALLOC_ENABLE", False))
        self._slice_targets: Dict[str, int] = dict(
            getattr(settings, "PRB_CONST_ALLOC_MAP", {}) or {}
        )
        self._log_interval = max(1, int(getattr(settings, "PRB_CONST_LOG_INTERVAL", 500)))
        self._step = 0
        self._logger = logging.getLogger(self.xapp_id)

    def start(self):
        if not self.enabled:
            return
        if not self._slice_targets:
            self._logger.warning(
                "%s enabled but no slice PRB targets configured; disabling.", self.xapp_id
            )
            self.enabled = False
            return
        self._logger.info(
            "%s enabled. Constant slice PRBs: %s", self.xapp_id, self._slice_targets
        )

    def step(self):
        if not self.enabled:
            return
        self._step += 1
        for cell in self.cell_list.values():
            self._apply_targets(cell)
        if self._step % self._log_interval == 0:
            self._log_slice_metrics()

    # ------------------------------------------------------------------ Helpers
    def _apply_targets(self, cell):
        quotas = dict(getattr(cell, "slice_dl_prb_quota", {}) or {})
        for sl, target in self._slice_targets.items():
            quotas[sl] = int(max(0, target))
        max_prb = int(getattr(cell, "max_dl_prb", 0) or 0)
        total = sum(quotas.values())
        if max_prb > 0 and total > max_prb:
            scale = max_prb / float(max(1, total))
            for sl in quotas:
                quotas[sl] = int(quotas[sl] * scale)
        cell.slice_dl_prb_quota = quotas

    def _log_slice_metrics(self):
        for cell in self.cell_list.values():
            data = self._aggregate_metrics(cell)
            for sl, stats in data.items():
                prb = stats["slice_prb"]
                tx_mbps = stats["tx_mbps"]
                per_prb_mbps = tx_mbps / max(1.0, prb)
                bytes_per_prb = (per_prb_mbps * 1e6) / 8.0
                self._logger.info(
                    "%s slice=%s prb=%d tx=%.3f Mbps (%.3f Mbps/PRB, %.1f kB/s/PRB) "
                    "buf=%.1f kB latency=%.3f ms demand=%.1f granted=%.1f sat=%.2f",
                    cell.cell_id,
                    sl,
                    prb,
                    tx_mbps,
                    per_prb_mbps,
                    bytes_per_prb / 1024.0,
                    stats["buf_bytes"] / 1024.0,
                    stats["latency"] * 1e3,
                    stats["prb_req"],
                    stats["prb_granted"],
                    stats["satisfaction"],
                )

    def _aggregate_metrics(self, cell):
        quota_map = dict(getattr(cell, "slice_dl_prb_quota", {}) or {})
        agg = {
            sl: {
                "slice_prb": float(quota_map.get(sl, 0.0)),
                "tx_mbps": 0.0,
                "buf_bytes": 0.0,
                "latency_sum": 0.0,
                "latency_count": 0,
                "prb_req": 0.0,
                "prb_granted": 0.0,
                "satisfaction": 0.0,
            }
            for sl in self._slice_targets.keys()
        }
        req_map = getattr(cell, "dl_total_prb_demand", {}) or {}
        alloc_map = getattr(cell, "prb_ue_allocation_dict", {}) or {}
        for ue in getattr(cell, "connected_ue_list", {}).values():
            sl = getattr(ue, "slice_type", None)
            if sl not in agg:
                continue
            agg[sl]["tx_mbps"] += float(getattr(ue, "served_downlink_bitrate", 0.0) or 0.0) / 1e6
            agg[sl]["buf_bytes"] += float(getattr(ue, "dl_buffer_bytes", 0.0) or 0.0)
            agg[sl]["latency_sum"] += float(getattr(ue, "downlink_latency", 0.0) or 0.0)
            agg[sl]["latency_count"] += 1
            imsi = getattr(ue, "ue_imsi", None)
            if imsi:
                agg[sl]["prb_req"] += float(req_map.get(imsi, 0.0) or 0.0)
                alloc = alloc_map.get(imsi, {}) or {}
                agg[sl]["prb_granted"] += float(alloc.get("downlink", 0.0) or 0.0)
        for sl, stats in agg.items():
            cnt = max(1, stats["latency_count"])
            stats["latency"] = (stats["latency_sum"] / cnt) if cnt > 0 else 0.0
            demand = max(0.0, stats["prb_req"])
            granted = max(0.0, stats["prb_granted"])
            denom = demand if demand > 0 else max(1.0, stats["slice_prb"])
            stats["satisfaction"] = min(1.0, granted / max(1.0, denom))
        return agg
