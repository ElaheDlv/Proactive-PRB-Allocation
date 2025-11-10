import math
import logging
import settings
from utils import dist_between, estimate_throughput


class Cell:
    def __init__(self, base_station, cell_init_data):
        assert base_station is not None, "Base station cannot be None"
        assert cell_init_data is not None, "Cell init data cannot be None"
        self.base_station = base_station

        self.cell_id = cell_init_data["cell_id"]
        self.frequency_band = cell_init_data["frequency_band"]
        self.carrier_frequency_MHz = cell_init_data["carrier_frequency_MHz"]
        self.bandwidth_Hz = cell_init_data["bandwidth_Hz"]
        self.max_prb = cell_init_data["max_prb"]
        self.max_dl_prb = cell_init_data["max_dl_prb"]
        self.max_ul_prb = cell_init_data["max_ul_prb"]
        self.cell_radius = cell_init_data["cell_radius"]
        self.transmit_power_dBm = cell_init_data["transmit_power_dBm"]
        self.cell_individual_offset_dBm = cell_init_data["cell_individual_offset_dBm"]
        self.frequency_priority = cell_init_data["frequency_priority"]
        self.qrx_level_min = cell_init_data["qrx_level_min"]

        self.prb_ue_allocation_dict = {}  # { "ue_imsi": {"downlink": 30, "uplink": 5}}
        self.connected_ue_list = {}
        self.ue_uplink_signal_strength_dict = {}
        # Live control: per-UE PRB cap (None means unlimited)
        self.prb_per_ue_cap = None
        # KPI exposure: per-UE requested DL PRBs in the current allocation round
        # Map: {ue_imsi: required_dl_prbs}
        self.dl_total_prb_demand = {}
        # Static per-slice DL PRB quotas (absolute PRBs per slice)
        # Initialized from settings.RAN_SLICE_DL_PRB_SPLIT_DEFAULT
        self.slice_dl_prb_quota = self._init_slice_quota()
        # Tracking allocated PRBs by slice in the last round
        self.allocated_dl_prb_by_slice = {s: 0 for s in self.slice_dl_prb_quota.keys()}

    def __repr__(self):
        return f"Cell({self.cell_id}, base_station={self.base_station.bs_id}, frequency_band={self.frequency_band}, carrier_frequency_MHz={self.carrier_frequency_MHz})"

    @property
    def allocated_dl_prb(self):
        return sum(
            [
                self.prb_ue_allocation_dict[ue_imsi]["downlink"]
                for ue_imsi in self.connected_ue_list.keys()
            ]
        )

    @property
    def allocated_ul_prb(self):
        return sum(
            [
                self.prb_ue_allocation_dict[ue_imsi]["uplink"]
                for ue_imsi in self.connected_ue_list.keys()
            ]
        )

    @property
    def allocated_prb(self):
        return sum(
            [
                self.prb_ue_allocation_dict[ue_imsi]["uplink"]
                + self.prb_ue_allocation_dict[ue_imsi]["downlink"]
                for ue_imsi in self.connected_ue_list.keys()
            ]
        )

    @property
    def current_load(self):
        return self.allocated_prb / self.max_prb

    @property
    def current_dl_load(self):
        return self.allocated_dl_prb / self.max_dl_prb

    @property
    def current_ul_load(self):
        return self.allocated_ul_prb / self.max_ul_prb

    @property
    def position_x(self):
        return self.base_station.position_x

    @property
    def position_y(self):
        return self.base_station.position_y

    def register_ue(self, ue):
        self.connected_ue_list[ue.ue_imsi] = ue
        self.prb_ue_allocation_dict[ue.ue_imsi] = {
            "downlink": 0,
            "uplink": 0,
        }

    def monitor_ue_signal_strength(self):
        self.ue_uplink_signal_strength_dict = {}
        pass_loss_model = settings.CHANNEL_PASS_LOSS_MODEL_MAP[
            settings.CHANNEL_PASS_LOSS_MODEL_URBAN_MACRO_NLOS
        ]
        # monitor the ue uplink signal strength
        for ue in self.connected_ue_list.values():
            # calculate the received power based on distance and transmit power
            distance = dist_between(
                self.position_x,
                self.position_y,
                ue.position_x,
                ue.position_y,
            )
            received_power = ue.uplink_transmit_power_dBm - pass_loss_model(
                distance_m=distance, frequency_ghz=self.carrier_frequency_MHz / 1000
            )
            self.ue_uplink_signal_strength_dict[ue.ue_imsi] = received_power

    def select_ue_mcs(self):
        for ue in self.connected_ue_list.values():
            ue.set_downlink_mcs_index(-1)
            ue.set_downlink_mcs_data(None)
            ue_cqi_mcs_data = settings.UE_CQI_MCS_SPECTRAL_EFFICIENCY_TABLE.get(
                ue.downlink_cqi, None
            )
            if ue.downlink_cqi == 0 or ue_cqi_mcs_data is None:
                continue

            ue_cqi_eff = ue_cqi_mcs_data["spectral_efficiency"]
            max_mcs_index = 0
            for (
                mcs_index,
                mcs_eff,
            ) in settings.RAN_MCS_SPECTRAL_EFFICIENCY_TABLE.items():
                if mcs_eff["spectral_efficiency"] <= ue_cqi_eff:
                    max_mcs_index = mcs_index
                else:
                    break
            ue.set_downlink_mcs_index(max_mcs_index)
            downlink_mcs_data = settings.RAN_MCS_SPECTRAL_EFFICIENCY_TABLE.get(
                max_mcs_index, None
            )
            if downlink_mcs_data is None:
                ue.set_downlink_mcs_data(None)
            else:
                # copy the dictionary to avoid modifying the original data
                ue.set_downlink_mcs_data(downlink_mcs_data.copy())

    def step(self, delta_time):
        self.monitor_ue_signal_strength()

        # select modulation and coding scheme (MCS) for each UE based on CQI
        self.select_ue_mcs()

        # allocate PRBs dynamically based on each UE's QoS profile and channel conditions
        self.allocate_prb()

        # for each UE, estimate the downlink, uplink bitrate and latency
        self.estimate_ue_bitrate_and_latency()

    def _init_slice_quota(self):
        """Compute initial absolute PRB quotas per slice from defaults in settings.
        Any leftover PRBs (due to rounding) remain unused.
        """
        quotas = {}
        try:
            default_frac = settings.RAN_SLICE_DL_PRB_SPLIT_DEFAULT
        except Exception:
            default_frac = {"eMBB": 1.0, "URLLC": 0.0, "mMTC": 0.0}
        total_assigned = 0
        for s, frac in default_frac.items():
            frac = max(0.0, float(frac))
            cnt = int(self.max_dl_prb * frac)
            quotas[s] = cnt
            total_assigned += cnt
        # Ensure no negative and no overflow before applying overrides
        if total_assigned > self.max_dl_prb and total_assigned > 0:
            scale = self.max_dl_prb / total_assigned
            for s in list(quotas.keys()):
                quotas[s] = int(quotas[s] * scale)
        # Apply absolute overrides if provided
        try:
            overrides = getattr(settings, "RAN_SLICE_DL_PRB_OVERRIDES", {})
        except Exception:
            overrides = {}
        for s, val in overrides.items():
            try:
                quotas[s] = max(0, int(val))
            except Exception:
                continue
        total_assigned = sum(quotas.values())
        if total_assigned > self.max_dl_prb and total_assigned > 0:
            scale = self.max_dl_prb / total_assigned
            for s in list(quotas.keys()):
                quotas[s] = int(quotas[s] * scale)
        return quotas

    def set_slice_quota_by_fraction(self, frac_map):
        """Update slice quotas using fractions, per this cell's max_dl_prb."""
        if not isinstance(frac_map, dict):
            return
        quotas = {}
        total = 0
        for s, f in frac_map.items():
            f = max(0.0, float(f))
            cnt = int(self.max_dl_prb * f)
            quotas[s] = cnt
            total += cnt
        if total > self.max_dl_prb and total > 0:
            scale = self.max_dl_prb / total
            for s in list(quotas.keys()):
                quotas[s] = int(quotas[s] * scale)
        self.slice_dl_prb_quota = quotas

    def adjust_slice_quota_move_rb(self, src_slice: str, dst_slice: str, prb_step: int = 1):
        """Move PRBs from src_slice to dst_slice within this cell.

        prb_step: number of PRBs to move (treating a paper's RB as N PRBs).
        Ensures quotas are non-negative and do not exceed cell.max_dl_prb.
        """
        if prb_step <= 0:
            return False
        if src_slice == dst_slice:
            return False
        # Ensure keys exist
        for s in (src_slice, dst_slice):
            if s not in self.slice_dl_prb_quota:
                self.slice_dl_prb_quota[s] = 0
        move = min(prb_step, self.slice_dl_prb_quota.get(src_slice, 0))
        if move <= 0:
            return False
        # Apply move
        self.slice_dl_prb_quota[src_slice] -= move
        self.slice_dl_prb_quota[dst_slice] = self.slice_dl_prb_quota.get(dst_slice, 0) + move
        # Cap total to max_dl_prb
        total_quota = sum(self.slice_dl_prb_quota.values())
        if total_quota > self.max_dl_prb:
            # Reduce dst back to respect max
            overflow = total_quota - self.max_dl_prb
            self.slice_dl_prb_quota[dst_slice] = max(0, self.slice_dl_prb_quota[dst_slice] - overflow)
        return True

    def allocate_prb(self):
        # QoS-aware Proportional Fair Scheduling (PFS)

        # reset PRB allocation for all UEs
        for ue in self.connected_ue_list.values():
            self.prb_ue_allocation_dict[ue.ue_imsi]["downlink"] = 0
            self.prb_ue_allocation_dict[ue.ue_imsi]["uplink"] = 0

        # sample QoS and channel condition-aware PRB allocation
        ue_prb_requirements = {}
        # reset demand map
        self.dl_total_prb_demand = {}
        # Reset per-slice allocation tracking
        self.allocated_dl_prb_by_slice = {s: 0 for s in self.slice_dl_prb_quota.keys()}

        sim_dt = float(getattr(settings, "SIM_STEP_TIME_DEFAULT", 1) or 1.0)

        # Step 1: Calculate required PRBs using the observed replay backlog.
        # We still honour GBR contracts when sustained demand exists, but avoid
        # allocating excess PRBs when very little traffic is queued.
        for ue in self.connected_ue_list.values():
            dl_gbr = ue.qos_profile["GBR_DL"]
            dl_mcs = ue.downlink_mcs_data  # Assume this attribute exists
            if dl_mcs is None:
                print(
                    f"Cell {self.cell_id}: UE {ue.ue_imsi} has no downlink MCS data. Skipping."
                )
                continue
            dl_throughput_per_prb = estimate_throughput(
                dl_mcs["modulation_order"], dl_mcs["target_code_rate"], 1
            )
            # GBR expressed as PRBs per scheduler step (floor when demand persists).
            baseline_prbs = math.ceil(dl_gbr / dl_throughput_per_prb) if dl_throughput_per_prb > 0 else 0

            # Observe queued trace traffic (prefer gNB replay buffer, fall back to UE mirror).
            backlog_bytes = 0
            if hasattr(self.base_station, "get_dl_buf_bytes"):
                try:
                    backlog_bytes = int(self.base_station.get_dl_buf_bytes(ue.ue_imsi) or 0)
                except Exception:
                    backlog_bytes = 0
            if backlog_bytes <= 0:
                backlog_bytes = int(getattr(ue, "dl_buffer_bytes", 0) or 0)

            # Translate queued bytes into PRBs needed to drain the buffer in this step.
            backlog_prbs = 0
            if backlog_bytes > 0 and dl_throughput_per_prb > 0 and sim_dt > 0:
                backlog_bits = float(backlog_bytes) * 8.0
                required_rate = backlog_bits / sim_dt
                backlog_prbs = math.ceil(required_rate / dl_throughput_per_prb)

            # Only enforce the GBR floor when the queue persists at or above the GBR load.
            gbr_bytes_per_step = (dl_gbr * sim_dt) / 8.0 if dl_gbr > 0 and sim_dt > 0 else 0.0
            enforce_baseline = backlog_bytes >= gbr_bytes_per_step and backlog_prbs > 0
            active_baseline = baseline_prbs if enforce_baseline else 0

            # Resulting demand matches real traffic, with GBR acting as a floor only
            # when sustained load warrants it.
            total_required_prbs = max(backlog_prbs, active_baseline)
            ue_prb_requirements[ue.ue_imsi] = {
                "dl_required_prbs": total_required_prbs,
                "baseline_prbs": baseline_prbs,
                "backlog_prbs": backlog_prbs,
                "dl_throughput_per_prb": dl_throughput_per_prb,
            }
            # Expose requested PRBs for KPI xApps
            self.dl_total_prb_demand[ue.ue_imsi] = total_required_prbs

        # Step 2: Allocate within slice quotas
        # Group UEs by slice
        ues_by_slice = {}
        for ue in self.connected_ue_list.values():
            s = getattr(ue, "slice_type", None)
            if s is None:
                # If slice unknown, treat as eMBB
                s = "eMBB"
            ues_by_slice.setdefault(s, []).append(ue.ue_imsi)

        for s, quota in self.slice_dl_prb_quota.items():
            ue_ids = ues_by_slice.get(s, [])
            if not ue_ids or quota <= 0:
                continue
            demands = {uid: ue_prb_requirements.get(uid, {}).get("dl_required_prbs", 0) for uid in ue_ids}
            total_demand = sum(demands.values())
            # Baseline: guarantee at least 1 PRB to each UE if possible
            remaining = quota
            if remaining >= len(ue_ids):
                for uid in ue_ids:
                    self.prb_ue_allocation_dict[uid]["downlink"] = 1
                remaining -= len(ue_ids)
            else:
                # Not enough quota for all: assign 1 PRB to first N UEs
                for uid in ue_ids[:remaining]:
                    self.prb_ue_allocation_dict[uid]["downlink"] = 1
                self.allocated_dl_prb_by_slice[s] += remaining
                continue

            # Proportional distribution of remaining PRBs according to demand
            if remaining > 0 and total_demand > 0:
                for uid in ue_ids:
                    share = demands[uid] / total_demand if total_demand > 0 else 0
                    add = int(share * remaining)
                    self.prb_ue_allocation_dict[uid]["downlink"] += add
            # Track allocated by slice
            alloc_slice = sum(self.prb_ue_allocation_dict[uid]["downlink"] for uid in ue_ids)
            self.allocated_dl_prb_by_slice[s] = alloc_slice

        # Enforce per-UE live cap if set
        if self.prb_per_ue_cap is not None:
            for ue_imsi in list(self.connected_ue_list.keys()):
                current_alloc = self.prb_ue_allocation_dict[ue_imsi]["downlink"]
                if current_alloc > self.prb_per_ue_cap:
                    self.prb_ue_allocation_dict[ue_imsi]["downlink"] = self.prb_per_ue_cap

        # # Logging
        # for ue_imsi, allocation in self.prb_ue_allocation_dict.items():
        #     print(
        #         f"Cell: {self.cell_id} allocated {allocation['downlink']} DL PRBs for UE {ue_imsi}"
        #     )

    def estimate_ue_bitrate_and_latency(self):
        for ue in self.connected_ue_list.values():
            if ue.downlink_mcs_data is None:
                print(
                    f"Cell {self.cell_id}: UE {ue.ue_imsi} has no downlink MCS data. Skipping."
                )
                continue
            ue_modulation_order = ue.downlink_mcs_data["modulation_order"]
            ue_code_rate = ue.downlink_mcs_data["target_code_rate"]
            ue_dl_prb = self.prb_ue_allocation_dict[ue.ue_imsi]["downlink"]
            # TODO: uplink bitrate
            cap_bps = estimate_throughput(
                ue_modulation_order, ue_code_rate, ue_dl_prb
            )
            # Save achievable capacity (bps)
            ue.achievable_downlink_bitrate = cap_bps
            # If a trace is attached (UE-managed) or BS manages DL replay for this UE, serve from buffer up to capacity
            dt = getattr(settings, "SIM_STEP_TIME_DEFAULT", 1) or 1
            has_replay = (getattr(ue, "_trace_samples", None) is not None) or (
                hasattr(self.base_station, "has_dl_replay") and self.base_station.has_dl_replay(ue.ue_imsi)
            )
            if has_replay:
                # How many bytes can we serve this step given capacity?
                cap_bytes = int((cap_bps * dt) / 8)
                if hasattr(self.base_station, "pull_dl_bytes") and self.base_station.has_dl_replay(ue.ue_imsi):
                    take = self.base_station.pull_dl_bytes(ue.ue_imsi, cap_bytes)
                    # Mirror remaining queue for UI/backcompat
                    ue.dl_buffer_bytes = int(getattr(self.base_station, "get_dl_buf_bytes", lambda x: 0)(ue.ue_imsi))
                # else:  ### This  part might be useful in some cases .But for now we do not need it
                #     take = min(max(0, int(ue.dl_buffer_bytes)), max(0, cap_bytes))
                #     ue.dl_buffer_bytes = max(0, ue.dl_buffer_bytes - take)
                # Compute served bitrate
                served_bps = (take * 8) / dt if dt > 0 else 0
                ue.served_downlink_bitrate = served_bps
                # Trace debug counters
                try:
                    ue._trace_served_dl_last = take
                    ue._trace_served_dl_total += take
                    if getattr(settings, "TRACE_DEBUG", False) and (
                        not getattr(settings, "TRACE_DEBUG_IMSI", set())
                        or ue.ue_imsi in getattr(settings, "TRACE_DEBUG_IMSI", set())
                    ):
                        logging.getLogger("trace_replay").info(
                            f"[trace] {ue.ue_imsi}: served_dl={take}B step_cap={cap_bytes}B buf_after={ue.dl_buffer_bytes}B rate={served_bps/1e6:.3f}Mbps"
                        )
                except Exception:
                    pass
                # Always show actual served traffic derived from the trace/buffer.
                # If no bytes were available to send this step, served_bps=0 and we
                # display 0 — never show radio capacity as a stand‑in.
                ue.set_downlink_bitrate(served_bps)
            else:
                # No trace attached for this UE: do not fabricate throughput.
                ue.served_downlink_bitrate = 0.0
                ue.set_downlink_bitrate(0.0)
            # TODO: downlink and uplink latency
            # After serving downlink data (after buffer update)
            if has_replay:
                # Transmission delay for this step
                #transmission_delay = (take * 8) / served_bps if served_bps > 0 else 0
                #print("Transmission Delay:", transmission_delay)
                transmission_delay = 0.0
                # Queuing delay: remaining buffer to be served. When capacity collapses
                # to zero we still expose a backlog-derived latency using a tiny
                # fallback rate so dashboards/xApps can see the stall instead of 0.
                backlog_bytes = max(0, int(getattr(ue, "dl_buffer_bytes", 0) or 0))
                backlog_bits = backlog_bytes * 8
                if cap_bps > 0:
                    queuing_delay = backlog_bits / cap_bps
                elif backlog_bits > 0:
                    stall_bps = max(
                        1e-9, float(getattr(settings, "TRACE_LATENCY_STALL_BPS", 1.0) or 1.0)
                    )
                    queuing_delay = backlog_bits / stall_bps
                else:
                    queuing_delay = 0.0
                #print("Queuing Delay:", queuing_delay)
                # Total downlink latency
                ue.downlink_latency = transmission_delay + queuing_delay
            else:
                # No trace: assume no queuing, only transmission delay for a nominal packet
                #nominal_packet_size = 1500  # bytes, typical MTU
                #transmission_delay = (nominal_packet_size * 8) / cap_bps if cap_bps > 0 else 0
                #ue.downlink_latency = transmission_delay
                ue.downlink_latency = 0.0

    def deregister_ue(self, ue):
        if ue.ue_imsi in self.prb_ue_allocation_dict:
            del self.prb_ue_allocation_dict[ue.ue_imsi]
            print(f"Cell {self.cell_id}: Released resources for UE {ue.ue_imsi}")
        else:
            print(f"Cell {self.cell_id}: No resources to release for UE {ue.ue_imsi}")

        if ue.ue_imsi in self.connected_ue_list:
            del self.connected_ue_list[ue.ue_imsi]
            print(f"Cell {self.cell_id}: Deregistered UE {ue.ue_imsi}")
        else:
            print(f"Cell {self.cell_id}: No UE {ue.ue_imsi} to deregister")

    def to_json(self):
        return {
            "cell_id": self.cell_id,
            "frequency_band": self.frequency_band,
            "carrier_frequency_MHz": self.carrier_frequency_MHz,
            "bandwidth_Hz": self.bandwidth_Hz,
            "max_prb": self.max_prb,
            "cell_radius": self.cell_radius,
            "vis_cell_radius": self.cell_radius
            * settings.REAL_LIFE_DISTANCE_MULTIPLIER,
            "position_x": self.position_x,
            "position_y": self.position_y,
            "vis_position_x": self.position_x * settings.REAL_LIFE_DISTANCE_MULTIPLIER,
            "vis_position_y": self.position_y * settings.REAL_LIFE_DISTANCE_MULTIPLIER,
            "prb_ue_allocation_dict": self.prb_ue_allocation_dict,
            "max_dl_prb": self.max_dl_prb,
            "max_ul_prb": self.max_ul_prb,
            "allocated_dl_prb": self.allocated_dl_prb,
            "allocated_ul_prb": self.allocated_ul_prb,
            "current_dl_load": self.allocated_dl_prb / self.max_dl_prb,
            "current_ul_load": self.allocated_ul_prb / self.max_ul_prb,
            "current_load": self.current_load,
            "connected_ue_list": list(self.connected_ue_list.keys()),
            "slice_dl_prb_quota": self.slice_dl_prb_quota,
            "allocated_dl_prb_by_slice": self.allocated_dl_prb_by_slice,
        }
