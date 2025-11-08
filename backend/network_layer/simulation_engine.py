import asyncio
import json
import random

from utils import (
    get_random_ue_operational_region,
    load_raw_packet_csv,
)
from .core_network import CoreNetwork
from .base_station import BaseStation
from .cell import Cell
from .ric import RIC
from .ue import UE
import settings
import utils
import logging

logger = logging.getLogger(__name__)


class SimulationEngine(metaclass=utils.SingletonMeta):
    def __init__(self):
        self.websocket = utils.WebSocketSingleton().get_websocket()
        self.core_network = None
        self.ric = None

        self.base_station_list = {}
        self.cell_list = {}
        self.ue_list = {}

        self.sim_started = False
        self.sim_step = 0

        self.logs = []
        # cache loaded traces by file path
        self._trace_cache = {}
        # Track UEs that still need a trace attachment (e.g., awaiting slice info)
        self._pending_trace_attach = set()

    def add_base_station(self, bs):
        assert isinstance(bs, BaseStation)
        assert bs.simulation_engine == self
        assert bs.bs_id is not None
        assert bs.bs_id not in self.base_station_list
        self.base_station_list[bs.bs_id] = bs
        for cell in bs.cell_list.values():
            self.add_cell(cell)

    def add_cell(self, cell):
        assert isinstance(cell, Cell)
        assert cell.cell_id is not None
        assert cell.cell_id not in self.cell_list
        self.cell_list[cell.cell_id] = cell

    def reset_network(self):
        logger.info("Resetting network...")
        self.base_station_list = {}
        self.cell_list = {}
        self.ue_list = {}
        self.global_UE_counter = 0
        self.sim_started = False
        self.sim_step = 0
        self.logs = []
        self.core_network = None
        self.ric = None
        self._pending_trace_attach.clear()
        logger.info("Network reset complete.")

    def network_setup(self):
        self.core_network = CoreNetwork(self)

        # init base station list
        for bs_init_data in settings.RAN_DEFAULT_BS_LIST:
            assert (
                bs_init_data["bs_id"] not in self.base_station_list
            ), f"Base station ID {bs_init_data["bs_id"]} already exists"
            self.add_base_station(
                BaseStation(
                    simulation_engine=self,
                    bs_init_data=bs_init_data,
                )
            )

        # for the moment, the ric must be initialized after the core network and the base stations.
        # so that the xApps can subscribe information from the base stations.
        self.ric = RIC(self)
        self.ric.load_xApps()

    def spawn_random_ue(self):
        ue_operation_region = get_random_ue_operational_region()

        position_x = random.randint(
            ue_operation_region["min_x"], ue_operation_region["max_x"]
        )
        position_y = random.randint(
            ue_operation_region["min_y"], ue_operation_region["max_y"]
        )
        target_x = random.randint(
            ue_operation_region["min_x"], ue_operation_region["max_x"]
        )
        target_y = random.randint(
            ue_operation_region["min_y"], ue_operation_region["max_y"]
        )
        speed_mps = random.randint(settings.UE_speed_mps_MIN, settings.UE_speed_mps_MAX)

        # get the next available UE IMSI
        new_ue_IMSI = None
        for i in range(settings.UE_DEFAULT_MAX_COUNT):
            ue_imsi = f"IMSI_{i}"
            if ue_imsi not in self.ue_list:
                new_ue_IMSI = ue_imsi
                break

        if new_ue_IMSI is None:
            logger.error("No available IMSI for new UE. Cannot spawn UE.")
            return None

        ue = UE(
            ue_imsi=new_ue_IMSI,
            operation_region=ue_operation_region,
            position_x=position_x,
            position_y=position_y,
            target_x=target_x,
            target_y=target_y,
            speed_mps=speed_mps,
            simulation_engine=self,
        )
        if getattr(settings, "SIM_FREEZE_MOBILITY", False):
            ue.speed_mps = 0
            ue.set_target(ue.position_x, ue.position_y)
        if not ue.power_up():
            logger.error(
                f"UE {ue.ue_imsi} power up procedures failed. Cannot register UE."
            )
            return None
        self.add_ue(ue)
        logger.info(f"UE {ue.ue_imsi} registered to network: {repr(ue)}")
        return ue

    def add_ue(self, ue):
        assert isinstance(ue, UE)
        assert ue.ue_imsi is not None
        assert ue.ue_imsi not in self.ue_list
        self.ue_list[ue.ue_imsi] = ue
        self.global_UE_counter += 1
        # Attach trace if mapped for this IMSI
        if not self._attach_trace_if_mapped(ue):
            self._pending_trace_attach.add(ue.ue_imsi)

    def _attach_trace_if_mapped(self, ue: UE) -> bool:
        """Attach a configured raw CSV trace to the given UE, if present.

        Flow:
        1) Look up this UE's IMSI inside settings.TRACE_RAW_MAP (provided via
           --trace-raw-map). Each entry is a dict {imsi, file, ue_ip}.
        2) If found, compute a cache key and load/aggregate the CSV into
           (t, dl_bytes, ul_bytes) samples using utils.load_raw_packet_csv.
           Results are cached per (file, ue_ip, bin).
        3) If samples exist, attach them to the UE with the configured
           TRACE_SPEEDUP so the UE starts replaying on subsequent steps.
        """
        # Prefer raw packet mapping if present for this IMSI
        raw_map = getattr(settings, "TRACE_RAW_MAP", []) or []  # list of dicts
        raw_entry = None
        # Resolution priority: exact IMSI > slice mapping > wildcard ALL
        ue_imsi = (ue.ue_imsi or "").strip()
        for item in raw_map:
            if not isinstance(item, dict):
                continue
            imsi_key = (item.get("imsi") or "").strip()
            if imsi_key and imsi_key == ue_imsi:
                raw_entry = item
                break
        if raw_entry is None:
            # Try slice-based rule
            ue_slice = (getattr(ue, "slice_type", None) or "").strip()
            for item in raw_map:
                if not isinstance(item, dict):
                    continue
                sl = (item.get("slice") or "").strip()
                if sl and ue_slice and sl.lower() == ue_slice.lower():
                    raw_entry = item
                    break
        if raw_entry is None:
            # Try wildcard rule
            for item in raw_map:
                if not isinstance(item, dict):
                    continue
                imsi_key = (item.get("imsi") or "").strip().upper()
                if imsi_key in ("*", "ALL"):
                    raw_entry = item
                    break

        if raw_entry:
            # Extract mapping fields and playback/aggregation settings
            path = raw_entry.get("file")             # CSV path on disk
            ue_ip = raw_entry.get("ue_ip")            # UE IP to classify DL/UL
            bin_s = getattr(settings, "TRACE_BIN", 1.0)  # aggregation bin (s)
            overhead = getattr(settings, "TRACE_OVERHEAD_BYTES", 0)  # per-packet subtraction
            if not path:
                return False  # nothing to load

            # Cache key so repeated UEs/files don't re-parse the CSV
            cache_key = ("raw", path, ue_ip or "AUTO", float(bin_s))
            if cache_key not in self._trace_cache:
                # Load and aggregate: returns List[(t_s, dl_bytes, ul_bytes)]
                try:
                    samples = load_raw_packet_csv(
                        path, ue_ip=ue_ip, bin_s=float(bin_s), overhead_sub_bytes=int(overhead)
                    )
                except Exception as e:
                    logger.error(f"Failed to load RAW trace for {ue.ue_imsi} from {path}: {e}")
                    samples = []
                # Store in cache (may be empty if parsing failed)
                self._trace_cache[cache_key] = samples
            else:
                # Reuse cached samples for the same (file, ue_ip, bin)
                samples = self._trace_cache[cache_key]

            # If we have any samples, attach DL replay to BS (preferred)
            if samples:
                speed = getattr(settings, "TRACE_SPEEDUP", 1.0)
                bs = ue.current_bs
                try:
                    if bs and hasattr(bs, "attach_dl_trace"):
                        bs.attach_dl_trace(ue.ue_imsi, samples, speed)
                        logger.info(
                            f"Attached RAW DL trace (BS) to {ue.ue_imsi} from {path} (samples={len(samples)}, bin={bin_s}s, speedup={speed}, ue_ip={ue_ip or 'AUTO'})"
                        )
                    else:
                        ue.attach_trace(samples, speed)
                        logger.info(
                            f"Attached RAW trace (UE) to {ue.ue_imsi} from {path} (samples={len(samples)}, bin={bin_s}s, speedup={speed}, ue_ip={ue_ip or 'AUTO'})"
                        )
                except Exception as e:
                    logger.error(f"Failed to attach DL trace for {ue.ue_imsi}: {e}")
                return True
        return False  # no mapping or failed attachment

        # Aggregated CSV mapping is disabled in this configuration; prefer RAW traces only
        return

    def spawn_UEs(self):
        # Avoid background spawns during Gym-driven episodes or before init completes.
        if getattr(settings, "PRB_GYM_ENABLE", False):
            return
        if self.sim_step <= 0:
            return
        current_ue_count = len(self.ue_list.keys())
        logger.info(f"Current UE count: {current_ue_count}")
        if current_ue_count >= settings.UE_DEFAULT_MAX_COUNT:
            # the reason why current UE count can be larger than the maximum count is that
            # new UEs can be spawned by the user chat agents.
            logger.info(
                "UE count reached the maximum limit. No more UEs will be spawned."
            )
            return

        number_of_UEs_to_spawn = random.randint(
            settings.UE_DEFAULT_SPAWN_RATE_MIN,
            settings.UE_DEFAULT_SPAWN_RATE_MAX,
        )
        number_of_UEs_to_spawn = min(
            number_of_UEs_to_spawn, settings.UE_DEFAULT_MAX_COUNT - current_ue_count
        )
        logger.info(f"Spawning {number_of_UEs_to_spawn} UEs:")
        num_us_spawned = 0
        while num_us_spawned < number_of_UEs_to_spawn:
            ue = self.spawn_random_ue()
            if ue is None:
                continue
            num_us_spawned += 1

    def step_UEs(self, delta_time):
        ue_to_remove = []
        for ue in self.ue_list.values():
            ue.step(delta_time)
            if ue.target_reached:
                logger.info(
                    f"UE {ue.ue_imsi} reached target: ({ue.target_x}, {ue.target_y})"
                )
                # assign a new target for the UE
                target_x = random.randint(
                    ue.operation_region["min_x"],
                    ue.operation_region["max_x"],
                )
                target_y = random.randint(
                    ue.operation_region["min_y"],
                    ue.operation_region["max_y"],
                )
                ue.set_target(target_x, target_y)

            if not ue.connected:
                ue_to_remove.append(ue)

        for ue in ue_to_remove:
            self.remove_UE(ue)
            logger.info(f"UE {ue.ue_imsi} deregistered and removed from simulation.")

    def remove_UE(self, ue):
        assert isinstance(ue, UE)
        assert ue.ue_imsi in self.ue_list
        del self.ue_list[ue.ue_imsi]
        logger.info(f"UE {ue.ue_imsi} deregistered and removed from simulation.")

    def deregister_ue(self, ue_imsi):
        """
        Deregisters and removes the UE from the simulation and all internal references.
        """
        ue = self.ue_list.get(ue_imsi)
        if not ue:
            logger.warning(f"UE {ue_imsi} not found in simulation.")
            return False
        # Deregister from CoreNetwork
        if self.core_network:
            self.core_network.handle_deregistration_request(ue)
        # Remove from SimulationEngine's list
        del self.ue_list[ue_imsi]
        logger.info(f"UE {ue_imsi} deregistered and fully removed from simulation.")
        return True

    def register_ue(self, ue_imsi, subscribed_slices, register_slice=None):
        """
        Register a new UE with IMSI and slice subscription list.
        Optionally pick 'register_slice' for initial attachment, otherwise use the first in list.
        Returns True if successful, False otherwise.
        """
        if self.core_network is None:
            logger.error("Core network not initialized.")
            return False
        if ue_imsi in self.ue_list:
            logger.warning(
                f"UE {ue_imsi} already present in simulation. Cannot register again."
            )
            return False
        if not isinstance(subscribed_slices, list) or not subscribed_slices:
            logger.error(
                f"Subscribed_slices for UE {ue_imsi} is not a valid list: {subscribed_slices}"
            )
            return False
        # Update core network with slice subscription
        self.core_network.ue_subscription_data[ue_imsi] = subscribed_slices
        # Validate register_slice
        attach_slice = register_slice if register_slice else subscribed_slices[0]
        if attach_slice not in subscribed_slices:
            logger.error(
                f"Selected register_slice '{attach_slice}' is not in subscription list for UE {ue_imsi}."
            )
            return False
        # Generate parameters for new UE
        op_region = {"min_x": 0, "min_y": 0, "max_x": 2000, "max_y": 2000}
        pos_x = random.randint(op_region["min_x"], op_region["max_x"])
        pos_y = random.randint(op_region["min_y"], op_region["max_y"])
        target_x = random.randint(op_region["min_x"], op_region["max_x"])
        target_y = random.randint(op_region["min_y"], op_region["max_y"])
        speed_mps = random.randint(settings.UE_speed_mps_MIN, settings.UE_speed_mps_MAX)
        from .ue import UE

        ue = UE(
            ue_imsi=ue_imsi,
            operation_region=op_region,
            position_x=pos_x,
            position_y=pos_y,
            target_x=target_x,
            target_y=target_y,
            speed_mps=speed_mps,
            simulation_engine=self,
        )
        if getattr(settings, "SIM_FREEZE_MOBILITY", False):
            ue.speed_mps = 0
            ue.set_target(ue.position_x, ue.position_y)
        # Power up (attach+auth registration via core network)
        powered = ue.power_up()
        if powered:
            # Attach/authenticate with the chosen slice
            self.core_network.handle_ue_authentication_and_registration(
                ue, requested_slice=attach_slice
            )
            self.ue_list[ue_imsi] = ue
            # Attach trace if mapped (or queue for retry)
            if not self._attach_trace_if_mapped(ue):
                self._pending_trace_attach.add(ue_imsi)
            logger.info(
                f"UE {ue_imsi} added and registered at runtime. Subscribed to slices: {subscribed_slices}. Registered on: {attach_slice}"
            )
            return True
        else:
            logger.error(f"Failed to register UE {ue_imsi} at runtime.")
            return False

    def step_BSs(self, delta_time):
        for bs in self.base_station_list.values():
            bs.step(delta_time)

    def step_ric(self, delta_time):
        if self.ric is not None:
            self.ric.step(delta_time)
        else:
            logger.warning("RIC is not initialized. Skipping RIC step.")

    def step(self, delta_time):
        logger.info(
            f"Simulation step {self.sim_step} started with delta_time {delta_time} seconds."
        )

        self.logs = []

        # spawn new UEs if needed
        logger.info("Spawning new UEs if needed...")
        self.spawn_UEs()

        # Retry pending trace attachments (e.g., slice-based rules once slice is known)
        for imsi in list(self._pending_trace_attach):
            ue = self.ue_list.get(imsi)
            if ue is None:
                self._pending_trace_attach.discard(imsi)
                continue
            if self._attach_trace_if_mapped(ue):
                self._pending_trace_attach.discard(imsi)

        # move UEs towards their targets, monitor signal quality, report measurement events ...
        logger.info("Stepping through UEs...")
        self.step_UEs(delta_time)

        # dynamically allocate resources for UEs
        logger.info("Stepping through Base Stations...")
        self.step_BSs(delta_time)
        
        logger.info("Stepping through RIC...")
        self.step_ric(delta_time)

    async def start_simulation(self):
        assert not self.sim_started
        self.sim_step = 0
        self.sim_started = True

        while self.sim_started and self.sim_step < settings.SIM_MAX_STEP:
            #print(f"\n========= TIME STEP: {self.sim_step} ==========\n")
            self.sim_step += 1
            self.step(settings.SIM_STEP_TIME_DEFAULT)
            if self.websocket is None:
                print("No websocket connection. Cannot send simulation state updates.")
                continue
            await self.websocket.send(
                json.dumps(
                    {
                        "layer": "network_layer",
                        "command": "simulation_state_update",
                        "response": self.to_json(),
                        "error": None,
                    }
                )
            )
            await asyncio.sleep(settings.SIM_STEP_TIME_DEFAULT)

        print("simulation ended")

    def stop(self):
        self.sim_started = False
        self.logs.append("Simulation stopped")
        print("Simulation stopped")

    def to_json(self):
        return {
            "time_step": self.sim_step,
            "base_stations": [bs.to_json() for bs in self.base_station_list.values()],
            "cells": [cell.to_json() for cell in self.cell_list.values()],
            "ric": self.ric.to_json() if self.ric else None,
            "UE_list": [ue.to_json() for ue in self.ue_list.values()],
            "logs": self.logs,
        }
