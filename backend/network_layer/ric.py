import importlib
import pkgutil
import inspect
import os
from network_layer.xApps.xapp_base import xAppBase
from .ai_service_subscription_manager import AIServiceSubscriptionManager

import logging

logger = logging.getLogger(__name__)


class RIC:
    # Ran Intelligent Controller
    def __init__(self, simulation_engine=None):
        self.ric_id = "RIC"
        self.simulation_engine = simulation_engine
        self.xapp_list = {}
        self.ai_service_subscription_manager = AIServiceSubscriptionManager(ric=self)

    @property
    def base_station_list(self):
        return self.simulation_engine.base_station_list

    @property
    def cell_list(self):
        return self.simulation_engine.cell_list

    @property
    def ue_list(self):
        return self.simulation_engine.ue_list

    def load_xApps(self):
        # dynamically load xApps from the xApp directory
        self.xapp_list = {}

        xapp_classes = []
        xapps_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "xApps"))
        for _, module_name, is_pkg in pkgutil.iter_modules([xapps_dir]):
            if is_pkg or module_name == "xapp_base":
                continue  # skip subpackages and base class

            module = importlib.import_module(f"network_layer.xApps.{module_name}")

            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, xAppBase) and obj is not xAppBase:
                    xapp_classes.append(obj)

        for xapp_cls in xapp_classes:
            xapp_instance = xapp_cls(ric=self)
            assert (
                xapp_instance.xapp_id not in self.xapp_list
            ), f"xApp {xapp_instance.xapp_id} already exists"
            self.xapp_list[xapp_instance.xapp_id] = xapp_instance
            logger.info(f"RIC: Loaded xApp: {xapp_instance.xapp_id}")

        for xapp in self.xapp_list.values():
            xapp.start()

    def step(self, delta_time):
        # Step through all xApps
        for xapp in self.xapp_list.values():
            xapp.step()

        # Step through AI service subscription manager
        self.ai_service_subscription_manager.step()

    def to_json(self):
        return {
            "ric_id": self.ric_id,
            "xapp_list": [xapp.to_json() for xapp in self.xapp_list.values()],
            "ai_service_subscription_manager": self.ai_service_subscription_manager.to_json(),
        }
