from .xapp_base import xAppBase

import logging


logger = logging.getLogger(__name__)


class xAppAIServiceMonitor(xAppBase):
    """
    xApp that monitors the performance of AI services through AI service traffic across base stations.

    For the moment we store the events in memory. in the future if necessary we can add databases.
    """

    def __init__(self, ric=None):
        super().__init__(ric=ric)
        self.enabled = True

        self.per_ue_memory_size = 20

        self.ai_service_event_memory = {}

    def handle_ai_service_event(self, event):
        #  {
        #             "ue_imsi": ue_imsi,
        #             "request": {
        #                 "ai_service_name": ai_service_name,
        #                 "ue_imsi": ue_imsi,
        #                 "request_data": traffic_data["data"],
        #                 "request_files": files,
        #                 "request_files_size": request_files_size,
        #             },
        #             "response": response,
        #             "service_response_time_ms": end_time - start_time,
        #         }
        # response: {
        #     "error": None,
        #     "response": response,
        #     "process_time": process_time,
        #     "node_id": node_id,
        #     "k8s_pod_name": k8s_pod_name,
        # }
        ue_imsi = event["ue_imsi"]

        if ue_imsi not in self.ai_service_event_memory:
            self.ai_service_event_memory[ue_imsi] = []

        if len(self.ai_service_event_memory[ue_imsi]) >= self.per_ue_memory_size:
            self.ai_service_event_memory[ue_imsi].pop(0)

        self.ai_service_event_memory[ue_imsi].append(event)

        logger.info(
            f"AI service event recorded for UE {ue_imsi}: "
            f"Service: {event['request']['ai_service_name']}, "
            f"Response Time: {event['service_response_time_ms']} ms"
        )

    def start(self):
        if not self.enabled:
            print(f"{self.xapp_id}: xApp is not enabled")
            return

        # subcribe events from all base stations
        for bs in self.base_station_list.values():
            bs.init_ai_service_event_handler(self.handle_ai_service_event)

    def slim_ai_service_event(self, event):
        return {
            "ue_imsi": event["ue_imsi"],
            "request": {
                "ai_service_name": event["request"]["ai_service_name"],
                "request_data": event["request"]["request_data"],
                "request_files_size": event["request"].get("request_files_size", None),
            },
            "response": {
                "error": event["response"]["error"],
                "process_time": event["response"]["process_time"],
                "node_id": event["response"]["node_id"],
                "k8s_pod_name": event["response"]["k8s_pod_name"],
            },
            "service_response_time_ms": event["service_response_time_ms"],
        }

    def to_json(self):
        res = super().to_json()
        res["per_ue_memory_size"] = self.per_ue_memory_size
        res["ai_service_event_memory"] = {
            ue_imsi: [self.slim_ai_service_event(e) for e in events]
            for ue_imsi, events in self.ai_service_event_memory.items()
        }
        return res
