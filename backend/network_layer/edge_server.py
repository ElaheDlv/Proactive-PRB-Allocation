from utils import (
    send_post_request,
    parse_memory_usage_string,
    start_ai_service_in_docker,
    remove_ai_service_in_docker,
)
from settings import AI_SERVICE_UNDEPLOYMENT_COUNT_DOWN_STEPS
import logging

logger = logging.getLogger(__name__)


class EdgeServer:
    def __init__(self, base_station, edge_server_init_data):
        self.base_station = base_station
        self.edge_id = base_station.bs_id + "_edge"
        self.node_id = edge_server_init_data.get("node_id", "default_edge_server_node")
        self.device_type = edge_server_init_data.get("device_type", "DeviceType.CPU")
        self.cpu_memory_GB = edge_server_init_data.get("cpu_memory_GB", 10.0)
        self.device_memory_GB = edge_server_init_data.get("device_memory_GB", 0.0)

        self.ai_service_deployments = {}

    def to_json(self):
        return {
            "edge_id": self.edge_id,
            "node_id": self.node_id,
            "device_type": self.device_type,
            "cpu_memory_GB": self.cpu_memory_GB,
            "device_memory_GB": self.device_memory_GB,
            "ai_service_deployments": {
                sub_id: {
                    "ai_service_subscription": deployment[
                        "ai_service_subscription"
                    ].subscription_id,
                    "ai_service_name": deployment[
                        "ai_service_subscription"
                    ].ai_service_name,
                    "ai_service_endpoint": deployment["ai_service_endpoint"],
                    "image_repository_url": deployment["image_repository_url"],
                    "container_name": deployment["container_name"],
                    "edge_specific_cpu_memory_usage_GB": deployment[
                        "edge_specific_cpu_memory_usage_GB"
                    ],
                    "edge_specific_device_memory_usage_GB": deployment[
                        "edge_specific_device_memory_usage_GB"
                    ],
                    "countdown_steps": deployment["countdown_steps"],
                    "ue_id_list": deployment["ai_service_subscription"].ue_id_list,
                }
                for sub_id, deployment in self.ai_service_deployments.items()
            },
            "available_cpu_memory_GB": self.available_cpu_memory_GB,
            "available_device_memory_GB": self.available_device_memory_GB,
        }

    @property
    def available_cpu_memory_GB(self):
        cpu_memory_used_GB = 0
        # below code is used when you have real multiple edge servers
        # for deployed_ai_service in self.ai_service_deployments.values():
        #     cpu_memory_used_GB += deployed_ai_service.get(
        #         "edge_specific_cpu_memory_usage_GB", 0.0
        #     )

        # below code is used when you have a single edge server that's shared by all base stations
        for (
            base_station
        ) in self.base_station.simulation_engine.base_station_list.values():
            if base_station.edge_server is not None:
                for (
                    deployed_ai_service
                ) in base_station.edge_server.ai_service_deployments.values():
                    cpu_memory_used_GB += deployed_ai_service.get(
                        "edge_specific_cpu_memory_usage_GB", 0.0
                    )
        return self.cpu_memory_GB - cpu_memory_used_GB

    @property
    def available_device_memory_GB(self):
        device_memory_used_GB = 0
        # below code is used when you have real multiple edge servers
        # for deployed_ai_service in self.ai_service_deployments.values():
        #     device_memory_used_GB += deployed_ai_service.get(
        #         "edge_specific_device_memory_usage_GB", 0.0
        #     )

        # below code is used when you have a single edge server that's shared by all base stations
        for (
            base_station
        ) in self.base_station.simulation_engine.base_station_list.values():
            if base_station.edge_server is not None:
                for (
                    deployed_ai_service
                ) in base_station.edge_server.ai_service_deployments.values():
                    device_memory_used_GB += deployed_ai_service.get(
                        "edge_specific_device_memory_usage_GB", 0.0
                    )

        return self.device_memory_GB - device_memory_used_GB

    def format_container_name(self, ai_service_subscription):
        """
        Format the container name for the AI service deployment.

        Args:
            ai_service_subscription (AIServiceSubscription): The AI service subscription object.

        Returns:
            str: The formatted container name.
        """
        return f"{self.edge_id}_{ai_service_subscription.subscription_id}_{ai_service_subscription.ai_service_name.replace(' ', '_')}"

    def create_ai_service_deployment(self, ai_service_subscription):
        """
        Create an AI service deployment for the given AI service subscription.
        AI service is per subscription basis for the moment, until in the future we support multiple deployments per subscription to enable load balancing.

        returns:
            error (str): Error message if any, otherwise None.
            ai_service_deployment (dict): The AI service deployment data if created or found, otherwise None.
        """

        if ai_service_subscription.subscription_id in self.ai_service_deployments:
            return (
                f"AI service subscription {ai_service_subscription.subscription_id} already exists on edge server {self.edge_id}.",
                self.ai_service_deployments[ai_service_subscription.subscription_id],
            )

        subscription_id = ai_service_subscription.subscription_id
        ai_service_name = ai_service_subscription.ai_service_name

        # check if the edge server has enough resources to deploy the AI serivce
        edge_specific_profile = None
        for profile in ai_service_subscription.ai_service_data["profiles"]:
            if profile["node_id"] == self.node_id:
                edge_specific_profile = profile
                break

        if edge_specific_profile is None:
            # the AI service has not been tested on this edge server yet
            return (
                f"This {ai_service_name} AI service is not compatible with the edge server {self.edge_id}.",
                None,
            )
        edge_specific_cpu_memory_usage_GB = parse_memory_usage_string(
            edge_specific_profile.get("idle_container_cpu_memory_usage")
        )

        edge_specific_device_memory_usage_GB = parse_memory_usage_string(
            edge_specific_profile.get("idle_container_device_memory_usage")
        )

        available_cpu_memory_GB = self.available_cpu_memory_GB
        available_device_memory_GB = self.available_device_memory_GB

        if (
            edge_specific_cpu_memory_usage_GB > available_cpu_memory_GB
            or edge_specific_device_memory_usage_GB > available_device_memory_GB
        ):
            return (
                f"""Not enough resources to deploy {ai_service_name} AI service on the edge server {self.edge_id}.
                Available CPU memory: {available_cpu_memory_GB} GB,
                Available device memory: {available_device_memory_GB} GB,
                Required CPU memory: {edge_specific_cpu_memory_usage_GB} GB,
                Required device memory: {edge_specific_device_memory_usage_GB} GB.""",
                None,
            )

        # deploy the AI service and return the deployment data
        ai_service_data = ai_service_subscription.ai_service_data
        # AI service docker image repository url
        # e.g., docker.io/cranfield6g/cranfield-edge-trpakov-vit-face-expression
        image_repository_url = ai_service_data["image_repository_url"]

        container_name = self.format_container_name(ai_service_subscription)
        error, ai_service_endpoint = start_ai_service_in_docker(
            ai_service_image_url=image_repository_url,
            container_name=container_name,
        )

        if error:
            return (
                f"Failed to start the AI service {ai_service_name} on the edge server {self.edge_id}: {error}",
                None,
            )

        self.ai_service_deployments[subscription_id] = {
            "ai_service_subscription": ai_service_subscription,
            "base_station_id": self.base_station.bs_id,
            "edge_id": self.edge_id,
            "node_id": self.node_id,
            "ai_service_endpoint": ai_service_endpoint,
            "ai_service_data": ai_service_data,
            "image_repository_url": image_repository_url,
            "container_name": self.format_container_name(ai_service_subscription),
            "edge_specific_cpu_memory_usage_GB": edge_specific_cpu_memory_usage_GB,
            "edge_specific_device_memory_usage_GB": edge_specific_device_memory_usage_GB,
            "countdown_steps": AI_SERVICE_UNDEPLOYMENT_COUNT_DOWN_STEPS,
        }

        logger.info(
            f"Deployed AI service {ai_service_name} on edge server {self.edge_id} with endpoint {ai_service_endpoint}."
        )
        return None, self.ai_service_deployments[subscription_id]

    def undeploy_ai_service(self, ai_service_subscription):
        """
        Stop the AI service deployment if it exists for the given AI service subscription.

        Args:
            ai_service_subscription (AIServiceSubscription): The AI service subscription object.
        """
        ai_service_deployment = self.get_ai_service_deployment(ai_service_subscription)
        if ai_service_deployment:
            logger.info(
                f"Undeploying AI service {ai_service_subscription.ai_service_name} for subscription {ai_service_subscription.subscription_id} on edge server {self.edge_id}."
            )
            remove_ai_service_in_docker(
                container_name=ai_service_deployment["container_name"],
            )
            del self.ai_service_deployments[ai_service_subscription.subscription_id]

    def get_ai_service_deployment(self, ai_service_subscription):
        """
        Get the AI service deployment data for the given subscription ID.

        Args:
            ai_service_subscription (AIServiceSubscription): The AI service subscription object.

        Returns:
            dict: The AI service deployment data if found, otherwise None.
        """
        return self.ai_service_deployments.get(
            ai_service_subscription.subscription_id, None
        )

    def check_ue_subscription(self, ai_service_name, ue_imsi):
        for deployment in self.ai_service_deployments.values():
            if (
                deployment["ai_service_subscription"].ai_service_name == ai_service_name
                and ue_imsi in deployment["ai_service_subscription"].ue_id_list
            ):
                return deployment["ai_service_subscription"]
        return None

    def handle_ai_service_request(
        self, ai_service_subscription, request_data, request_files=None
    ):
        """
        Handle the AI service request from a User Equipment (UE).

        Args:
            ai_service_subscription (AIServiceSubscription): The AI service subscription object.
            request_data (dict): The request data to be sent to the AI service.
            request_files (dict, optional): Files to be sent with the request.

        Returns:
            dict: The response from the AI service.
        """
        ai_service_deployment = self.get_ai_service_deployment(ai_service_subscription)
        if ai_service_deployment is None:
            return {
                "response": None,
                "error": f"AI service {ai_service_subscription.ai_service_name} is not deployed on edge server {self.edge_id}.",
            }

        ai_service_endpoint = ai_service_deployment.get("ai_service_endpoint", "")
        if not ai_service_endpoint:
            return {
                "response": None,
                "error": f"AI service {ai_service_subscription.ai_service_name} is not deployed on edge server {self.edge_id}.",
            }

        response, process_time, node_id, k8s_pod_name = send_post_request(
            url=f"http://{ai_service_endpoint}/model/run",
            data=request_data,
            files=request_files or {},
        )

        logger.info(
            f"AI service {ai_service_subscription.ai_service_name} response: {response}, process time: {process_time}, node ID: {node_id}, k8s pod name: {k8s_pod_name}"
        )

        if response is None:
            return {
                "response": None,
                "error": f"Failed to process request for {ai_service_subscription.ai_service_name} on edge server {self.edge_id}.",
            }

        return {
            "error": None,
            "response": response,
            "process_time": process_time,
            "node_id": node_id,
            "k8s_pod_name": k8s_pod_name,
        }
