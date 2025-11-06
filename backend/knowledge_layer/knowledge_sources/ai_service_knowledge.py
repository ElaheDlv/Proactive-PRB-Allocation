import json
from ..knowledge_entry import knowledge_entry
from ..tags import KnowledgeTag
import os
from utils import bytes_pretty_printer

AI_SERVICE_DATA_FILE_PATH = os.path.join(
    os.path.dirname(__file__), "./cranfield_ai_services__1748783109.2438045.json"
)
with open(AI_SERVICE_DATA_FILE_PATH, "r") as file:
    cranfield_ai_services_data = json.load(file)


IMAGE_REPO_URL_MASK = "docker.io/cranfield6g/cranfield-edge-"
AI_SERVICE_TASK_MAP = {}
AI_SERVICE_NAME_MAP = {}


# this is a temporary fix to handle the YOLOv8 series
def ai_service_name_mapper(image_repository_url: str) -> str:
    # docker.io/cranfield6g/cranfield-edge-ultralytics-yolov8-yolov8s -> ultralytics-yolov8-yolov8s
    return image_repository_url.replace(IMAGE_REPO_URL_MASK, "")


def ai_service_url_mapper(service_name: str):
    # ultralytics-yolov8-yolov8s -> docker.io/cranfield6g/cranfield-edge-ultralytics-yolov8-yolov8s
    return IMAGE_REPO_URL_MASK + service_name


for ai_service_data in cranfield_ai_services_data:
    task = ai_service_data["task"]
    image_repository_url = ai_service_data["image_repository_url"]

    if task not in AI_SERVICE_TASK_MAP:
        AI_SERVICE_TASK_MAP[task] = []

    AI_SERVICE_TASK_MAP[task].append(ai_service_data)
    AI_SERVICE_NAME_MAP[ai_service_name_mapper(image_repository_url)] = ai_service_data


@knowledge_entry(
    key="/docs/ai_services",
    tags=[KnowledgeTag.AI_SERVICE, KnowledgeTag.KNOWLEDGE_GUIDE],
    related=[],
)
def ai_services_help(sim, knowledge_router, query_key, params):
    return (
        "📘 **Welcome to the AI Service Knowledge Base**\n\n"
        "You can query all the AI serivces ready-for-deployment across the network here.\n\n"
        "### Available Endpoints:\n"
        "- **Overview available AI tasks and services**: `/ai_services`\n"
        # "- **Get overview of AI services for a specific task**: `/ai_services?task={task_name}`\n"
        "- **Get details of a specific AI service**: `/ai_services/{ai_service_name}`\n\n"
        # "### Supported AI Tasks:\n"
        # f"{'\n '.join(AI_SERVICE_TASK_MAP.keys())}\n\n"
        # "### Available AI services:\n"
        # f"{'\n '.join([ai_service_name_mapper(service['image_repository_url']) for service in cranfield_ai_services_data])}\n\n"
    )


@knowledge_entry(
    key="/ai_services",
    tags=[KnowledgeTag.AI_SERVICE, KnowledgeTag.KNOWLEDGE_GUIDE],
    related=[],
)
def ai_services_overview(sim, knowledge_router, query_key, params):
    response = f"There are in total {len(cranfield_ai_services_data)} AI services covering {len(AI_SERVICE_TASK_MAP)} different tasks:"

    for task, ai_services in AI_SERVICE_TASK_MAP.items():
        response += f"""\n### AI services for task "{task}":\n"""
        for ai_service in ai_services:
            response += (
                f"- {ai_service_name_mapper(ai_service['image_repository_url'])}\n"
            )

    return response


@knowledge_entry(
    key="/ai_services/{ai_service_name}",
    tags=[KnowledgeTag.AI_SERVICE],
    related=[],
)
def ai_service_detail(sim, knowledge_router, query_key, params):
    ai_service_name = params["ai_service_name"]

    if not ai_service_name or ai_service_name.strip() == "":
        return "Please provide a valid AI service name."

    ai_service_name = ai_service_name.strip()

    if ai_service_name not in AI_SERVICE_NAME_MAP:
        return f"AI service {ai_service_name} is currently not supported. Please check the supported services by querying /ai_services"

    ai_service = AI_SERVICE_NAME_MAP[ai_service_name]

    response = f"""Service Details
AI Service Name:\t{ai_service_name_mapper(ai_service['image_repository_url'])}
Service Image URL:\t{ai_service['image_repository_url']}
Service Image Size:\t{bytes_pretty_printer(ai_service["service_disk_size_bytes"])}

{ai_service['task_detail']}
"""
    return response


# Below entries are for internal use only as their output are can take any format instead of agent-friendly strings/paragraphs


@knowledge_entry(
    key="/ai_services/{ai_service_name}/raw",
    tags=[KnowledgeTag.AI_SERVICE],
    related=[],
)
def ai_service_raw(sim, knowledge_router, query_key, params):
    ai_service_name = params["ai_service_name"]

    if not ai_service_name or ai_service_name.strip() == "":
        return None

    ai_service_name = ai_service_name.strip()

    if ai_service_name not in AI_SERVICE_NAME_MAP:
        return None

    return AI_SERVICE_NAME_MAP[ai_service_name].copy()
