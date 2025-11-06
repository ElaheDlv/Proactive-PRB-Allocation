import os
import random
import copy
import cv2

AI_SERVICE_UNDEPLOYMENT_COUNT_DOWN_STEPS = 20

AI_SERVICE_SAMPLE_REQUEST_DATA = []
AI_SERVICE_SAMPLE_IMAGE_FILES = ["puppy_in_cup.png", "dog_and_kitten.jpg", "squirrel.png"]


for image_file_name in AI_SERVICE_SAMPLE_IMAGE_FILES:
    image_file_path = os.path.join(
        os.path.dirname(__file__), "..", "assets", image_file_name
    )
    bgr_img = cv2.imread(image_file_path)
    # Convert BGR to RGB before encoding
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    success, buffer = cv2.imencode(".png", rgb_img)

    if not success:
        raise ValueError("Image encoding failed")

    file_bytes = buffer.tobytes()
    AI_SERVICE_SAMPLE_REQUEST_DATA.append(
        {
            "files": {
                "file": file_bytes,
            },
            "size": len(file_bytes),
            "name": image_file_name,
        }
    )


def get_random_ai_service_request_data():
    return copy.deepcopy(random.choice(AI_SERVICE_SAMPLE_REQUEST_DATA))


def prepare_ai_service_sample_request(ai_service_name: str, ue_id: str, files: dict):
    return {
        "url": f"http://cranfield_6G.com/ai_services/{ai_service_name}",
        "data": {
            "ue_id": ue_id,
        },
        "files": files,
    }
