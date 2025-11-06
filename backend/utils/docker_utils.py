import subprocess
import logging
import socket
import requests

logger = logging.getLogger(__name__)


def get_available_port() -> int:
    """Get an available port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def start_ai_service_in_docker(ai_service_image_url: str, container_name: str):
    """Start an AI service using Docker.
    Args:
        ai_service_image_url (str): The URL of the AI service Docker image, e.g., "docker.io/cranfield6g/cranfield-edge-trpakov-vit-face-expression"
        container_name (str): The name of the Docker container to be created, e.g., "cranfield-edge-trpakov-vit-face-expression"

    Returns:
        error (str): Error message if any, otherwise None.
        ai_service_endpoint (str): The URL where the AI service is accessible, e.g., "localhost:8000"
    """
    # ---------------------------------
    # Check if any container of the same name is already running
    # ---------------------------------
    try:
        subprocess.run(
            ["docker", "inspect", container_name],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info(f"Docker container {container_name} already exists.")

        # get the port that's mapped to the 8000 of the existing container
        service_port = (
            subprocess.run(
                ["docker", "port", container_name, "8000"],
                capture_output=True,
                text=True,
                check=True,
            )
            .stdout.strip()
            .split(":")[-1]
        )
        logger.info(
            f"Docker container {container_name} is already running on port {service_port}:8000."
        )
        return None, f"localhost:{service_port}"
    except subprocess.CalledProcessError:
        logger.info(
            f"Docker container {container_name} does not exist. It will be created."
        )

    try:
        # --------------------------------
        # Ensure the Docker image is available
        # ---------------------------------
        image_available_locally = False
        try:
            # Check if the image exists locally; if so, skip pulling
            subprocess.run(
                ["docker", "image", "inspect", ai_service_image_url],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            image_available_locally = True
            logger.info(
                f"Docker image {ai_service_image_url} found locally. Skipping pull."
            )
        except subprocess.CalledProcessError:
            logger.info(
                f"Docker image {ai_service_image_url} not found locally. Attempting to pull..."
            )

        if not image_available_locally:
            logger.info(f"Pulling Docker image {ai_service_image_url} ...")
            subprocess.run(
                ["docker", "pull", ai_service_image_url],
                check=True,
            )
            logger.info(f"Docker image {ai_service_image_url} pulled successfully.")

        # ----------------------------------
        # Save the disk size of the pulled docker image
        # ----------------------------------
        docker_image_size_bytes = subprocess.run(
            [
                "docker",
                "image",
                "inspect",
                ai_service_image_url,
                "--format={{.Size}}",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        logger.info(f"Docker image size: {docker_image_size_bytes} bytes.")

        # --------------------------------
        # Run the docker container
        # ---------------------------------
        available_port = get_available_port()
        cmd = [
            "docker",
            "run",
            "-d",
            "--name",
            container_name,
            "-p",
            f"{available_port}:8000",
            "--health-cmd",
            "python healthcheck.py",
            "--health-interval=5s",
            "--health-timeout=2s",
            "--health-retries=3",
            ai_service_image_url,
        ]
        print(f"Running command: {' '.join(cmd)}")
        subprocess.run(
            cmd,
            check=True,
        )
        print(f"Docker container {container_name} started successfully.")
        print(f"Access the server at http://localhost:{available_port}/run")

        # return f"localhost:{available_port}"
        return None, f"localhost:{available_port}"
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start Docker container {container_name}: {e}")
        return (
            f"Failed to start Docker container {container_name}: {e}",
            None,
        )


def remove_ai_service_in_docker(container_name: str):
    """Stop and delete the Docker container for the AI service.

    Args:
        container_name (str): The name of the Docker container to be removed.
    """
    logger.info(f"Removing Docker container {container_name} ...")
    try:
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            check=True,
        )
        logger.info(f"Docker container {container_name} removed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to remove Docker container {container_name}: {e}")


def send_post_request(url, data, files):
    """Send request to run AI service and display AI service responses."""
    try:
        response = requests.post(url, files=files, data=data)
        # get the process time, node id and k8s pod name from the response headers
        process_time = response.headers.get("X-Process-Time")
        node_id = response.headers.get("X-NODE-ID")
        k8s_pod_name = response.headers.get("X-K8S-POD-NAME")
        if response.status_code == 200:
            return response.json(), process_time, node_id, k8s_pod_name
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return None, None, None, None
    except Exception as e:
        print(f"Request failed: {e}")
        return None, None, None, None
