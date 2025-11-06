from functools import cache
from agents import function_tool
from pymongo import MongoClient
from pydantic import BaseModel, Field
from typing import Any
import os

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "cranfield_ai_services_saved"
COLLECTION_NAME = "ai_services"

@cache
def get_mongo_collection():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    return db[COLLECTION_NAME]

@function_tool
def fetch_task_types() -> list[str]:
    """Return all unique task types in the collection."""
    print("LOG: Fetching the task types")
    collection = get_mongo_collection()
    task_types = collection.distinct("task")
    return task_types

@function_tool
def fetch_model_info(task_types: list[str]) -> list[dict[str, Any]]:
    """
    Fetch model documents for the given task_types, with selected fields.
    """
    print(f"LOG: Fetching model_info for the task_types: {task_types}")
    collection = get_mongo_collection()
    query = {"task": {"$in": task_types}}
    projection = {
        "_id": 1,
        "model_name": 1,
        "model_url": 1,
        "task": 1,
        "task_detail": 1,
        "image_repository_url": 1,
        "code.readme_content": 1
    }
    docs = list(collection.find(query, projection))
    for doc in docs:
        doc['id'] = str(doc['_id'])
        del doc['_id']
    return docs

class SuggestedModel(BaseModel):
    id: str = Field(description = "The Id of the model in the database")
    model_name: str =  Field(description = "The name of the model")
    repository_url: str = Field(description = "The repository url of the model")
    rationale: str = Field(description = "detailed justification, referencing relevant readme or metadata for choosing the model")

class SuggestedModelsOutput(BaseModel):
    models: list[SuggestedModel] = Field(description = "The list of suggested models for the user")


class OrchestratorOutput(BaseModel):
    models: None | list[SuggestedModel]
    network_slice: None | str = Field(description= "The network slice that is suggested to the user")
    deployment_location: None | str = Field(description= "The deployment location suggested to the user")
    questions: None | str = Field(description="Clarification questions to the user, to clarify the requirement and to properly recommend attributes")
    message: None | str = Field(description="Any other message to the user, only if the recommendation is not possible")

from agents import Agent

model_suggestion_agent = Agent(
    name="ModelSuggestionAgent",
    instructions="""
You are an expert AI model recommender.

- You will receive: user requirements and final categories as input.
- Use the tool 'fetch_model_info' to get candidate models that fall into the final categories.
- Carefully review each candidate model's fields, especially 'code.readme_content'.
- Select the most suitable model(s) based on a deep understanding of the user's needs and the model capabilities.
- For each recommended model, provide:
  - id (the document id from the database)
  - model_name
  - repository_url (use the 'image_repository_url' field)
  - rationale (provide a detailed justification, referencing the relevant readme content or metadata)

""",
    tools=[fetch_model_info],
    output_type=SuggestedModelsOutput,
)


category_matching_agent = Agent(
    name="category_matching_agent",
    instructions="""
 Your task is to match the user's requirement to the most appropriate category(ies) and model(s) available in the database.

    1. You must use the tool (fetch_task_types) to fetch all the task types, which are the categories of AI/ML models.
    2. Find the most suitable category(ies) that suit the user's requirement.
    3. Don't worry about the specific model; just hand off the task with the suitable category name. The model_suggestion_agent will select the appropriate model from the database.

 If you are not sure, you may ask clarification questions to the user before making a decision.

""",
    tools=[fetch_task_types],
    handoffs= [model_suggestion_agent]
)
model_recommender_orchestrator = Agent(
    name="model_recommender_orchestrator",
    instructions=f"""
    You are a great orchestrator and analyzer. Your goal is to understand the user's requirement and recommend the best option for their needs without using technical terms that a user may not know.

    First part of your task is to understand the requirement to provide proper recommendation, do not use any tool if you have any clarification question and  do not make any recommendations at this stage
    When you or your tools asks for clarification, leave the other fields apart from questions empty.
    Once you understand the user's requirement, proceed with the recommendations, which should include:

    1. The appropriate type of network slice, which could be one among eMBB, URLLC, or mMTC.
    2. The best location for deployment, which could be Cloud or Edge.
    3. To get the most suitable model for the user, you may need to use two tools:
        - Use the tool category_matching_tool to get the most relevant category or categories for the user from the database.
        - At this stage if you are unable to match any suitable category from the tool, let the user know by providing relevant message in the message field
        - Use the tool model_suggestion_tool and pass the category or categories; the tool will provide the most suitable models in JSON format as shown below:

                model_suggestion_tool output format:
                {SuggestedModel.model_json_schema()}

    4. Once you receive the output, you can parse it in JSON format.
    [IMPORTANT: 
    1. Don't keep on asking question, ask quesition to properly understand the requirement and don't exploit this flexibility
    2. Don't reply to the user, unless you have quesitons or you are unable to recommend or you are done with all the recommendations
]
""",
    tools=[category_matching_agent.as_tool(
        tool_name="category_matching_tool",
        tool_description="""
        This tool will get the list of categories from the database and understands the user's requirement and provide matching category[ies] for the user
        Note: This tool can understand natural language and needs clear requirement from the conversation with user to make decision
        """
    ), model_suggestion_agent.as_tool(
        tool_name="model_suggestion_tool",
        tool_description=f"""This tool takes the list of categories as input and provides the most matching models from the database
        Note: This tool can understand natural language and needs clear requirement from the convesation with user to make decision
        If there are list of categories, pass them to me in one shot
        Output: This tool outputs a structured json in the format of {OrchestratorOutput.model_json_schema()}
    """,
)],
output_type=OrchestratorOutput
)