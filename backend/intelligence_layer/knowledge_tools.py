from agents import function_tool


from knowledge_layer import KnowledgeRouter

knowledge_router = KnowledgeRouter()  # get the singleton instance of KnowledgeRouter


@function_tool
async def get_knowledge(knowledge_query_key: str) -> str:
    """Get the knowledge of a specific key in the knowledge database.

    Args:
        knowledge_query_key (str): The key to query in the knowledge layer.
    """
    return knowledge_router.query_knowledge(knowledge_query_key)


@function_tool
async def get_knowledge_bulk(knowledge_query_key_list: list[str]) -> str:
    """Get the knowledge of a list of keys in the knowledge database.

    Args:
        knowledge_query_key_list (list[str]): The list of keys to query in the knowledge layer.
    """

    response_text = ""

    for knowledge_query_key in knowledge_query_key_list:
        if knowledge_query_key.strip():
            response_text += f"Query {knowledge_query_key}: \n{knowledge_router.query_knowledge(knowledge_query_key)}\n\n-----------------------------\n\n"

    return response_text
