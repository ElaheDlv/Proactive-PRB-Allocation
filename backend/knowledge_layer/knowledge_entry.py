from typing import Optional, List, Tuple, Callable
from .relationships import KnowledgeRelationship
from .tags import KnowledgeTag

knowledge_entry_registry = {}


def knowledge_entry(
    key: str,
    tags: Optional[List[KnowledgeTag]] = None,
    related: Optional[List[Tuple[KnowledgeRelationship, str]]] = None,
):
    def decorator(func: Callable):
        knowledge_entry_registry[key] = {
            "func": func,
            "tags": tags or [],
            "related": related or [],
        }
        return func

    return decorator
