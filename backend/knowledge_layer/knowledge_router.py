from typing import Callable, Dict, List, Tuple, Optional
from .relationships import KnowledgeRelationship
from .tags import KnowledgeTag
from .knowledge_route import KnowledgeRoute
import utils
from .knowledge_entry import knowledge_entry_registry
from .knowledge_sources import *  # Import all modules to load entries


class KnowledgeRouter(metaclass=utils.SingletonMeta):
    def __init__(self):
        self.routes: List[KnowledgeRoute] = []

    def register_route(
        self,
        pattern: str,
        handler: Callable,
        tags: Optional[List[KnowledgeTag]] = None,
        related: Optional[List[Tuple[KnowledgeRelationship, str]]] = None,
    ):
        route = KnowledgeRoute(pattern, handler, tags, related)
        self.routes.append(route)

    def import_routes(self, sim=None):
        for reg_key, entry in knowledge_entry_registry.items():
            handler_func = entry["func"]
            tags = entry["tags"] if "tags" in entry else []
            related = entry["related"] if "related" in entry else []

            def wrapped_handler(query_key, params, f=handler_func):
                return f(sim, self, query_key, params)

            self.register_route(
                pattern=reg_key,
                handler=wrapped_handler,
                tags=tags,
                related=related,
            )

    def _find_route(self, key: str) -> Tuple[KnowledgeRoute, Dict[str, str]]:
        for route in self.routes:
            params = route.match(key)
            if params is not None:
                return route, params
        raise KeyError(f"Query key '{key}' not recognized.")

    def query_knowledge(self, query_key: str):
        try:
            route, params = self._find_route(query_key)
            print(f"Querying knowledge for key: {query_key}")
            print(f"Route found: {route.pattern}")
            print(f"Parameters extracted: {params}")
            knowledge = route.handler(query_key, params)
            if route.related:
                knowledge += "\n\nRelated knowledge:\n"
                for rel, pat in route.related:
                    knowledge += f"- {rel.value}: {pat}\n"
            return knowledge
        except Exception as e:
            return f"Error recognising the knowledge key {query_key}: {str(e)}"

    def get_routes(self):
        return [
            {
                "pattern": route.pattern,
                "tags": [tag.value for tag in route.tags],
                "related": [
                    {"relationship": r.value, "pattern": p} for r, p in route.related
                ],
            }
            for route in self.routes
        ]
