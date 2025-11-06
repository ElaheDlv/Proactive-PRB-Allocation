import re
from typing import Callable, List, Tuple, Optional
from .tags import KnowledgeTag
from .relationships import KnowledgeRelationship

class KnowledgeRoute:
    def __init__(
        self,
        pattern: str,
        handler: Callable,
        tags: Optional[List[KnowledgeTag]] = None,
        related: Optional[List[Tuple[KnowledgeRelationship, str]]] = None,
    ):
        self.pattern = pattern
        self.regex, self.param_names = self._compile_pattern(pattern)
        self.handler = handler
        self.tags = tags or []
        self.related = related or []

    def _compile_pattern(self, pattern: str):
        param_names = []
        regex_pattern = re.sub(
            r"\{(\w+)\}", lambda m: self._param_replacer(m, param_names), pattern
        )
        return re.compile(f"^{regex_pattern}$"), param_names

    def _param_replacer(self, match, param_names):
        name = match.group(1)
        param_names.append(name)
        return f"(?P<{name}>[^/]+)"

    def match(self, key: str):
        match = self.regex.match(key)
        return match.groupdict() if match else None