from enum import Enum


class KnowledgeTag(Enum):
    # network element types
    UE = "UE"
    BS = "BS"
    CELL = "CELL"
    RIC = "RIC"

    # element attribute categories
    ID = "ID"
    LOCATION = "LOCATION"
    MOBILITY = "MOBILITY"
    QoS = "QoS"
    CODE = "CODE"

    # simulation related
    SIMULATION = "SIMULATION"

    # knowledge layer related
    KNOWLEDGE_GUIDE = "KNOWLEDGE_GUIDE"

    # AI services related
    AI_SERVICE = "AI_SERVICE"