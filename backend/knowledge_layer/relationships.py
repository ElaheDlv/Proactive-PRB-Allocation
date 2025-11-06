from enum import Enum


class KnowledgeRelationship(Enum):
    RELATED_STANDARD = "related_standard"
    BELONGS_TO = "belongs_to"

    # attribute-attribute relationships
    DEPENDS_ON = "depends_on"
    DERIVED_FROM = "derived_from"
    AFFECTS = "affects"
    CONSTRAINED_BY = "constrained_by"
    ASSOCIATED_WITH = "associated_with"

    # class-attribute relationships
    ATTRIBUTE_OF = "attribute_of"
    HAS_ATTRIBUTE = "has_attribute"

    # method-method relationships
    CALL_METHOD = "call_method"  # expects another method that is called by this method
    CALLED_BY_METHOD = (
        "called_by_method"  # expects another method that calls this method
    )

    # method-attribute relationships
    SET_ATTRIBUTE = "set_attribute"  # expects an attribute that is set by this method
    SET_BY_METHOD = "set_by_method"  # expects another method that sets this attribute
    USED_BY_METHOD = "used_by_method"  # expects another method that uses this attribute
    USES_ATTRIBUTE = (
        "uses_attribute"  # expects an attribute that is used by this method
    )
