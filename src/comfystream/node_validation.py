"""Utilities for validating ComfyUI node definitions"""
import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def validate_node_class_for_serialization(node_class: type, class_name: str) -> List[str]:
    """
    Validate that a ComfyUI node class has JSON-serializable attributes.
    
    This helps prevent "Object of type method is not JSON serializable" errors
    when ComfyUI tries to serialize node information for the /object_info endpoint.
    
    Args:
        node_class: The node class to validate
        class_name: Name of the class for error reporting
        
    Returns:
        List of error messages, empty if no issues found
    """
    errors = []
    
    # Check all class attributes that ComfyUI might try to serialize
    attributes_to_check = [
        'RETURN_TYPES', 'RETURN_NAMES', 'FUNCTION', 'CATEGORY', 
        'OUTPUT_NODE', 'DESCRIPTION', 'INPUT_TYPES'
    ]
    
    for attr_name in attributes_to_check:
        if hasattr(node_class, attr_name):
            attr_value = getattr(node_class, attr_name)
            
            # Special handling for INPUT_TYPES method
            if attr_name == 'INPUT_TYPES' and callable(attr_value):
                try:
                    # Call the method and test serialization of the result
                    input_types_result = attr_value()
                    json.dumps(input_types_result)
                except (TypeError, ValueError) as e:
                    errors.append(f"{class_name}.{attr_name}() result: {e}")
                except Exception as e:
                    errors.append(f"{class_name}.{attr_name}() call failed: {e}")
            else:
                # Test direct serialization of the attribute
                try:
                    json.dumps(attr_value)
                except (TypeError, ValueError) as e:
                    errors.append(f"{class_name}.{attr_name}: {type(attr_value).__name__} - {e}")
    
    return errors


def validate_all_node_classes(node_class_mappings: Dict[str, type]) -> None:
    """
    Validate all node classes in a mapping and log any issues.
    
    Args:
        node_class_mappings: Dictionary mapping class names to node classes
    """
    all_errors = []
    
    for class_name, node_class in node_class_mappings.items():
        errors = validate_node_class_for_serialization(node_class, class_name)
        all_errors.extend(errors)
    
    if all_errors:
        logger.error(f"Found {len(all_errors)} JSON serialization issues in node classes:")
        for error in all_errors:
            logger.error(f"  - {error}")
        logger.error("These issues may cause ComfyUI's /object_info endpoint to fail with 'Object of type method is not JSON serializable'")
    else:
        logger.debug("All node classes passed JSON serialization validation")