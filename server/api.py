"""
ComfyUI-specific Pydantic models for ComfyStream API parameters.

This module defines ComfyUI-specific BaseModel classes for validating and serializing
API request parameters. Core streaming models are imported from pytrickle.
"""

import json
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator

# Import core streaming models from pytrickle
from pytrickle.api import (
    StreamStartRequest as BaseStreamStartRequest,
    StreamParamsUpdateRequest as BaseStreamParamsUpdateRequest
)

from comfystream.server.workflows import get_default_workflow
DEFAULT_WORKFLOW_JSON = get_default_workflow()
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512

class ComfyUIParams(BaseModel):
    class Config:
        extra = "forbid"

    prompts: Union[str, List[Union[str, Dict[str, Any]]]] = [DEFAULT_WORKFLOW_JSON]
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT

    @classmethod
    def merge_with_defaults(cls, updates: Dict[str, Any], current_width: int = DEFAULT_WIDTH, current_height: int = DEFAULT_HEIGHT) -> 'ComfyUIParams':
        """
        Merge parameter updates with current values, using ComfyUI defaults as fallbacks.
        
        Args:
            updates: Dictionary of parameter updates
            current_width: Current width value (used as default if width not in updates)
            current_height: Current height value (used as default if height not in updates)
            
        Returns:
            New ComfyUIParams instance with merged values
        """
        merged_params = {
            'width': updates.get('width', current_width),
            'height': updates.get('height', current_height),
        }
        
        # Only include prompts if explicitly provided in updates
        if 'prompts' in updates:
            merged_params['prompts'] = updates['prompts']
        
        # Add any other parameters from updates
        for key, value in updates.items():
            if key not in merged_params:
                merged_params[key] = value
        
        return cls.model_validate(merged_params)

    @field_validator('prompts', mode='before')
    @classmethod
    def validate_prompts(cls, v) -> List[Dict[str, Any]]:
        if v == "":
            return [DEFAULT_WORKFLOW_JSON]
        
        # Handle list input (could be list of strings or list of dicts)
        if isinstance(v, list):
            result = []
            for item in v:
                if isinstance(item, str):
                    try:
                        parsed = json.loads(item)
                        if isinstance(parsed, dict):
                            result.append(parsed)
                        else:
                            raise ValueError("Each JSON string in prompts must parse to a dictionary")
                    except json.JSONDecodeError:
                        raise ValueError(f"Could not parse JSON string: {item}")
                elif isinstance(item, dict):
                    result.append(item)
                else:
                    raise ValueError("Each item in prompts list must be either a JSON string or dict")
            return result
            
        # Handle single dict input
        if isinstance(v, dict):
            return [v]
        
        # Handle single string input
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, dict):
                    return [parsed]
                elif isinstance(parsed, list):
                    # Handle case where string is a JSON array
                    return cls.validate_prompts(parsed)  # Recurse to handle the list
                else:
                    raise ValueError("Provided JSON string must parse to a dictionary or array of dictionaries")
            except json.JSONDecodeError:
                raise ValueError("Provided prompt string must be valid JSON")
        
        raise ValueError("Prompts must be either a JSON string, dictionary, or list of JSON strings/dictionaries")

    @field_validator('width', 'height', mode='before')
    @classmethod
    def convert_dimensions_to_int(cls, v):
        if isinstance(v, str):
            try:
                return int(v)
            except ValueError:
                raise ValueError(f"Invalid dimension value: {v} cannot be converted to integer")
        return v

class StreamStartRequest(BaseStreamStartRequest):
    """ComfyUI-specific stream start request with ComfyUI parameters."""
    
    def get_comfy_params(self) -> ComfyUIParams:
        """Get the ComfyUI parameters from the params dict or use defaults."""
        if self.params is not None:
            # Parse the params dict into ComfyUIParams with validation
            return ComfyUIParams.model_validate(self.params)
        
        return ComfyUIParams()

class StreamParamsUpdateRequest(BaseStreamParamsUpdateRequest):
    """ComfyUI-specific request model for updating stream parameters with prompts."""
    
    def get_comfy_params(self) -> ComfyUIParams:
        """Parse the parameters into a ComfyUIParams object."""
        # Use model_dump() to get all the fields including extra ones
        params_dict = self.model_dump(exclude_none=True)
        
        # Handle special case where no params were provided - use defaults
        if not params_dict:
            return ComfyUIParams()
        
        # Parse into ComfyUIParams with validation
        return ComfyUIParams.model_validate(params_dict)
    
    def get_width(self) -> int:
        """Get validated width parameter."""
        comfy_params = self.get_comfy_params()
        return comfy_params.width
    
    def get_height(self) -> int:
        """Get validated height parameter."""
        comfy_params = self.get_comfy_params()
        return comfy_params.height
    
    def get_prompts(self) -> List[Dict[str, Any]]:
        """Get validated prompts parameter."""
        comfy_params = self.get_comfy_params()
        return comfy_params.prompts