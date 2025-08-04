"""
ComfyUI-specific Pydantic models for ComfyStream API parameters.

This module defines ComfyUI-specific BaseModel classes for validating and serializing
API request parameters. Core streaming models are imported from pytrickle.
"""

import json
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator

# Import core streaming models from pytrickle
from pytrickle.api_spec import (
    StreamStartRequest as BaseStreamStartRequest,
    StreamParamsUpdateRequest as BaseStreamParamsUpdateRequest, 
    StreamResponse,
    StreamStatusResponse,
    HealthCheckResponse,
    ServiceInfoResponse
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
    
    # Make params optional with default values
    params: Optional[ComfyUIParams] = Field(
        default=None, 
        description="ComfyUI workflow parameters (prompt, width, height). If not provided, defaults will be used."
    )
    
    def get_comfy_params(self) -> ComfyUIParams:
        """Get the ComfyUI parameters, either from params field or from top-level fields, or use defaults."""
        if self.params is not None:
            return self.params
        
        return ComfyUIParams()

class StreamParamsUpdateRequest(BaseStreamParamsUpdateRequest):
    """ComfyUI-specific request model for updating stream parameters with prompts."""
    width: int = Field(default=DEFAULT_WIDTH, description="Width of the generated video")
    height: int = Field(default=DEFAULT_HEIGHT, description="Height of the generated video")  
    prompts: Optional[Union[str, List[Union[str, Dict[str, Any]]]]] = Field(..., description="ComfyUI workflow as JSON string or dict")

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

# StreamResponse, StreamStatusResponse, HealthCheckResponse, and ServiceInfoResponse
# are now imported from pytrickle.api_spec 