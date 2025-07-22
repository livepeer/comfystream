"""
Pydantic models for ComfyStream API parameters.

This module defines reusable BaseModel classes for validating and serializing
API request parameters used across the ComfyStream trickle API endpoints.
"""

import json
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator

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

class StreamStartRequest(BaseModel):
    subscribe_url: str = Field(..., description="URL for subscribing to input video stream")
    publish_url: str = Field(..., description="URL for publishing output video stream")
    control_url: Optional[str] = Field(default=None, description="URL for control channel communication")
    events_url: Optional[str] = Field(default=None, description="URL for events channel communication")
    data_url: Optional[str] = Field(default=None, description="URL for publishing text data output from inference")
    gateway_request_id: str = Field(..., description="Unique identifier for the stream request")
    
    # Optional fields that may be present in the request
    manifest_id: Optional[str] = Field(default=None, description="Manifest identifier")
    model_id: Optional[str] = Field(default=None, description="Model identifier")
    stream_id: Optional[str] = Field(default=None, description="Stream identifier")
    
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

class StreamParamsUpdateRequest(BaseModel):
    """Request model for updating stream parameters with flat structure."""
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

class StreamResponse(BaseModel):
    status: str = Field(..., description="Operation status (success/error)")
    message: str = Field(..., description="Human-readable message")
    request_id: Optional[str] = Field(default=None, description="Stream request ID")
    config: Optional[dict] = Field(default=None, description="Stream configuration details")

class StreamStatusResponse(BaseModel):
    processing_active: bool = Field(..., description="Whether stream processing is active")
    stream_count: int = Field(..., description="Number of active streams")
    message: Optional[str] = Field(default=None, description="Status message")
    current_stream: Optional[dict] = Field(default=None, description="Current stream details")
    all_streams: Optional[dict] = Field(default=None, description="All active streams")

class HealthCheckResponse(BaseModel):
    status: str = Field(..., description="Service health status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    stream_manager_ready: Optional[bool] = Field(default=None, description="Whether stream manager is ready")
    error: Optional[str] = Field(default=None, description="Error message if unhealthy")

class ServiceInfoResponse(BaseModel):
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    description: str = Field(..., description="Service description")
    capabilities: list = Field(..., description="List of service capabilities")
    endpoints: dict = Field(..., description="Available API endpoints") 