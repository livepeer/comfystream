# Multiple Outputs Architecture in ComfyStream

## Overview

ComfyStream now supports workflows with multiple outputs of different types (video, audio, text). This document explains how the architecture handles multiple outputs and how to use them effectively.

## Current Architecture

### Output Types and Queues

The system uses separate queues for each output type:

- **Video outputs**: `tensor_cache.image_outputs` → `get_video_output()`
- **Audio outputs**: `tensor_cache.audio_outputs` → `get_audio_output()`  
- **Text outputs**: `tensor_cache.text_outputs` → `get_text_output()`

### Node Types

Each output type has corresponding save nodes:

- `SaveTensor` - saves video/image tensors
- `SaveAudioTensor` - saves audio tensors  
- `SaveTextTensor` - saves text strings

## Multiple Outputs Support

### 1. Validation Changes

The `convert_prompt` function in `utils.py` now supports up to 2 outputs:

```python
# ~~Only handle single output for now~~
# WIP: handle multiple outputs (up to 2)
if num_outputs > 2:
    raise Exception("too many outputs in prompt")
```

### 2. Coordinated Output Collection

New methods have been added to handle multiple outputs:

#### Pipeline Level (`pipeline.py`)
```python
async def get_multiple_outputs(self, output_types: List[str]) -> Dict[str, Any]:
    """Get multiple outputs of different types in a coordinated way."""
```

#### Client Level (`client.py`)
```python
async def get_multiple_outputs(self, output_types: List[str]) -> Dict[str, Any]:
    """Get multiple outputs of different types in a coordinated way."""
```

### 3. Trickle Integration

The trickle integration has been enhanced to handle multiple outputs:

- Modified `_collect_outputs()` to use `get_multiple_outputs(['video', 'text'])`
- Added `last_text_output` attribute to store text outputs
- Text outputs are logged and can be used in control messages

## Usage Examples

### Basic Multiple Outputs

```python
# Get both video and text outputs
outputs = await client.get_multiple_outputs(['video', 'text'])

if outputs.get('video') is not None:
    print(f"Video output shape: {outputs['video'].shape}")
    
if outputs.get('text') is not None:
    print(f"Text output: {outputs['text']}")
```

### Workflow Example

The `text-tensor-example-api.json` workflow demonstrates multiple outputs:

```json
{
  "1": {
    "inputs": {"images": ["2", 0]},
    "class_type": "SaveTensor"
  },
  "2": {
    "inputs": {},
    "class_type": "LoadTensor"
  },
  "3": {
    "inputs": {"text": "Hello from ComfyStream!"},
    "class_type": "SaveTextTensor"
  }
}
```

This workflow produces:
- **Video output**: From SaveTensor node
- **Text output**: From SaveTextTensor node

## Implementation Details

### Parallel Collection

Outputs are collected in parallel using `asyncio.gather()` to avoid blocking:

```python
tasks = []
for output_type in output_types:
    if output_type == 'video':
        tasks.append(self.get_video_output())
    elif output_type == 'text':
        tasks.append(self.get_text_output())

outputs = await asyncio.wait_for(
    asyncio.gather(*tasks, return_exceptions=True),
    timeout=10.0
)
```

### Error Handling

Each output type is handled independently with error isolation:

```python
for i, output_type in enumerate(output_types):
    if isinstance(outputs[i], Exception):
        logger.error(f"Error getting {output_type} output: {outputs[i]}")
        results[output_type] = None
    else:
        results[output_type] = outputs[i]
```

### Timeout Management

A 10-second timeout ensures the system doesn't hang waiting for outputs:

```python
except asyncio.TimeoutError:
    logger.error("Timeout waiting for multiple outputs")
    for output_type in output_types:
        results[output_type] = None
```

## Best Practices

### 1. Output Type Selection

Only request the output types you need:

```python
# Good: Only request needed outputs
outputs = await client.get_multiple_outputs(['video', 'text'])

# Avoid: Requesting unused outputs
outputs = await client.get_multiple_outputs(['video', 'audio', 'text'])  # if you don't need audio
```

### 2. Error Handling

Always check for None values:

```python
outputs = await client.get_multiple_outputs(['video', 'text'])

video_output = outputs.get('video')
if video_output is not None:
    # Process video output
    process_video(video_output)
else:
    # Handle missing video output
    logger.warning("No video output received")
```

### 3. Workflow Design

Design workflows with clear output purposes:

- Use `SaveTensor` for video/image outputs
- Use `SaveTextTensor` for text outputs  
- Use `SaveAudioTensor` for audio outputs
- Keep outputs focused and purposeful

## Limitations

### Current Limitations

1. **Maximum 2 outputs**: The validation currently limits workflows to 2 outputs
2. **Single type per queue**: Each output type has its own queue, so multiple outputs of the same type may not be handled optimally
3. **Synchronous coordination**: Outputs are collected together but may not be perfectly synchronized

### Future Enhancements

1. **Multiple outputs per type**: Support for multiple video/audio/text outputs
2. **Output metadata**: Associate metadata with outputs (timestamps, frame numbers, etc.)
3. **Output routing**: Route different outputs to different destinations
4. **Output buffering**: Buffer outputs for better synchronization

## Troubleshooting

### Common Issues

1. **"too many outputs" error**: Ensure your workflow has ≤ 2 output nodes
2. **Missing outputs**: Check that output nodes are properly connected in the workflow
3. **Timeout errors**: Increase timeout or check if outputs are being generated
4. **Type mismatches**: Ensure you're requesting the correct output types

### Debug Tips

1. **Log output collection**: Enable debug logging to see output collection
2. **Test individual outputs**: Test each output type separately first
3. **Check workflow validation**: Ensure your workflow passes `convert_prompt` validation
4. **Monitor queues**: Check if output queues are being populated

## Conclusion

The multiple outputs architecture provides a flexible foundation for complex workflows that need to produce different types of outputs. The coordinated collection approach ensures efficient processing while maintaining error isolation and timeout protection.

For more complex use cases, consider extending the architecture with output metadata, routing, and buffering capabilities. 