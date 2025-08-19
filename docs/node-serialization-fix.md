# ComfyUI Node JSON Serialization Fix

## Problem
ComfyUI frontend was breaking when adding whisper custom nodes with the error:
```
TypeError: Object of type method is not JSON serializable
```

This error occurred in ComfyUI's `/object_info` endpoint when it tried to serialize node class information to JSON.

## Root Cause
The issue was caused by ComfyUI node classes having method objects assigned to class attributes that should contain simple serializable values (strings, tuples, etc.).

Common causes:
1. **Method objects assigned to class attributes**: `RETURN_NAMES = INPUT_TYPES` (where INPUT_TYPES is a method)
2. **Mismatch between RETURN_TYPES and execute() return values**: Can cause processing inconsistencies
3. **Missing OUTPUT_NODE flag**: For nodes that don't output to other nodes

## Solution

### 1. Fixed ComfyStreamUIPreview Node
**Before:**
```python
class ComfyStreamUIPreview:
    RETURN_TYPES = ()
    FUNCTION = "execute"
    CATEGORY = "ComfyStream"
    
    def execute(self):
        return ("UI Preview Node Executed",)  # Mismatch with RETURN_TYPES
```

**After:**
```python
class ComfyStreamUIPreview:
    RETURN_TYPES = ()
    OUTPUT_NODE = True  # Added - this is a UI node
    FUNCTION = "execute"
    CATEGORY = "ComfyStream"
    
    def execute(self):
        return ()  # Fixed - matches RETURN_TYPES
```

### 2. Added Node Validation Utility
Created `src/comfystream/node_validation.py` to detect and prevent similar issues:

```python
from comfystream.node_validation import validate_all_node_classes
validate_all_node_classes(NODE_CLASS_MAPPINGS)
```

This utility:
- Validates all node class attributes are JSON serializable
- Checks INPUT_TYPES method results
- Logs warnings for any issues found
- Helps prevent future serialization errors

### 3. Integration
Added validation to `nodes/__init__.py` to automatically check all nodes when the package is loaded.

## Testing
Created comprehensive tests to verify:
1. Fixed node passes JSON serialization
2. Problematic patterns are detected
3. Execute return values match RETURN_TYPES
4. Method objects in class attributes are caught

## Prevention
To prevent similar issues in custom nodes:

1. **Never assign method objects to class attributes**:
   ```python
   # ❌ Wrong - method object
   RETURN_NAMES = INPUT_TYPES
   
   # ✅ Correct - tuple of strings
   RETURN_NAMES = ("output1", "output2")
   ```

2. **Match RETURN_TYPES with execute() return**:
   ```python
   # ✅ Both empty for output nodes
   RETURN_TYPES = ()
   def execute(self):
       return ()
   
   # ✅ Both match for processing nodes
   RETURN_TYPES = ("IMAGE", "STRING")
   def execute(self):
       return (image, text)
   ```

3. **Use OUTPUT_NODE for UI/terminal nodes**:
   ```python
   # ✅ For nodes that don't output to other nodes
   OUTPUT_NODE = True
   RETURN_TYPES = ()
   ```

This fix resolves the ComfyUI frontend breaking issue and provides protection against similar problems in the future.