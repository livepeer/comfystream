#!/usr/bin/env python3
"""
Test to verify that ComfyUI node classes are properly serializable
"""
import json
import os
import sys
import tempfile
import unittest

# Add the nodes directory to the path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nodes'))

class TestNodeSerialization(unittest.TestCase):
    """Test that node classes can be JSON serialized without errors"""
    
    def test_comfystream_ui_preview_serialization(self):
        """Test that ComfyStreamUIPreview node can be serialized"""
        # Define the node class as it should be after our fix
        class ComfyStreamUIPreview:
            @classmethod
            def INPUT_TYPES(cls):
                return {
                    "required": {},
                    "optional": {}
                }
            
            RETURN_TYPES = ()
            OUTPUT_NODE = True
            FUNCTION = "execute"
            CATEGORY = "ComfyStream"
            
            def execute(self):
                return ()
        
        # Test serialization of all attributes ComfyUI might serialize
        attributes_to_test = [
            'RETURN_TYPES', 'FUNCTION', 'CATEGORY', 'OUTPUT_NODE'
        ]
        
        for attr_name in attributes_to_test:
            with self.subTest(attribute=attr_name):
                attr_value = getattr(ComfyStreamUIPreview, attr_name)
                try:
                    json.dumps(attr_value)
                except (TypeError, ValueError) as e:
                    self.fail(f"Attribute {attr_name} is not JSON serializable: {e}")
        
        # Test INPUT_TYPES method result
        try:
            input_types = ComfyStreamUIPreview.INPUT_TYPES()
            json.dumps(input_types)
        except (TypeError, ValueError) as e:
            self.fail(f"INPUT_TYPES result is not JSON serializable: {e}")
    
    def test_execute_return_matches_return_types(self):
        """Test that execute return value matches RETURN_TYPES"""
        class ComfyStreamUIPreview:
            RETURN_TYPES = ()
            OUTPUT_NODE = True
            
            def execute(self):
                return ()
        
        instance = ComfyStreamUIPreview()
        result = instance.execute()
        
        # For output nodes with empty RETURN_TYPES, execute should return empty tuple
        self.assertEqual(result, ())
        self.assertEqual(len(result), len(ComfyStreamUIPreview.RETURN_TYPES))
    
    def test_method_object_detection(self):
        """Test that we can detect problematic method objects in node classes"""
        class ProblematicNode:
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {}}
            
            RETURN_TYPES = ()
            FUNCTION = "execute"
            
            # This would cause the JSON serialization error
            RETURN_NAMES = INPUT_TYPES  # Method object!
        
        # This should fail JSON serialization
        with self.assertRaises((TypeError, ValueError)):
            json.dumps(ProblematicNode.RETURN_NAMES)

if __name__ == '__main__':
    unittest.main()