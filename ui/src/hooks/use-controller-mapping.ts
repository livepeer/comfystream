import { useState, useEffect, useCallback } from 'react';
import { ControllerMapping, MappingStorage, ButtonMapping } from '@/types/controller';

// Define a type for legacy mappings that might be in storage
interface LegacyMapping {
  type: string;
  nodeId: string;
  fieldName: string;
  [key: string]: any; // Allow any additional properties for legacy types
}

const STORAGE_KEY = 'comfystream_controller_mappings';

export function useControllerMapping() {
  const [mappings, setMappings] = useState<Record<string, Record<string, ControllerMapping>>>({});
  
  // Load mappings from localStorage on mount
  useEffect(() => {
    try {
      console.log('Attempting to load controller mappings from localStorage');
      const savedMappings = localStorage.getItem(STORAGE_KEY);
      
      if (savedMappings) {
        console.log('Found saved controller mappings:', savedMappings);
        const parsedMappings = JSON.parse(savedMappings) as MappingStorage;
        console.log('Parsed mappings:', parsedMappings);
        
        // Handle legacy mappings if needed
        const updatedMappings = { ...parsedMappings.mappings };
        let hasUpdated = false;
        
        // Check for any legacy mappings and update them to the new format
        Object.entries(updatedMappings).forEach(([nodeId, fields]) => {
          Object.entries(fields).forEach(([fieldName, mapping]) => {
            const legacyMapping = mapping as LegacyMapping;
            
            if (legacyMapping.type === 'promptList') {
              // Convert promptList mapping to button with series mode
              console.log(`Converting legacy promptList mapping for ${nodeId}.${fieldName} to button with series mode`);
              
              updatedMappings[nodeId][fieldName] = {
                type: 'button',
                nodeId,
                fieldName,
                buttonIndex: legacyMapping.nextButtonIndex || 0,
                mode: 'series',
                valueWhenPressed: '1',
                valueWhenReleased: '0',
                nextButtonIndex: legacyMapping.prevButtonIndex,
                valuesList: ['1', '2', '3'], // Default values
                currentValueIndex: 0
              } as ButtonMapping;
              
              hasUpdated = true;
            } else if (legacyMapping.type === 'button' && !('mode' in legacyMapping)) {
              // Handle legacy button mappings that don't have mode
              console.log(`Updating legacy button mapping for ${nodeId}.${fieldName}`);
              
              updatedMappings[nodeId][fieldName] = {
                ...legacyMapping,
                mode: legacyMapping.toggleMode ? 'toggle' : 'momentary'
              } as ButtonMapping;
              
              hasUpdated = true;
            }
          });
        });
        
        if (hasUpdated) {
          console.log('Updated legacy mappings format to new format');
          setMappings(updatedMappings);
          localStorage.setItem(STORAGE_KEY, JSON.stringify({ mappings: updatedMappings }));
        } else {
          setMappings(parsedMappings.mappings);
        }
      } else {
        console.log('No saved controller mappings found in localStorage');
      }
    } catch (error) {
      console.error('Failed to load controller mappings:', error);
    }
  }, []);
  
  // Add a custom event for when mappings change
  const MAPPING_CHANGE_EVENT = 'comfystream:mappingChanged';
  
  // Save mappings to localStorage whenever they change
  useEffect(() => {
    try {
      // Only save if there are actually mappings
      if (Object.keys(mappings).length > 0) {
        console.log('Saving controller mappings to localStorage:', mappings);
        localStorage.setItem(STORAGE_KEY, JSON.stringify({ mappings }));
        console.log('Saved controller mappings successfully');
        
        // Dispatch a custom event to notify other components
        const event = new CustomEvent(MAPPING_CHANGE_EVENT, { 
          detail: { action: 'save', mappings } 
        });
        window.dispatchEvent(event);
      }
    } catch (error) {
      console.error('Failed to save controller mappings:', error);
    }
  }, [mappings]);
  
  // Add or update a mapping
  const saveMapping = useCallback((nodeId: string, fieldName: string, mapping: ControllerMapping) => {
    console.log(`Saving mapping for ${nodeId}.${fieldName}:`, mapping);
    
    // Force refresh storage before updating
    try {
      // First read current mappings from storage to ensure we have the latest
      const savedMappings = localStorage.getItem(STORAGE_KEY);
      if (savedMappings) {
        const parsedMappings = JSON.parse(savedMappings) as MappingStorage;
        setMappings(prev => {
          // Merge with our current state
          const freshMappings = { ...parsedMappings.mappings };
          
          // Update with the new mapping
          if (!freshMappings[nodeId]) {
            freshMappings[nodeId] = {};
          }
          freshMappings[nodeId][fieldName] = mapping;
          
          return freshMappings;
        });
        return;
      }
    } catch (error) {
      console.error('Error refreshing mappings from storage:', error);
    }
    
    // Fallback to normal update if the above fails
    setMappings(prev => {
      const updated = { ...prev };
      
      // Initialize the node entry if it doesn't exist
      if (!updated[nodeId]) {
        updated[nodeId] = {};
      }
      
      // Add or update the field mapping
      updated[nodeId][fieldName] = mapping;
      
      return updated;
    });
  }, []);
  
  // Remove a mapping
  const removeMapping = useCallback((nodeId: string, fieldName: string) => {
    console.log(`Removing mapping for ${nodeId}.${fieldName}`);
    
    // Force refresh storage before removing
    try {
      // First read current mappings from storage to ensure we have the latest
      const savedMappings = localStorage.getItem(STORAGE_KEY);
      if (savedMappings) {
        const parsedMappings = JSON.parse(savedMappings) as MappingStorage;
        setMappings(prev => {
          // Start with fresh mappings from storage
          const freshMappings = { ...parsedMappings.mappings };
          
          // Remove the mapping if it exists
          if (freshMappings[nodeId] && freshMappings[nodeId][fieldName]) {
            const { [fieldName]: _, ...rest } = freshMappings[nodeId];
            freshMappings[nodeId] = rest;
            
            // If node has no more mappings, remove the node entry
            if (Object.keys(freshMappings[nodeId]).length === 0) {
              const { [nodeId]: __, ...remainingNodes } = freshMappings;
              return remainingNodes;
            }
          }
          
          return freshMappings;
        });
        return;
      }
    } catch (error) {
      console.error('Error refreshing mappings from storage:', error);
    }
    
    // Fallback to normal update if the above fails
    setMappings(prev => {
      const updated = { ...prev };
      
      // If node exists and has the field mapping, remove it
      if (updated[nodeId] && updated[nodeId][fieldName]) {
        const { [fieldName]: _, ...rest } = updated[nodeId];
        updated[nodeId] = rest;
        
        // If node has no more mappings, remove the node entry
        if (Object.keys(updated[nodeId]).length === 0) {
          const { [nodeId]: __, ...remainingNodes } = updated;
          return remainingNodes;
        }
      }
      
      return updated;
    });
  }, []);
  
  // Get a mapping by nodeId and fieldName
  const getMapping = useCallback((nodeId: string, fieldName: string): ControllerMapping | undefined => {
    const mapping = mappings[nodeId]?.[fieldName];
    // Only log when we're actually finding a mapping, not for every check
    if (mapping) {
      console.log(`Retrieved mapping for ${nodeId}.${fieldName}:`, mapping);
    }
    return mapping;
  }, [mappings]);
  
  // Check if a mapping exists
  const hasMapping = useCallback((nodeId: string, fieldName: string): boolean => {
    return !!mappings[nodeId]?.[fieldName];
  }, [mappings]);
  
  return {
    mappings,
    saveMapping,
    removeMapping,
    getMapping,
    hasMapping
  };
} 