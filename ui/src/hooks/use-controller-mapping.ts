import { useState, useEffect, useCallback } from 'react';
import { ControllerMapping, MappingStorage } from '@/types/controller';

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
        setMappings(parsedMappings.mappings);
      } else {
        console.log('No saved controller mappings found in localStorage');
      }
    } catch (error) {
      console.error('Failed to load controller mappings:', error);
    }
  }, []);
  
  // Save mappings to localStorage whenever they change
  useEffect(() => {
    try {
      // Only save if there are actually mappings
      if (Object.keys(mappings).length > 0) {
        console.log('Saving controller mappings to localStorage:', mappings);
        localStorage.setItem(STORAGE_KEY, JSON.stringify({ mappings }));
        console.log('Saved controller mappings successfully');
      }
    } catch (error) {
      console.error('Failed to save controller mappings:', error);
    }
  }, [mappings]);
  
  // Add or update a mapping
  const saveMapping = useCallback((nodeId: string, fieldName: string, mapping: ControllerMapping) => {
    console.log(`Saving mapping for ${nodeId}.${fieldName}:`, mapping);
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