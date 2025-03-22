"use client";

import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { MouseYMapping } from '@/types/controller';
import { MinMaxFields } from './common';
import { BaseMappingFormProps } from './base-mapping-form';

export function MouseYMappingForm({ 
  nodeId, 
  fieldName,
  inputMin,
  inputMax,
  currentMapping,
  onSaveMapping
}: BaseMappingFormProps) {
  // Form state for mouse-y mapping
  const [multiplier, setMultiplier] = useState<number>(0.01);
  const [minOverride, setMinOverride] = useState<number | undefined>(inputMin);
  const [maxOverride, setMaxOverride] = useState<number | undefined>(inputMax);
  
  // Update state from current mapping when it changes
  useEffect(() => {
    if (currentMapping && currentMapping.type === 'mouse-y') {
      const mouseYMapping = currentMapping as MouseYMapping;
      setMultiplier(mouseYMapping.multiplier || 0.01);
      setMinOverride(mouseYMapping.minOverride);
      setMaxOverride(mouseYMapping.maxOverride);
    }
  }, [currentMapping]);
  
  // Function to reset value
  const handleResetValue = () => {
    if (onSaveMapping) {
      onSaveMapping({
        type: 'mouse-y',
        nodeId,
        fieldName,
        multiplier: 0
      } as MouseYMapping);
    }
  };
  
  // Save the current mapping
  const handleSave = () => {
    const mapping: MouseYMapping = {
      type: 'mouse-y',
      nodeId,
      fieldName,
      multiplier: multiplier || 0.01,
      minOverride,
      maxOverride
    };
    
    onSaveMapping(mapping);
  };
  
  return (
    <div className="space-y-4">
      <div>
        <div className="flex justify-between items-center mb-1">
          <Label htmlFor="multiplier-mouse-y">Sensitivity</Label>
          <Button 
            variant="outline" 
            size="sm"
            onClick={handleResetValue}
            className="text-xs"
          >
            Reset Value
          </Button>
        </div>
        <Input 
          id="multiplier-mouse-y"
          type="number" 
          step="0.001"
          value={multiplier} 
          onChange={(e) => setMultiplier(parseFloat(e.target.value) || 0.01)} 
        />
        <p className="text-xs text-gray-500 mt-1">
          Higher values = more sensitive vertical movement
        </p>
      </div>
      
      <MinMaxFields
        minValue={minOverride}
        maxValue={maxOverride}
        setMinValue={setMinOverride}
        setMaxValue={setMaxOverride}
        inputMin={inputMin}
        inputMax={inputMax}
      />
      
      <Button 
        className="w-full mt-4"
        onClick={handleSave}
      >
        Save Mapping
      </Button>
    </div>
  );
} 