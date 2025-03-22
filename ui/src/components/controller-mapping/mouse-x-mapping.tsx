"use client";

import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { MouseXMapping } from '@/types/controller';
import { MinMaxFields } from './common';
import { BaseMappingFormProps } from './base-mapping-form';

export function MouseXMappingForm({ 
  nodeId, 
  fieldName,
  inputMin,
  inputMax,
  currentMapping,
  onSaveMapping
}: BaseMappingFormProps) {
  // Form state for mouse-x mapping
  const [multiplier, setMultiplier] = useState<number>(0.01);
  const [minOverride, setMinOverride] = useState<number | undefined>(inputMin);
  const [maxOverride, setMaxOverride] = useState<number | undefined>(inputMax);
  
  // Update state from current mapping when it changes
  useEffect(() => {
    if (currentMapping && currentMapping.type === 'mouse-x') {
      const mouseXMapping = currentMapping as MouseXMapping;
      setMultiplier(mouseXMapping.multiplier || 0.01);
      setMinOverride(mouseXMapping.minOverride);
      setMaxOverride(mouseXMapping.maxOverride);
    }
  }, [currentMapping]);
  
  // Function to reset value
  const handleResetValue = () => {
    if (onSaveMapping) {
      onSaveMapping({
        type: 'mouse-x',
        nodeId,
        fieldName,
        multiplier: 0
      } as MouseXMapping);
    }
  };
  
  // Save the current mapping
  const handleSave = () => {
    const mapping: MouseXMapping = {
      type: 'mouse-x',
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
          <Label htmlFor="multiplier-mouse-x">Sensitivity</Label>
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
          id="multiplier-mouse-x"
          type="number" 
          step="0.001"
          value={multiplier} 
          onChange={(e) => setMultiplier(parseFloat(e.target.value) || 0.01)} 
        />
        <p className="text-xs text-gray-500 mt-1">
          Higher values = more sensitive horizontal movement
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