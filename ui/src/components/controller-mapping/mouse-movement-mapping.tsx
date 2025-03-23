"use client";

import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { MouseMovementMapping } from '@/types/controller';
import { MinMaxFields } from './common';
import { BaseMappingFormProps } from './base-mapping-form';

interface MouseMovementMappingFormProps extends BaseMappingFormProps {
  axis: 'x' | 'y';
}

export function MouseMovementMappingForm({ 
  nodeId, 
  fieldName,
  inputMin,
  inputMax,
  currentMapping,
  onSaveMapping,
  axis
}: MouseMovementMappingFormProps) {
  // Form state for mouse movement mapping
  const [multiplier, setMultiplier] = useState<number>(0.01);
  const [minOverride, setMinOverride] = useState<number | undefined>(inputMin);
  const [maxOverride, setMaxOverride] = useState<number | undefined>(inputMax);
  
  // Update state from current mapping when it changes
  useEffect(() => {
    if (currentMapping && currentMapping.type === 'mouse-movement' && 
        (currentMapping as MouseMovementMapping).axis === axis) {
      const mouseMapping = currentMapping as MouseMovementMapping;
      setMultiplier(mouseMapping.multiplier || 0.01);
      setMinOverride(mouseMapping.minOverride);
      setMaxOverride(mouseMapping.maxOverride);
    }
  }, [currentMapping, axis]);
  
  // Function to reset value
  const handleResetValue = () => {
    if (onSaveMapping) {
      onSaveMapping({
        type: 'mouse-movement',
        nodeId,
        fieldName,
        axis,
        multiplier: 0
      } as MouseMovementMapping);
    }
  };
  
  // Save the current mapping
  const handleSave = () => {
    const mapping: MouseMovementMapping = {
      type: 'mouse-movement',
      nodeId,
      fieldName,
      axis,
      multiplier: multiplier || 0.01,
      minOverride,
      maxOverride
    };
    
    onSaveMapping(mapping);
  };
  
  const axisLabel = axis === 'x' ? 'horizontal' : 'vertical';
  
  return (
    <div className="space-y-4">
      <div>
        <div className="flex justify-between items-center mb-1">
          <Label htmlFor={`multiplier-mouse-${axis}`}>Sensitivity</Label>
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
          id={`multiplier-mouse-${axis}`}
          type="number" 
          step="0.001"
          value={multiplier} 
          onChange={(e) => setMultiplier(parseFloat(e.target.value) || 0.01)} 
        />
        <p className="text-xs text-gray-500 mt-1">
          Higher values = more sensitive {axisLabel} movement
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