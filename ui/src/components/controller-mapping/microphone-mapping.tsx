"use client";

import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { MicrophoneMapping } from '@/types/controller';
import { MinMaxFields } from './common';
import { BaseMappingFormProps } from './base-mapping-form';

export function MicrophoneMappingForm({ 
  nodeId, 
  fieldName,
  inputMin,
  inputMax,
  currentMapping,
  onSaveMapping,
}: BaseMappingFormProps) {
  // Form state for microphone mapping
  const [multiplier, setMultiplier] = useState<number>(1);
  const [minOverride, setMinOverride] = useState<number | undefined>(inputMin);
  const [maxOverride, setMaxOverride] = useState<number | undefined>(inputMax);
  const [smoothing, setSmoothing] = useState<number>(0.3); // Default smoothing value
  
  // Update state from current mapping when it changes
  useEffect(() => {
    if (currentMapping && currentMapping.type === 'microphone') {
      const micMapping = currentMapping as MicrophoneMapping;
      setMultiplier(micMapping.multiplier);
      setMinOverride(micMapping.minOverride);
      setMaxOverride(micMapping.maxOverride);
      setSmoothing(micMapping.smoothing || 0.3);
    }
  }, [currentMapping]);
  
  // Save the current mapping
  const handleSave = () => {
    const mapping: MicrophoneMapping = {
      type: 'microphone',
      nodeId,
      fieldName,
      audioFeature: 'volume',
      multiplier,
      minOverride,
      maxOverride,
      smoothing
    };
    
    onSaveMapping(mapping);
  };
  
  return (
    <div className="space-y-4">
      <div>
        <Label htmlFor="multiplier">Multiplier</Label>
        <Input 
          id="multiplier"
          type="number" 
          value={multiplier} 
          onChange={(e) => setMultiplier(parseFloat(e.target.value) || 1)} 
        />
      </div>
      
      <div>
        <Label htmlFor="smoothing">Smoothing</Label>
        <div className="flex gap-2 items-center">
          <Input 
            id="smoothing"
            type="range" 
            min="0"
            max="0.95"
            step="0.05"
            value={smoothing} 
            onChange={(e) => setSmoothing(parseFloat(e.target.value))} 
          />
          <span className="text-sm w-12">{smoothing.toFixed(2)}</span>
        </div>
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