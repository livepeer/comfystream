"use client";

import React from 'react';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { ControllerMapping } from '@/types/controller';
import { BaseMappingFormProps } from './base-mapping-form';

export type MappingFormProps = BaseMappingFormProps;

export function MinMaxFields({ 
  minValue, 
  maxValue, 
  setMinValue, 
  setMaxValue, 
  inputMin, 
  inputMax 
}: { 
  minValue?: number; 
  maxValue?: number; 
  setMinValue: (value: number | undefined) => void; 
  setMaxValue: (value: number | undefined) => void; 
  inputMin?: number; 
  inputMax?: number; 
}) {
  return (
    <div className="grid grid-cols-2 gap-2">
      <div>
        <Label htmlFor="min-override">Min Value</Label>
        <Input 
          id="min-override"
          type="number" 
          value={minValue !== undefined ? minValue : ''} 
          placeholder={inputMin?.toString() || 'Default'}
          onChange={(e) => setMinValue(e.target.value ? parseFloat(e.target.value) : undefined)} 
        />
      </div>
      <div>
        <Label htmlFor="max-override">Max Value</Label>
        <Input 
          id="max-override"
          type="number" 
          value={maxValue !== undefined ? maxValue : ''} 
          placeholder={inputMax?.toString() || 'Default'}
          onChange={(e) => setMaxValue(e.target.value ? parseFloat(e.target.value) : undefined)} 
        />
      </div>
    </div>
  );
}

// Define a type for button modes that can be reused
export type ButtonModeType = 'momentary' | 'toggle' | 'series' | 'increment' | 'axis';

export function ButtonModeSelector({
  buttonMode,
  setButtonMode
}: {
  buttonMode: ButtonModeType;
  setButtonMode: (mode: ButtonModeType) => void;
}) {
  return (
    <div className="space-y-2 mt-4">
      <Label>Button Mode</Label>
      <div className="grid grid-cols-4 gap-2">
        <div className="flex items-center space-x-2">
          <input
            type="radio"
            id="mode-momentary"
            checked={buttonMode === 'momentary'}
            onChange={() => setButtonMode('momentary')}
            className="w-4 h-4"
          />
          <Label htmlFor="mode-momentary">Momentary</Label>
        </div>
        <div className="flex items-center space-x-2">
          <input
            type="radio"
            id="mode-toggle"
            checked={buttonMode === 'toggle'}
            onChange={() => setButtonMode('toggle')}
            className="w-4 h-4"
          />
          <Label htmlFor="mode-toggle">Toggle</Label>
        </div>
        <div className="flex items-center space-x-2">
          <input
            type="radio"
            id="mode-series"
            checked={buttonMode === 'series'}
            onChange={() => setButtonMode('series')}
            className="w-4 h-4"
          />
          <Label htmlFor="mode-series">Series</Label>
        </div>
        <div className="flex items-center space-x-2">
          <input
            type="radio"
            id="mode-increment"
            checked={buttonMode === 'increment'}
            onChange={() => setButtonMode('increment')}
            className="w-4 h-4"
          />
          <Label htmlFor="mode-increment">Increment</Label>
        </div>
      </div>
    </div>
  );
}

export function ValuesList({
  valuesList,
  setValuesList,
  currentValueIndex,
  setCurrentValueIndex
}: {
  valuesList: string[];
  setValuesList: (values: string[]) => void;
  currentValueIndex: number;
  setCurrentValueIndex: (index: number) => void;
}) {
  return (
    <div>
      <Label>Values to Cycle Through</Label>
      <div className="border rounded-md p-2 bg-gray-50 space-y-2 mt-1 max-h-40 overflow-y-auto">
        {valuesList.map((value, index) => (
          <div key={index} className="flex gap-2 items-center">
            <Input
              value={value}
              onChange={(e) => {
                const newList = [...valuesList];
                newList[index] = e.target.value;
                setValuesList(newList);
              }}
              className="flex-1"
            />
            <Button
              variant="outline" 
              size="sm"
              onClick={() => {
                const newList = valuesList.filter((_, i) => i !== index);
                setValuesList(newList);
                if (currentValueIndex >= newList.length) {
                  setCurrentValueIndex(newList.length - 1);
                }
              }}
            >
              ✕
            </Button>
          </div>
        ))}
        <Button
          variant="outline"
          size="sm"
          className="w-full"
          onClick={() => setValuesList([...valuesList, ''])}
        >
          Add Value
        </Button>
      </div>
    </div>
  );
}

export function IncrementModeFields({
  incrementStep,
  setIncrementStep,
  minOverride,
  setMinOverride,
  maxOverride,
  setMaxOverride,
  inputMin,
  inputMax
}: {
  incrementStep: number;
  setIncrementStep: (step: number) => void;
  minOverride?: number;
  setMinOverride: (min?: number) => void;
  maxOverride?: number;
  setMaxOverride: (max?: number) => void;
  inputMin?: number;
  inputMax?: number;
}) {
  return (
    <>
      <div>
        <Label htmlFor="increment-step">Increment Step</Label>
        <Input 
          id="increment-step"
          type="number" 
          step="0.1"
          value={incrementStep} 
          onChange={(e) => setIncrementStep(parseFloat(e.target.value) || 1)} 
        />
        <p className="text-xs text-gray-500 mt-1">Amount to change value by when activated</p>
      </div>
      
      <MinMaxFields
        minValue={minOverride}
        maxValue={maxOverride}
        setMinValue={setMinOverride}
        setMaxValue={setMaxOverride}
        inputMin={inputMin}
        inputMax={inputMax}
      />
    </>
  );
} 