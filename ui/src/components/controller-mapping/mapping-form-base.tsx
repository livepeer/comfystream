"use client";

import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { ButtonModeSelector, ValuesList, IncrementModeFields, ButtonModeType } from './common';
import { BaseMappingFormProps } from './base-mapping-form';
import { ControllerMapping } from '@/types/controller';

export interface InputDetectionProps {
  isListening: boolean;
  setIsListening: (isListening: boolean) => void;
  detectingTarget: "primary" | "next";
  setDetectingTarget: (target: "primary" | "next") => void;
  detectedInput?: string;
  onStopListening: () => void;
  title: string;
  instructions: string;
}

export interface ModeBasedInputProps<T extends ControllerMapping> {
  buttonMode: ButtonModeType;
  valueWhenPressed: string;
  setValueWhenPressed: (value: string) => void;
  valueWhenReleased: string;
  setValueWhenReleased: (value: string) => void;
  valuesList: string[];
  setValuesList: (values: string[]) => void;
  currentValueIndex: number;
  setCurrentValueIndex: (index: number) => void;
  incrementStep: number;
  setIncrementStep: (step: number) => void;
  minOverride?: number;
  setMinOverride: (min?: number) => void;
  maxOverride?: number;
  setMaxOverride: (max?: number) => void;
  inputMin?: number;
  inputMax?: number;
}

export interface MappingFormBaseProps<T extends ControllerMapping> extends BaseMappingFormProps {
  mappingType: T['type'];
  createMapping: (commonState: ModeBasedInputProps<T>) => T;
  detectElement: {
    primaryLabel: string;
    nextLabel?: string;
    primaryDetectionComponent: React.ReactNode;
    nextDetectionComponent?: React.ReactNode;
  };
}

export function InputDetectionUI({ 
  isListening, 
  setIsListening, 
  detectingTarget, 
  onStopListening, 
  detectedInput,
  title,
  instructions 
}: InputDetectionProps) {
  return (
    <div className="p-3 mt-2 bg-blue-50 rounded border border-blue-200">
      <div className="flex justify-between items-center mb-2">
        <p className="text-sm font-medium">{title}</p>
        <Button 
          variant="outline" 
          size="sm"
          onClick={onStopListening}
        >
          Stop
        </Button>
      </div>
      
      <p className="text-xs mb-2">{instructions}</p>
      
      {detectedInput && (
        <p className="text-xs bg-white p-2 rounded border border-blue-100 font-medium text-blue-700">
          Detected: {detectedInput}
        </p>
      )}
    </div>
  );
}

export function MappingFormBase<T extends ControllerMapping>({ 
  nodeId, 
  fieldName,
  inputMin,
  inputMax,
  currentMapping,
  onSaveMapping,
  mappingType,
  createMapping,
  detectElement,
  children
}: MappingFormBaseProps<T> & { children?: React.ReactNode }) {
  const [buttonMode, setButtonMode] = useState<ButtonModeType>('momentary');
  const [valueWhenPressed, setValueWhenPressed] = useState<string>('1');
  const [valueWhenReleased, setValueWhenReleased] = useState<string>('0');
  const [valuesList, setValuesList] = useState<Array<string>>([
    'A beautiful landscape with mountains and a lake, trending on artstation',
    'Portrait of a cyberpunk character with neon lights, highly detailed', 
    'A futuristic city at night with flying cars, cinematic lighting',
    'A fantasy forest with magical creatures and glowing elements',
    'Abstract digital art with vibrant colors and geometric shapes'
  ]);
  const [currentValueIndex, setCurrentValueIndex] = useState<number>(0);
  const [incrementStep, setIncrementStep] = useState<number>(1);
  const [minOverride, setMinOverride] = useState<number | undefined>(inputMin);
  const [maxOverride, setMaxOverride] = useState<number | undefined>(inputMax);
  
  // State for the common methods we'll pass to child components
  const modeBasedInputProps: ModeBasedInputProps<T> = {
    buttonMode,
    valueWhenPressed,
    setValueWhenPressed,
    valueWhenReleased,
    setValueWhenReleased,
    valuesList,
    setValuesList,
    currentValueIndex,
    setCurrentValueIndex,
    incrementStep,
    setIncrementStep,
    minOverride,
    setMinOverride,
    maxOverride,
    setMaxOverride,
    inputMin,
    inputMax,
  };
  
  // Handle save
  const handleSave = () => {
    const mapping = createMapping(modeBasedInputProps);
    onSaveMapping(mapping);
  };
  
  return (
    <div className="space-y-2">
      {children}
      
      <ButtonModeSelector
        buttonMode={buttonMode}
        setButtonMode={setButtonMode}
      />
      
      {buttonMode !== 'series' && buttonMode !== 'increment' && (
        <div>
          <Label htmlFor="value-pressed">Value When Pressed</Label>
          <Input 
            id="value-pressed"
            value={valueWhenPressed} 
            onChange={(e) => setValueWhenPressed(e.target.value)} 
          />
        </div>
      )}
      
      {buttonMode !== 'series' && buttonMode !== 'increment' && (
        <div>
          <Label htmlFor="value-released">Value When Released {buttonMode === 'toggle' && '(Toggle Off)'}</Label>
          <Input 
            id="value-released"
            value={valueWhenReleased} 
            onChange={(e) => setValueWhenReleased(e.target.value)} 
          />
        </div>
      )}
      
      {buttonMode === 'series' && (
        <>
          {detectElement.nextLabel && detectElement.nextDetectionComponent && (
            <div>
              <Label htmlFor="next-input">{detectElement.nextLabel}</Label>
              {detectElement.nextDetectionComponent}
            </div>
          )}
          
          <ValuesList
            valuesList={valuesList}
            setValuesList={setValuesList}
            currentValueIndex={currentValueIndex}
            setCurrentValueIndex={setCurrentValueIndex}
          />
        </>
      )}
      
      {buttonMode === 'increment' && (
        <>
          {detectElement.nextLabel && detectElement.nextDetectionComponent && (
            <div>
              <Label htmlFor="next-input">{detectElement.nextLabel}</Label>
              {detectElement.nextDetectionComponent}
            </div>
          )}
          
          <IncrementModeFields
            incrementStep={incrementStep}
            setIncrementStep={setIncrementStep}
            minOverride={minOverride}
            setMinOverride={setMinOverride}
            maxOverride={maxOverride}
            setMaxOverride={setMaxOverride}
            inputMin={inputMin}
            inputMax={inputMax}
          />
        </>
      )}
      
      <Button 
        className="w-full mt-4"
        onClick={handleSave}
      >
        Save Mapping
      </Button>
    </div>
  );
} 