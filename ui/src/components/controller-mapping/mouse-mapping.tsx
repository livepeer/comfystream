"use client";

import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { MouseMapping } from '@/types/controller';
import { MappingFormBase, InputDetectionUI, ModeBasedInputProps } from './mapping-form-base';
import { BaseMappingFormProps } from './base-mapping-form';
import { MinMaxFields } from './common';

type MouseActionType = 'wheel' | 'button';

export function MouseMappingForm({ 
  nodeId, 
  fieldName,
  inputMin,
  inputMax,
  currentMapping,
  onSaveMapping
}: BaseMappingFormProps) {
  const [isListeningForMouse, setIsListeningForMouse] = useState(false);
  const [detectingButtonTarget, setDetectingButtonTarget] = useState<"primary" | "next">("primary");
  
  // Mouse-specific state
  const [mouseAction, setMouseAction] = useState<MouseActionType>('button');
  const [mouseButtonIndex, setMouseButtonIndex] = useState<number>(0);
  const [nextMouseButtonIndex, setNextMouseButtonIndex] = useState<number>(-1);
  const [multiplier, setMultiplier] = useState<number>(0.01);
  const [minOverride, setMinOverride] = useState<number | undefined>(inputMin);
  const [maxOverride, setMaxOverride] = useState<number | undefined>(inputMax);

  // Update state from current mapping when it changes
  useEffect(() => {
    if (currentMapping && currentMapping.type === 'mouse') {
      const mouseMapping = currentMapping as MouseMapping;
      setMouseAction(mouseMapping.action);
      
      if (mouseMapping.action === 'button') {
        if (mouseMapping.buttonIndex !== undefined) {
          setMouseButtonIndex(mouseMapping.buttonIndex);
        }
        
        if (mouseMapping.nextButtonIndex !== undefined) {
          setNextMouseButtonIndex(mouseMapping.nextButtonIndex);
        }
      } else {
        // For mouse wheel
        setMultiplier(mouseMapping.multiplier || 0.01);
        setMinOverride(mouseMapping.minOverride);
        setMaxOverride(mouseMapping.maxOverride);
      }
    }
  }, [currentMapping]);
  
  // Handle mouse detection
  useEffect(() => {
    if (!isListeningForMouse || mouseAction !== 'button') return;
    
    const handleMouseDown = (event: MouseEvent) => {
      // Prevent default
      event.preventDefault();
      
      console.log(`Mouse button detected: ${event.button}`);
      
      // Set the detected button
      if (detectingButtonTarget === "primary") {
        setMouseButtonIndex(event.button);
      } else {
        setNextMouseButtonIndex(event.button);
      }
      
      // Stop listening
      setIsListeningForMouse(false);
    };
    
    // Add event listener for mouse down
    window.addEventListener('mousedown', handleMouseDown);
    
    // Clean up
    return () => {
      window.removeEventListener('mousedown', handleMouseDown);
    };
  }, [isListeningForMouse, detectingButtonTarget, mouseAction]);
  
  // Start listening for mouse input
  const handleStartMouseListening = (target: "primary" | "next") => {
    setDetectingButtonTarget(target);
    setIsListeningForMouse(true);
  };
  
  // Stop listening for mouse input
  const handleStopMouseListening = () => {
    setIsListeningForMouse(false);
  };
  
  // Create a mapping from the common state
  const createMapping = (commonState: ModeBasedInputProps<MouseMapping>): MouseMapping => {
    const { 
      buttonMode, 
      valueWhenPressed, 
      valueWhenReleased, 
      valuesList, 
      currentValueIndex, 
      incrementStep, 
      minOverride: modeMinOverride, 
      maxOverride: modeMaxOverride 
    } = commonState;
    
    let mapping: MouseMapping;
    
    if (mouseAction === 'button') {
      mapping = {
        type: 'mouse',
        nodeId,
        fieldName,
        action: mouseAction,
        buttonIndex: mouseButtonIndex,
        mode: buttonMode,
        valueWhenPressed: parseFloat(valueWhenPressed) || valueWhenPressed,
        valueWhenReleased: parseFloat(valueWhenReleased) || valueWhenReleased
      };
      
      // Add series-specific properties if in series mode
      if (buttonMode === 'series') {
        mapping.nextAction = 'button';
        mapping.nextButtonIndex = nextMouseButtonIndex !== -1 ? nextMouseButtonIndex : undefined;
        mapping.valuesList = valuesList.map(v => parseFloat(v) || v);
        mapping.currentValueIndex = currentValueIndex;
      } else if (buttonMode === 'increment') {
        mapping.nextAction = 'button';
        mapping.nextButtonIndex = nextMouseButtonIndex !== -1 ? nextMouseButtonIndex : undefined;
        mapping.incrementStep = incrementStep;
        mapping.minOverride = modeMinOverride;
        mapping.maxOverride = modeMaxOverride;
      }
    } else {
      // Mouse wheel mapping
      mapping = {
        type: 'mouse',
        nodeId,
        fieldName,
        action: mouseAction,
        mode: 'axis', // Wheel always uses axis mode
        multiplier: multiplier,
        minOverride: minOverride,
        maxOverride: maxOverride
      };
    }
    
    return mapping;
  };
  
  // Primary mouse button detection component
  const PrimaryMouseButtonDetection = (
    <div className="flex gap-2">
      <Input 
        id="mouse-button-index"
        type="number" 
        min="0" 
        max="2"
        value={mouseButtonIndex} 
        onChange={(e) => setMouseButtonIndex(parseInt(e.target.value) || 0)} 
      />
      <Button 
        onClick={() => handleStartMouseListening("primary")} 
        disabled={isListeningForMouse}
        size="sm"
      >
        Detect
      </Button>
    </div>
  );
  
  // Next mouse button detection component
  const NextMouseButtonDetection = (
    <div className="flex gap-2">
      <Input 
        id="next-mouse-button"
        type="number" 
        min="-1" 
        max="2"
        value={nextMouseButtonIndex} 
        onChange={(e) => setNextMouseButtonIndex(parseInt(e.target.value) || -1)} 
        placeholder="No second button"
      />
      <Button 
        onClick={() => handleStartMouseListening("next")} 
        disabled={isListeningForMouse}
        size="sm"
      >
        Detect
      </Button>
    </div>
  );
  
  // Action selection component
  const ActionSelection = (
    <div className="space-y-2">
      <Label>Mouse Action</Label>
      <div className="flex space-x-4">
        <div className="flex items-center space-x-2">
          <Button
            type="button"
            variant={mouseAction === 'button' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setMouseAction('button')}
            className="px-3 py-1 h-auto"
          >
            Button
          </Button>
        </div>
        <div className="flex items-center space-x-2">
          <Button
            type="button"
            variant={mouseAction === 'wheel' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setMouseAction('wheel')}
            className="px-3 py-1 h-auto"
          >
            Mouse Wheel
          </Button>
        </div>
      </div>
    </div>
  );
  
  // Conditional rendering based on mouseAction
  if (mouseAction === 'wheel') {
    return (
      <div className="space-y-2">
        {ActionSelection}
        
        <div>
          <Label htmlFor="wheel-multiplier">Wheel Sensitivity</Label>
          <Input 
            id="wheel-multiplier"
            type="number" 
            step="0.001"
            min="0.001"
            max="1"
            value={multiplier} 
            onChange={(e) => setMultiplier(parseFloat(e.target.value) || 0.01)} 
          />
          <p className="text-xs text-gray-500 mt-1">Smaller values make the wheel less sensitive</p>
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
          onClick={() => {
            const mapping: MouseMapping = {
              type: 'mouse',
              nodeId,
              fieldName,
              action: mouseAction,
              mode: 'axis',
              multiplier,
              minOverride,
              maxOverride
            };
            onSaveMapping(mapping);
          }}
        >
          Save Mapping
        </Button>
      </div>
    );
  }

  return (
    <>
      <div className="space-y-2 mb-4">
        {ActionSelection}
      </div>
      
      <MappingFormBase<MouseMapping>
        nodeId={nodeId}
        fieldName={fieldName}
        inputMin={inputMin}
        inputMax={inputMax}
        currentMapping={currentMapping}
        onSaveMapping={onSaveMapping}
        mappingType="mouse"
        createMapping={createMapping}
        detectElement={{
          primaryLabel: "Mouse Button",
          nextLabel: "Next Value Button (Optional)",
          primaryDetectionComponent: PrimaryMouseButtonDetection,
          nextDetectionComponent: NextMouseButtonDetection
        }}
      >
        {/* Mouse Detection UI - conditionally rendered */}
        {isListeningForMouse && (
          <InputDetectionUI
            isListening={isListeningForMouse}
            setIsListening={setIsListeningForMouse}
            detectingTarget={detectingButtonTarget}
            setDetectingTarget={setDetectingButtonTarget}
            onStopListening={handleStopMouseListening}
            title="Mouse Detection Mode"
            instructions="Click any mouse button"
          />
        )}
        
        <div>
          <Label htmlFor="mouse-button-index">Button</Label>
          {PrimaryMouseButtonDetection}
          <p className="text-xs text-gray-500 mt-1">0 = left, 1 = middle, 2 = right</p>
        </div>
      </MappingFormBase>
    </>
  );
} 