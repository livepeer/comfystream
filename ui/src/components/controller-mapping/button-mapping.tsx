"use client";

import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { ButtonMapping } from '@/types/controller';
import { MappingFormBase, InputDetectionUI, ModeBasedInputProps } from './mapping-form-base';
import { ControllerMappingFormProps } from './base-mapping-form';

export function ButtonMappingForm({ 
  nodeId, 
  fieldName,
  inputMin,
  inputMax,
  currentMapping,
  onSaveMapping,
  controllers,
  isControllerConnected
}: ControllerMappingFormProps) {
  const [isListening, setIsListening] = useState(false);
  const [detectedInput, setDetectedInput] = useState<string>('');
  const [detectingButtonTarget, setDetectingButtonTarget] = useState<"primary" | "next">("primary");
  
  // Button-specific state
  const [buttonIndex, setButtonIndex] = useState<number>(0);
  const [nextButtonIndex, setNextButtonIndex] = useState<number>(-1);

  // Update state from current mapping when it changes
  useEffect(() => {
    if (currentMapping && currentMapping.type === 'button') {
      const buttonMapping = currentMapping as ButtonMapping;
      setButtonIndex(buttonMapping.buttonIndex);
      
      // Handle nextButtonIndex - could be undefined in some modes
      if (buttonMapping.nextButtonIndex !== undefined) {
        setNextButtonIndex(buttonMapping.nextButtonIndex);
      } else {
        setNextButtonIndex(-1); // Default value when not specified
      }
    }
  }, [currentMapping]);
  
  // Fix the controller detection in the listening mode
  useEffect(() => {
    if (!isListening || controllers.length === 0) return;
    
    console.log("Entering listening mode for controller input");
    console.log("Available controllers:", controllers.length);
    
    // Get the first controller
    const controller = controllers[0];
    console.log(`Using controller: ${controller.id}`);
    console.log(`Buttons: ${controller.buttons.length}, Axes: ${controller.axes.length}`);
    
    // Store initial button states for calibration
    const initialButtonStates = Array.from(controller.buttons).map(b => b.pressed);
    
    const detectChanges = () => {
      // Get fresh gamepad data
      const gamepads = navigator.getGamepads();
      const freshController = gamepads[controller.index];
      
      if (!freshController) {
        console.warn("Controller lost during detection");
        return;
      }
      
      // Check for button presses (any button that wasn't pressed initially but is now)
      freshController.buttons.forEach((button, index) => {
        if (button.pressed && !initialButtonStates[index]) {
          console.log(`Button ${index} pressed`);
          setDetectedInput(`Button ${index}`);
          
          if (detectingButtonTarget === "primary") {
            setButtonIndex(index);
          } else if (detectingButtonTarget === "next") {
            setNextButtonIndex(index);
          }
        }
      });
    };
    
    // Poll for input changes
    const intervalId = setInterval(detectChanges, 50);
    
    // Cleanup interval on unmount or when listening stops
    return () => {
      console.log("Exiting listening mode");
      clearInterval(intervalId);
    };
  }, [isListening, controllers, detectingButtonTarget]);
  
  // Start listening for controller input
  const handleStartListening = (target: "primary" | "next") => {
    // Clear previous detection state
    setDetectedInput('');
    
    // Set which button we're detecting
    setDetectingButtonTarget(target);
    
    // Force a refresh of controllers
    if (typeof navigator.getGamepads === 'function') {
      console.log("Refreshing controllers before detection");
      const gamepads = navigator.getGamepads();
      console.log(`Found ${gamepads.length} controllers via getGamepads()`);
    }
    
    // Start listening mode
    setIsListening(true);
  };
  
  // Stop listening
  const handleStopListening = () => {
    setIsListening(false);
  };
  
  // Create a mapping from the common state
  const createMapping = (commonState: ModeBasedInputProps<ButtonMapping>): ButtonMapping => {
    const { 
      buttonMode, 
      valueWhenPressed, 
      valueWhenReleased, 
      valuesList, 
      currentValueIndex, 
      incrementStep, 
      minOverride, 
      maxOverride 
    } = commonState;
    
    // Button mapping - differentiate based on mode
    const mapping: ButtonMapping = {
      type: 'button',
      nodeId,
      fieldName,
      buttonIndex,
      mode: buttonMode, 
      valueWhenPressed: parseFloat(valueWhenPressed) || valueWhenPressed,
      valueWhenReleased: parseFloat(valueWhenReleased) || valueWhenReleased
    };
    
    // Add series-specific properties if in series mode
    if (buttonMode === 'series') {
      mapping.nextButtonIndex = nextButtonIndex !== -1 ? nextButtonIndex : undefined;
      mapping.valuesList = valuesList.map(v => parseFloat(v) || v);
      mapping.currentValueIndex = currentValueIndex;
    } else if (buttonMode === 'increment') {
      mapping.nextButtonIndex = nextButtonIndex;
      mapping.incrementStep = incrementStep;
      mapping.minOverride = minOverride;
      mapping.maxOverride = maxOverride;
    }
    
    return mapping;
  };
  
  // Primary button detection component
  const PrimaryButtonDetection = (
    <div className="flex gap-2">
      <Input 
        id="button-index"
        type="number" 
        min="0" 
        max="20"
        value={buttonIndex} 
        onChange={(e) => setButtonIndex(parseInt(e.target.value) || 0)} 
      />
      <Button 
        onClick={() => handleStartListening("primary")} 
        disabled={!isControllerConnected || isListening}
        size="sm"
      >
        Detect
      </Button>
    </div>
  );
  
  // Next button detection component
  const NextButtonDetection = (
    <div className="flex gap-2">
      <Input 
        id="next-button"
        type="number" 
        min="-1" 
        max="20"
        value={nextButtonIndex} 
        onChange={(e) => setNextButtonIndex(parseInt(e.target.value) || -1)} 
        placeholder="No second button"
      />
      <Button 
        onClick={() => handleStartListening("next")} 
        disabled={!isControllerConnected || isListening}
        size="sm"
      >
        Detect
      </Button>
    </div>
  );
  
  return (
    <MappingFormBase<ButtonMapping>
      nodeId={nodeId}
      fieldName={fieldName}
      inputMin={inputMin}
      inputMax={inputMax}
      currentMapping={currentMapping}
      onSaveMapping={onSaveMapping}
      mappingType="button"
      createMapping={createMapping}
      detectElement={{
        primaryLabel: "Button",
        nextLabel: "Next Value Button (Optional)",
        primaryDetectionComponent: PrimaryButtonDetection,
        nextDetectionComponent: NextButtonDetection
      }}
    >
      {/* Controller Detection UI - conditionally rendered */}
      {isListening && (
        <InputDetectionUI
          isListening={isListening}
          setIsListening={setIsListening}
          detectingTarget={detectingButtonTarget}
          setDetectingTarget={setDetectingButtonTarget}
          detectedInput={detectedInput}
          onStopListening={handleStopListening}
          title="Controller Detection Mode"
          instructions="Press the button you want to map"
        />
      )}
      
      <div>
        <Label htmlFor="button-index">Button</Label>
        {PrimaryButtonDetection}
      </div>
    </MappingFormBase>
  );
} 