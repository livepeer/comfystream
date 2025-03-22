"use client";

import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { ButtonMapping } from '@/types/controller';
import { ButtonModeSelector, ValuesList, IncrementModeFields, ButtonModeType } from './common';
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
  
  // Form state for button mapping
  const [buttonIndex, setButtonIndex] = useState<number>(0);
  const [buttonMode, setButtonMode] = useState<ButtonModeType>('momentary');
  const [valueWhenPressed, setValueWhenPressed] = useState<string>('1');
  const [valueWhenReleased, setValueWhenReleased] = useState<string>('0');
  const [nextButtonIndex, setNextButtonIndex] = useState<number>(-1);
  const [valuesList, setValuesList] = useState<Array<string>>(['1', '2', '3']);
  const [currentValueIndex, setCurrentValueIndex] = useState<number>(0);
  const [incrementStep, setIncrementStep] = useState<number>(1);
  const [minOverride, setMinOverride] = useState<number | undefined>(inputMin);
  const [maxOverride, setMaxOverride] = useState<number | undefined>(inputMax);
  
  // Update state from current mapping when it changes
  useEffect(() => {
    if (currentMapping && currentMapping.type === 'button') {
      const buttonMapping = currentMapping as ButtonMapping;
      setButtonIndex(buttonMapping.buttonIndex);
      setButtonMode(buttonMapping.mode || 'momentary');
      setValueWhenPressed(buttonMapping.valueWhenPressed?.toString() || '1');
      setValueWhenReleased(buttonMapping.valueWhenReleased?.toString() || '0');
      
      if (buttonMapping.mode === 'series') {
        setNextButtonIndex(buttonMapping.nextButtonIndex || -1);
        setValuesList(buttonMapping.valuesList?.map(v => v.toString()) || ['1', '2', '3']);
        setCurrentValueIndex(buttonMapping.currentValueIndex || 0);
      } else if (buttonMapping.mode === 'increment') {
        setNextButtonIndex(buttonMapping.nextButtonIndex || -1);
        setIncrementStep(buttonMapping.incrementStep || 1);
        setMinOverride(buttonMapping.minOverride);
        setMaxOverride(buttonMapping.maxOverride);
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
  
  // Save the current mapping
  const handleSave = () => {
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
    
    onSaveMapping(mapping);
  };
  
  return (
    <div className="space-y-2">
      {/* Controller Detection UI */}
      {isListening && (
        <div className="p-3 mt-2 bg-blue-50 rounded border border-blue-200">
          <div className="flex justify-between items-center mb-2">
            <p className="text-sm font-medium">Controller Detection Mode</p>
            <Button 
              variant="outline" 
              size="sm"
              onClick={handleStopListening}
            >
              Stop
            </Button>
          </div>
          
          <p className="text-xs mb-2">
            Press the button you want to map
          </p>
          
          {detectedInput && (
            <p className="text-xs bg-white p-2 rounded border border-blue-100 font-medium text-blue-700">
              Detected: {detectedInput}
              {detectingButtonTarget === "primary" && <> → Setting {buttonMode === 'series' ? 'Previous' : ''} Button to {buttonIndex}</>}
              {detectingButtonTarget === "next" && <> → Setting Next Button to {nextButtonIndex}</>}
            </p>
          )}
        </div>
      )}
      
      <div>
        <Label htmlFor="button-index">{buttonMode === 'series' ? 'Previous Button' : 'Button'}</Label>
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
      </div>
      
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
          <div>
            <Label htmlFor="next-button">Next Value Button (Optional)</Label>
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
            <p className="text-xs text-gray-500 mt-1">Set to -1 to disable second button</p>
          </div>
          
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
          <div>
            <Label htmlFor="next-button">Decrement Button</Label>
            <div className="flex gap-2">
              <Input 
                id="next-button"
                type="number" 
                min="-1" 
                max="20"
                value={nextButtonIndex} 
                onChange={(e) => setNextButtonIndex(parseInt(e.target.value) || -1)} 
              />
              <Button 
                onClick={() => handleStartListening("next")} 
                disabled={!isControllerConnected || isListening}
                size="sm"
              >
                Detect
              </Button>
            </div>
          </div>
          
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