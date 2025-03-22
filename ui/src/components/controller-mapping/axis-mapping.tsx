"use client";

import React, { useState, useEffect, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { AxisMapping } from '@/types/controller';
import { MinMaxFields } from './common';
import { ControllerMappingFormProps } from './base-mapping-form';

export function AxisMappingForm({ 
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
  
  // Form state for axis mapping
  const [axisIndex, setAxisIndex] = useState<number>(0);
  const [multiplier, setMultiplier] = useState<number>(1);
  const [minOverride, setMinOverride] = useState<number | undefined>(inputMin);
  const [maxOverride, setMaxOverride] = useState<number | undefined>(inputMax);
  
  // Add a state to track axis values for feedback
  const [currentAxisValue, setCurrentAxisValue] = useState<number | null>(null);
  
  // Reference to store detection state without causing re-renders
  const detectionStateRef = useRef({
    lastAxisValues: [] as number[],
    loopCount: 0
  });
  
  // Update state from current mapping when it changes
  useEffect(() => {
    if (currentMapping && currentMapping.type === 'axis') {
      const axisMapping = currentMapping as AxisMapping;
      setAxisIndex(axisMapping.axisIndex);
      setMultiplier(axisMapping.multiplier);
      setMinOverride(axisMapping.minOverride);
      setMaxOverride(axisMapping.maxOverride);
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
    
    // Store initial axis values for calibration
    const initialAxisValues = Array.from(controller.axes);
    
    console.log("Initial axis values:", initialAxisValues);
    
    const detectChanges = () => {
      // Get fresh gamepad data
      const gamepads = navigator.getGamepads();
      const freshController = gamepads[controller.index];
      
      if (!freshController) {
        console.warn("Controller lost during detection");
        return;
      }
      
      // Check for significant axis movements COMPARED TO INITIAL VALUES
      freshController.axes.forEach((axisValue, index) => {
        const initialValue = initialAxisValues[index];
        const movement = Math.abs(axisValue - initialValue);
        
        // Use movement threshold for detection
        if (movement > 0.3) {
          console.log(`Axis ${index} moved: from ${initialValue.toFixed(2)} to ${axisValue.toFixed(2)} (change: ${movement.toFixed(2)})`);
          setDetectedInput(`Axis ${index}`);
          setAxisIndex(index);
          setCurrentAxisValue(axisValue);
        }
        
        // Always update current value for visual feedback when on the mapped axis
        if (index === axisIndex) {
          setCurrentAxisValue(axisValue);
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
  }, [isListening, controllers, axisIndex]);
  
  // Start listening for controller input
  const handleStartListening = () => {
    // Clear previous detection state
    setDetectedInput('');
    
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
    const mapping: AxisMapping = {
      type: 'axis',
      nodeId,
      fieldName,
      axisIndex,
      multiplier,
      minOverride,
      maxOverride
    };
    
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
            Move a joystick or trigger you want to map
          </p>
          
          {detectedInput && (
            <p className="text-xs bg-white p-2 rounded border border-blue-100 font-medium text-blue-700">
              Detected: {detectedInput}
              {detectedInput && <> → Setting Axis to {axisIndex}</>}
            </p>
          )}
        </div>
      )}
      
      {/* Add visual feedback for axis movement during detection */}
      {isListening && currentAxisValue !== null && (
        <div className="mt-2 bg-blue-50 p-2 rounded border border-blue-200">
          <p className="text-xs font-medium">Current Axis Value:</p>
          <div className="relative h-4 bg-gray-200 rounded-full mt-1">
            <div 
              className="absolute top-0 bottom-0 bg-blue-600 rounded-full"
              style={{ 
                left: '50%', 
                width: `${Math.abs(currentAxisValue) * 100}%`, 
                transform: `translateX(${currentAxisValue < 0 ? '-100%' : '0'})`,
                transformOrigin: 'left center'
              }}
            />
            <div className="absolute top-0 bottom-0 w-px bg-gray-400 left-1/2" />
          </div>
          <p className="text-xs text-right mt-1">{currentAxisValue.toFixed(3)}</p>
        </div>
      )}
      
      <div>
        <Label htmlFor="axis-index">Axis</Label>
        <div className="flex gap-2">
          <Input 
            id="axis-index"
            type="number" 
            min="0" 
            max="20"
            value={axisIndex} 
            onChange={(e) => setAxisIndex(parseInt(e.target.value) || 0)} 
          />
          <Button 
            onClick={handleStartListening} 
            disabled={!isControllerConnected || isListening}
            size="sm"
          >
            Detect
          </Button>
        </div>
      </div>
      
      <div>
        <Label htmlFor="multiplier">Multiplier</Label>
        <Input 
          id="multiplier"
          type="number" 
          value={multiplier} 
          onChange={(e) => setMultiplier(parseFloat(e.target.value) || 1)} 
        />
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