"use client";

import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { MouseMapping } from '@/types/controller';
import { ButtonModeSelector, ValuesList, IncrementModeFields, ButtonModeType } from './common';
import { BaseMappingFormProps } from './base-mapping-form';

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
  
  // Form state for mouse mapping
  const [mouseAction, setMouseAction] = useState<'wheel' | 'button'>('button');
  const [mouseButtonIndex, setMouseButtonIndex] = useState<number>(0);
  const [nextMouseButtonIndex, setNextMouseButtonIndex] = useState<number>(-1);
  const [buttonMode, setButtonMode] = useState<ButtonModeType>('momentary');
  const [valueWhenPressed, setValueWhenPressed] = useState<string>('1');
  const [valueWhenReleased, setValueWhenReleased] = useState<string>('0');
  const [valuesList, setValuesList] = useState<Array<string>>(['1', '2', '3']);
  const [currentValueIndex, setCurrentValueIndex] = useState<number>(0);
  const [incrementStep, setIncrementStep] = useState<number>(1);
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
        
        setButtonMode(mouseMapping.mode || 'momentary');
        setValueWhenPressed(mouseMapping.valueWhenPressed?.toString() || '1');
        setValueWhenReleased(mouseMapping.valueWhenReleased?.toString() || '0');
        
        if (mouseMapping.mode === 'series') {
          setNextMouseButtonIndex(mouseMapping.nextButtonIndex || -1);
          setValuesList(mouseMapping.valuesList?.map(v => v.toString()) || ['1', '2', '3']);
          setCurrentValueIndex(mouseMapping.currentValueIndex || 0);
        } else if (mouseMapping.mode === 'increment') {
          setNextMouseButtonIndex(mouseMapping.nextButtonIndex || -1);
          setIncrementStep(mouseMapping.incrementStep || 1);
          setMinOverride(mouseMapping.minOverride);
          setMaxOverride(mouseMapping.maxOverride);
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
  
  // Save the current mapping
  const handleSave = () => {
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
        mapping.minOverride = minOverride;
        mapping.maxOverride = maxOverride;
      }
    } else {
      // Mouse wheel - explicitly set mode to 'axis'
      mapping = {
        type: 'mouse',
        nodeId,
        fieldName,
        action: mouseAction,
        mode: 'axis', // Always 'axis' for wheel
        multiplier: multiplier,
        minOverride,
        maxOverride
      };
    }
    
    onSaveMapping(mapping);
  };
  
  return (
    <div className="space-y-2">
      <div>
        <Label htmlFor="mouse-action">Mouse Action</Label>
        <select
          id="mouse-action"
          value={mouseAction}
          onChange={(e) => setMouseAction(e.target.value as 'wheel' | 'button')}
          className="p-2 border rounded w-full"
        >
          <option value="button">Mouse Button</option>
          <option value="wheel">Mouse Wheel</option>
        </select>
      </div>
      
      {mouseAction === 'button' && (
        <>
          <div>
            <Label htmlFor="mouse-button">Mouse Button</Label>
            <div className="flex gap-2">
              <Input 
                id="mouse-button"
                type="number" 
                min="0" 
                max="4"
                value={mouseButtonIndex} 
                onChange={(e) => setMouseButtonIndex(parseInt(e.target.value) || 0)} 
                placeholder="0 = left, 1 = middle, 2 = right"
              />
              <Button 
                onClick={() => handleStartMouseListening("primary")} 
                disabled={isListeningForMouse}
                size="sm"
              >
                Detect
              </Button>
            </div>
            <p className="text-xs text-gray-500 mt-1">0 = left, 1 = middle, 2 = right</p>
          </div>
          
          {isListeningForMouse && detectingButtonTarget === "primary" && (
            <div className="p-3 mt-2 bg-blue-50 rounded border border-blue-200">
              <div className="flex justify-between items-center mb-2">
                <p className="text-sm font-medium">Mouse Button Detection Mode</p>
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={handleStopMouseListening}
                >
                  Stop
                </Button>
              </div>
              <p className="text-xs mb-2">Click any mouse button</p>
            </div>
          )}
          
          <ButtonModeSelector
            buttonMode={buttonMode}
            setButtonMode={setButtonMode}
          />
          
          {buttonMode !== 'series' && buttonMode !== 'increment' && (
            <div>
              <Label htmlFor="value-pressed-mouse">Value When Pressed</Label>
              <Input 
                id="value-pressed-mouse"
                value={valueWhenPressed} 
                onChange={(e) => setValueWhenPressed(e.target.value)} 
              />
            </div>
          )}
          
          {buttonMode !== 'series' && buttonMode !== 'increment' && (
            <div>
              <Label htmlFor="value-released-mouse">Value When Released {buttonMode === 'toggle' && '(Toggle Off)'}</Label>
              <Input 
                id="value-released-mouse"
                value={valueWhenReleased} 
                onChange={(e) => setValueWhenReleased(e.target.value)} 
              />
            </div>
          )}
          
          {buttonMode === 'series' && (
            <>
              <div>
                <Label htmlFor="next-mouse-button">Next Value Button (Optional)</Label>
                <div className="flex gap-2">
                  <Input 
                    id="next-mouse-button"
                    type="number" 
                    min="-1" 
                    max="4"
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
                <p className="text-xs text-gray-500 mt-1">Set to -1 to disable second button</p>
              </div>
              
              {isListeningForMouse && detectingButtonTarget === "next" && (
                <div className="p-3 mt-2 bg-blue-50 rounded border border-blue-200">
                  <div className="flex justify-between items-center mb-2">
                    <p className="text-sm font-medium">Mouse Button Detection Mode</p>
                    <Button 
                      variant="outline" 
                      size="sm"
                      onClick={handleStopMouseListening}
                    >
                      Stop
                    </Button>
                  </div>
                  <p className="text-xs mb-2">Click any mouse button</p>
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
              <div>
                <Label htmlFor="next-mouse-button-inc">Decrement Button</Label>
                <div className="flex gap-2">
                  <Input 
                    id="next-mouse-button-inc"
                    type="number" 
                    min="-1" 
                    max="4"
                    value={nextMouseButtonIndex} 
                    onChange={(e) => setNextMouseButtonIndex(parseInt(e.target.value) || -1)} 
                  />
                  <Button 
                    onClick={() => handleStartMouseListening("next")} 
                    disabled={isListeningForMouse}
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
        </>
      )}
      
      {mouseAction === 'wheel' && (
        <>
          <div>
            <Label htmlFor="multiplier-mouse">Sensitivity</Label>
            <Input 
              id="multiplier-mouse"
              type="number" 
              step="0.001"
              value={multiplier} 
              onChange={(e) => setMultiplier(parseFloat(e.target.value) || 0.01)} 
            />
            <p className="text-xs text-gray-500 mt-1">
              Higher values = more sensitive wheel scrolling
            </p>
          </div>
          
          <div className="grid grid-cols-2 gap-2">
            <div>
              <Label htmlFor="min-override-mouse">Min Value</Label>
              <Input 
                id="min-override-mouse"
                type="number" 
                value={minOverride !== undefined ? minOverride : ''} 
                placeholder="Default"
                onChange={(e) => setMinOverride(e.target.value ? parseFloat(e.target.value) : undefined)} 
              />
            </div>
            <div>
              <Label htmlFor="max-override-mouse">Max Value</Label>
              <Input 
                id="max-override-mouse"
                type="number" 
                value={maxOverride !== undefined ? maxOverride : ''} 
                placeholder="Default"
                onChange={(e) => setMaxOverride(e.target.value ? parseFloat(e.target.value) : undefined)} 
              />
            </div>
          </div>
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