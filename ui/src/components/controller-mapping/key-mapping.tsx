"use client";

import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { KeyMapping } from '@/types/controller';
import { ButtonModeSelector, ValuesList, IncrementModeFields, ButtonModeType } from './common';
import { BaseMappingFormProps } from './base-mapping-form';

export function KeyMappingForm({ 
  nodeId, 
  fieldName,
  inputMin,
  inputMax,
  currentMapping,
  onSaveMapping
}: BaseMappingFormProps) {
  const [isListeningForKey, setIsListeningForKey] = useState(false);
  const [detectingButtonTarget, setDetectingButtonTarget] = useState<"primary" | "next">("primary");
  
  // Form state for key mapping
  const [keyCode, setKeyCode] = useState<string>('');
  const [nextKeyCode, setNextKeyCode] = useState<string>('');
  const [buttonMode, setButtonMode] = useState<ButtonModeType>('momentary');
  const [valueWhenPressed, setValueWhenPressed] = useState<string>('1');
  const [valueWhenReleased, setValueWhenReleased] = useState<string>('0');
  const [valuesList, setValuesList] = useState<Array<string>>(['1', '2', '3']);
  const [currentValueIndex, setCurrentValueIndex] = useState<number>(0);
  const [incrementStep, setIncrementStep] = useState<number>(1);
  const [minOverride, setMinOverride] = useState<number | undefined>(inputMin);
  const [maxOverride, setMaxOverride] = useState<number | undefined>(inputMax);
  
  // Update state from current mapping when it changes
  useEffect(() => {
    if (currentMapping && currentMapping.type === 'key') {
      const keyMapping = currentMapping as KeyMapping;
      setKeyCode(keyMapping.keyCode || '');
      setButtonMode(keyMapping.mode || 'momentary');
      setValueWhenPressed(keyMapping.valueWhenPressed?.toString() || '1');
      setValueWhenReleased(keyMapping.valueWhenReleased?.toString() || '0');
      
      if (keyMapping.mode === 'series') {
        setNextKeyCode(keyMapping.nextKeyCode || '');
        setValuesList(keyMapping.valuesList?.map(v => v.toString()) || ['1', '2', '3']);
        setCurrentValueIndex(keyMapping.currentValueIndex || 0);
      } else if (keyMapping.mode === 'increment') {
        setNextKeyCode(keyMapping.nextKeyCode || '');
        setIncrementStep(keyMapping.incrementStep || 1);
        setMinOverride(keyMapping.minOverride);
        setMaxOverride(keyMapping.maxOverride);
      }
    }
  }, [currentMapping]);
  
  // Handle keyboard detection
  useEffect(() => {
    if (!isListeningForKey) return;
    
    const handleKeyDown = (event: KeyboardEvent) => {
      // Prevent default to avoid triggering browser shortcuts
      event.preventDefault();
      
      console.log(`Key detected: ${event.code}`);
      
      // Set the detected key
      if (detectingButtonTarget === "primary") {
        setKeyCode(event.code);
      } else {
        setNextKeyCode(event.code);
      }
      
      // Stop listening
      setIsListeningForKey(false);
    };
    
    // Add event listener for key down
    window.addEventListener('keydown', handleKeyDown);
    
    // Clean up
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [isListeningForKey, detectingButtonTarget]);
  
  // Start listening for keyboard input
  const handleStartKeyListening = (target: "primary" | "next") => {
    setDetectingButtonTarget(target);
    setIsListeningForKey(true);
  };
  
  // Stop listening for keyboard input
  const handleStopKeyListening = () => {
    setIsListeningForKey(false);
  };
  
  // Save the current mapping
  const handleSave = () => {
    // Keyboard mapping
    const mapping: KeyMapping = {
      type: 'key',
      nodeId,
      fieldName,
      keyCode,
      mode: buttonMode,
      valueWhenPressed: parseFloat(valueWhenPressed) || valueWhenPressed,
      valueWhenReleased: parseFloat(valueWhenReleased) || valueWhenReleased
    };
    
    // Add series-specific properties if in series mode
    if (buttonMode === 'series') {
      mapping.nextKeyCode = nextKeyCode || undefined;
      mapping.valuesList = valuesList.map(v => parseFloat(v) || v);
      mapping.currentValueIndex = currentValueIndex;
    } else if (buttonMode === 'increment') {
      mapping.nextKeyCode = nextKeyCode || undefined;
      mapping.incrementStep = incrementStep;
      mapping.minOverride = minOverride;
      mapping.maxOverride = maxOverride;
    }
    
    onSaveMapping(mapping);
  };
  
  return (
    <div className="space-y-2">
      <div>
        <Label htmlFor="key-code">{buttonMode === 'series' ? 'Previous Key' : 'Key'}</Label>
        <div className="flex gap-2">
          <Input 
            id="key-code"
            value={keyCode} 
            onChange={(e) => setKeyCode(e.target.value)}
            placeholder="Press Detect to capture a key"
            readOnly
          />
          <Button 
            onClick={() => handleStartKeyListening("primary")} 
            disabled={isListeningForKey}
            size="sm"
          >
            Detect
          </Button>
        </div>
      </div>
      
      {isListeningForKey && detectingButtonTarget === "primary" && (
        <div className="p-3 mt-2 bg-blue-50 rounded border border-blue-200">
          <div className="flex justify-between items-center mb-2">
            <p className="text-sm font-medium">Keyboard Detection Mode</p>
            <Button 
              variant="outline" 
              size="sm"
              onClick={handleStopKeyListening}
            >
              Stop
            </Button>
          </div>
          <p className="text-xs mb-2">Press any key on your keyboard</p>
        </div>
      )}
      
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
            <Label htmlFor="next-key">Next Value Key (Optional)</Label>
            <div className="flex gap-2">
              <Input 
                id="next-key"
                value={nextKeyCode} 
                onChange={(e) => setNextKeyCode(e.target.value)}
                placeholder="Press Detect to capture a key"
                readOnly
              />
              <Button 
                onClick={() => handleStartKeyListening("next")} 
                disabled={isListeningForKey}
                size="sm"
              >
                Detect
              </Button>
            </div>
            <p className="text-xs text-gray-500 mt-1">Leave empty to disable second key</p>
          </div>
          
          {isListeningForKey && detectingButtonTarget === "next" && (
            <div className="p-3 mt-2 bg-blue-50 rounded border border-blue-200">
              <div className="flex justify-between items-center mb-2">
                <p className="text-sm font-medium">Keyboard Detection Mode</p>
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={handleStopKeyListening}
                >
                  Stop
                </Button>
              </div>
              <p className="text-xs mb-2">Press any key on your keyboard</p>
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
            <Label htmlFor="next-key">Decrement Key</Label>
            <div className="flex gap-2">
              <Input 
                id="next-key"
                value={nextKeyCode} 
                onChange={(e) => setNextKeyCode(e.target.value)}
                placeholder="Press Detect to capture a key"
                readOnly
              />
              <Button 
                onClick={() => handleStartKeyListening("next")} 
                disabled={isListeningForKey}
                size="sm"
              >
                Detect
              </Button>
            </div>
          </div>
          
          {isListeningForKey && detectingButtonTarget === "next" && (
            <div className="p-3 mt-2 bg-blue-50 rounded border border-blue-200">
              <div className="flex justify-between items-center mb-2">
                <p className="text-sm font-medium">Keyboard Detection Mode</p>
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={handleStopKeyListening}
                >
                  Stop
                </Button>
              </div>
              <p className="text-xs mb-2">Press any key on your keyboard</p>
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