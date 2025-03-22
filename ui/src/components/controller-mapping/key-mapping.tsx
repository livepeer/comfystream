"use client";

import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { KeyMapping } from '@/types/controller';
import { MappingFormBase, InputDetectionUI, ModeBasedInputProps } from './mapping-form-base';
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
  
  // Key-specific state
  const [keyCode, setKeyCode] = useState<string>('');
  const [nextKeyCode, setNextKeyCode] = useState<string>('');

  // Update state from current mapping when it changes
  useEffect(() => {
    if (currentMapping && currentMapping.type === 'key') {
      const keyMapping = currentMapping as KeyMapping;
      setKeyCode(keyMapping.keyCode || '');
      
      if (keyMapping.nextKeyCode) {
        setNextKeyCode(keyMapping.nextKeyCode);
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
  
  // Create a mapping from the common state
  const createMapping = (commonState: ModeBasedInputProps<KeyMapping>): KeyMapping => {
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
    
    return mapping;
  };
  
  // Primary key detection component
  const PrimaryKeyDetection = (
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
  );
  
  // Next key detection component
  const NextKeyDetection = (
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
  );
  
  return (
    <MappingFormBase<KeyMapping>
      nodeId={nodeId}
      fieldName={fieldName}
      inputMin={inputMin}
      inputMax={inputMax}
      currentMapping={currentMapping}
      onSaveMapping={onSaveMapping}
      mappingType="key"
      createMapping={createMapping}
      detectElement={{
        primaryLabel: "Key",
        nextLabel: "Next Value Key (Optional)",
        primaryDetectionComponent: PrimaryKeyDetection,
        nextDetectionComponent: NextKeyDetection
      }}
    >
      {/* Keyboard Detection UI - conditionally rendered */}
      {isListeningForKey && (
        <InputDetectionUI
          isListening={isListeningForKey}
          setIsListening={setIsListeningForKey}
          detectingTarget={detectingButtonTarget}
          setDetectingTarget={setDetectingButtonTarget}
          onStopListening={handleStopKeyListening}
          title="Keyboard Detection Mode"
          instructions="Press any key on your keyboard"
        />
      )}
      
      <div>
        <Label htmlFor="key-code">Key</Label>
        {PrimaryKeyDetection}
      </div>
    </MappingFormBase>
  );
} 