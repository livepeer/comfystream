"use client";

import React, { useState, useEffect } from 'react';
import { useController } from '@/hooks/use-controller';
import { useControllerMapping } from '@/hooks/use-controller-mapping';
import { ControllerMapping } from '@/types/controller';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { GamepadIcon } from 'lucide-react';

// Import all mapping forms
import { AxisMappingForm } from './axis-mapping';
import { ButtonMappingForm } from './button-mapping';
import { KeyMappingForm } from './key-mapping';
import { MouseMappingForm } from './mouse-mapping';
import { MouseXMappingForm } from './mouse-x-mapping';
import { MouseYMappingForm } from './mouse-y-mapping';
import { ControllerInfo } from './controller-info';
import { BaseMappingFormProps, ControllerMappingFormProps } from './base-mapping-form';

interface ControllerMappingProps {
  nodeId: string;
  fieldName: string;
  inputType: string;
  inputMin?: number;
  inputMax?: number;
  onChange?: (value: any) => void;
}

export function ControllerMappingButton({ 
  nodeId, 
  fieldName,
  inputType,
  inputMin,
  inputMax,
  onChange
}: ControllerMappingProps) {
  const { controllers, isControllerConnected } = useController();
  const { getMapping, saveMapping, removeMapping, hasMapping } = useControllerMapping();
  const [isOpen, setIsOpen] = useState(false);
  const [mappingType, setMappingType] = useState<'axis' | 'button' | 'key' | 'mouse' | 'mouse-x' | 'mouse-y'>('axis');
  const [currentMapping, setCurrentMapping] = useState<ControllerMapping | undefined>(
    getMapping(nodeId, fieldName)
  );

  // Debug function to check controllers
  const checkControllers = () => {
    console.log("Manual check for controllers:");
    console.log(`Controllers detected: ${controllers.length}`);
    controllers.forEach((c, i) => {
      console.log(`Controller ${i}: ${c.id} - ${c.buttons.length} buttons, ${c.axes.length} axes`);
    });
    
    // Attempt to force refresh controllers
    if (typeof navigator.getGamepads === 'function') {
      const gamepads = navigator.getGamepads();
      console.log(`Raw gamepad API returned ${gamepads.length} gamepads`);
    }
  };

  // Update local state when mapping changes
  useEffect(() => {
    const mapping = getMapping(nodeId, fieldName);
    setCurrentMapping(mapping);
    
    if (mapping) {
      setMappingType(mapping.type);
    }
  }, [nodeId, fieldName, getMapping]);

  // Save the current mapping
  const handleSaveMapping = (mapping: ControllerMapping) => {
    saveMapping(nodeId, fieldName, mapping);
    setCurrentMapping(mapping);
    setIsOpen(false);
  };

  // Remove the current mapping
  const handleRemoveMapping = () => {
    removeMapping(nodeId, fieldName);
    setCurrentMapping(undefined);
    setIsOpen(false);
  };

  // Common props for all mapping forms
  const baseProps: BaseMappingFormProps = {
    currentMapping,
    nodeId,
    fieldName,
    inputMin,
    inputMax,
    onSaveMapping: handleSaveMapping
  };

  // Extended props for controller-based mapping forms
  const controllerProps: ControllerMappingFormProps = {
    ...baseProps,
    controllers,
    isControllerConnected
  };

  return (
    <>
      <Button
        variant={hasMapping(nodeId, fieldName) ? "default" : "outline"}
        size="sm"
        className="h-8 gap-1 ml-2"
        onClick={() => setIsOpen(true)}
      >
        <GamepadIcon className="h-4 w-4" />
        {hasMapping(nodeId, fieldName) ? 'Mapped' : 'Map'}
      </Button>
      
      <Dialog open={isOpen} onOpenChange={setIsOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Controller Mapping</DialogTitle>
          </DialogHeader>
          
          <div className="space-y-4">
            <ControllerInfo />
            
            <div className="space-y-2">
              <label htmlFor="mapping-type" className="text-sm font-medium">Mapping Type</label>
              <select
                id="mapping-type"
                value={mappingType}
                onChange={(e) => setMappingType(e.target.value as 'axis' | 'button' | 'key' | 'mouse' | 'mouse-x' | 'mouse-y')}
                className="p-2 border rounded w-full"
              >
                <option value="axis">Controller Axis</option>
                <option value="button">Controller Button</option>
                <option value="key">Keyboard Key</option>
                <option value="mouse">Mouse Button/Wheel</option>
                <option value="mouse-x">Mouse X Movement</option>
                <option value="mouse-y">Mouse Y Movement</option>
              </select>
            </div>
            
            {/* Render the appropriate mapping form based on type */}
            {mappingType === 'axis' && (
              <AxisMappingForm {...controllerProps} />
            )}
            
            {mappingType === 'button' && (
              <ButtonMappingForm {...controllerProps} />
            )}
            
            {mappingType === 'key' && (
              <KeyMappingForm {...baseProps} />
            )}
            
            {mappingType === 'mouse' && (
              <MouseMappingForm {...baseProps} />
            )}
            
            {mappingType === 'mouse-x' && (
              <MouseXMappingForm {...baseProps} />
            )}
            
            {mappingType === 'mouse-y' && (
              <MouseYMappingForm {...baseProps} />
            )}
            
            <div className="flex justify-between mt-6">
              {currentMapping ? (
                <Button variant="destructive" size="sm" onClick={handleRemoveMapping}>
                  Remove Mapping
                </Button>
              ) : (
                <div></div>
              )}
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}