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
import { MouseMovementMappingForm } from './mouse-movement-mapping';
import { ControllerInfo } from './controller-info';
import { BaseMappingFormProps, ControllerMappingFormProps } from './base-mapping-form';

// Custom event name for control panel refresh
const CONTROL_PANEL_REFRESH_EVENT = 'comfystream:refreshControlPanel';

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
  const [mappingType, setMappingType] = useState<'axis' | 'button' | 'key' | 'mouse' | 'mouse-movement'>('axis');
  const [mouseMovementAxis, setMouseMovementAxis] = useState<'x' | 'y'>('x');
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
      if (mapping.type === 'mouse-movement') {
        setMappingType('mouse-movement');
        setMouseMovementAxis(mapping.axis);
      } else {
        setMappingType(mapping.type);
      }
    }
  }, [nodeId, fieldName, getMapping]);

  // Function to dispatch refresh event
  const triggerControlPanelRefresh = () => {
    console.log('Triggering control panel refresh');
    // Create and dispatch a custom event
    const refreshEvent = new CustomEvent(CONTROL_PANEL_REFRESH_EVENT);
    window.dispatchEvent(refreshEvent);
  };

  // Function to force a deep refresh of the controller system
  const forceSystemRefresh = () => {
    // 1. Clear any cached data in local component state
    setCurrentMapping(undefined);
    
    // 2. Try to reset controller state (even though we're not directly controlling it)
    if (typeof navigator.getGamepads === 'function') {
      console.log('Refreshing gamepad connection state');
      // This doesn't actually refresh the controllers, but it might help trigger reconnection
      navigator.getGamepads();
    }
    
    // 3. Dispatch refresh events - this is the key part
    console.log('Dispatching control panel refresh event');
    triggerControlPanelRefresh();
    
    // 4. NO PAGE RELOAD - it breaks the app
  };

  // Save the current mapping
  const handleSaveMapping = (mapping: ControllerMapping) => {
    saveMapping(nodeId, fieldName, mapping);
    setCurrentMapping(mapping);
    setIsOpen(false);
    
    // Simple timeout to force re-open the dialog to make sure changes are applied
    setTimeout(() => {
      triggerControlPanelRefresh();
    }, 500);
  };

  // Remove the current mapping
  const handleRemoveMapping = () => {
    removeMapping(nodeId, fieldName);
    setCurrentMapping(undefined);
    setIsOpen(false);
    
    // Simple timeout to force re-open the dialog to make sure changes are applied
    setTimeout(() => {
      triggerControlPanelRefresh();
    }, 500);
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

  // Handle mouse movement axis change
  const handleMouseMovementAxisChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setMouseMovementAxis(e.target.value as 'x' | 'y');
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
                onChange={(e) => setMappingType(e.target.value as 'axis' | 'button' | 'key' | 'mouse' | 'mouse-movement')}
                className="p-2 border rounded w-full"
              >
                <option value="axis">Controller Axis</option>
                <option value="button">Controller Button</option>
                <option value="key">Keyboard Key</option>
                <option value="mouse">Mouse Button/Wheel</option>
                <option value="mouse-movement">Mouse Movement</option>
              </select>
            </div>
            
            {/* Show axis selector for mouse movement */}
            {mappingType === 'mouse-movement' && (
              <div className="space-y-2">
                <label htmlFor="mouse-axis" className="text-sm font-medium">Mouse Axis</label>
                <select
                  id="mouse-axis"
                  value={mouseMovementAxis}
                  onChange={handleMouseMovementAxisChange}
                  className="p-2 border rounded w-full"
                >
                  <option value="x">Horizontal (X Axis)</option>
                  <option value="y">Vertical (Y Axis)</option>
                </select>
              </div>
            )}
            
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
            
            {mappingType === 'mouse-movement' && (
              <MouseMovementMappingForm {...baseProps} axis={mouseMovementAxis} />
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