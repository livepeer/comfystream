"use client";

import React, { useState, useEffect, useRef } from 'react';
import { useController } from '@/hooks/use-controller';
import { useControllerMapping } from '@/hooks/use-controller-mapping';
import { AxisMapping, ButtonMapping, ControllerMapping, PromptListMapping } from '@/types/controller';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select } from '@/components/ui/select';
import { GamepadIcon } from 'lucide-react';

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
  const [isListening, setIsListening] = useState(false);
  const [detectedInput, setDetectedInput] = useState<string>('');
  const [mappingType, setMappingType] = useState<'axis' | 'button' | 'promptList'>('axis');
  const [currentMapping, setCurrentMapping] = useState<ControllerMapping | undefined>(
    getMapping(nodeId, fieldName)
  );
  
  // Form state for axis mapping
  const [axisIndex, setAxisIndex] = useState<number>(0);
  const [multiplier, setMultiplier] = useState<number>(1);
  const [minOverride, setMinOverride] = useState<number | undefined>(inputMin);
  const [maxOverride, setMaxOverride] = useState<number | undefined>(inputMax);
  
  // Form state for button mapping
  const [buttonIndex, setButtonIndex] = useState<number>(0);
  const [toggleMode, setToggleMode] = useState<boolean>(false);
  const [valueWhenPressed, setValueWhenPressed] = useState<string>('1');
  const [valueWhenReleased, setValueWhenReleased] = useState<string>('0');
  
  // Form state for prompt list mapping
  const [nextButtonIndex, setNextButtonIndex] = useState<number>(1);
  const [prevButtonIndex, setPrevButtonIndex] = useState<number>(0);
  
  // Track the type of input that was detected to avoid rerenders
  const [detectedInputType, setDetectedInputType] = useState<"none" | "button" | "axis">("none");
  
  // Reference to store detection state without causing re-renders
  const detectionStateRef = useRef({
    lastButtonStates: [] as boolean[],
    lastAxisValues: [] as number[],
    controllerIndex: -1,
    loopCount: 0,
    lastLogTime: 0
  });
  
  // Update local state when mapping changes
  useEffect(() => {
    const mapping = getMapping(nodeId, fieldName);
    setCurrentMapping(mapping);
    
    if (mapping) {
      setMappingType(mapping.type);
      
      if (mapping.type === 'axis') {
        const axisMapping = mapping as AxisMapping;
        setAxisIndex(axisMapping.axisIndex);
        setMultiplier(axisMapping.multiplier);
        setMinOverride(axisMapping.minOverride);
        setMaxOverride(axisMapping.maxOverride);
      } else if (mapping.type === 'button') {
        const buttonMapping = mapping as ButtonMapping;
        setButtonIndex(buttonMapping.buttonIndex);
        setToggleMode(buttonMapping.toggleMode);
        setValueWhenPressed(buttonMapping.valueWhenPressed.toString());
        setValueWhenReleased(buttonMapping.valueWhenReleased?.toString() || '0');
      } else if (mapping.type === 'promptList') {
        const promptListMapping = mapping as PromptListMapping;
        setNextButtonIndex(promptListMapping.nextButtonIndex);
        setPrevButtonIndex(promptListMapping.prevButtonIndex);
      }
    }
  }, [nodeId, fieldName, getMapping]);
  
  // Fix the controller detection in the listening mode
  useEffect(() => {
    if (!isListening || controllers.length === 0) return;
    
    console.log("Entering listening mode for controller input");
    console.log("Available controllers:", controllers.length);
    
    // Get the first controller
    const controller = controllers[0];
    console.log(`Using controller: ${controller.id}`);
    console.log(`Buttons: ${controller.buttons.length}, Axes: ${controller.axes.length}`);
    
    // Store initial button states and axis values for calibration
    const initialButtonStates = Array.from(controller.buttons).map(b => b.pressed);
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
      
      // Check for button presses (any button that wasn't pressed initially but is now)
      freshController.buttons.forEach((button, index) => {
        if (button.pressed && !initialButtonStates[index]) {
          console.log(`Button ${index} pressed`);
          setDetectedInput(`Button ${index}`);
          
          if (mappingType === 'button') {
            setButtonIndex(index);
          } else if (mappingType === 'promptList') {
            // For prompt list, first press sets nextButton, second press sets prevButton
            if (detectedInput === '') {
              setNextButtonIndex(index);
            } else {
              setPrevButtonIndex(index);
            }
          }
        }
      });
      
      // Check for significant axis movements COMPARED TO INITIAL VALUES
      // This is the key difference - compare to initial values, not just checking absolute values
      if (mappingType === 'axis') {
        freshController.axes.forEach((axisValue, index) => {
          const initialValue = initialAxisValues[index];
          const movement = Math.abs(axisValue - initialValue);
          
          // Use movement threshold similar to old branch
          if (movement > 0.3) {
            console.log(`Axis ${index} moved: from ${initialValue.toFixed(2)} to ${axisValue.toFixed(2)} (change: ${movement.toFixed(2)})`);
            setDetectedInput(`Axis ${index}`);
            setAxisIndex(index);
          }
        });
      }
    };
    
    // Poll for input changes
    const intervalId = setInterval(detectChanges, 50);
    
    // Cleanup interval on unmount or when listening stops
    return () => {
      console.log("Exiting listening mode");
      clearInterval(intervalId);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isListening, controllers.length]); // Only depend on listening state and if controllers are available
  
  // Add debug button to manually check controllers
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
  
  // Save the current mapping
  const handleSaveMapping = () => {
    let mapping: ControllerMapping;
    
    if (mappingType === 'axis') {
      mapping = {
        type: 'axis',
        nodeId,
        fieldName,
        axisIndex,
        multiplier,
        minOverride,
        maxOverride
      } as AxisMapping;
    } else if (mappingType === 'button') {
      mapping = {
        type: 'button',
        nodeId,
        fieldName,
        buttonIndex,
        toggleMode,
        valueWhenPressed: parseFloat(valueWhenPressed) || valueWhenPressed,
        valueWhenReleased: parseFloat(valueWhenReleased) || valueWhenReleased
      } as ButtonMapping;
    } else {
      mapping = {
        type: 'promptList',
        nodeId,
        fieldName,
        nextButtonIndex,
        prevButtonIndex,
        currentPromptIndex: 0
      } as PromptListMapping;
    }
    
    saveMapping(nodeId, fieldName, mapping);
    setCurrentMapping(mapping);
    setIsListening(false);
    setIsOpen(false);
  };
  
  // Remove the current mapping
  const handleRemoveMapping = () => {
    removeMapping(nodeId, fieldName);
    setCurrentMapping(undefined);
    setIsOpen(false);
  };
  
  // Start listening for controller input
  const handleStartListening = () => {
    // Clear previous detection state
    setDetectedInput('');
    
    // Force a refresh of controllers
    if (typeof navigator.getGamepads === 'function') {
      console.log("Refreshing controllers before detection");
      const gamepads = navigator.getGamepads();
      console.log(`Found ${gamepads.length} controllers via getGamepads()`);
      
      // If we have controllers, let's use them
      const activePads: Gamepad[] = [];
      for (let i = 0; i < gamepads.length; i++) {
        if (gamepads[i]) {
          console.log(`Found active controller: ${gamepads[i]?.id}`);
          activePads.push(gamepads[i]!);
        }
      }
      
      // If we don't have any controllers in state but found some now, log a warning
      if (controllers.length === 0 && activePads.length > 0) {
        console.warn("Controllers were found but not in state - detection may not work correctly");
      }
    }
    
    // Start listening mode
    setIsListening(true);
  };
  
  // Stop listening
  const handleStopListening = () => {
    setIsListening(false);
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
            <div className="flex justify-between items-center">
              {isControllerConnected ? (
                <span className="text-xs text-green-500">Controller Connected</span>
              ) : (
                <span className="text-xs text-red-500">No Controller Detected</span>
              )}
              <Button 
                variant="outline" 
                size="sm" 
                onClick={checkControllers}
              >
                Refresh Controllers
              </Button>
            </div>
            
            {/* Debug Info */}
            <div className="bg-gray-100 p-2 rounded text-xs">
              <p>Controllers: {controllers.length}</p>
              {controllers.map((controller, idx) => (
                <div key={idx} className="mt-1">
                  <p>#{idx}: {controller.id}</p>
                  <p>{controller.buttons.length} buttons, {controller.axes.length} axes</p>
                </div>
              ))}
            </div>
            
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
                  {mappingType === 'axis' 
                    ? 'Move a joystick or trigger you want to map' 
                    : mappingType === 'button' 
                      ? 'Press the button you want to map' 
                      : 'First press button for "Next", then press button for "Previous"'}
                </p>
                
                {detectedInput && (
                  <p className="text-xs bg-white p-2 rounded border border-blue-100 font-medium text-blue-700">
                    Detected: {detectedInput}
                    {mappingType === 'axis' && <> → Setting Axis to {axisIndex}</>}
                    {mappingType === 'button' && <> → Setting Button to {buttonIndex}</>}
                    {mappingType === 'promptList' && !detectedInput.includes('Next') && <> → Setting Next Button to {nextButtonIndex}</>}
                    {mappingType === 'promptList' && detectedInput.includes('Next') && <> → Setting Prev Button to {prevButtonIndex}</>}
                  </p>
                )}
              </div>
            )}
            
            <div className="space-y-2">
              <Label htmlFor="mapping-type">Mapping Type</Label>
              <select
                id="mapping-type"
                value={mappingType}
                onChange={(e) => setMappingType(e.target.value as 'axis' | 'button' | 'promptList')}
                className="p-2 border rounded w-full"
              >
                <option value="axis">Axis</option>
                <option value="button">Button</option>
                <option value="promptList">Prompt List</option>
              </select>
            </div>
            
            {mappingType === 'axis' && (
              <div className="space-y-2">
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
                
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <Label htmlFor="min-override">Min Value</Label>
                    <Input 
                      id="min-override"
                      type="number" 
                      value={minOverride !== undefined ? minOverride : ''} 
                      placeholder={inputMin?.toString() || 'Default'}
                      onChange={(e) => setMinOverride(e.target.value ? parseFloat(e.target.value) : undefined)} 
                    />
                  </div>
                  <div>
                    <Label htmlFor="max-override">Max Value</Label>
                    <Input 
                      id="max-override"
                      type="number" 
                      value={maxOverride !== undefined ? maxOverride : ''} 
                      placeholder={inputMax?.toString() || 'Default'}
                      onChange={(e) => setMaxOverride(e.target.value ? parseFloat(e.target.value) : undefined)} 
                    />
                  </div>
                </div>
              </div>
            )}
            
            {mappingType === 'button' && (
              <div className="space-y-2">
                <div>
                  <Label htmlFor="button-index">Button</Label>
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
                      onClick={handleStartListening} 
                      disabled={!isControllerConnected || isListening}
                      size="sm"
                    >
                      Detect
                    </Button>
                  </div>
                </div>
                
                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="toggle-mode"
                    checked={toggleMode}
                    onChange={(e) => setToggleMode(e.target.checked)}
                    className="w-4 h-4"
                  />
                  <Label htmlFor="toggle-mode">Toggle Mode</Label>
                </div>
                
                <div>
                  <Label htmlFor="value-pressed">Value When Pressed</Label>
                  <Input 
                    id="value-pressed"
                    value={valueWhenPressed} 
                    onChange={(e) => setValueWhenPressed(e.target.value)} 
                  />
                </div>
                
                <div>
                  <Label htmlFor="value-released">Value When Released {toggleMode && '(Toggle Off)'}</Label>
                  <Input 
                    id="value-released"
                    value={valueWhenReleased} 
                    onChange={(e) => setValueWhenReleased(e.target.value)} 
                  />
                </div>
              </div>
            )}
            
            {mappingType === 'promptList' && (
              <div className="space-y-2">
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <Label htmlFor="next-button">Next Prompt Button ({nextButtonIndex})</Label>
                    <Button 
                      onClick={handleStartListening} 
                      disabled={!isControllerConnected || isListening}
                      size="sm"
                    >
                      Detect Both
                    </Button>
                  </div>
                  <Input 
                    id="next-button"
                    type="number" 
                    min="0" 
                    max="20"
                    value={nextButtonIndex} 
                    onChange={(e) => setNextButtonIndex(parseInt(e.target.value) || 0)} 
                  />
                </div>
                
                <div className="mt-2">
                  <Label htmlFor="prev-button">Previous Prompt Button ({prevButtonIndex})</Label>
                  <Input 
                    id="prev-button"
                    type="number" 
                    min="0" 
                    max="20"
                    value={prevButtonIndex} 
                    onChange={(e) => setPrevButtonIndex(parseInt(e.target.value) || 0)} 
                  />
                </div>
              </div>
            )}
            
            <div className="flex justify-between mt-6">
              {currentMapping ? (
                <>
                  <Button variant="destructive" size="sm" onClick={handleRemoveMapping}>
                    Remove Mapping
                  </Button>
                  <Button variant="default" size="sm" onClick={handleSaveMapping}>
                    Update Mapping
                  </Button>
                </>
              ) : (
                <Button className="ml-auto" onClick={handleSaveMapping}>
                  Save Mapping
                </Button>
              )}
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
} 