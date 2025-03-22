"use client";

import React, { useState, useEffect, useRef } from 'react';
import { useController } from '@/hooks/use-controller';
import { useControllerMapping } from '@/hooks/use-controller-mapping';
import { AxisMapping, ButtonMapping, ControllerMapping, KeyMapping, MouseMapping, MouseXMapping, MouseYMapping } from '@/types/controller';
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
  const [mappingType, setMappingType] = useState<'axis' | 'button' | 'key' | 'mouse' | 'mouse-x' | 'mouse-y'>('axis');
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
  const [buttonMode, setButtonMode] = useState<'toggle' | 'momentary' | 'series' | 'axis'>('momentary');
  const [valueWhenPressed, setValueWhenPressed] = useState<string>('1');
  const [valueWhenReleased, setValueWhenReleased] = useState<string>('0');
  const [nextButtonIndex, setNextButtonIndex] = useState<number>(-1);
  const [valuesList, setValuesList] = useState<Array<string>>(['1', '2', '3']);
  const [currentValueIndex, setCurrentValueIndex] = useState<number>(0);
  
  // Track the type of input that was detected to avoid rerenders
  const [detectedInputType, setDetectedInputType] = useState<"none" | "button" | "axis" | "key" | "mouse">("none");
  
  // Track which button we're detecting (primary or next)
  const [detectingButtonTarget, setDetectingButtonTarget] = useState<"primary" | "next">("primary");
  
  // State for keyboard mapping
  const [keyCode, setKeyCode] = useState<string>('');
  const [nextKeyCode, setNextKeyCode] = useState<string>('');
  const [isListeningForKey, setIsListeningForKey] = useState(false);
  
  // State for mouse mapping
  const [mouseAction, setMouseAction] = useState<'move-x' | 'move-y' | 'wheel' | 'button'>('button');
  const [mouseButtonIndex, setMouseButtonIndex] = useState<number>(0);
  const [nextMouseButtonIndex, setNextMouseButtonIndex] = useState<number>(-1);
  const [isListeningForMouse, setIsListeningForMouse] = useState(false);
  
  // Reference to store detection state without causing re-renders
  const detectionStateRef = useRef({
    lastButtonStates: [] as boolean[],
    lastAxisValues: [] as number[],
    controllerIndex: -1,
    loopCount: 0,
    lastLogTime: 0
  });
  
  // Add a state to track axis values for feedback
  const [currentAxisValue, setCurrentAxisValue] = useState<number | null>(null);
  
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
        setButtonMode(buttonMapping.mode || 'momentary');
        setValueWhenPressed(buttonMapping.valueWhenPressed.toString());
        setValueWhenReleased(buttonMapping.valueWhenReleased?.toString() || '0');
        
        if (buttonMapping.mode === 'series') {
          setNextButtonIndex(buttonMapping.nextButtonIndex || -1);
          setValuesList(buttonMapping.valuesList?.map(v => v.toString()) || ['1', '2', '3']);
          setCurrentValueIndex(buttonMapping.currentValueIndex || 0);
        }
      } else if (mapping.type === 'key') {
        const keyMapping = mapping as KeyMapping;
        setKeyCode(keyMapping.keyCode);
        setButtonMode(keyMapping.mode || 'momentary');
        setValueWhenPressed(keyMapping.valueWhenPressed.toString());
        setValueWhenReleased(keyMapping.valueWhenReleased?.toString() || '0');
        
        if (keyMapping.mode === 'series') {
          setNextKeyCode(keyMapping.nextKeyCode || '');
          setValuesList(keyMapping.valuesList?.map(v => v.toString()) || ['1', '2', '3']);
          setCurrentValueIndex(keyMapping.currentValueIndex || 0);
        }
      } else if (mapping.type === 'mouse-x') {
        const mouseXMapping = mapping as MouseXMapping;
        setMultiplier(mouseXMapping.multiplier || 0.01);
        setMinOverride(mouseXMapping.minOverride);
        setMaxOverride(mouseXMapping.maxOverride);
      } else if (mapping.type === 'mouse-y') {
        const mouseYMapping = mapping as MouseYMapping;
        setMultiplier(mouseYMapping.multiplier || 0.01);
        setMinOverride(mouseYMapping.minOverride);
        setMaxOverride(mouseYMapping.maxOverride);
      } else if (mapping.type === 'mouse') {
        const mouseMapping = mapping as MouseMapping;
        setMouseAction(mouseMapping.action);
        
        if (mouseMapping.buttonIndex !== undefined) {
          setMouseButtonIndex(mouseMapping.buttonIndex);
        }
        
        if (mouseMapping.action === 'button') {
          setButtonMode(mouseMapping.mode || 'momentary');
          setValueWhenPressed(mouseMapping.valueWhenPressed?.toString() || '1');
          setValueWhenReleased(mouseMapping.valueWhenReleased?.toString() || '0');
          
          if (mouseMapping.mode === 'series') {
            setNextMouseButtonIndex(mouseMapping.nextButtonIndex || -1);
            setValuesList(mouseMapping.valuesList?.map(v => v.toString()) || ['1', '2', '3']);
            setCurrentValueIndex(mouseMapping.currentValueIndex || 0);
          }
        } else {
          // For mouse movement or wheel
          setMultiplier(mouseMapping.multiplier || 0.01);
          setMinOverride(mouseMapping.minOverride);
          setMaxOverride(mouseMapping.maxOverride);
        }
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
            if (detectingButtonTarget === "primary") {
              setButtonIndex(index);
            } else if (detectingButtonTarget === "next") {
              setNextButtonIndex(index);
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
            setCurrentAxisValue(axisValue);
          }
          
          // Always update current value for visual feedback when on the mapped axis
          if (mappingType === 'axis' && index === axisIndex) {
            setCurrentAxisValue(axisValue);
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
      // Button mapping - differentiate based on mode
      mapping = {
        type: 'button',
        nodeId,
        fieldName,
        buttonIndex,
        mode: buttonMode, 
        valueWhenPressed: parseFloat(valueWhenPressed) || valueWhenPressed,
        valueWhenReleased: parseFloat(valueWhenReleased) || valueWhenReleased
      } as ButtonMapping;
      
      // Add series-specific properties if in series mode
      if (buttonMode === 'series') {
        (mapping as ButtonMapping).nextButtonIndex = nextButtonIndex !== -1 ? nextButtonIndex : undefined;
        (mapping as ButtonMapping).valuesList = valuesList.map(v => parseFloat(v) || v);
        (mapping as ButtonMapping).currentValueIndex = currentValueIndex;
      }
    } else if (mappingType === 'key') {
      // Keyboard mapping
      mapping = {
        type: 'key',
        nodeId,
        fieldName,
        keyCode,
        mode: buttonMode,
        valueWhenPressed: parseFloat(valueWhenPressed) || valueWhenPressed,
        valueWhenReleased: parseFloat(valueWhenReleased) || valueWhenReleased
      } as KeyMapping;
      
      // Add series-specific properties if in series mode
      if (buttonMode === 'series') {
        (mapping as KeyMapping).nextKeyCode = nextKeyCode || undefined;
        (mapping as KeyMapping).valuesList = valuesList.map(v => parseFloat(v) || v);
        (mapping as KeyMapping).currentValueIndex = currentValueIndex;
      }
    } else if (mappingType === 'mouse-x') {
      mapping = {
        type: 'mouse-x',
        nodeId,
        fieldName,
        multiplier: multiplier || 0.01,
        minOverride,
        maxOverride
      } as MouseXMapping;
    } else if (mappingType === 'mouse-y') {
      mapping = {
        type: 'mouse-y',
        nodeId,
        fieldName,
        multiplier: multiplier || 0.01,
        minOverride,
        maxOverride
      } as MouseYMapping;
    } else if (mappingType === 'mouse') {
      // Mouse mapping
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
        } as MouseMapping;
        
        // Add series-specific properties if in series mode
        if (buttonMode === 'series') {
          (mapping as MouseMapping).nextAction = 'button';
          (mapping as MouseMapping).nextButtonIndex = nextMouseButtonIndex !== -1 ? nextMouseButtonIndex : undefined;
          (mapping as MouseMapping).valuesList = valuesList.map(v => parseFloat(v) || v);
          (mapping as MouseMapping).currentValueIndex = currentValueIndex;
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
        } as MouseMapping;
      }
    } else {
      // Default to a button mapping if something goes wrong
      mapping = {
        type: 'button',
        nodeId,
        fieldName,
        buttonIndex: 0,
        mode: 'momentary',
        valueWhenPressed: 1,
        valueWhenReleased: 0
      } as ButtonMapping;
    }
    
    saveMapping(nodeId, fieldName, mapping);
    setCurrentMapping(mapping);
    setIsListening(false);
    setIsListeningForKey(false);
    setIsListeningForMouse(false);
    setIsOpen(false);
  };
  
  // Remove the current mapping
  const handleRemoveMapping = () => {
    removeMapping(nodeId, fieldName);
    setCurrentMapping(undefined);
    setIsOpen(false);
  };
  
  // Start listening for controller input
  const handleStartListening = (target?: "primary" | "next") => {
    // Clear previous detection state
    setDetectedInput('');
    
    // Set which button we're detecting
    setDetectingButtonTarget(target || "primary");
    
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
  
  // Start listening for keyboard input
  const handleStartKeyListening = (target: "primary" | "next") => {
    setDetectingButtonTarget(target);
    setIsListeningForKey(true);
  };
  
  // Stop listening for keyboard input
  const handleStopKeyListening = () => {
    setIsListeningForKey(false);
  };
  
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
  
  // Start listening for mouse input
  const handleStartMouseListening = (target: "primary" | "next") => {
    setDetectingButtonTarget(target);
    setIsListeningForMouse(true);
  };
  
  // Stop listening for mouse input
  const handleStopMouseListening = () => {
    setIsListeningForMouse(false);
  };
  
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
                    : 'Press the button you want to map'}
                </p>
                
                {detectedInput && (
                  <p className="text-xs bg-white p-2 rounded border border-blue-100 font-medium text-blue-700">
                    Detected: {detectedInput}
                    {mappingType === 'axis' && <> → Setting Axis to {axisIndex}</>}
                    {mappingType === 'button' && detectingButtonTarget === "primary" && <> → Setting {buttonMode === 'series' ? 'Previous' : ''} Button to {buttonIndex}</>}
                    {mappingType === 'button' && detectingButtonTarget === "next" && <> → Setting Next Button to {nextButtonIndex}</>}
                  </p>
                )}
              </div>
            )}
            
            {/* Add visual feedback for axis movement during detection */}
            {mappingType === 'axis' && isListening && currentAxisValue !== null && (
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
            
            <div className="space-y-2">
              <Label htmlFor="mapping-type">Mapping Type</Label>
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
                      onClick={() => handleStartListening("primary")} 
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

<div className="space-y-2 mt-4">
                  <Label>Button Mode</Label>
                  <div className="grid grid-cols-3 gap-2">
                    <div className="flex items-center space-x-2">
                      <input
                        type="radio"
                        id="mode-momentary"
                        checked={buttonMode === 'momentary'}
                        onChange={() => setButtonMode('momentary')}
                        className="w-4 h-4"
                      />
                      <Label htmlFor="mode-momentary">Momentary</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <input
                        type="radio"
                        id="mode-toggle"
                        checked={buttonMode === 'toggle'}
                        onChange={() => setButtonMode('toggle')}
                        className="w-4 h-4"
                      />
                      <Label htmlFor="mode-toggle">Toggle</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <input
                        type="radio"
                        id="mode-series"
                        checked={buttonMode === 'series'}
                        onChange={() => setButtonMode('series')}
                        className="w-4 h-4"
                      />
                      <Label htmlFor="mode-series">Series</Label>
                    </div>
                  </div>
                </div>
            
            {mappingType === 'button' && (
              <div className="space-y-2">
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
                
                {buttonMode !== 'series' && (
                  <div>
                    <Label htmlFor="value-pressed">Value When Pressed</Label>
                    <Input 
                      id="value-pressed"
                      value={valueWhenPressed} 
                      onChange={(e) => setValueWhenPressed(e.target.value)} 
                    />
                  </div>
                )}
                
                {buttonMode !== 'series' && (
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
                    
                    <div>
                      <Label>Values to Cycle Through</Label>
                      <div className="border rounded-md p-2 bg-gray-50 space-y-2 mt-1 max-h-40 overflow-y-auto">
                        {valuesList.map((value, index) => (
                          <div key={index} className="flex gap-2 items-center">
                            <Input
                              value={value}
                              onChange={(e) => {
                                const newList = [...valuesList];
                                newList[index] = e.target.value;
                                setValuesList(newList);
                              }}
                              className="flex-1"
                            />
                            <Button
                              variant="outline" 
                              size="sm"
                              onClick={() => {
                                const newList = valuesList.filter((_, i) => i !== index);
                                setValuesList(newList);
                                if (currentValueIndex >= newList.length) {
                                  setCurrentValueIndex(newList.length - 1);
                                }
                              }}
                            >
                              ✕
                            </Button>
                          </div>
                        ))}
                        <Button
                          variant="outline"
                          size="sm"
                          className="w-full"
                          onClick={() => setValuesList([...valuesList, ''])}
                        >
                          Add Value
                        </Button>
                      </div>
                    </div>
                  </>
                )}
              </div>
            )}
            
            {mappingType === 'key' && (
              <div className="space-y-2">
                <div>
                  <Label htmlFor="key-code">Key</Label>
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
                
                {buttonMode !== 'series' && (
                  <div>
                    <Label htmlFor="value-pressed">Value When Pressed</Label>
                    <Input 
                      id="value-pressed"
                      value={valueWhenPressed} 
                      onChange={(e) => setValueWhenPressed(e.target.value)} 
                    />
                  </div>
                )}
                
                {buttonMode !== 'series' && (
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
                    
                    <div>
                      <Label>Values to Cycle Through</Label>
                      <div className="border rounded-md p-2 bg-gray-50 space-y-2 mt-1 max-h-40 overflow-y-auto">
                        {valuesList.map((value, index) => (
                          <div key={index} className="flex gap-2 items-center">
                            <Input
                              value={value}
                              onChange={(e) => {
                                const newList = [...valuesList];
                                newList[index] = e.target.value;
                                setValuesList(newList);
                              }}
                              className="flex-1"
                            />
                            <Button
                              variant="outline" 
                              size="sm"
                              onClick={() => {
                                const newList = valuesList.filter((_, i) => i !== index);
                                setValuesList(newList);
                                if (currentValueIndex >= newList.length) {
                                  setCurrentValueIndex(newList.length - 1);
                                }
                              }}
                            >
                              ✕
                            </Button>
                          </div>
                        ))}
                        <Button
                          variant="outline"
                          size="sm"
                          className="w-full"
                          onClick={() => setValuesList([...valuesList, ''])}
                        >
                          Add Value
                        </Button>
                      </div>
                    </div>
                  </>
                )}
              </div>
            )}
            
            {/* Mouse X Movement Mapping UI */}
            {mappingType === 'mouse-x' && (
              <div className="space-y-2">
                <div>
                  <div className="flex justify-between items-center mb-1">
                    <Label htmlFor="multiplier-mouse-x">Sensitivity</Label>
                    <Button 
                      variant="outline" 
                      size="sm"
                      onClick={() => onChange && onChange(0)}
                      className="text-xs"
                    >
                      Reset Value
                    </Button>
                  </div>
                  <Input 
                    id="multiplier-mouse-x"
                    type="number" 
                    step="0.001"
                    value={multiplier} 
                    onChange={(e) => setMultiplier(parseFloat(e.target.value) || 0.01)} 
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Higher values = more sensitive horizontal movement
                  </p>
                </div>
                
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <Label htmlFor="min-override-mouse-x">Min Value</Label>
                    <Input 
                      id="min-override-mouse-x"
                      type="number" 
                      value={minOverride !== undefined ? minOverride : ''} 
                      placeholder={inputMin?.toString() || 'Default'}
                      onChange={(e) => setMinOverride(e.target.value ? parseFloat(e.target.value) : undefined)} 
                    />
                  </div>
                  <div>
                    <Label htmlFor="max-override-mouse-x">Max Value</Label>
                    <Input 
                      id="max-override-mouse-x"
                      type="number" 
                      value={maxOverride !== undefined ? maxOverride : ''} 
                      placeholder={inputMax?.toString() || 'Default'}
                      onChange={(e) => setMaxOverride(e.target.value ? parseFloat(e.target.value) : undefined)} 
                    />
                  </div>
                </div>
              </div>
            )}
            
            {/* Mouse Y Movement Mapping UI */}
            {mappingType === 'mouse-y' && (
              <div className="space-y-2">
                <div>
                  <div className="flex justify-between items-center mb-1">
                    <Label htmlFor="multiplier-mouse-y">Sensitivity</Label>
                    <Button 
                      variant="outline" 
                      size="sm"
                      onClick={() => onChange && onChange(0)}
                      className="text-xs"
                    >
                      Reset Value
                    </Button>
                  </div>
                  <Input 
                    id="multiplier-mouse-y"
                    type="number" 
                    step="0.001"
                    value={multiplier} 
                    onChange={(e) => setMultiplier(parseFloat(e.target.value) || 0.01)} 
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Higher values = more sensitive vertical movement
                  </p>
                </div>
                
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <Label htmlFor="min-override-mouse-y">Min Value</Label>
                    <Input 
                      id="min-override-mouse-y"
                      type="number" 
                      value={minOverride !== undefined ? minOverride : ''} 
                      placeholder={inputMin?.toString() || 'Default'}
                      onChange={(e) => setMinOverride(e.target.value ? parseFloat(e.target.value) : undefined)} 
                    />
                  </div>
                  <div>
                    <Label htmlFor="max-override-mouse-y">Max Value</Label>
                    <Input 
                      id="max-override-mouse-y"
                      type="number" 
                      value={maxOverride !== undefined ? maxOverride : ''} 
                      placeholder={inputMax?.toString() || 'Default'}
                      onChange={(e) => setMaxOverride(e.target.value ? parseFloat(e.target.value) : undefined)} 
                    />
                  </div>
                </div>
              </div>
            )}
            
            {mappingType === 'mouse' && (
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
                    
                    <div className="space-y-2 mt-4">
                      <Label>Button Mode</Label>
                      <div className="grid grid-cols-3 gap-2">
                        <div className="flex items-center space-x-2">
                          <input
                            type="radio"
                            id="mode-momentary-mouse"
                            checked={buttonMode === 'momentary'}
                            onChange={() => setButtonMode('momentary')}
                            className="w-4 h-4"
                          />
                          <Label htmlFor="mode-momentary-mouse">Momentary</Label>
                        </div>
                        <div className="flex items-center space-x-2">
                          <input
                            type="radio"
                            id="mode-toggle-mouse"
                            checked={buttonMode === 'toggle'}
                            onChange={() => setButtonMode('toggle')}
                            className="w-4 h-4"
                          />
                          <Label htmlFor="mode-toggle-mouse">Toggle</Label>
                        </div>
                        <div className="flex items-center space-x-2">
                          <input
                            type="radio"
                            id="mode-series-mouse"
                            checked={buttonMode === 'series'}
                            onChange={() => setButtonMode('series')}
                            className="w-4 h-4"
                          />
                          <Label htmlFor="mode-series-mouse">Series</Label>
                        </div>
                      </div>
                    </div>
                    
                    {buttonMode !== 'series' && (
                      <div>
                        <Label htmlFor="value-pressed-mouse">Value When Pressed</Label>
                        <Input 
                          id="value-pressed-mouse"
                          value={valueWhenPressed} 
                          onChange={(e) => setValueWhenPressed(e.target.value)} 
                        />
                      </div>
                    )}
                    
                    {buttonMode !== 'series' && (
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
                        
                        <div>
                          <Label>Values to Cycle Through</Label>
                          <div className="border rounded-md p-2 bg-gray-50 space-y-2 mt-1 max-h-40 overflow-y-auto">
                            {valuesList.map((value, index) => (
                              <div key={index} className="flex gap-2 items-center">
                                <Input
                                  value={value}
                                  onChange={(e) => {
                                    const newList = [...valuesList];
                                    newList[index] = e.target.value;
                                    setValuesList(newList);
                                  }}
                                  className="flex-1"
                                />
                                <Button
                                  variant="outline" 
                                  size="sm"
                                  onClick={() => {
                                    const newList = valuesList.filter((_, i) => i !== index);
                                    setValuesList(newList);
                                    if (currentValueIndex >= newList.length) {
                                      setCurrentValueIndex(newList.length - 1);
                                    }
                                  }}
                                >
                                  ✕
                                </Button>
                              </div>
                            ))}
                            <Button
                              variant="outline"
                              size="sm"
                              className="w-full"
                              onClick={() => setValuesList([...valuesList, ''])}
                            >
                              Add Value
                            </Button>
                          </div>
                        </div>
                      </>
                    )}
                  </>
                )}
                
                {(mouseAction === 'wheel') && (
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