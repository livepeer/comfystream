import { useEffect, useRef } from 'react';
import { useController } from './use-controller';
import { AxisMapping, ButtonMapping, ControllerMapping } from '@/types/controller';

// Hook for processing controller input according to a mapping
export function useControllerInput(
  mapping: ControllerMapping | undefined,
  onValueChange: (value: any) => void
) {
  const { controllers, wasButtonJustPressed, hasAxisChanged } = useController();
  const valueRef = useRef<any>(null); // Track current value to avoid unnecessary updates
  const mappingRef = useRef<ControllerMapping | undefined>(mapping);
  const lastPollTimeRef = useRef(0);
  
  // Button state tracking to improve detection
  const buttonStatesRef = useRef<Record<number, boolean>>({});
  
  // Update the mapping ref when the mapping changes
  useEffect(() => {
    if (mapping !== mappingRef.current) {
      console.log('Controller mapping changed:', mapping);
      mappingRef.current = mapping;
      
      // Reset value reference when mapping changes
      valueRef.current = null;
      
      // Reset button states
      buttonStatesRef.current = {};
    }
  }, [mapping]);
  
  useEffect(() => {
    if (!mapping || controllers.length === 0) return;
    
    console.log('Setting up controller input processing for:', mapping);
    
    const controller = controllers[0]; // Use the first controller for now
    const controllerIndex = 0;
    
    // Different processing based on mapping type
    const intervalId = setInterval(() => {
      // Rate limit polling to reduce console spam
      const now = Date.now();
      const shouldLog = now - lastPollTimeRef.current > 5000; // Log every 5 seconds
      if (shouldLog) {
        lastPollTimeRef.current = now;
        console.log('Processing controller input for', mapping.type, 'mapping');
      }
      
      // Get fresh gamepad data each time
      const gamepads = navigator.getGamepads();
      const freshController = gamepads[controller.index];
      
      if (!freshController) {
        if (shouldLog) console.warn('Controller lost during processing');
        return;
      }
      
      switch (mapping.type) {
        case 'axis': {
          const axisMapping = mapping as AxisMapping;
          const axisIndex = axisMapping.axisIndex;
          
          // Validate axis index
          if (axisIndex < 0 || axisIndex >= freshController.axes.length) {
            if (shouldLog) console.warn(`Invalid axis index: ${axisIndex}`);
            return;
          }
          
          // For axes, we should always check the current value, not just when it changes
          // This ensures continuous updates for smoother control
          const axisValue = freshController.axes[axisIndex] || 0;
          
          if (shouldLog) {
            console.log(`Axis ${axisIndex} value:`, axisValue.toFixed(4));
          }
          
          // Apply scaling and constraints
          let scaledValue = axisValue * (axisMapping.multiplier || 1);
          
          // Apply min/max overrides if provided
          if (axisMapping.minOverride !== undefined && axisMapping.maxOverride !== undefined) {
            // Map from [-1, 1] to [min, max]
            scaledValue = 
              axisMapping.minOverride + 
              ((scaledValue + 1) / 2) * (axisMapping.maxOverride - axisMapping.minOverride);
            
            if (shouldLog) {
              console.log(`Scaled value (with min/max):`, scaledValue.toFixed(4));
            }
          }
          
          // For more precise control, round to a reasonable number of decimal places
          // This prevents tiny fluctuations from triggering updates
          const roundedValue = parseFloat(scaledValue.toFixed(3));
          
          // Only update if the value has changed significantly
          if (valueRef.current === null || Math.abs(valueRef.current - roundedValue) > 0.002) {
            if (shouldLog || Math.abs(valueRef.current - roundedValue) > 0.02) {
              console.log(`Sending axis value update:`, roundedValue);
            }
            valueRef.current = roundedValue;
            onValueChange(roundedValue);
          }
          break;
        }
        
        case 'button': {
          const buttonMapping = mapping as ButtonMapping;
          const buttonIndex = buttonMapping.buttonIndex;
          
          // Validate button index
          if (buttonIndex < 0 || buttonIndex >= freshController.buttons.length) {
            if (shouldLog) console.warn(`Invalid button index: ${buttonIndex}`);
            return;
          }
          
          const isPressed = freshController.buttons[buttonIndex]?.pressed || false;
          
          // Store previous state
          const wasPressed = buttonStatesRef.current[buttonIndex] || false;
          
          // Update button state cache
          buttonStatesRef.current[buttonIndex] = isPressed;
          
          // Handle based on button mode
          if (buttonMapping.mode === 'toggle') {
            // Toggle mode - change value only when button is just pressed (rising edge)
            if (isPressed && !wasPressed) {
              const newValue = valueRef.current === buttonMapping.valueWhenPressed 
                ? (buttonMapping.valueWhenReleased || '') 
                : buttonMapping.valueWhenPressed;
              
              console.log(`Button ${buttonIndex} toggled to:`, newValue);
              valueRef.current = newValue;
              onValueChange(newValue);
            }
          } else if (buttonMapping.mode === 'series') {
            // Series mode - cycle through values when button is just pressed
            if (isPressed && !wasPressed) {
              // Check if we have values to cycle through
              if (buttonMapping.valuesList && buttonMapping.valuesList.length > 0) {
                // Use currentValueIndex or initialize to 0
                let currentIndex = buttonMapping.currentValueIndex || 0;
                
                // Move to next value in the list
                currentIndex = (currentIndex + 1) % buttonMapping.valuesList.length;
                
                const newValue = buttonMapping.valuesList[currentIndex];
                console.log(`Button ${buttonIndex} cycled to value ${currentIndex}:`, newValue);
                
                // Store the new index in the mapping reference
                if (mappingRef.current && mappingRef.current.type === 'button') {
                  (mappingRef.current as ButtonMapping).currentValueIndex = currentIndex;
                }
                
                valueRef.current = newValue;
                onValueChange(newValue);
              }
            }
            
            // Check for nextButtonIndex if defined (for backward cycling)
            if (buttonMapping.nextButtonIndex !== undefined && 
                buttonMapping.nextButtonIndex >= 0 &&
                buttonMapping.nextButtonIndex < freshController.buttons.length) {
              
              const nextIsPressed = freshController.buttons[buttonMapping.nextButtonIndex]?.pressed || false;
              const nextWasPressed = buttonStatesRef.current[buttonMapping.nextButtonIndex] || false;
              
              // Update next button state
              buttonStatesRef.current[buttonMapping.nextButtonIndex] = nextIsPressed;
              
              // Handle backward cycling
              if (nextIsPressed && !nextWasPressed && 
                  buttonMapping.valuesList && buttonMapping.valuesList.length > 0) {
                
                // Use currentValueIndex or initialize to 0
                let currentIndex = buttonMapping.currentValueIndex || 0;
                
                // Move to previous value in the list (with wrap-around)
                currentIndex = (currentIndex - 1 + buttonMapping.valuesList.length) % buttonMapping.valuesList.length;
                
                const newValue = buttonMapping.valuesList[currentIndex];
                console.log(`Button ${buttonMapping.nextButtonIndex} cycled to previous value ${currentIndex}:`, newValue);
                
                // Store the new index in the mapping reference
                if (mappingRef.current && mappingRef.current.type === 'button') {
                  (mappingRef.current as ButtonMapping).currentValueIndex = currentIndex;
                }
                
                valueRef.current = newValue;
                onValueChange(newValue);
              }
            }
          } else {
            // Momentary mode (default) - value follows button state
            const newValue = isPressed 
              ? buttonMapping.valueWhenPressed 
              : (buttonMapping.valueWhenReleased !== undefined ? buttonMapping.valueWhenReleased : valueRef.current);
            
            // Only send updates on state change
            if (isPressed !== wasPressed || valueRef.current === null) {
              console.log(`Button ${buttonIndex} state changed to:`, isPressed, 'value:', newValue);
              valueRef.current = newValue;
              onValueChange(newValue);
            }
          }
          break;
        }
      }
    }, 16); // Poll at 60Hz for smoother control
    
    return () => {
      console.log('Cleaning up controller input processing');
      clearInterval(intervalId);
    };
  }, [controllers, mapping, onValueChange, wasButtonJustPressed, hasAxisChanged]);
  
  return null; // This hook doesn't return anything, it just applies the effects
} 