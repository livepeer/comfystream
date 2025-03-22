import { useEffect, useRef } from 'react';
import { useController } from './use-controller';
import { AxisMapping, ButtonMapping, ControllerMapping, PromptListMapping } from '@/types/controller';

// Hook for processing controller input according to a mapping
export function useControllerInput(
  mapping: ControllerMapping | undefined,
  onValueChange: (value: any) => void
) {
  const { controllers, wasButtonJustPressed, hasAxisChanged } = useController();
  const valueRef = useRef<any>(null); // Track current value to avoid unnecessary updates
  const mappingRef = useRef<ControllerMapping | undefined>(mapping);
  const lastPollTimeRef = useRef(0);
  
  // Update the mapping ref when the mapping changes
  useEffect(() => {
    if (mapping !== mappingRef.current) {
      console.log('Controller mapping changed:', mapping);
      mappingRef.current = mapping;
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
          
          if (hasAxisChanged(controllerIndex, axisIndex)) {
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
            
            // Only update if the value has changed significantly
            if (valueRef.current === null || Math.abs(valueRef.current - scaledValue) > 0.01) {
              if (shouldLog) {
                console.log(`Sending value update:`, scaledValue);
              }
              valueRef.current = scaledValue;
              onValueChange(scaledValue);
            }
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
          
          if (buttonMapping.toggleMode) {
            // Toggle mode - change value only when button is just pressed
            if (wasButtonJustPressed(controllerIndex, buttonIndex)) {
              const newValue = valueRef.current === buttonMapping.valueWhenPressed 
                ? (buttonMapping.valueWhenReleased || '') 
                : buttonMapping.valueWhenPressed;
              
              console.log(`Button ${buttonIndex} toggled to:`, newValue);
              valueRef.current = newValue;
              onValueChange(newValue);
            }
          } else {
            // Direct mode - value follows button state
            const isPressed = freshController.buttons[buttonIndex]?.pressed || false;
            const newValue = isPressed 
              ? buttonMapping.valueWhenPressed 
              : (buttonMapping.valueWhenReleased !== undefined ? buttonMapping.valueWhenReleased : valueRef.current);
            
            if (valueRef.current !== newValue) {
              if (shouldLog) {
                console.log(`Button ${buttonIndex} state changed to:`, isPressed, 'value:', newValue);
              }
              valueRef.current = newValue;
              onValueChange(newValue);
            }
          }
          break;
        }
        
        case 'promptList': {
          const promptListMapping = mapping as PromptListMapping;
          
          // Validate button indices
          if (promptListMapping.nextButtonIndex < 0 || 
              promptListMapping.nextButtonIndex >= freshController.buttons.length ||
              promptListMapping.prevButtonIndex < 0 || 
              promptListMapping.prevButtonIndex >= freshController.buttons.length) {
            if (shouldLog) console.warn('Invalid button indices for prompt list navigation');
            return;
          }
          
          // Check for next/prev button presses
          if (wasButtonJustPressed(controllerIndex, promptListMapping.nextButtonIndex)) {
            console.log('Next prompt button pressed');
            onValueChange('__NEXT_PROMPT__'); // Special value handled by the control panel
          }
          
          if (wasButtonJustPressed(controllerIndex, promptListMapping.prevButtonIndex)) {
            console.log('Previous prompt button pressed');
            onValueChange('__PREV_PROMPT__'); // Special value handled by the control panel
          }
          break;
        }
      }
    }, 50); // Poll at 20Hz
    
    return () => {
      console.log('Cleaning up controller input processing');
      clearInterval(intervalId);
    };
  }, [controllers, mapping, onValueChange, wasButtonJustPressed, hasAxisChanged]);
  
  return null; // This hook doesn't return anything, it just applies the effects
} 