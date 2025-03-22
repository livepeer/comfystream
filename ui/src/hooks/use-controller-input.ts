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
  
  useEffect(() => {
    if (!mapping || controllers.length === 0) return;
    
    const controller = controllers[0]; // Use the first controller for now
    const controllerIndex = 0;
    
    // Different processing based on mapping type
    const intervalId = setInterval(() => {
      switch (mapping.type) {
        case 'axis': {
          const axisMapping = mapping as AxisMapping;
          if (hasAxisChanged(controllerIndex, axisMapping.axisIndex)) {
            const axisValue = controller.axes[axisMapping.axisIndex] || 0;
            
            // Apply scaling and constraints
            let scaledValue = axisValue * (axisMapping.multiplier || 1);
            
            // Apply min/max overrides if provided
            if (axisMapping.minOverride !== undefined && axisMapping.maxOverride !== undefined) {
              // Map from [-1, 1] to [min, max]
              scaledValue = 
                axisMapping.minOverride + 
                ((scaledValue + 1) / 2) * (axisMapping.maxOverride - axisMapping.minOverride);
            }
            
            // Only update if the value has changed significantly
            if (valueRef.current === null || Math.abs(valueRef.current - scaledValue) > 0.01) {
              valueRef.current = scaledValue;
              onValueChange(scaledValue);
            }
          }
          break;
        }
        
        case 'button': {
          const buttonMapping = mapping as ButtonMapping;
          const buttonIndex = buttonMapping.buttonIndex;
          
          if (buttonMapping.toggleMode) {
            // Toggle mode - change value only when button is just pressed
            if (wasButtonJustPressed(controllerIndex, buttonIndex)) {
              const newValue = valueRef.current === buttonMapping.valueWhenPressed 
                ? (buttonMapping.valueWhenReleased || '') 
                : buttonMapping.valueWhenPressed;
              
              valueRef.current = newValue;
              onValueChange(newValue);
            }
          } else {
            // Direct mode - value follows button state
            const isPressed = controller.buttons[buttonIndex]?.pressed || false;
            const newValue = isPressed 
              ? buttonMapping.valueWhenPressed 
              : (buttonMapping.valueWhenReleased !== undefined ? buttonMapping.valueWhenReleased : valueRef.current);
            
            if (valueRef.current !== newValue) {
              valueRef.current = newValue;
              onValueChange(newValue);
            }
          }
          break;
        }
        
        case 'promptList': {
          const promptListMapping = mapping as PromptListMapping;
          
          // Check for next/prev button presses
          if (wasButtonJustPressed(controllerIndex, promptListMapping.nextButtonIndex)) {
            onValueChange('__NEXT_PROMPT__'); // Special value handled by the control panel
          }
          
          if (wasButtonJustPressed(controllerIndex, promptListMapping.prevButtonIndex)) {
            onValueChange('__PREV_PROMPT__'); // Special value handled by the control panel
          }
          break;
        }
      }
    }, 50); // Poll at 20Hz
    
    return () => clearInterval(intervalId);
  }, [controllers, mapping, onValueChange, wasButtonJustPressed, hasAxisChanged]);
  
  return null; // This hook doesn't return anything, it just applies the effects
} 