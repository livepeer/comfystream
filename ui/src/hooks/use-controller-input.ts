import { useEffect, useRef } from 'react';
import { useController } from './use-controller';
import { useControllerMapping } from './use-controller-mapping';
import { AxisMapping, ButtonMapping, ControllerMapping, KeyMapping, MouseMapping, MouseMovementMapping, MicrophoneMapping } from '@/types/controller';

// Constants for button hold & repeat functionality
const INITIAL_DELAY_MS = 500; // Initial delay before repeating starts
const REPEAT_INTERVAL_MS = 100; // Interval between repeats after initial delay

// Hook for processing controller input according to a mapping
export function useControllerInput(
  mapping: ControllerMapping | undefined,
  onValueChange: (value: any) => void
) {
  const { 
    controllers, 
    wasButtonJustPressed, 
    hasAxisChanged,
    // Keyboard and mouse API
    toggleKeyMouseInput,
    isKeyPressed,
    wasKeyJustPressed,
    isMouseButtonPressed,
    wasMouseButtonJustPressed,
    getMousePositionDelta,
    getMousePosition,
    getMouseWheelDelta,
    isMicrophoneEnabled,
    enableMicrophone,
    getMicrophoneLevel,
    disableMicrophone
  } = useController();
  
  // Access the controller mapping to update global mapping state
  const { saveMapping } = useControllerMapping();
  
  const valueRef = useRef<any>(null); // Track current value to avoid unnecessary updates
  const mappingRef = useRef<ControllerMapping | undefined>(mapping ? JSON.parse(JSON.stringify(mapping)) : undefined); // Deep clone the mapping
  const lastPollTimeRef = useRef(0);
  
  // Button state tracking to improve detection
  const buttonStatesRef = useRef<Record<number, boolean>>({});
  
  // Key state tracking
  const keyStatesRef = useRef<Record<string, boolean>>({});
  
  // Mouse button state tracking
  const mouseButtonStatesRef = useRef<Record<number, boolean>>({});
  
  // Mouse position tracking
  const mousePositionRef = useRef<{ x: number, y: number }>({ x: 0, y: 0 });
  
  // Add accumulated value tracking for mouse-x and mouse-y
  // For mouse position based mappings, we need to track accumulated values
  const mouseAccumulatedXRef = useRef<number>(0);
  const mouseAccumulatedYRef = useRef<number>(0);
  
  // Button hold timer state for increment/decrement functionality
  const incrementTimerRef = useRef<NodeJS.Timeout | null>(null);
  const decrementTimerRef = useRef<NodeJS.Timeout | null>(null);
  const buttonHoldStartTimeRef = useRef<Record<number, number>>({});
  const keyHoldStartTimeRef = useRef<Record<string, number>>({});
  const mouseButtonHoldStartTimeRef = useRef<Record<number, number>>({});
  
  // Process microphone input mapping
  const processMicrophoneMapping = (mapping: MicrophoneMapping, shouldLog: boolean) => {
    if (mapping.audioFeature !== 'volume') {
      // For MVP we only support volume - other features would go here
      return;
    }
    
    // Get the current microphone level (0-1)
    const level = getMicrophoneLevel();
    
    // Skip processing if the level is 0 (no input)
    if (level === 0) return;
    
    if (shouldLog) {
      console.log(`Microphone level: ${level.toFixed(3)}`);
    }
    
    // Apply multiplier
    let value = level * mapping.multiplier;
    
    // Apply min/max overrides if needed
    const minValue = mapping.minOverride !== undefined ? mapping.minOverride : 0;
    const maxValue = mapping.maxOverride !== undefined ? mapping.maxOverride : 1;
    
    // Scale the value to the min/max range
    value = minValue + value * (maxValue - minValue);
    
    // Only update if the value has changed significantly
    if (valueRef.current === null || Math.abs(valueRef.current - value) > 0.001) {
      valueRef.current = value;
      onValueChange(value);
    }
  };
  
  // Update the mapping ref when the mapping changes
  useEffect(() => {
    if (mapping !== mappingRef.current) {
      console.log('Controller mapping changed:', mapping);
      mappingRef.current = mapping ? JSON.parse(JSON.stringify(mapping)) : undefined; // Deep clone the mapping
      
      // Reset value reference when mapping changes
      valueRef.current = null;
      
      // Reset button states
      buttonStatesRef.current = {};
      keyStatesRef.current = {};
      mouseButtonStatesRef.current = {};
      
      // Reset accumulated mouse values
      mouseAccumulatedXRef.current = 0;
      mouseAccumulatedYRef.current = 0;
      
      // Enable keyboard and mouse input if needed
      if (mapping && (mapping.type === 'key' || mapping.type === 'mouse' || mapping.type === 'mouse-movement')) {
        toggleKeyMouseInput(true);
      }
      
      // Enable microphone input if needed
      if (mapping && mapping.type === 'microphone') {
        enableMicrophone();
      } else {
        // Disable microphone if not needed
        disableMicrophone();
      }
      
      // Set the initial value for series mode to ensure the UI is initialized correctly
      if (mapping && mapping.type === 'button' && mapping.mode === 'series' && 
          mapping.valuesList && mapping.valuesList.length > 0) {
        const buttonMapping = mapping as ButtonMapping;
        const currentIndex = buttonMapping.currentValueIndex ?? 0;
        
        // Set the initial value
        if (buttonMapping.valuesList && buttonMapping.valuesList.length > 0) {
          const initialValue = buttonMapping.valuesList[currentIndex];
          if (initialValue !== undefined) {
            console.log(`Setting initial series value to ${initialValue} (index ${currentIndex})`);
            valueRef.current = initialValue;
            
            // Use a setTimeout to break the potential update cycle
            setTimeout(() => {
              onValueChange(initialValue);
            }, 0);
          }
        }
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mapping, toggleKeyMouseInput, enableMicrophone, disableMicrophone]);
  
  useEffect(() => {
    if (!mapping) return;
    
    // Enable keyboard and mouse input if needed
    if (mapping.type === 'key' || mapping.type === 'mouse' || mapping.type === 'mouse-movement') {
      toggleKeyMouseInput(true);
    }
    
    // Enable microphone if needed
    if (mapping.type === 'microphone') {
      enableMicrophone();
    }
    
    console.log('Setting up input processing for:', mapping);
    
    // Common polling function for all input types
    const intervalId = setInterval(() => {
      // Rate limit polling to reduce console spam
      const now = Date.now();
      const shouldLog = now - lastPollTimeRef.current > 5000; // Log every 5 seconds
      if (shouldLog) {
        lastPollTimeRef.current = now;
        console.log('Processing input for', mapping.type, 'mapping');
      }
      
      // Process based on mapping type
      switch (mapping.type) {
        case 'axis': {
          if (controllers.length === 0) return;
          
          const controller = controllers[0]; // Use the first controller for now
          const axisMapping = mapping as AxisMapping;
          const axisIndex = axisMapping.axisIndex;
      
      // Get fresh gamepad data each time
      const gamepads = navigator.getGamepads();
      const freshController = gamepads[controller.index];
      
      if (!freshController) {
        if (shouldLog) console.warn('Controller lost during processing');
        return;
      }
          
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
          if (controllers.length === 0) return;
          
          const controller = controllers[0]; // Use the first controller for now
          const buttonMapping = mapping as ButtonMapping;
          const buttonIndex = buttonMapping.buttonIndex;
          
          // Get fresh gamepad data each time
          const gamepads = navigator.getGamepads();
          const freshController = gamepads[controller.index];
          
          if (!freshController) {
            if (shouldLog) console.warn('Controller lost during processing');
            return;
          }
          
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
                // Get the current mapping from our local ref to ensure we have the latest state
                const currentButtonMapping = mappingRef.current?.type === 'button' 
                  ? mappingRef.current as ButtonMapping
                  : buttonMapping;
                
                // Use currentValueIndex from our ref or initialize to 0
                let currentIndex = currentButtonMapping.currentValueIndex ?? 0;
                
                // Move to next value in the list
                currentIndex = (currentIndex + 1) % buttonMapping.valuesList.length;
                
                const newValue = buttonMapping.valuesList[currentIndex];
                console.log(`Button ${buttonIndex} cycled to value ${currentIndex}:`, newValue);
                
                // Update the index in our local ref
                if (mappingRef.current && mappingRef.current.type === 'button') {
                  (mappingRef.current as ButtonMapping).currentValueIndex = currentIndex;
                }
                
                // Also update the index in the global mapping store so all panels stay in sync
                if (mapping.nodeId && mapping.fieldName) {
                  // Create a copy of the mapping with the updated index
                  const updatedMapping = { ...buttonMapping, currentValueIndex: currentIndex };
                  saveMapping(mapping.nodeId, mapping.fieldName, updatedMapping);
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
                
                // Get the current mapping from our local ref to ensure we have the latest state
                const currentButtonMapping = mappingRef.current?.type === 'button' 
                  ? mappingRef.current as ButtonMapping
                  : buttonMapping;
                
                // Use currentValueIndex from our ref or initialize to 0
                let currentIndex = currentButtonMapping.currentValueIndex ?? 0;
                
                // Move to previous value in the list (with wrap-around)
                currentIndex = (currentIndex - 1 + buttonMapping.valuesList.length) % buttonMapping.valuesList.length;
                
                const newValue = buttonMapping.valuesList[currentIndex];
                console.log(`Button ${buttonMapping.nextButtonIndex} cycled to previous value ${currentIndex}:`, newValue);
                
                // Update the index in our local ref
                if (mappingRef.current && mappingRef.current.type === 'button') {
                  (mappingRef.current as ButtonMapping).currentValueIndex = currentIndex;
                }
                
                // Also update the index in the global mapping store so all panels stay in sync
                if (mapping.nodeId && mapping.fieldName) {
                  // Create a copy of the mapping with the updated index
                  const updatedMapping = { ...buttonMapping, currentValueIndex: currentIndex };
                  saveMapping(mapping.nodeId, mapping.fieldName, updatedMapping);
                }
                
                valueRef.current = newValue;
                onValueChange(newValue);
              }
            }
          } else if (buttonMapping.mode === 'increment') {
            // Increment/decrement mode - change the value by increment step when buttons are pressed
            // Primary button increments, next button decrements

            // Handle button press and release with repeat functionality
            const handleIncrementRepeat = () => {
              // Get current value or initialize to inputMin (or 0 if not defined)
              const currentValue = typeof valueRef.current === 'number' 
                ? valueRef.current 
                : (buttonMapping.minOverride !== undefined ? buttonMapping.minOverride : 0);
              
              // Calculate new value by adding increment step
              const incrementStep = buttonMapping.incrementStep || 1;
              let newValue = currentValue + incrementStep;
              
              // Apply upper bound if defined
              if (buttonMapping.maxOverride !== undefined) {
                newValue = Math.min(newValue, buttonMapping.maxOverride);
              }
              
              // Only update if value changed (avoid repeating at limits)
              if (newValue !== currentValue) {
                console.log(`Button ${buttonIndex} incremented value from ${currentValue} to ${newValue}`);
                valueRef.current = newValue;
                onValueChange(newValue);
                
                // Schedule next repeat
                incrementTimerRef.current = setTimeout(handleIncrementRepeat, REPEAT_INTERVAL_MS);
              }
            };
            
            const handleDecrementRepeat = () => {
              // Get current value or initialize
              const currentValue = typeof valueRef.current === 'number' 
                ? valueRef.current 
                : (buttonMapping.maxOverride !== undefined ? buttonMapping.maxOverride : 0);
              
              // Calculate new value by subtracting increment step
              const incrementStep = buttonMapping.incrementStep || 1;
              let newValue = currentValue - incrementStep;
              
              // Apply lower bound if defined
              if (buttonMapping.minOverride !== undefined) {
                newValue = Math.max(newValue, buttonMapping.minOverride);
              }
              
              // Only update if value changed (avoid repeating at limits)
              if (newValue !== currentValue) {
                console.log(`Button ${buttonMapping.nextButtonIndex} decremented value from ${currentValue} to ${newValue}`);
                valueRef.current = newValue;
                onValueChange(newValue);
                
                // Schedule next repeat
                decrementTimerRef.current = setTimeout(handleDecrementRepeat, REPEAT_INTERVAL_MS);
              }
            };
            
            // Handle increment (primary button)
            if (isPressed) {
              if (!wasPressed) {
                // Button was just pressed, start the hold timer after initial press
                const currentValue = typeof valueRef.current === 'number' 
                  ? valueRef.current 
                  : (buttonMapping.minOverride !== undefined ? buttonMapping.minOverride : 0);
                
                // Calculate new value by adding increment step
                const incrementStep = buttonMapping.incrementStep || 1;
                let newValue = currentValue + incrementStep;
                
                // Apply upper bound if defined
                if (buttonMapping.maxOverride !== undefined) {
                  newValue = Math.min(newValue, buttonMapping.maxOverride);
                }
                
                // Update immediately for the first press
                console.log(`Button ${buttonIndex} incremented value from ${currentValue} to ${newValue}`);
                valueRef.current = newValue;
                onValueChange(newValue);
                
                // Record press time and start repeat timer
                buttonHoldStartTimeRef.current[buttonIndex] = Date.now();
                incrementTimerRef.current = setTimeout(handleIncrementRepeat, INITIAL_DELAY_MS);
              }
            } else if (wasPressed) {
              // Button was released, clear the repeat timer
              if (incrementTimerRef.current) {
                clearTimeout(incrementTimerRef.current);
                incrementTimerRef.current = null;
              }
            }
            
            // Handle decrement (next button)
            if (buttonMapping.nextButtonIndex !== undefined && 
                buttonMapping.nextButtonIndex >= 0 &&
                buttonMapping.nextButtonIndex < freshController.buttons.length) {
              
              const nextIsPressed = freshController.buttons[buttonMapping.nextButtonIndex]?.pressed || false;
              const nextWasPressed = buttonStatesRef.current[buttonMapping.nextButtonIndex] || false;
              
              // Update next button state
              buttonStatesRef.current[buttonMapping.nextButtonIndex] = nextIsPressed;
              
              // Handle decrement with repeat
              if (nextIsPressed) {
                if (!nextWasPressed) {
                  // Button was just pressed, handle first press immediately
                  const currentValue = typeof valueRef.current === 'number' 
                    ? valueRef.current 
                    : (buttonMapping.maxOverride !== undefined ? buttonMapping.maxOverride : 0);
                  
                  // Calculate new value by subtracting increment step
                  const incrementStep = buttonMapping.incrementStep || 1;
                  let newValue = currentValue - incrementStep;
                  
                  // Apply lower bound if defined
                  if (buttonMapping.minOverride !== undefined) {
                    newValue = Math.max(newValue, buttonMapping.minOverride);
                  }
                  
                  // Update immediately for the first press
                  console.log(`Button ${buttonMapping.nextButtonIndex} decremented value from ${currentValue} to ${newValue}`);
                  valueRef.current = newValue;
                  onValueChange(newValue);
                  
                  // Record press time and start repeat timer
                  buttonHoldStartTimeRef.current[buttonMapping.nextButtonIndex] = Date.now();
                  decrementTimerRef.current = setTimeout(handleDecrementRepeat, INITIAL_DELAY_MS);
                }
              } else if (nextWasPressed) {
                // Button was released, clear the repeat timer
                if (decrementTimerRef.current) {
                  clearTimeout(decrementTimerRef.current);
                  decrementTimerRef.current = null;
                }
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
        
        case 'key': {
          const keyMapping = mapping as KeyMapping;
          const keyCode = keyMapping.keyCode;
          
          // Check if key is pressed
          const isKeyDown = isKeyPressed(keyCode);
          
          // Store previous state
          const wasKeyDown = keyStatesRef.current[keyCode] || false;
          
          // Update key state cache
          keyStatesRef.current[keyCode] = isKeyDown;
          
          // Handle based on key mode
          if (keyMapping.mode === 'toggle') {
            // Toggle mode - change value only when key is just pressed (rising edge)
            if (isKeyDown && !wasKeyDown) {
              const newValue = valueRef.current === keyMapping.valueWhenPressed 
                ? (keyMapping.valueWhenReleased || '') 
                : keyMapping.valueWhenPressed;
              
              console.log(`Key ${keyCode} toggled to:`, newValue);
              valueRef.current = newValue;
              onValueChange(newValue);
            }
          } else if (keyMapping.mode === 'series') {
            // Series mode - cycle through values when key is just pressed
            if (isKeyDown && !wasKeyDown) {
              // Check if we have values to cycle through
              if (keyMapping.valuesList && keyMapping.valuesList.length > 0) {
                // Use currentValueIndex or initialize to 0
                let currentIndex = keyMapping.currentValueIndex || 0;
                
                // Move to next value in the list
                currentIndex = (currentIndex + 1) % keyMapping.valuesList.length;
                
                const newValue = keyMapping.valuesList[currentIndex];
                console.log(`Key ${keyCode} cycled to value ${currentIndex}:`, newValue);
                
                // Update the index in our local ref
                if (mappingRef.current && mappingRef.current.type === 'key') {
                  (mappingRef.current as KeyMapping).currentValueIndex = currentIndex;
                }
                
                // Also update the index in the global mapping store so all panels stay in sync
                if (mapping.nodeId && mapping.fieldName) {
                  // Create a copy of the mapping with the updated index
                  const updatedMapping = { ...keyMapping, currentValueIndex: currentIndex };
                  saveMapping(mapping.nodeId, mapping.fieldName, updatedMapping);
                }
                
                valueRef.current = newValue;
                onValueChange(newValue);
              }
            }
            
            // Check for nextKeyCode if defined (for backward cycling)
            if (keyMapping.nextKeyCode) {
              const nextIsPressed = isKeyPressed(keyMapping.nextKeyCode);
              const nextWasPressed = keyStatesRef.current[keyMapping.nextKeyCode] || false;
              
              // Update next key state
              keyStatesRef.current[keyMapping.nextKeyCode] = nextIsPressed;
              
              // Handle backward cycling
              if (nextIsPressed && !nextWasPressed && 
                  keyMapping.valuesList && keyMapping.valuesList.length > 0) {
                
                // Use currentValueIndex or initialize to 0
                let currentIndex = keyMapping.currentValueIndex || 0;
                
                // Move to previous value in the list (with wrap-around)
                currentIndex = (currentIndex - 1 + keyMapping.valuesList.length) % keyMapping.valuesList.length;
                
                const newValue = keyMapping.valuesList[currentIndex];
                console.log(`Key ${keyMapping.nextKeyCode} cycled to previous value ${currentIndex}:`, newValue);
                
                // Update the index in our local ref
                if (mappingRef.current && mappingRef.current.type === 'key') {
                  (mappingRef.current as KeyMapping).currentValueIndex = currentIndex;
                }
                
                // Also update the index in the global mapping store so all panels stay in sync
                if (mapping.nodeId && mapping.fieldName) {
                  // Create a copy of the mapping with the updated index
                  const updatedMapping = { ...keyMapping, currentValueIndex: currentIndex };
                  saveMapping(mapping.nodeId, mapping.fieldName, updatedMapping);
                }
                
                valueRef.current = newValue;
                onValueChange(newValue);
              }
            }
          } else if (keyMapping.mode === 'increment') {
            // Increment/decrement mode - change the value by increment step when keys are pressed
            // Primary key increments, next key decrements

            // Handle key press and release with repeat functionality
            const handleKeyIncrementRepeat = () => {
              // Get current value or initialize to inputMin (or 0 if not defined)
              const currentValue = typeof valueRef.current === 'number' 
                ? valueRef.current 
                : (keyMapping.minOverride !== undefined ? keyMapping.minOverride : 0);
              
              // Calculate new value by adding increment step
              const incrementStep = keyMapping.incrementStep || 1;
              let newValue = currentValue + incrementStep;
              
              // Apply upper bound if defined
              if (keyMapping.maxOverride !== undefined) {
                newValue = Math.min(newValue, keyMapping.maxOverride);
              }
              
              // Only update if value changed (avoid repeating at limits)
              if (newValue !== currentValue) {
                console.log(`Key ${keyCode} incremented value from ${currentValue} to ${newValue}`);
                valueRef.current = newValue;
                onValueChange(newValue);
                
                // Schedule next repeat
                incrementTimerRef.current = setTimeout(handleKeyIncrementRepeat, REPEAT_INTERVAL_MS);
              }
            };
            
            const handleKeyDecrementRepeat = () => {
              // Get current value or initialize
              const currentValue = typeof valueRef.current === 'number' 
                ? valueRef.current 
                : (keyMapping.maxOverride !== undefined ? keyMapping.maxOverride : 0);
              
              // Calculate new value by subtracting increment step
              const incrementStep = keyMapping.incrementStep || 1;
              let newValue = currentValue - incrementStep;
              
              // Apply lower bound if defined
              if (keyMapping.minOverride !== undefined) {
                newValue = Math.max(newValue, keyMapping.minOverride);
              }
              
              // Only update if value changed (avoid repeating at limits)
              if (newValue !== currentValue) {
                console.log(`Key ${keyMapping.nextKeyCode} decremented value from ${currentValue} to ${newValue}`);
                valueRef.current = newValue;
                onValueChange(newValue);
                
                // Schedule next repeat
                decrementTimerRef.current = setTimeout(handleKeyDecrementRepeat, REPEAT_INTERVAL_MS);
              }
            };
            
            // Handle increment (primary key)
            if (isKeyDown) {
              if (!wasKeyDown) {
                // Key was just pressed, start the hold timer after initial press
                const currentValue = typeof valueRef.current === 'number' 
                  ? valueRef.current 
                  : (keyMapping.minOverride !== undefined ? keyMapping.minOverride : 0);
                
                // Calculate new value by adding increment step
                const incrementStep = keyMapping.incrementStep || 1;
                let newValue = currentValue + incrementStep;
                
                // Apply upper bound if defined
                if (keyMapping.maxOverride !== undefined) {
                  newValue = Math.min(newValue, keyMapping.maxOverride);
                }
                
                // Update immediately for the first press
                console.log(`Key ${keyCode} incremented value from ${currentValue} to ${newValue}`);
                valueRef.current = newValue;
                onValueChange(newValue);
                
                // Record press time and start repeat timer
                keyHoldStartTimeRef.current[keyCode] = Date.now();
                incrementTimerRef.current = setTimeout(handleKeyIncrementRepeat, INITIAL_DELAY_MS);
              }
            } else if (wasKeyDown) {
              // Key was released, clear the repeat timer
              if (incrementTimerRef.current) {
                clearTimeout(incrementTimerRef.current);
                incrementTimerRef.current = null;
              }
            }
            
            // Handle decrement (next key)
            if (keyMapping.nextKeyCode) {
              const nextIsPressed = isKeyPressed(keyMapping.nextKeyCode);
              const nextWasPressed = keyStatesRef.current[keyMapping.nextKeyCode] || false;
              
              // Update next key state
              keyStatesRef.current[keyMapping.nextKeyCode] = nextIsPressed;
              
              // Handle decrement with repeat
              if (nextIsPressed) {
                if (!nextWasPressed) {
                  // Key was just pressed, handle first press immediately
                  const currentValue = typeof valueRef.current === 'number' 
                    ? valueRef.current 
                    : (keyMapping.maxOverride !== undefined ? keyMapping.maxOverride : 0);
                  
                  // Calculate new value by subtracting increment step
                  const incrementStep = keyMapping.incrementStep || 1;
                  let newValue = currentValue - incrementStep;
                  
                  // Apply lower bound if defined
                  if (keyMapping.minOverride !== undefined) {
                    newValue = Math.max(newValue, keyMapping.minOverride);
                  }
                  
                  // Update immediately for the first press
                  console.log(`Key ${keyMapping.nextKeyCode} decremented value from ${currentValue} to ${newValue}`);
                  valueRef.current = newValue;
                  onValueChange(newValue);
                  
                  // Record press time and start repeat timer
                  keyHoldStartTimeRef.current[keyMapping.nextKeyCode] = Date.now();
                  decrementTimerRef.current = setTimeout(handleKeyDecrementRepeat, INITIAL_DELAY_MS);
                }
              } else if (nextWasPressed) {
                // Key was released, clear the repeat timer
                if (decrementTimerRef.current) {
                  clearTimeout(decrementTimerRef.current);
                  decrementTimerRef.current = null;
                }
              }
            }
          } else {
            // Momentary mode (default) - value follows key state
            const newValue = isKeyDown 
              ? keyMapping.valueWhenPressed 
              : (keyMapping.valueWhenReleased !== undefined ? keyMapping.valueWhenReleased : valueRef.current);
            
            // Only send updates on state change
            if (isKeyDown !== wasKeyDown || valueRef.current === null) {
              console.log(`Key ${keyCode} state changed to:`, isKeyDown, 'value:', newValue);
              valueRef.current = newValue;
              onValueChange(newValue);
            }
          }
          break;
        }
        
        case 'mouse': {
          const mouseMapping = mapping as MouseMapping;
          
          // Handle based on mouse action type
          switch (mouseMapping.action) {
            case 'button': {
              if (mouseMapping.buttonIndex === undefined) return;
              
              const isPressed = isMouseButtonPressed(mouseMapping.buttonIndex);
              
              // Store previous state
              const wasPressed = mouseButtonStatesRef.current[mouseMapping.buttonIndex] || false;
              
              // Update button state cache
              mouseButtonStatesRef.current[mouseMapping.buttonIndex] = isPressed;
              
              // Handle based on button mode
              if (mouseMapping.mode === 'toggle') {
                // Toggle mode - change value only when button is just pressed (rising edge)
                if (isPressed && !wasPressed) {
                  const newValue = valueRef.current === mouseMapping.valueWhenPressed 
                    ? (mouseMapping.valueWhenReleased || '') 
                    : mouseMapping.valueWhenPressed;
                  
                  console.log(`Mouse button ${mouseMapping.buttonIndex} toggled to:`, newValue);
                  valueRef.current = newValue;
                  onValueChange(newValue);
                }
              } else if (mouseMapping.mode === 'series') {
                // Series mode - cycle through values when button is just pressed
                if (isPressed && !wasPressed) {
                  // Check if we have values to cycle through
                  if (mouseMapping.valuesList && mouseMapping.valuesList.length > 0) {
                    // Use currentValueIndex or initialize to 0
                    let currentIndex = mouseMapping.currentValueIndex || 0;
                    
                    // Move to next value in the list
                    currentIndex = (currentIndex + 1) % mouseMapping.valuesList.length;
                    
                    const newValue = mouseMapping.valuesList[currentIndex];
                    console.log(`Mouse button ${mouseMapping.buttonIndex} cycled to value ${currentIndex}:`, newValue);
                    
                    // Update the index in our local ref
                    if (mappingRef.current && mappingRef.current.type === 'mouse') {
                      (mappingRef.current as MouseMapping).currentValueIndex = currentIndex;
                    }
                    
                    // Also update the index in the global mapping store so all panels stay in sync
                    if (mapping.nodeId && mapping.fieldName) {
                      // Create a copy of the mapping with the updated index
                      const updatedMapping = { ...mouseMapping, currentValueIndex: currentIndex };
                      saveMapping(mapping.nodeId, mapping.fieldName, updatedMapping);
                    }
                    
                    valueRef.current = newValue;
                    onValueChange(newValue);
                  }
                }
                
                // Check for nextButtonIndex if defined (for backward cycling)
                if (mouseMapping.nextButtonIndex !== undefined && mouseMapping.nextAction === 'button') {
                  const nextIsPressed = isMouseButtonPressed(mouseMapping.nextButtonIndex);
                  const nextWasPressed = mouseButtonStatesRef.current[mouseMapping.nextButtonIndex] || false;
                  
                  // Update next button state
                  mouseButtonStatesRef.current[mouseMapping.nextButtonIndex] = nextIsPressed;
                  
                  // Handle backward cycling
                  if (nextIsPressed && !nextWasPressed && 
                      mouseMapping.valuesList && mouseMapping.valuesList.length > 0) {
                    
                    // Use currentValueIndex or initialize to 0
                    let currentIndex = mouseMapping.currentValueIndex || 0;
                    
                    // Move to previous value in the list (with wrap-around)
                    currentIndex = (currentIndex - 1 + mouseMapping.valuesList.length) % mouseMapping.valuesList.length;
                    
                    const newValue = mouseMapping.valuesList[currentIndex];
                    console.log(`Mouse button ${mouseMapping.nextButtonIndex} cycled to previous value ${currentIndex}:`, newValue);
                    
                    // Update the index in our local ref
                    if (mappingRef.current && mappingRef.current.type === 'mouse') {
                      (mappingRef.current as MouseMapping).currentValueIndex = currentIndex;
                    }
                    
                    // Also update the index in the global mapping store so all panels stay in sync
                    if (mapping.nodeId && mapping.fieldName) {
                      // Create a copy of the mapping with the updated index
                      const updatedMapping = { ...mouseMapping, currentValueIndex: currentIndex };
                      saveMapping(mapping.nodeId, mapping.fieldName, updatedMapping);
                    }
                    
                    valueRef.current = newValue;
                    onValueChange(newValue);
                  }
                }
              } else if (mouseMapping.mode === 'increment') {
                // Increment/decrement mode - change the value by increment step when mouse buttons are pressed
                // Primary button increments, next button decrements

                // Handle mouse button press and release with repeat functionality
                const handleMouseIncrementRepeat = () => {
                  // Get current value or initialize to inputMin (or 0 if not defined)
                  const currentValue = typeof valueRef.current === 'number' 
                    ? valueRef.current 
                    : (mouseMapping.minOverride !== undefined ? mouseMapping.minOverride : 0);
                  
                  // Calculate new value by adding increment step
                  const incrementStep = mouseMapping.incrementStep || 1;
                  let newValue = currentValue + incrementStep;
                  
                  // Apply upper bound if defined
                  if (mouseMapping.maxOverride !== undefined) {
                    newValue = Math.min(newValue, mouseMapping.maxOverride);
                  }
                  
                  // Only update if value changed (avoid repeating at limits)
                  if (newValue !== currentValue) {
                    console.log(`Mouse button ${mouseMapping.buttonIndex} incremented value from ${currentValue} to ${newValue}`);
                    valueRef.current = newValue;
                    onValueChange(newValue);
                    
                    // Schedule next repeat
                    incrementTimerRef.current = setTimeout(handleMouseIncrementRepeat, REPEAT_INTERVAL_MS);
                  }
                };
                
                const handleMouseDecrementRepeat = () => {
                  // Get current value or initialize
                  const currentValue = typeof valueRef.current === 'number' 
                    ? valueRef.current 
                    : (mouseMapping.maxOverride !== undefined ? mouseMapping.maxOverride : 0);
                  
                  // Calculate new value by subtracting increment step
                  const incrementStep = mouseMapping.incrementStep || 1;
                  let newValue = currentValue - incrementStep;
                  
                  // Apply lower bound if defined
                  if (mouseMapping.minOverride !== undefined) {
                    newValue = Math.max(newValue, mouseMapping.minOverride);
                  }
                  
                  // Only update if value changed (avoid repeating at limits)
                  if (newValue !== currentValue) {
                    console.log(`Mouse button ${mouseMapping.nextButtonIndex} decremented value from ${currentValue} to ${newValue}`);
                    valueRef.current = newValue;
                    onValueChange(newValue);
                    
                    // Schedule next repeat
                    decrementTimerRef.current = setTimeout(handleMouseDecrementRepeat, REPEAT_INTERVAL_MS);
                  }
                };
                
                // Handle increment (primary button)
                if (isPressed) {
                  if (!wasPressed) {
                    // Button was just pressed, start the hold timer after initial press
                    const currentValue = typeof valueRef.current === 'number' 
                      ? valueRef.current 
                      : (mouseMapping.minOverride !== undefined ? mouseMapping.minOverride : 0);
                      
                    // Calculate new value by adding increment step
                    const incrementStep = mouseMapping.incrementStep || 1;
                    let newValue = currentValue + incrementStep;
                    
                    // Apply upper bound if defined
                    if (mouseMapping.maxOverride !== undefined) {
                      newValue = Math.min(newValue, mouseMapping.maxOverride);
                    }
                    
                    // Update immediately for the first press
                    console.log(`Mouse button ${mouseMapping.buttonIndex} incremented value from ${currentValue} to ${newValue}`);
                    valueRef.current = newValue;
                    onValueChange(newValue);
                    
                    // Record press time and start repeat timer
                    mouseButtonHoldStartTimeRef.current[mouseMapping.buttonIndex] = Date.now();
                    incrementTimerRef.current = setTimeout(handleMouseIncrementRepeat, INITIAL_DELAY_MS);
                  }
                } else if (wasPressed) {
                  // Button was released, clear the repeat timer
                  if (incrementTimerRef.current) {
                    clearTimeout(incrementTimerRef.current);
                    incrementTimerRef.current = null;
                  }
                }
                
                // Handle decrement (next button)
                if (mouseMapping.nextButtonIndex !== undefined && mouseMapping.nextAction === 'button') {
                  const nextIsPressed = isMouseButtonPressed(mouseMapping.nextButtonIndex);
                  const nextWasPressed = mouseButtonStatesRef.current[mouseMapping.nextButtonIndex] || false;
                  
                  // Update next button state
                  mouseButtonStatesRef.current[mouseMapping.nextButtonIndex] = nextIsPressed;
                  
                  // Handle decrement with repeat
                  if (nextIsPressed) {
                    if (!nextWasPressed) {
                      // Button was just pressed, handle first press immediately
                      const currentValue = typeof valueRef.current === 'number' 
                        ? valueRef.current 
                        : (mouseMapping.maxOverride !== undefined ? mouseMapping.maxOverride : 0);
                        
                      // Calculate new value by subtracting increment step
                      const incrementStep = mouseMapping.incrementStep || 1;
                      let newValue = currentValue - incrementStep;
                      
                      // Apply lower bound if defined
                      if (mouseMapping.minOverride !== undefined) {
                        newValue = Math.max(newValue, mouseMapping.minOverride);
                      }
                      
                      // Update immediately for the first press
                      console.log(`Mouse button ${mouseMapping.nextButtonIndex} decremented value from ${currentValue} to ${newValue}`);
                      valueRef.current = newValue;
                      onValueChange(newValue);
                      
                      // Record press time and start repeat timer
                      mouseButtonHoldStartTimeRef.current[mouseMapping.nextButtonIndex] = Date.now();
                      decrementTimerRef.current = setTimeout(handleMouseDecrementRepeat, INITIAL_DELAY_MS);
                    }
                  } else if (nextWasPressed) {
                    // Button was released, clear the repeat timer
                    if (decrementTimerRef.current) {
                      clearTimeout(decrementTimerRef.current);
                      decrementTimerRef.current = null;
                    }
                  }
                }
              } else {
                // Momentary mode (default) - value follows button state
                const newValue = isPressed 
                  ? mouseMapping.valueWhenPressed 
                  : (mouseMapping.valueWhenReleased !== undefined ? mouseMapping.valueWhenReleased : valueRef.current);
                
                // Only send updates on state change
                if (isPressed !== wasPressed || valueRef.current === null) {
                  console.log(`Mouse button ${mouseMapping.buttonIndex} state changed to:`, isPressed, 'value:', newValue);
                  valueRef.current = newValue;
                  onValueChange(newValue);
                }
              }
              break;
            }
            
            case 'wheel': {
              if (mouseMapping.mode !== 'axis') return;
              
              const deltaY = getMouseWheelDelta();
              
              // Only process significant movement
              if (Math.abs(deltaY) > 0) {
                // Apply scaling
                let scaledValue = deltaY * (mouseMapping.multiplier || 0.01);
                
                // Apply min/max overrides if provided
                if (mouseMapping.minOverride !== undefined && mouseMapping.maxOverride !== undefined) {
                  // Clamp value to min/max
                  scaledValue = Math.max(mouseMapping.minOverride, 
                                  Math.min(mouseMapping.maxOverride, scaledValue));
                }
                
                // Round to reasonable precision
                const roundedValue = parseFloat(scaledValue.toFixed(3));
                
                // Always update on wheel movement
                valueRef.current = roundedValue;
                onValueChange(roundedValue);
              }
              break;
            }
          }
          break;
        }
        
        case 'mouse-movement': {
          const mouseMovementMapping = mapping as MouseMovementMapping;
          const position = getMousePosition();
          
          // Extract the axis to determine which direction to track
          const axis = mouseMovementMapping.axis;
          
          // Calculate delta from last position
          const deltaX = position.x - (mousePositionRef.current?.x || position.x);
          const deltaY = position.y - (mousePositionRef.current?.y || position.y);
          
          // Store current position
          mousePositionRef.current = position;
          
          // Handle X or Y based on the axis
          if (axis === 'x') {
            // Only process significant movement
            if (Math.abs(deltaX) > 1) {
              // Scale the delta by the multiplier
              const scaledDelta = deltaX * (mouseMovementMapping.multiplier || 0.01);
              
              // Add the delta to our accumulated value
              mouseAccumulatedXRef.current += scaledDelta;
              
              // Get min/max values (from mapping or defaults)
              const minValue = mouseMovementMapping.minOverride !== undefined ? mouseMovementMapping.minOverride : -1;
              const maxValue = mouseMovementMapping.maxOverride !== undefined ? mouseMovementMapping.maxOverride : 1;
              
              // Clamp the accumulated value within bounds
              mouseAccumulatedXRef.current = Math.max(minValue, Math.min(maxValue, mouseAccumulatedXRef.current));
              
              // Round to reasonable precision
              const roundedValue = parseFloat(mouseAccumulatedXRef.current.toFixed(3));
              
              // Only update on significant changes
              if (valueRef.current === null || Math.abs(valueRef.current - roundedValue) > 0.001) {
                valueRef.current = roundedValue;
                onValueChange(roundedValue);
              }
            }
          } else if (axis === 'y') {
            // Only process significant movement
            if (Math.abs(deltaY) > 1) {
              // Scale the delta by the multiplier (note: invert Y for intuitive up/down)
              const scaledDelta = -deltaY * (mouseMovementMapping.multiplier || 0.01); // Negative because screen Y is inverted
              
              // Add the delta to our accumulated value
              mouseAccumulatedYRef.current += scaledDelta;
              
              // Get min/max values (from mapping or defaults)
              const minValue = mouseMovementMapping.minOverride !== undefined ? mouseMovementMapping.minOverride : -1;
              const maxValue = mouseMovementMapping.maxOverride !== undefined ? mouseMovementMapping.maxOverride : 1;
              
              // Clamp the accumulated value within bounds
              mouseAccumulatedYRef.current = Math.max(minValue, Math.min(maxValue, mouseAccumulatedYRef.current));
              
              // Round to reasonable precision
              const roundedValue = parseFloat(mouseAccumulatedYRef.current.toFixed(3));
              
              // Only update on significant changes
              if (valueRef.current === null || Math.abs(valueRef.current - roundedValue) > 0.001) {
                valueRef.current = roundedValue;
                onValueChange(roundedValue);
              }
            }
          }
          break;
        }
        
        case 'microphone': {
          processMicrophoneMapping(mapping as MicrophoneMapping, shouldLog);
          break;
        }
      }
    }, 16); // Poll at 60Hz for smoother control
    
    return () => {
      console.log('Cleaning up input processing');
      clearInterval(intervalId);
      
      // Clear any pending increment/decrement timers
      if (incrementTimerRef.current) {
        clearTimeout(incrementTimerRef.current);
        incrementTimerRef.current = null;
      }
      
      if (decrementTimerRef.current) {
        clearTimeout(decrementTimerRef.current);
        decrementTimerRef.current = null;
      }
      
      // Disable keyboard and mouse input when unmounting
      if (mapping.type === 'key' || mapping.type === 'mouse' || mapping.type === 'mouse-movement') {
        toggleKeyMouseInput(false);
      }
    };
  }, [
    controllers, 
    mapping, 
    onValueChange, 
    wasButtonJustPressed, 
    hasAxisChanged,
    toggleKeyMouseInput,
    isKeyPressed,
    wasKeyJustPressed,
    isMouseButtonPressed,
    wasMouseButtonJustPressed,
    getMousePositionDelta,
    getMousePosition,
    getMouseWheelDelta,
    isMicrophoneEnabled,
    enableMicrophone,
    getMicrophoneLevel,
    processMicrophoneMapping,
    disableMicrophone,
    saveMapping
  ]);
  
  return null; // This hook doesn't return anything, it just applies the effects
} 