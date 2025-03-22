import { useState, useEffect, useRef, useCallback } from "react";

// Polling interval for controllers in milliseconds
const CONTROLLER_POLL_INTERVAL = 16; // Faster polling (60fps)

// Controller hook that detects controllers and provides input state
export function useController() {
  const [controllers, setControllers] = useState<Gamepad[]>([]);
  const [isControllerConnected, setIsControllerConnected] = useState(false);
  const prevButtonStates = useRef<Record<number, Record<number, boolean>>>({});
  const prevAxisValues = useRef<Record<number, Record<number, number>>>({});
  const requestRef = useRef<number | null>(null);
  const frameCount = useRef(0);
  
  // Handler for when a gamepad is connected
  const handleGamepadConnected = useCallback((event: GamepadEvent) => {
    console.log("Gamepad connected:", event.gamepad.id);
    console.log("Gamepad details:", {
      index: event.gamepad.index,
      connected: event.gamepad.connected,
      buttons: event.gamepad.buttons.length,
      axes: event.gamepad.axes.length
    });
    
    // Force an immediate poll to update the controllers
    pollGamepads();
    setIsControllerConnected(true);
  }, []);

  // Handler for when a gamepad is disconnected
  const handleGamepadDisconnected = useCallback((event: GamepadEvent) => {
    console.log("Gamepad disconnected:", event.gamepad.id);
    console.log("Disconnected gamepad index:", event.gamepad.index);
    
    // Force an immediate poll to update the controllers
    pollGamepads();
  }, []);

  // Poll for gamepad state updates
  const pollGamepads = useCallback(() => {
    frameCount.current += 1;
    const shouldLog = frameCount.current % 300 === 0; // Log every ~5 seconds
    
    try {
      const gamepads = navigator.getGamepads();
      if (shouldLog) {
        console.log("Polling controllers, raw array:", gamepads);
      }
      
      const activePads: Gamepad[] = [];
      
      for (let i = 0; i < gamepads.length; i++) {
        const pad = gamepads[i];
        if (pad && pad.connected) {
          activePads.push(pad);
          if (shouldLog) {
            console.log(`Active controller ${i}: ${pad.id}, buttons: ${pad.buttons.length}, axes: ${pad.axes.length}`);
          }
        }
      }

      if (shouldLog) {
        console.log(`Found ${activePads.length} active controllers`);
      }
      
      setControllers(activePads);
      setIsControllerConnected(activePads.length > 0);
    } catch (error) {
      console.error("Error polling gamepads:", error);
    }
    
    // Set up next poll
    requestRef.current = requestAnimationFrame(pollGamepads);
  }, []);

  // Check if a button was just pressed (for edge detection)
  const wasButtonJustPressed = useCallback((controllerIndex: number, buttonIndex: number): boolean => {
    const controller = controllers[controllerIndex];
    if (!controller) return false;
    
    // Initialize controller state if needed
    if (!prevButtonStates.current[controllerIndex]) {
      prevButtonStates.current[controllerIndex] = {};
    }
    
    const buttonState = controller.buttons[buttonIndex]?.pressed || false;
    const wasPressed = buttonState && !prevButtonStates.current[controllerIndex][buttonIndex];
    
    // Update previous state
    prevButtonStates.current[controllerIndex][buttonIndex] = buttonState;
    
    return wasPressed;
  }, [controllers]);
  
  // Check if an axis value has changed beyond threshold
  const hasAxisChanged = useCallback(
    (controllerIndex: number, axisIndex: number, threshold = 0.01): boolean => {
      const controller = controllers[controllerIndex];
      if (!controller) return false;
      
      // Initialize controller state if needed
      if (!prevAxisValues.current[controllerIndex]) {
        prevAxisValues.current[controllerIndex] = {};
      }
      
      const axisValue = controller.axes[axisIndex] || 0;
      const prevValue = prevAxisValues.current[controllerIndex][axisIndex] || 0;
      const hasChanged = Math.abs(axisValue - prevValue) > threshold;
      
      // Update previous state
      prevAxisValues.current[controllerIndex][axisIndex] = axisValue;
      
      return hasChanged;
    },
    [controllers]
  );

  // Set up event listeners and polling
  useEffect(() => {
    console.log("Setting up controller event listeners and polling");
    
    window.addEventListener("gamepadconnected", handleGamepadConnected);
    window.addEventListener("gamepaddisconnected", handleGamepadDisconnected);
    
    // Check for already connected controllers that might have been missed
    try {
      if (navigator.getGamepads) {
        const initialGamepads = navigator.getGamepads();
        console.log("Initial gamepad check:", initialGamepads);
        
        const hasActiveControllers = Array.from(initialGamepads).some(gp => gp && gp.connected);
        if (hasActiveControllers) {
          console.log("Found already connected controllers on initial check");
        }
      }
    } catch (e) {
      console.error("Error checking initial gamepads:", e);
    }
    
    // Start polling for controllers
    requestRef.current = requestAnimationFrame(pollGamepads);
    
    return () => {
      console.log("Cleaning up controller listeners and polling");
      window.removeEventListener("gamepadconnected", handleGamepadConnected);
      window.removeEventListener("gamepaddisconnected", handleGamepadDisconnected);
      
      if (requestRef.current) {
        cancelAnimationFrame(requestRef.current);
      }
    };
  }, [handleGamepadConnected, handleGamepadDisconnected, pollGamepads]);

  return {
    controllers,
    isControllerConnected,
    wasButtonJustPressed,
    hasAxisChanged,
    refreshControllers: pollGamepads // Expose the poll function for manual refreshes
  };
} 