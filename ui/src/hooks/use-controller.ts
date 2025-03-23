import { useState, useEffect, useRef, useCallback } from "react";

// Polling interval for controllers in milliseconds
const CONTROLLER_POLL_INTERVAL = 16; // Faster polling (60fps)

// Key state tracking interface
interface KeyboardState {
  keys: Record<string, boolean>;
  previousKeys: Record<string, boolean>;
}

// Mouse state tracking interface
interface MouseState {
  buttons: boolean[];
  previousButtons: boolean[];
  position: { x: number, y: number };
  previousPosition: { x: number, y: number };
  wheelDelta: number;
}

// Add AudioContext TypeScript interface if needed
declare global {
  interface Window {
    webkitAudioContext?: typeof AudioContext;
  }
}

// Controller hook that detects controllers and provides input state
export function useController() {
  const [controllers, setControllers] = useState<Gamepad[]>([]);
  const [isControllerConnected, setIsControllerConnected] = useState(false);
  const [isKeyMouseEnabled, setIsKeyMouseEnabled] = useState(false);
  
  // Add microphone state
  const [isMicrophoneEnabled, setIsMicrophoneEnabled] = useState(false);
  const [isMicrophoneAvailable, setIsMicrophoneAvailable] = useState(false);
  
  // Add refs for microphone audio processing
  const microphoneStreamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const dataArrayRef = useRef<Uint8Array | null>(null);
  
  const prevButtonStates = useRef<Record<number, Record<number, boolean>>>({});
  const prevAxisValues = useRef<Record<number, Record<number, number>>>({});
  const requestRef = useRef<number | null>(null);
  const frameCount = useRef(0);
  const lastControllerCount = useRef(0);
  
  // Keyboard and mouse state tracking
  const keyboardState = useRef<KeyboardState>({
    keys: {},
    previousKeys: {}
  });
  
  const mouseState = useRef<MouseState>({
    buttons: [false, false, false],
    previousButtons: [false, false, false],
    position: { x: 0, y: 0 },
    previousPosition: { x: 0, y: 0 },
    wheelDelta: 0
  });
  
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
            
            // Log the current axes values periodically for debugging
            if (shouldLog) {
              console.log(`Controller ${i} axes values:`, Array.from(pad.axes).map(v => v.toFixed(2)));
            }
          }
        }
      }

      // Only update the state if it actually changed
      if (activePads.length !== lastControllerCount.current) {
        if (shouldLog) {
          console.log(`Controller count changed from ${lastControllerCount.current} to ${activePads.length}`);
        }
        
        setControllers(activePads);
        setIsControllerConnected(activePads.length > 0);
        lastControllerCount.current = activePads.length;
        
        // Clear previous button and axis states when controllers change
        if (activePads.length === 0) {
          prevButtonStates.current = {};
          prevAxisValues.current = {};
        }
      }
    } catch (error) {
      console.error("Error polling gamepads:", error);
    }
    
    // Set up next poll
    requestRef.current = requestAnimationFrame(pollGamepads);
  }, []);

  // Keyboard event handlers
  const handleKeyDown = useCallback((event: KeyboardEvent) => {
    if (!isKeyMouseEnabled) return;
    
    // Store previous state
    keyboardState.current.previousKeys[event.code] = keyboardState.current.keys[event.code] || false;
    
    // Update current state
    keyboardState.current.keys[event.code] = true;
    
    // Only log occasionally to avoid spam
    if (frameCount.current % 60 === 0) {
      console.log(`Key pressed: ${event.code}`);
    }
  }, [isKeyMouseEnabled]);
  
  const handleKeyUp = useCallback((event: KeyboardEvent) => {
    if (!isKeyMouseEnabled) return;
    
    // Store previous state
    keyboardState.current.previousKeys[event.code] = keyboardState.current.keys[event.code] || false;
    
    // Update current state
    keyboardState.current.keys[event.code] = false;
    
    // Only log occasionally
    if (frameCount.current % 60 === 0) {
      console.log(`Key released: ${event.code}`);
    }
  }, [isKeyMouseEnabled]);
  
  // Mouse event handlers
  const handleMouseMove = useCallback((event: MouseEvent) => {
    if (!isKeyMouseEnabled) return;
    
    // Store previous position
    mouseState.current.previousPosition = { ...mouseState.current.position };
    
    // Update current position
    mouseState.current.position = { x: event.clientX, y: event.clientY };
  }, [isKeyMouseEnabled]);
  
  const handleMouseDown = useCallback((event: MouseEvent) => {
    if (!isKeyMouseEnabled) return;
    
    // Store previous button state
    mouseState.current.previousButtons[event.button] = mouseState.current.buttons[event.button];
    
    // Update current button state
    mouseState.current.buttons[event.button] = true;
    
    if (frameCount.current % 60 === 0) {
      console.log(`Mouse button ${event.button} pressed`);
    }
  }, [isKeyMouseEnabled]);
  
  const handleMouseUp = useCallback((event: MouseEvent) => {
    if (!isKeyMouseEnabled) return;
    
    // Store previous button state
    mouseState.current.previousButtons[event.button] = mouseState.current.buttons[event.button];
    
    // Update current button state
    mouseState.current.buttons[event.button] = false;
    
    if (frameCount.current % 60 === 0) {
      console.log(`Mouse button ${event.button} released`);
    }
  }, [isKeyMouseEnabled]);
  
  const handleMouseWheel = useCallback((event: WheelEvent) => {
    if (!isKeyMouseEnabled) return;
    
    // Update wheel delta
    mouseState.current.wheelDelta = event.deltaY;
    
    // Reset wheel delta after a short delay
    setTimeout(() => {
      mouseState.current.wheelDelta = 0;
    }, 50);
  }, [isKeyMouseEnabled]);

  // Check if a button was just pressed (for edge detection)
  const wasButtonJustPressed = useCallback((controllerIndex: number, buttonIndex: number): boolean => {
    const controller = controllers[controllerIndex];
    if (!controller) return false;
    
    // Get fresh gamepad data to ensure we have the latest state
    const gamepads = navigator.getGamepads();
    const freshController = gamepads[controller.index];
    
    if (!freshController) {
      console.warn("Controller lost during button check");
      return false;
    }
    
    // Initialize controller state if needed
    if (!prevButtonStates.current[controllerIndex]) {
      prevButtonStates.current[controllerIndex] = {};
    }
    
    // Ensure the button index is valid
    if (buttonIndex < 0 || buttonIndex >= freshController.buttons.length) {
      return false;
    }
    
    const buttonState = freshController.buttons[buttonIndex]?.pressed || false;
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
      
      // Get fresh gamepad data to ensure we have the latest state
      const gamepads = navigator.getGamepads();
      const freshController = gamepads[controller.index];
      
      if (!freshController) {
        console.warn("Controller lost during axis check");
        return false;
      }
      
      // Initialize controller state if needed
      if (!prevAxisValues.current[controllerIndex]) {
        prevAxisValues.current[controllerIndex] = {};
      }
      
      // Ensure the axis index is valid
      if (axisIndex < 0 || axisIndex >= freshController.axes.length) {
        return false;
      }
      
      const axisValue = freshController.axes[axisIndex] || 0;
      const prevValue = prevAxisValues.current[controllerIndex][axisIndex] || 0;
      const hasChanged = Math.abs(axisValue - prevValue) > threshold;
      
      // Update previous state
      prevAxisValues.current[controllerIndex][axisIndex] = axisValue;
      
      return hasChanged;
    },
    [controllers]
  );
  
  // Keyboard and mouse helper methods
  const isKeyPressed = useCallback((keyCode: string): boolean => {
    return !!keyboardState.current.keys[keyCode];
  }, []);
  
  const wasKeyJustPressed = useCallback((keyCode: string): boolean => {
    return keyboardState.current.keys[keyCode] && !keyboardState.current.previousKeys[keyCode];
  }, []);
  
  const isMouseButtonPressed = useCallback((button: number): boolean => {
    return !!mouseState.current.buttons[button];
  }, []);
  
  const wasMouseButtonJustPressed = useCallback((button: number): boolean => {
    return mouseState.current.buttons[button] && !mouseState.current.previousButtons[button];
  }, []);
  
  const getMousePositionDelta = useCallback(() => {
    return {
      x: mouseState.current.position.x - mouseState.current.previousPosition.x,
      y: mouseState.current.position.y - mouseState.current.previousPosition.y
    };
  }, []);
  
  const getMousePosition = useCallback(() => {
    return { ...mouseState.current.position };
  }, []);
  
  const getMouseWheelDelta = useCallback(() => {
    return mouseState.current.wheelDelta;
  }, []);
  
  // Toggle keyboard and mouse input
  const toggleKeyMouseInput = useCallback((enabled: boolean) => {
    setIsKeyMouseEnabled(enabled);
  }, []);

  // Add function to get microphone level
  const getMicrophoneLevel = useCallback(() => {
    if (!isMicrophoneEnabled || !analyserRef.current || !dataArrayRef.current) {
      return 0;
    }
    
    // Get audio data from analyzer
    analyserRef.current.getByteFrequencyData(dataArrayRef.current);
    
    // Calculate average volume (0-1)
    const data = dataArrayRef.current;
    let sum = 0;
    for (let i = 0; i < data.length; i++) {
      sum += data[i];
    }
    const average = sum / data.length;
    
    return average / 255;
  }, [isMicrophoneEnabled]);
  
  // Function to enable microphone input
  const enableMicrophone = useCallback(async () => {
    if (isMicrophoneEnabled) return true;
    
    try {
      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      microphoneStreamRef.current = stream;
      
      // Set up audio context and analyzer
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      audioContextRef.current = audioContext;
      
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256; // Small FFT for efficiency
      analyserRef.current = analyser;
      
      const source = audioContext.createMediaStreamSource(stream);
      source.connect(analyser);
      
      // Create data array for analysis
      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      dataArrayRef.current = dataArray;
      
      setIsMicrophoneEnabled(true);
      setIsMicrophoneAvailable(true);
      return true;
    } catch (error) {
      console.error('Error accessing microphone:', error);
      setIsMicrophoneAvailable(false);
      return false;
    }
  }, [isMicrophoneEnabled]);
  
  // Function to disable microphone input
  const disableMicrophone = useCallback(() => {
    if (microphoneStreamRef.current) {
      microphoneStreamRef.current.getTracks().forEach(track => track.stop());
      microphoneStreamRef.current = null;
    }
    
    if (audioContextRef.current) {
      audioContextRef.current.close().catch(console.error);
      audioContextRef.current = null;
    }
    
    analyserRef.current = null;
    dataArrayRef.current = null;
    
    setIsMicrophoneEnabled(false);
  }, []);
  
  // Check for microphone capability on mount
  useEffect(() => {
    const checkMicrophoneAvailability = async () => {
      try {
        // Just check if we can access the device list
        await navigator.mediaDevices.enumerateDevices();
        setIsMicrophoneAvailable(true);
      } catch (error) {
        console.error('Microphone enumeration failed:', error);
        setIsMicrophoneAvailable(false);
      }
    };
    
    checkMicrophoneAvailability();
    
    // Clean up on unmount
    return () => {
      disableMicrophone();
    };
  }, [disableMicrophone]);

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
  
  // Set up keyboard and mouse event listeners
  useEffect(() => {
    if (isKeyMouseEnabled) {
      console.log("Setting up keyboard and mouse event listeners");
      
      window.addEventListener("keydown", handleKeyDown);
      window.addEventListener("keyup", handleKeyUp);
      window.addEventListener("mousemove", handleMouseMove);
      window.addEventListener("mousedown", handleMouseDown);
      window.addEventListener("mouseup", handleMouseUp);
      window.addEventListener("wheel", handleMouseWheel);
      
      return () => {
        console.log("Cleaning up keyboard and mouse event listeners");
        window.removeEventListener("keydown", handleKeyDown);
        window.removeEventListener("keyup", handleKeyUp);
        window.removeEventListener("mousemove", handleMouseMove);
        window.removeEventListener("mousedown", handleMouseDown);
        window.removeEventListener("mouseup", handleMouseUp);
        window.removeEventListener("wheel", handleMouseWheel);
      };
    }
  }, [
    isKeyMouseEnabled, 
    handleKeyDown, 
    handleKeyUp, 
    handleMouseMove, 
    handleMouseDown, 
    handleMouseUp, 
    handleMouseWheel
  ]);

  return {
    // GamePad API
    controllers,
    isControllerConnected,
    wasButtonJustPressed,
    hasAxisChanged,
    refreshControllers: pollGamepads, // Expose the poll function for manual refreshes
    
    // Keyboard and mouse API
    isKeyMouseEnabled,
    toggleKeyMouseInput,
    isKeyPressed,
    wasKeyJustPressed,
    isMouseButtonPressed,
    wasMouseButtonJustPressed,
    getMousePositionDelta,
    getMousePosition,
    getMouseWheelDelta,
    
    // Add microphone functions
    isMicrophoneAvailable,
    isMicrophoneEnabled,
    enableMicrophone,
    disableMicrophone,
    getMicrophoneLevel
  };
} 