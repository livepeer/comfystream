// Base mapping interface
export interface ControllerMappingBase {
  type: 'axis' | 'button' | 'key' | 'mouse' | 'mouse-movement';
  nodeId: string;
  fieldName: string;
}

// Axis mapping for continuous inputs like sliders
export interface AxisMapping extends ControllerMappingBase {
  type: 'axis';
  axisIndex: number;
  multiplier: number;
  minOverride?: number;
  maxOverride?: number;
}

// Button mode type for reuse
export type ButtonModeType = 'toggle' | 'momentary' | 'series' | 'increment' | 'axis';

// Button mapping for toggles, momentary or cycling through series of values
export interface ButtonMapping extends ControllerMappingBase {
  type: 'button';
  buttonIndex: number;
  mode: ButtonModeType;
  valueWhenPressed: string | number;
  valueWhenReleased?: string | number;
  
  // For series mode
  nextButtonIndex?: number; // Optional second button for series navigation
  valuesList?: Array<string | number>; // List of values to cycle through
  currentValueIndex?: number; // Current index in the valuesList
  
  // For increment mode
  incrementStep?: number; // Amount to increment/decrement by
  minOverride?: number; // Minimum value
  maxOverride?: number; // Maximum value
}

// Keyboard key mapping
export interface KeyMapping extends ControllerMappingBase {
  type: 'key';
  keyCode: string; // Key code (e.g., 'KeyA', 'Space', 'ArrowUp')
  mode: ButtonModeType;
  valueWhenPressed: string | number;
  valueWhenReleased?: string | number;
  
  // For series mode
  nextKeyCode?: string; // Optional second key for series navigation
  valuesList?: Array<string | number>; // List of values to cycle through
  currentValueIndex?: number; // Current index in the valuesList
  
  // For increment mode
  incrementStep?: number; // Amount to increment/decrement by
  minOverride?: number; // Minimum value
  maxOverride?: number; // Maximum value
}

// Mouse movement mapping (horizontal or vertical movement)
export interface MouseMovementMapping extends ControllerMappingBase {
  type: 'mouse-movement';
  axis: 'x' | 'y';  // Which axis to track
  multiplier: number;
  minOverride?: number;
  maxOverride?: number;
}

// Mouse mapping for buttons and wheel
export interface MouseMapping extends ControllerMappingBase {
  type: 'mouse';
  action: 'wheel' | 'button';
  buttonIndex?: number; // 0 = left, 1 = middle, 2 = right
  mode: ButtonModeType;
  multiplier?: number; // For wheel
  valueWhenPressed?: string | number;
  valueWhenReleased?: string | number;
  minOverride?: number; // For wheel
  maxOverride?: number; // For wheel
  
  // For series mode
  nextAction?: 'button'; // Optional second action for series navigation
  nextButtonIndex?: number; // Button index for next action
  valuesList?: Array<string | number>; // List of values to cycle through
  currentValueIndex?: number; // Current index in the valuesList
  
  // For increment mode
  incrementStep?: number; // Amount to increment/decrement by
}

// Union type for all mapping types
export type ControllerMapping = AxisMapping | ButtonMapping | KeyMapping | MouseMapping | MouseMovementMapping;

// Mapping storage interface
export interface MappingStorage {
  mappings: Record<string, Record<string, ControllerMapping>>;
} 