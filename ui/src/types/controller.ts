// Base mapping interface
export interface ControllerMappingBase {
  type: 'axis' | 'button';
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

// Button mapping for toggles, momentary or cycling through series of values
export interface ButtonMapping extends ControllerMappingBase {
  type: 'button';
  buttonIndex: number;
  mode: 'toggle' | 'momentary' | 'series';
  valueWhenPressed: string | number;
  valueWhenReleased?: string | number;
  
  // For series mode
  nextButtonIndex?: number; // Optional second button for series navigation
  valuesList?: Array<string | number>; // List of values to cycle through
  currentValueIndex?: number; // Current index in the valuesList
}

// Union type for all mapping types
export type ControllerMapping = AxisMapping | ButtonMapping;

// Mapping storage interface
export interface MappingStorage {
  mappings: Record<string, Record<string, ControllerMapping>>;
} 