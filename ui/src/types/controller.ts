// Base mapping interface
export interface ControllerMappingBase {
  type: 'axis' | 'button' | 'promptList';
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

// Button mapping for toggles or incremental changes
export interface ButtonMapping extends ControllerMappingBase {
  type: 'button';
  buttonIndex: number;
  toggleMode: boolean;
  valueWhenPressed: string | number;
  valueWhenReleased?: string | number;
}

// Prompt list mapping for cycling through prompts
export interface PromptListMapping extends ControllerMappingBase {
  type: 'promptList';
  nextButtonIndex: number;
  prevButtonIndex: number;
  currentPromptIndex: number;
}

// Union type for all mapping types
export type ControllerMapping = AxisMapping | ButtonMapping | PromptListMapping;

// Mapping storage interface
export interface MappingStorage {
  mappings: Record<string, Record<string, ControllerMapping>>;
} 