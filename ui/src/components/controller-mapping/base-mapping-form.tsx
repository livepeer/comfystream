"use client";

import { ControllerMapping } from '@/types/controller';

// Base props that all mapping forms will use
export interface BaseMappingFormProps {
  currentMapping?: ControllerMapping;
  nodeId: string;
  fieldName: string;
  inputMin?: number;
  inputMax?: number;
  onChange?: (value: any) => void;
  onSaveMapping: (mapping: ControllerMapping) => void;
}

// Extended props for controller-based mappings
export interface ControllerMappingFormProps extends BaseMappingFormProps {
  controllers: Gamepad[];
  isControllerConnected: boolean;
} 