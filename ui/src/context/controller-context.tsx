"use client";

import * as React from "react";
import { createContext, useContext, ReactNode } from "react";
import { useController } from "@/hooks/use-controller";
import { useControllerMapping } from "@/hooks/use-controller-mapping";
import { ControllerMapping } from "@/types/controller";

interface ControllerContextType {
  isControllerConnected: boolean;
  mappings: Record<string, Record<string, ControllerMapping>>;
  saveMapping: (nodeId: string, fieldName: string, mapping: ControllerMapping) => void;
  removeMapping: (nodeId: string, fieldName: string) => void;
  getMapping: (nodeId: string, fieldName: string) => ControllerMapping | undefined;
  hasMapping: (nodeId: string, fieldName: string) => boolean;
}

const ControllerContext = createContext<ControllerContextType | undefined>(undefined);

export const useControllerContext = () => {
  const context = useContext(ControllerContext);
  if (!context) {
    throw new Error("useControllerContext must be used within a ControllerProvider");
  }
  return context;
};

interface ControllerProviderProps {
  children: ReactNode;
}

export const ControllerProvider = ({ children }: ControllerProviderProps) => {
  const { isControllerConnected } = useController();
  const { mappings, saveMapping, removeMapping, getMapping, hasMapping } = useControllerMapping();
  
  const value = {
    isControllerConnected,
    mappings,
    saveMapping,
    removeMapping,
    getMapping,
    hasMapping
  };
  
  return (
    <ControllerContext.Provider value={value}>
      {children}
    </ControllerContext.Provider>
  );
}; 