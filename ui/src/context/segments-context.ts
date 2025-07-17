import { Segments } from "@/lib/segments";
import * as React from "react";

export const SegmentsContext = React.createContext<Segments | undefined>(undefined);

export function useSegmentsContext() {
  const ctx = React.useContext(SegmentsContext);
  if (!ctx) {
    throw new Error("tried to access segments context outside of Segments component");
  }
  return ctx;
}
