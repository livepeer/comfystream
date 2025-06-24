import * as React from "react";
import { Prompt } from "@/types";
import { useSegments } from "@/hooks/use-segments";
import { SegmentsContext } from "@/context/segments-context";

export interface SegmentsProps extends React.HTMLAttributes<HTMLDivElement> {
  url: string;
  prompts: Prompt[] | null;
  connect: boolean;
  onConnected: () => void;
  onDisconnected: () => void;
  localStream: MediaStream | null;
  segmentTime: number;
  resolution?: {
    width: number;
    height: number;
  };
}

export const SegmentsConnector = (props: SegmentsProps) => {
  const segments = useSegments(props);

  console.log("[SegmentsConnector] Segments context value:", {
    isRecording: segments?.isRecording,
    isConnected: segments?.isConnected,
    lastSegmentTime: segments?.lastSegmentTime,
    error: segments?.error,
  });

  return (
    <div>
      {segments && (
        <SegmentsContext.Provider value={segments}>
          {props.children}
        </SegmentsContext.Provider>
      )}
    </div>
  );
};
