import { useState, useEffect, useRef, useCallback } from "react";
import { Prompt } from "@/types";
import { Segments } from "@/lib/segments";

interface SegmentsProps {
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

const MAX_SEND_RETRIES = 3;
const SEND_RETRY_INTERVAL = 1000;

export function useSegments(props: SegmentsProps): Segments {
  const {
    url,
    prompts,
    connect,
    onConnected,
    onDisconnected,
    localStream,
    segmentTime,
    resolution,
  } = props;

  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [lastSegmentTime, setLastSegmentTime] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const segmentIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const segmentCountRef = useRef<number>(0);

  const sendSegment = useCallback(
    async (blob: Blob, segmentIndex: number, retry: number = 0): Promise<void> => {
      try {
        const formData = new FormData();
        formData.append('segment', blob, `segment_${segmentIndex}.webm`);
        formData.append('segmentIndex', segmentIndex.toString());
        formData.append('timestamp', Date.now().toString());
        
        if (prompts) {
          formData.append('prompts', JSON.stringify(prompts));
        }
        
        if (resolution) {
          formData.append('resolution', JSON.stringify(resolution));
        }

        const response = await fetch(`${url}/api/segment`, {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`Segment send HTTP error: ${response.status}`);
        }

        const result = await response.json();
        console.log(`[useSegments] Segment ${segmentIndex} sent successfully:`, result);
        setLastSegmentTime(Date.now());
        setError(null);
      } catch (error) {
        console.error(`[useSegments] Error sending segment ${segmentIndex}:`, error);
        
        if (retry < MAX_SEND_RETRIES) {
          console.log(`[useSegments] Retrying segment ${segmentIndex}, attempt ${retry + 1}`);
          await new Promise((resolve) => setTimeout(resolve, SEND_RETRY_INTERVAL));
          return sendSegment(blob, segmentIndex, retry + 1);
        }
        
        setError(`Failed to send segment after ${MAX_SEND_RETRIES} retries`);
        throw error;
      }
    },
    [url, prompts, resolution],
  );

  const startRecording = useCallback(() => {
    if (!localStream || isRecording) return;

    try {
      // Get supported MIME type
      const mimeTypes = [
        'video/webm;codecs=vp9,opus',
        'video/webm;codecs=vp8,opus',
        'video/webm;codecs=h264,opus',
        'video/webm',
        'video/mp4',
      ];

      let mimeType = '';
      for (const type of mimeTypes) {
        if (MediaRecorder.isTypeSupported(type)) {
          mimeType = type;
          break;
        }
      }

      const mediaRecorder = new MediaRecorder(localStream, {
        mimeType: mimeType || undefined,
        videoBitsPerSecond: 2500000, // 2.5 Mbps
        audioBitsPerSecond: 128000,  // 128 kbps
      });

      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];
      segmentCountRef.current = 0;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        if (chunksRef.current.length > 0) {
          const blob = new Blob(chunksRef.current, { type: mimeType });
          const segmentIndex = segmentCountRef.current++;
          
          // Send the segment
          sendSegment(blob, segmentIndex).catch((error) => {
            console.error(`[useSegments] Failed to send segment ${segmentIndex}:`, error);
          });
          
          chunksRef.current = [];
        }

        // Auto-restart recording if we're still supposed to be recording
        setTimeout(() => {
          if (mediaRecorderRef.current && 
              mediaRecorderRef.current.state === 'inactive' && 
              isRecording && 
              localStream) {
            try {
              mediaRecorderRef.current.start();
            } catch (error) {
              console.error('[useSegments] Error restarting recording:', error);
            }
          }
        }, 100);
      };

      mediaRecorder.onerror = (event) => {
        console.error('[useSegments] MediaRecorder error:', event);
        setError('Recording error occurred');
      };

      // Start recording
      mediaRecorder.start();
      setIsRecording(true);
      setIsConnected(true);
      onConnected();

      console.log(`[useSegments] Started recording with ${segmentTime}s segments`);

      // Set up interval to capture segments
      segmentIntervalRef.current = setInterval(() => {
        if (mediaRecorderRef.current && 
            mediaRecorderRef.current.state === 'recording' && 
            isRecording) {
          // Stop current recording to get the segment
          try {
            mediaRecorderRef.current.stop();
          } catch (error) {
            console.error('[useSegments] Error stopping recording for segment:', error);
          }
        }
      }, segmentTime * 1000);

    } catch (error) {
      console.error('[useSegments] Error starting recording:', error);
      setError('Failed to start recording');
    }
  }, [localStream, isRecording, segmentTime, onConnected, sendSegment]);

  const stopRecording = useCallback(() => {
    if (segmentIntervalRef.current) {
      clearInterval(segmentIntervalRef.current);
      segmentIntervalRef.current = null;
    }

    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }

    setIsRecording(false);
    setIsConnected(false);
    onDisconnected();
    
    console.log('[useSegments] Stopped recording');
  }, [onDisconnected]);

  // Main effect to handle connection state
  useEffect(() => {
    if (connect && localStream) {
      startRecording();
    } else {
      stopRecording();
    }

    // Cleanup on unmount
    return () => {
      stopRecording();
    };
  }, [connect, localStream, startRecording, stopRecording]);

  // Effect to handle segment time changes during recording
  useEffect(() => {
    if (isRecording && segmentIntervalRef.current) {
      // Restart with new segment time
      clearInterval(segmentIntervalRef.current);
      
      segmentIntervalRef.current = setInterval(() => {
        if (mediaRecorderRef.current && 
            mediaRecorderRef.current.state === 'recording' && 
            isRecording) {
          try {
            mediaRecorderRef.current.stop();
          } catch (error) {
            console.error('[useSegments] Error stopping recording for segment time change:', error);
          }
        }
      }, segmentTime * 1000);

      console.log(`[useSegments] Updated segment time to ${segmentTime}s`);
    }
  }, [segmentTime, isRecording]);

  // Effect to handle resolution changes
  useEffect(() => {
    // Resolution changes will be included in the next segment automatically
    if (resolution && isConnected) {
      console.log('[useSegments] Resolution updated:', resolution);
    }
  }, [resolution, isConnected]);

  return {
    isRecording,
    isConnected,
    lastSegmentTime,
    error,
  };
}
