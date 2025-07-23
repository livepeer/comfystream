export interface Segments {
  isRecording: boolean;
  isConnected: boolean;
  lastSegmentTime: number | null;
  error: string | null;
}
