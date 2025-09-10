"use client";

import React, { useState, useEffect, useRef } from 'react';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Mic, MicOff, Volume2, VolumeX, Copy, Trash2 } from 'lucide-react';
import { toast } from 'sonner';

interface TranscriptionSegment {
  id: string;
  text: string;
  timestamp: Date;
  confidence?: number;
  isProcessing?: boolean;
}

interface TranscriptionViewerProps {
  isConnected: boolean;
  transcriptionData?: string;
}

export const TranscriptionViewer: React.FC<TranscriptionViewerProps> = ({
  isConnected,
  transcriptionData,
}) => {
  const [segments, setSegments] = useState<TranscriptionSegment[]>([]);
  const [isListening, setIsListening] = useState(true);
  const [isMuted, setIsMuted] = useState(false);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const lastSegmentRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to latest transcription
  useEffect(() => {
    if (lastSegmentRef.current) {
      lastSegmentRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [segments]);

  // Process incoming transcription data
  useEffect(() => {
    if (!transcriptionData || !isListening) return;

    try {
      // Handle different formats: text, json_segments, or json_words
      let newSegments: TranscriptionSegment[] = [];
      
      if (transcriptionData.startsWith('[') || transcriptionData.startsWith('{')) {
        // JSON format
        const parsed = JSON.parse(transcriptionData);
        if (Array.isArray(parsed)) {
          // json_segments or json_words format
          newSegments = parsed.map((item, index) => ({
            id: `${Date.now()}-${index}`,
            text: item.text || item.word || '',
            timestamp: new Date(),
            confidence: item.probability || 0.9,
          })).filter(seg => seg.text.trim() && !seg.text.includes('__WARMUP_SENTINEL__'));
        }
      } else if (typeof transcriptionData === 'string' && transcriptionData.trim()) {
        // Plain text format
        if (!transcriptionData.includes('__WARMUP_SENTINEL__')) {
          newSegments = [{
            id: `${Date.now()}`,
            text: transcriptionData.trim(),
            timestamp: new Date(),
            confidence: 0.9,
          }];
        }
      }

      if (newSegments.length > 0) {
        setSegments(prev => [...prev, ...newSegments].slice(-50)); // Keep last 50 segments
      }
    } catch (error) {
      console.error('Error parsing transcription data:', error);
    }
  }, [transcriptionData, isListening]);

  const copyAllText = () => {
    const allText = segments.map(seg => seg.text).join(' ');
    navigator.clipboard.writeText(allText);
    toast.success('Transcription copied to clipboard');
  };

  const clearTranscriptions = () => {
    setSegments([]);
    toast.success('Transcriptions cleared');
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
  };

  const getConfidenceColor = (confidence?: number) => {
    if (!confidence) return 'bg-gray-500';
    if (confidence >= 0.8) return 'bg-green-500';
    if (confidence >= 0.6) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  return (
    <Card className="w-full h-full bg-slate-900/95 border-slate-700 text-white">
      <CardHeader className="pb-3 space-y-0">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg font-semibold flex items-center gap-2">
            <div className="relative">
              <Mic className="w-5 h-5" />
              {isConnected && (
                <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-500 rounded-full animate-pulse" />
              )}
            </div>
            Live Transcription
          </CardTitle>
          <div className="flex items-center gap-2">
            <Badge 
              variant={isConnected ? "default" : "secondary"}
              className={isConnected ? "bg-green-600 text-white" : "bg-gray-600 text-gray-300"}
            >
              {isConnected ? "Connected" : "Disconnected"}
            </Badge>
          </div>
        </div>
        
        <div className="flex items-center justify-between pt-2">
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsListening(!isListening)}
              className={`text-xs ${isListening ? 'text-green-400 hover:text-green-300' : 'text-gray-400 hover:text-gray-300'}`}
            >
              {isListening ? <Mic className="w-4 h-4 mr-1" /> : <MicOff className="w-4 h-4 mr-1" />}
              {isListening ? 'Listening' : 'Paused'}
            </Button>
            
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsMuted(!isMuted)}
              className={`text-xs ${isMuted ? 'text-gray-400 hover:text-gray-300' : 'text-blue-400 hover:text-blue-300'}`}
            >
              {isMuted ? <VolumeX className="w-4 h-4 mr-1" /> : <Volume2 className="w-4 h-4 mr-1" />}
              {isMuted ? 'Muted' : 'Audio'}
            </Button>
          </div>
          
          <div className="flex items-center gap-1">
            <Button
              variant="ghost"
              size="sm"
              onClick={copyAllText}
              disabled={segments.length === 0}
              className="text-xs text-gray-400 hover:text-gray-300"
            >
              <Copy className="w-4 h-4" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={clearTranscriptions}
              disabled={segments.length === 0}
              className="text-xs text-gray-400 hover:text-gray-300"
            >
              <Trash2 className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </CardHeader>
      
      <CardContent className="p-0 flex-1 overflow-hidden">
        <ScrollArea className="h-full px-4" ref={scrollAreaRef}>
          <div className="space-y-3 pb-4">
            {segments.length === 0 ? (
              <div className="text-center text-gray-500 py-8">
                <Mic className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <p className="text-sm">
                  {isConnected 
                    ? (isListening ? "Waiting for speech..." : "Transcription paused")
                    : "Connect to start transcription"
                  }
                </p>
              </div>
            ) : (
              segments.map((segment, index) => (
                <div
                  key={segment.id}
                  ref={index === segments.length - 1 ? lastSegmentRef : undefined}
                  className="group relative bg-slate-800/50 rounded-lg p-3 border border-slate-700/50 hover:bg-slate-800/70 transition-colors"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex-1 min-w-0">
                      <p className="text-sm text-white leading-relaxed break-words">
                        {segment.text}
                      </p>
                    </div>
                    <div className="flex items-center gap-2 flex-shrink-0">
                      {segment.confidence && (
                        <div className="flex items-center gap-1">
                          <div 
                            className={`w-2 h-2 rounded-full ${getConfidenceColor(segment.confidence)}`}
                            title={`Confidence: ${Math.round(segment.confidence * 100)}%`}
                          />
                        </div>
                      )}
                      <span className="text-xs text-gray-400 font-mono">
                        {formatTime(segment.timestamp)}
                      </span>
                    </div>
                  </div>
                  
                  {/* Copy button for individual segment */}
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => {
                      navigator.clipboard.writeText(segment.text);
                      toast.success('Segment copied');
                    }}
                    className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity p-1 h-6 w-6 text-gray-400 hover:text-gray-300"
                  >
                    <Copy className="w-3 h-3" />
                  </Button>
                </div>
              ))
            )}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
};
