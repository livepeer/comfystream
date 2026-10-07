"use client";

import * as React from "react";
import { useCallback, useEffect, useRef, useState } from "react";
import { PeerContext } from "@/context/peer-context";
import { Peer } from "@/lib/peer";
import { connectViaBridge } from "@/lib/live-runner-client";
import { ensureBrowserSignerSession } from "@/lib/signer-session-browser";
import avPassthrough from "@/workflows/av-passthrough-api.json";
import invertColorAv from "@/workflows/invert-color-av-passthrough-api.json";

const WORKFLOWS = {
  "av-passthrough": avPassthrough,
  "invert-color-av": invertColorAv,
} as const;

export type OrchPeerConnectorProps = {
  connect: boolean;
  localStream: MediaStream | null;
  comfypeerOrigin: string;
  bridgeUrl: string;
  pipeline: keyof typeof WORKFLOWS;
  resolution: { width: number; height: number };
  onConnected: () => void;
  onDisconnected: () => void;
  children?: React.ReactNode;
};

export function OrchPeerConnector({
  connect,
  localStream,
  comfypeerOrigin,
  bridgeUrl,
  pipeline,
  resolution,
  onConnected,
  onDisconnected,
  children,
}: OrchPeerConnectorProps) {
  const [peer, setPeer] = useState<Peer | null>(null);
  const closeRef = useRef<(() => Promise<void>) | null>(null);
  const startingRef = useRef(false);

  const teardown = useCallback(async () => {
    if (closeRef.current) {
      await closeRef.current().catch(() => null);
      closeRef.current = null;
    }
    setPeer(null);
  }, []);

  useEffect(() => {
    if (!connect) {
      void teardown().then(() => onDisconnected());
      return;
    }
    if (!localStream || startingRef.current) return;

    let cancelled = false;
    startingRef.current = true;

    void (async () => {
      try {
        const signer = await ensureBrowserSignerSession(comfypeerOrigin);
        const prompts = WORKFLOWS[pipeline];
        const session = await connectViaBridge({
          bridgeUrl,
          localStream,
          accessToken: signer.access_token,
          discoveryUrl: signer.discovery_url,
          signerUrl: signer.signer_url,
          prompts,
          width: resolution.width,
          height: resolution.height,
          audio: true,
          onConnectionState: (state) => {
            if (state === "connected") onConnected();
            if (
              state === "failed" ||
              state === "disconnected" ||
              state === "closed"
            ) {
              onDisconnected();
            }
          },
        });
        if (cancelled) {
          await session.close();
          return;
        }
        closeRef.current = session.close;
        setPeer({
          peerConnection: session.pc,
          remoteStream: session.remoteStream,
          controlChannel: null,
          dataChannel: null,
          textOutputData: null,
        });
        if (session.pc.connectionState === "connected") {
          onConnected();
        }
      } catch (err) {
        console.error("[OrchPeerConnector]", err);
        onDisconnected();
      } finally {
        startingRef.current = false;
      }
    })();

    return () => {
      cancelled = true;
      void teardown();
    };
  }, [
    connect,
    localStream,
    comfypeerOrigin,
    bridgeUrl,
    pipeline,
    resolution.width,
    resolution.height,
    onConnected,
    onDisconnected,
    teardown,
  ]);

  return (
    <div>
      {peer ? (
        <PeerContext.Provider value={peer}>{children}</PeerContext.Provider>
      ) : (
        children
      )}
    </div>
  );
}
