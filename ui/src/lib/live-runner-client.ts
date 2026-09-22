/**
 * Session-control helpers for ComfyStream UI orch mode + WebRTC bridge.
 */

export type BridgeOfferResponse = {
  sdp: string;
  type: string;
  in?: string;
  out?: string;
};

export async function connectViaBridge(opts: {
  bridgeUrl: string;
  localStream: MediaStream;
  accessToken: string;
  discoveryUrl: string;
  signerUrl?: string | null;
  prompts: unknown;
  width?: number;
  height?: number;
  audio?: boolean;
  onRemoteStream?: (stream: MediaStream) => void;
  onConnectionState?: (state: RTCPeerConnectionState) => void;
}): Promise<{
  pc: RTCPeerConnection;
  remoteStream: MediaStream;
  close: () => Promise<void>;
}> {
  const bridgeBase = opts.bridgeUrl.replace(/\/$/, "");
  const pc = new RTCPeerConnection({
    iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
  });
  const remoteStream = new MediaStream();

  pc.ontrack = (ev) => {
    for (const track of ev.streams[0]?.getTracks() ?? [ev.track]) {
      remoteStream.addTrack(track);
    }
    opts.onRemoteStream?.(remoteStream);
  };
  pc.onconnectionstatechange = () => {
    opts.onConnectionState?.(pc.connectionState);
  };

  for (const track of opts.localStream.getTracks()) {
    pc.addTrack(track, opts.localStream);
  }

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);

  await new Promise<void>((resolve) => {
    if (pc.iceGatheringState === "complete") {
      resolve();
      return;
    }
    const check = () => {
      if (pc.iceGatheringState === "complete") {
        pc.removeEventListener("icegatheringstatechange", check);
        resolve();
      }
    };
    pc.addEventListener("icegatheringstatechange", check);
    setTimeout(resolve, 2000);
  });

  const local = pc.localDescription;
  if (!local?.sdp) {
    throw new Error("failed to create local SDP offer");
  }

  const res = await fetch(`${bridgeBase}/offer`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify({
      sdp: local.sdp,
      type: local.type,
      access_token: opts.accessToken,
      discovery_url: opts.discoveryUrl,
      signer_url: opts.signerUrl || undefined,
      prompts: opts.prompts,
      width: opts.width ?? 512,
      height: opts.height ?? 512,
      audio: opts.audio ?? true,
    }),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    pc.close();
    throw new Error(`bridge /offer failed (${res.status}): ${text}`);
  }
  const answer = (await res.json()) as BridgeOfferResponse;
  await pc.setRemoteDescription({
    sdp: answer.sdp,
    type: (answer.type as RTCSdpType) || "answer",
  });

  return {
    pc,
    remoteStream,
    close: async () => {
      pc.close();
    },
  };
}
