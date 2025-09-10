import * as React from "react";
import { useState, useCallback } from "react";

interface StreamControlProps {
  className?: string;
}

export function StreamControl({ className = "" }: StreamControlProps) {
  const [isLoading, setIsLoading] = useState(false);

  // Open popup which polls opener for stream and clones tracks locally (no postMessage MediaStream cloning)
  const openWebRTCPopup = useCallback(() => {
    // Must open first synchronous to user gesture to avoid blockers.
    let popup: Window | null = window.open('about:blank', '_blank', 'width=1024,height=1024');
    if (!popup) {
      // Attempt simplified open.
      popup = window.open('');
    }
    if (!popup) {
      alert('Popup blocked. Please allow popups for this site.');
      return;
    }
    const html = `<!DOCTYPE html><html><head><title>ComfyStream Preview</title><meta charset='utf-8' />
    <style>html,body{margin:0;background:#000;height:100%;display:flex;align-items:center;justify-content:center;font-family:sans-serif}video{max-width:100%;max-height:100%;background:#000}#status{color:#0f0;position:absolute;top:6px;left:8px;text-align:left;font:12px monospace;text-shadow:0 0 4px #000}</style>
    </head><body>
      <video id="webrtc_preview" autoplay playsinline muted></video>
      <div id="status">Initializing…</div>
      <script>
        (function(){
          const statusEl = document.getElementById('status');
          const video = document.getElementById('webrtc_preview');
          const openerRef = window.opener;
          let attempts = 0;
          const MAX_ATTEMPTS = 200; // ~60s at 300ms
          let localStream = new MediaStream();
          let clonedIds = new Set();
          let lastParentStream = null;
          function setStatus(msg){ if(statusEl) statusEl.textContent = msg; }
          function validateOpener(){
            try {
              if(!window.opener || window.opener !== openerRef){ setStatus('Opener lost. Closing…'); setTimeout(()=>window.close(),800); return false; }
              void window.opener.location.href; // access for same-origin check
              return true;
            } catch { setStatus('Cross-origin opener. Closing…'); setTimeout(()=>window.close(),800); return false; }
          }
          function attachVideo(){ if(video.srcObject !== localStream) video.srcObject = localStream; }
          function cloneTracks(){
            if(!validateOpener()) return;
            const parentStream = window.opener.__comfystreamRemoteStream;
            if(!parentStream){ setStatus('Waiting for stream…'); return; }
            if(lastParentStream && lastParentStream !== parentStream){
              localStream.getTracks().forEach(t=>{ try{t.stop();}catch{} });
              localStream = new MediaStream();
              clonedIds = new Set();
            }
            lastParentStream = parentStream;
            let added = false;
            parentStream.getTracks().forEach(src => {
              if(src.readyState === 'ended') return;
              if(!clonedIds.has(src.id)){
                try{ const c = src.clone(); localStream.addTrack(c); clonedIds.add(src.id); added = true; c.addEventListener('ended',()=>{ clonedIds.delete(src.id); }); }
                catch{ /* fallback skip */ }
              }
            });
            if(added){ attachVideo(); setStatus('Live'); if(video.play) video.play().catch(()=>{}); }
          }
          const interval = setInterval(()=>{
            attempts++;
            if(!validateOpener()){ clearInterval(interval); return; }
            // Parent cleared global (disconnect)
            if(!window.opener.__comfystreamRemoteStream){ setStatus('Parent stream ended'); clearInterval(interval); setTimeout(()=>window.close(),1200); return; }
            cloneTracks();
            // Remove ended local tracks (allow re-clone)
            localStream.getTracks().forEach(t=>{ if(t.readyState==='ended'){ localStream.removeTrack(t); try{t.stop();}catch{}; clonedIds.delete(t.id); }});
            if(attempts>=MAX_ATTEMPTS && localStream.getTracks().length===0){ setStatus('Timeout waiting for stream'); clearInterval(interval); setTimeout(()=>window.close(),1500); }
          },300);
          window.addEventListener('beforeunload', ()=>{ clearInterval(interval); localStream.getTracks().forEach(t=>{ try{t.stop();}catch{} }); });
          // Initial attempt
          cloneTracks();
        })();
      </script>
    </body></html>`;
    popup.document.write(html);
    popup.document.close();
  }, []);

  const openStreamWindow = () => {
    openWebRTCPopup();
  };

  return (
    <button 
  onClick={openStreamWindow}
  disabled={isLoading}
  className={`absolute bottom-4 right-4 z-10 p-2 bg-black/50 hover:bg-black/70 text-white rounded-full transition-colors ${className}`}
  title={"Open WebRTC preview (cloned tracks)"} 
  aria-label="Open WebRTC preview"
    >
      <svg 
        xmlns="http://www.w3.org/2000/svg" 
        viewBox="0 0 512 512" 
        className="w-6 h-6 fill-current"
      >
        <path d="M352 0c-12.9 0-24.6 7.8-29.6 19.8s-2.2 25.7 6.9 34.9L370.7 96 201.4 265.4c-12.5 12.5-12.5 32.8 0 45.3s32.8 12.5 45.3 0L416 141.3l41.4 41.4c9.2 9.2 22.9 11.9 34.9 6.9s19.8-16.6 19.8-29.6l0-128c0-17.7-14.3-32-32-32L352 0zM80 32C35.8 32 0 67.8 0 112L0 432c0 44.2 35.8 80 80 80l320 0c44.2 0 80-35.8 80-80l0-112c0-17.7-14.3-32-32-32s-32 14.3-32 32l0 112c0 8.8-7.2 16-16 16L80 448c-8.8 0-16-7.2-16-16l0-320c0-8.8 7.2-16 16-16l112 0c17.7 0 32-14.3 32-32s-14.3-32-32-32L80 32z"/>
      </svg>
    </button>
  );
}
