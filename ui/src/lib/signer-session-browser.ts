export const SIGNER_SESSION_STORAGE_KEY = "comfypeer-signer-session";

export type BrowserSignerSession = {
  access_token: string;
  token_type: "Bearer";
  expires_in: number;
  scope?: string;
  signer_url?: string | null;
  discovery_url: string;
  stored_at: number;
};

export function clearSignerSession(): void {
  try {
    sessionStorage.removeItem(SIGNER_SESSION_STORAGE_KEY);
  } catch {
    /* ignore */
  }
}

export function readSignerSession(): BrowserSignerSession | null {
  try {
    const raw = sessionStorage.getItem(SIGNER_SESSION_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as BrowserSignerSession;
    if (!parsed?.access_token || !parsed?.discovery_url) return null;
    return parsed;
  } catch {
    return null;
  }
}

export function isSignerSessionFresh(
  session: BrowserSignerSession,
  skewSec = 60,
): boolean {
  const expiresIn = Number(session.expires_in) || 0;
  if (expiresIn <= 0) return true;
  const ageSec = (Date.now() - (session.stored_at || 0)) / 1000;
  return ageSec < expiresIn - skewSec;
}

export async function mintBrowserSignerSession(
  comfypeerOrigin: string,
): Promise<BrowserSignerSession> {
  const base = comfypeerOrigin.replace(/\/$/, "");
  if (!base) {
    throw new Error("ComfyPeer origin is required for orchestrator mode");
  }
  const res = await fetch(`${base}/api/pymthouse/signer-session`, {
    method: "POST",
    credentials: "include",
    headers: { Accept: "application/json" },
  });
  if (!res.ok) {
    const body = (await res.json().catch(() => null)) as { error?: string } | null;
    throw new Error(body?.error || `signer-session failed (${res.status})`);
  }
  const data = (await res.json()) as Omit<BrowserSignerSession, "stored_at">;
  const envelope: BrowserSignerSession = {
    access_token: data.access_token,
    token_type: "Bearer",
    expires_in: Number(data.expires_in) || 0,
    scope: data.scope,
    signer_url: data.signer_url,
    discovery_url: data.discovery_url,
    stored_at: Date.now(),
  };
  sessionStorage.setItem(SIGNER_SESSION_STORAGE_KEY, JSON.stringify(envelope));
  return envelope;
}

export async function ensureBrowserSignerSession(
  comfypeerOrigin: string,
): Promise<BrowserSignerSession> {
  const existing = readSignerSession();
  if (existing && isSignerSessionFresh(existing)) {
    return existing;
  }
  return mintBrowserSignerSession(comfypeerOrigin);
}
