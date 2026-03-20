/**
 * Centralized API configuration.
 *
 * - Local dev  (vite dev)  → requests go to "/api/..." which Vite proxies to localhost:5000
 * - Production (deployed)  → requests go directly to the hosted backend
 *
 * Set the env var VITE_API_URL on Render (or .env) to override the backend URL.
 * Falls back to the hardcoded Render URL if not set.
 */

const FALLBACK_BACKEND = 'https://bdh-explainer-backend.onrender.com';

/**
 * Vite exposes env vars prefixed with VITE_ at build time via import.meta.env.
 * In local dev mode the proxy handles everything, so we use an empty base.
 */
function resolveBase(): string {
  // Check if we want to force the hosted backend locally via a flag, or just use it if we are in prod.
  // But for now, let's keep it simple:
  // If we are in DEV mode, we usually want localhost:5000 via proxy.
  // If the user wants to use hosted backend locally, they should change this or set VITE_API_URL.
  
  // However, if the user is running LOCALLY but wants to hit the REMOTE backend,
  // we need to detect that.
  // Let's allow VITE_API_URL to override even in DEV mode if it's explicitly set.
  if (import.meta.env.VITE_API_URL) {
    return import.meta.env.VITE_API_URL as string;
  }

  // Otherwise, in DEV, use the proxy
  if (import.meta.env.DEV) return '';

  // In PROD, fallback to the hardcoded URL
  return FALLBACK_BACKEND;
}

export const API_BASE: string = resolveBase();

/** Build a full API URL for a given path (e.g. "/api/run"). */
export function apiUrl(path: string): string {
  return `${API_BASE}${path}`;
}
