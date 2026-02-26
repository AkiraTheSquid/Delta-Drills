/* ================================================================
   SUPABASE-PRACTICE.JS — Supabase auth + practice state persistence
   ================================================================
   Provides:
   - Admin detection (sethbgibson@gmail.com)
   - Practice mode routing (backend / supabase / local)
   - Supabase Auth wrappers for non-admin users
   - Practice state load/save to Supabase practice_user_state table
   ================================================================ */

// --- Supabase client init ---

const SUPABASE_URL = "https://qaxtcaoydbpigomnfjpl.supabase.co";
const SUPABASE_ANON_KEY =
  "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFheHRjYW95ZGJwaWdvbW5manBsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzAxNzQ3MjQsImV4cCI6MjA4NTc1MDcyNH0.Mom-rTokqsvEbEshyvvfEjyL77AVa0LqJIg9FbpLvU4";

const ADMIN_EMAIL = "sethbgibson@gmail.com";

let supabaseClient = null;

function getSupabaseClient() {
  if (supabaseClient) return supabaseClient;
  if (typeof window.supabase === "undefined" || !window.supabase.createClient) {
    console.warn("Supabase JS SDK not loaded");
    return null;
  }
  try {
    supabaseClient = window.supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
  } catch (e) {
    console.error("Supabase init failed:", e);
  }
  return supabaseClient;
}

// --- Admin detection ---

function isAdminUser(email) {
  return (email || "").trim().toLowerCase() === ADMIN_EMAIL;
}

function isOnLocalhost() {
  const h = window.location.hostname;
  return h === "localhost" || h === "127.0.0.1" || h === "0.0.0.0";
}

function shouldUseLocalBackend(email) {
  return isAdminUser(email) && isOnLocalhost();
}

/**
 * Determine the practice mode:
 *   'backend'  — admin on localhost, use local FastAPI
 *   'supabase' — non-admin or deployed site, use Pyodide + Supabase
 *   'local'    — fallback, use Pyodide + localStorage
 */
function getPracticeMode(email) {
  if (shouldUseLocalBackend(email)) {
    return "backend";
  }
  const sb = getSupabaseClient();
  if (sb) {
    return "supabase";
  }
  return "local";
}

// --- Supabase Auth wrappers ---

async function supabaseSignIn(email, password) {
  const sb = getSupabaseClient();
  if (!sb) throw new Error("Supabase not available");
  const { data, error } = await sb.auth.signInWithPassword({ email, password });
  if (error) throw error;
  return data;
}

async function supabaseSignUp(email, password) {
  const sb = getSupabaseClient();
  if (!sb) throw new Error("Supabase not available");
  const { data, error } = await sb.auth.signUp({ email, password });
  if (error) throw error;
  return data;
}

async function supabaseSignOut() {
  const sb = getSupabaseClient();
  if (!sb) return;
  await sb.auth.signOut();
}

async function supabaseGetSession() {
  const sb = getSupabaseClient();
  if (!sb) return null;
  const { data } = await sb.auth.getSession();
  return data?.session || null;
}

// --- Session sync (deployed) ---

(async () => {
  try {
    const session = await supabaseGetSession();
    if (session?.access_token) {
      if (typeof setAuthState === "function") {
        setAuthState(session.access_token, session.user?.email || "");
      }
    } else if (localStorage.getItem("auth_token") === "supabase_session") {
      if (typeof setAuthState === "function") {
        setAuthState("", "");
      } else {
        localStorage.removeItem("auth_token");
        localStorage.removeItem("auth_email");
      }
    }
  } catch (e) {
    console.warn("Supabase session sync failed:", e);
  }
})();

// --- Practice state persistence (Supabase) ---

async function loadPracticeStateFromSupabase(email) {
  const sb = getSupabaseClient();
  if (!sb) return null;
  const { data, error } = await sb
    .from("practice_user_state")
    .select("state")
    .eq("user_email", email)
    .maybeSingle();
  if (error) {
    console.error("Failed to load practice state from Supabase:", error);
    return null;
  }
  return data?.state || null;
}

async function savePracticeStateToSupabase(email, state) {
  const sb = getSupabaseClient();
  if (!sb) return false;
  const { error } = await sb
    .from("practice_user_state")
    .upsert(
      { user_email: email, state: state, updated_at: new Date().toISOString() },
      { onConflict: "user_email" }
    );
  if (error) {
    console.error("Failed to save practice state to Supabase:", error);
    return false;
  }
  return true;
}
