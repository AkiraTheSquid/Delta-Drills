/* ================================================================
   APP.JS — Core app logic: auth, tab switching, API helper, account
   ================================================================ */

const tabs = document.querySelectorAll(".tab");
const authOnlyTabs = document.querySelectorAll(".auth-only");
const guestOnlyTabs = document.querySelectorAll(".guest-only");
const pages = document.querySelectorAll(".page");
const loginForm = document.getElementById("login-form");
const loginMessage = document.getElementById("login-message");
const signupForm = document.getElementById("signup-form");
const signupMessage = document.getElementById("signup-message");
const authStatus = document.getElementById("auth-status");
const logoutButton = document.getElementById("logout-button");
const accountForm = document.getElementById("account-form");
const accountMessage = document.getElementById("account-message");
const accountLogout = document.getElementById("account-logout");

const isLocalHost = ["localhost", "127.0.0.1", "0.0.0.0"].includes(window.location.hostname);
const defaultApiBase = isLocalHost
  ? "http://localhost:8000"
  : "https://delta-drills-backend.fly.dev";
let API_BASE = localStorage.getItem("api_base") || defaultApiBase;
let authToken = localStorage.getItem("auth_token") || "";
let authEmail = localStorage.getItem("auth_email") || "";

const authRequiredTabs = ["split-tool", "account", "learn", "course", "papers", "practice", "statistics"];

const switchTab = (tabName) => {
  if (authRequiredTabs.includes(tabName) && !authToken) {
    loginMessage.textContent = "Please log in to start a job.";
    tabName = "login";
  }
  tabs.forEach((t) => t.classList.toggle("active", t.dataset.tab === tabName));
  pages.forEach((p) => p.classList.toggle("hidden", p.id !== `page-${tabName}`));
};

tabs.forEach((t) => {
  t.addEventListener("click", () => switchTab(t.dataset.tab));
});

const updateTabVisibility = () => {
  authOnlyTabs.forEach((t) => t.classList.toggle("hidden", !authToken));
  guestOnlyTabs.forEach((t) => t.classList.toggle("hidden", !!authToken));
};

const setAuthState = (token, email) => {
  authToken = token || "";
  authEmail = email || "";
  if (authToken) {
    localStorage.setItem("auth_token", authToken);
    localStorage.setItem("auth_email", authEmail);
    authStatus.textContent = authEmail ? `Logged in as ${authEmail}` : "Logged in";
    switchTab("split-tool");
  } else {
    localStorage.removeItem("auth_token");
    localStorage.removeItem("auth_email");
    authStatus.textContent = "";
    switchTab("login");
  }
  updateTabVisibility();
};

logoutButton.addEventListener("click", () => {
  if (typeof supabaseSignOut === "function") supabaseSignOut();
  setAuthState("", "");
});
accountLogout.addEventListener("click", () => {
  if (typeof supabaseSignOut === "function") supabaseSignOut();
  setAuthState("", "");
});

const apiFetch = async (path, options = {}) => {
  const headers = options.headers ? { ...options.headers } : {};
  if (authToken) {
    headers.Authorization = `Bearer ${authToken}`;
  }
  const response = await fetch(`${API_BASE}${path}`, { ...options, headers });
  return response;
};

// --- Auth forms ---

loginForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  loginMessage.textContent = "Working...";
  const email = document.getElementById("email").value.trim();
  const password = document.getElementById("password").value;

  if (!email || !password) {
    loginMessage.textContent = "Enter an email and password.";
    return;
  }

  try {
    if (typeof shouldUseLocalBackend === "function" && shouldUseLocalBackend(email)) {
      // Admin on localhost — use local backend JWT auth
      const response = await fetch(`${API_BASE}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });
      const data = await response.json();
      if (!response.ok) {
        loginMessage.textContent = data.detail || "Login failed.";
        return;
      }
      loginMessage.textContent = "Logged in!";
      loginForm.reset();
      setAuthState(data.access_token, email);
    } else if (typeof supabaseSignIn === "function") {
      // Non-admin or deployed — use Supabase Auth
      const data = await supabaseSignIn(email, password);
      loginMessage.textContent = "Logged in!";
      loginForm.reset();
      setAuthState(data.session?.access_token || "", email);
    } else {
      // Fallback to local backend
      const response = await fetch(`${API_BASE}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });
      const data = await response.json();
      if (!response.ok) {
        loginMessage.textContent = data.detail || "Login failed.";
        return;
      }
      loginMessage.textContent = "Logged in!";
      loginForm.reset();
      setAuthState(data.access_token, email);
    }
    setTimeout(() => (loginMessage.textContent = ""), 1000);
  } catch (e) {
    loginMessage.textContent = e.message;
  }
});

signupForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  signupMessage.textContent = "Working...";
  const email = document.getElementById("signup-email").value.trim();
  const password = document.getElementById("signup-password").value;
  const confirm = document.getElementById("signup-confirm").value;

  if (!email || !password || !confirm) {
    signupMessage.textContent = "Enter an email and both password fields.";
    return;
  }

  if (password !== confirm) {
    signupMessage.textContent = "Passwords do not match.";
    return;
  }

  try {
    if (typeof shouldUseLocalBackend === "function" && shouldUseLocalBackend(email)) {
      // Admin on localhost — use local backend
      const response = await fetch(`${API_BASE}/auth/signup`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });
      const data = await response.json();
      if (!response.ok) {
        signupMessage.textContent = data.detail || "Signup failed.";
        return;
      }
      signupMessage.textContent = "Account created!";
      signupForm.reset();
      setAuthState(data.access_token, email);
    } else if (typeof supabaseSignUp === "function") {
      // Non-admin or deployed — use Supabase Auth
      const data = await supabaseSignUp(email, password);
      if (data.user && !data.session) {
        signupMessage.textContent = "Check your email to confirm your account.";
        signupForm.reset();
        return;
      }
      signupMessage.textContent = "Account created!";
      signupForm.reset();
      setAuthState(data.session?.access_token || "", email);
    } else {
      // Fallback to local backend
      const response = await fetch(`${API_BASE}/auth/signup`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });
      const data = await response.json();
      if (!response.ok) {
        signupMessage.textContent = data.detail || "Signup failed.";
        return;
      }
      signupMessage.textContent = "Account created!";
      signupForm.reset();
      setAuthState(data.access_token, email);
    }
    setTimeout(() => (signupMessage.textContent = ""), 1000);
  } catch (e) {
    signupMessage.textContent = e.message;
  }
});

// --- Account settings ---

accountForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const apiBase = document.getElementById("account-api-base").value.trim();
  const openaiKey = document.getElementById("account-openai-key").value.trim();
  const mathpixId = document.getElementById("account-mathpix-id").value.trim();
  const mathpixKey = document.getElementById("account-mathpix-key").value.trim();

  if (apiBase) {
    localStorage.setItem("api_base", apiBase);
    API_BASE = apiBase;
  } else {
    localStorage.removeItem("api_base");
    API_BASE = defaultApiBase;
  }
  localStorage.setItem("account_openai_key", openaiKey);
  localStorage.setItem("account_mathpix_id", mathpixId);
  localStorage.setItem("account_mathpix_key", mathpixKey);

  if (typeof saveUserSettingsToSupabase === "function" && authEmail) {
    const saved = await saveUserSettingsToSupabase(authEmail, openaiKey);
    accountMessage.textContent = saved ? "Saved to your account." : "Saved locally in this browser.";
  } else {
    accountMessage.textContent = "Saved locally in this browser.";
  }
  setTimeout(() => (accountMessage.textContent = ""), 2000);
});

// Load saved account settings into form
const savedApiBase = localStorage.getItem("api_base") || "";
const savedOpenai = localStorage.getItem("account_openai_key") || "";
const savedMathpixId = localStorage.getItem("account_mathpix_id") || "";
const savedMathpixKey = localStorage.getItem("account_mathpix_key") || "";
document.getElementById("account-api-base").value = savedApiBase;
document.getElementById("account-openai-key").value = savedOpenai;
document.getElementById("account-mathpix-id").value = savedMathpixId;
document.getElementById("account-mathpix-key").value = savedMathpixKey;

// --- Initial state ---

if (authToken) {
  authStatus.textContent = authEmail ? `Logged in as ${authEmail}` : "Logged in";
  switchTab("split-tool");
} else {
  switchTab("how-it-works");
}
updateTabVisibility();
