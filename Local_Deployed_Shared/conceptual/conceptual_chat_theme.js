/* ================================================================
   CONCEPTUAL CHAT — THEME. How <deep-chat> looks, as Deep Chat's own
   style objects plus one CSS string it injects into its shadow root.

   🔴 EVERY COLOUR IS A var(--token) FROM styles/variables.css. Deep Chat
   renders inside a shadow root, so our stylesheets cannot reach its
   elements — but CUSTOM PROPERTIES INHERIT through the shadow boundary, so a
   `var(--text)` written here resolves against <html data-theme> exactly as
   it does in feature CSS, and switching theme repaints the chat with no
   code. A literal colour here is a hole in two of the three themes.

   Maths needs NO KaTeX stylesheet in here: Deep Chat calls
   `window.katex.renderToString(..., {output: "mathml"})`, which the browser
   draws natively. (And `auxiliaryStyle` is a constructable stylesheet, where
   `@import` is refused with a console warning — don't try.)

   Look: ChatGPT/Claude — one centred 760px column, the learner's turn a
   soft right-aligned pill, the tutor's turn plain prose with no bubble,
   and a rounded composer floating at the bottom.
   ================================================================ */
(() => {
  const COLUMN = "760px";
  const tint = (a) => `rgb(var(--tint-rgb) / calc(${a} * var(--tint-k)))`;
  const well = (a) => `rgb(var(--well-rgb) / calc(${a} * var(--well-k)))`;

  /* 🔴 xmlns IS REQUIRED. Deep Chat parses `svg.content` with DOMParser as
     image/svg+xml; without the namespace the root is a plain Element with no
     `.style`, the icon never draws, and every hover throws in
     setElementsCSS ("Cannot convert undefined or null to object"). */
  const NS = 'xmlns="http://www.w3.org/2000/svg"';
  const SEND_SVG =
    `<svg ${NS} viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" ` +
    `stroke-linecap="round" stroke-linejoin="round"><path d="M12 19V5"/><path d="M5 12l7-7 7 7"/></svg>`;
  const STOP_SVG = `<svg ${NS} viewBox="0 0 24 24" fill="currentColor"><rect x="7" y="7" width="10" height="10" rx="2"/></svg>`;

  const roundButton = (bg, fg) => ({
    container: {
      default: {
        backgroundColor: bg, color: fg, borderRadius: "50%", width: "34px", height: "34px",
        padding: "0", display: "flex", alignItems: "center", justifyContent: "center",
        margin: "0", bottom: "7px", right: "8px", transition: "opacity .15s",
      },
      hover: { backgroundColor: bg, opacity: "0.85" },
      click: { backgroundColor: bg, opacity: "0.7" },
    },
  });
  const icon = (svg) => ({
    content: svg,
    styles: { default: { width: "18px", height: "18px", filter: "none" } },
  });

  const AUX = `
    :host { font-family: var(--ui-font); }
    #messages { padding: 24px 0 8px; }
    .outer-message-container { max-width: ${COLUMN}; margin: 0 auto; }
    .message-bubble { font-size: 15px; line-height: 1.7; }
    .ai-message-text p, .user-message-text p { margin: 0 0 .75em; }
    .ai-message-text p:last-child, .user-message-text p:last-child { margin-bottom: 0; }
    .ai-message-text h1, .ai-message-text h2, .ai-message-text h3 {
      color: var(--white); font-size: 1.05em; margin: 1.1em 0 .4em; line-height: 1.35;
    }
    .ai-message-text ul, .ai-message-text ol { padding-left: 1.4em; margin: 0 0 .75em; }
    .ai-message-text li { margin: .2em 0; }
    .ai-message-text a { color: var(--prose-link); }
    .ai-message-text strong { color: var(--white); }
    .ai-message-text blockquote {
      margin: .6em 0; padding: .1em 0 .1em 1em; border-left: 3px solid var(--border-strong); color: var(--muted);
    }
    .ai-message-text code, .user-message-text code {
      font-family: var(--code-font); font-size: .88em; padding: .12em .36em; border-radius: 5px;
      background: ${well(0.35)};
    }
    /* colour explicit: Deep Chat's default <pre> ink is white, which is
       invisible on the light theme's well. */
    .ai-message-text pre {
      color: var(--text); background: ${well(0.4)}; border: 1px solid var(--border); border-radius: 10px;
      padding: 12px 14px; overflow-x: auto; margin: .6em 0 .9em;
    }
    .ai-message-text pre code { background: none; padding: 0; font-size: 13px; line-height: 1.55; }
    .ai-message-text table { border-collapse: collapse; margin: .6em 0; font-size: .92em; }
    .ai-message-text th, .ai-message-text td { border: 1px solid var(--border); padding: 5px 10px; }
    .ai-message-text th { background: ${tint(0.05)}; }
    .ai-message-text .katex-display { overflow-x: auto; overflow-y: hidden; padding: 2px 0; }
    #text-input::-webkit-scrollbar, #messages::-webkit-scrollbar { width: 8px; }
    #messages::-webkit-scrollbar-thumb { background: var(--scroll-thumb); border-radius: 8px; }
    .cc-intro { text-align: center; padding: 12vh 16px 8px; }
    /* The welcome + suggestions are the EMPTY state; once the learner has
       spoken they are clutter above the thread (ChatGPT/Claude drop them). */
    #messages:has(.user-message-text) .cc-intro { display: none; }
    .cc-intro-title { font-size: 26px; font-weight: 650; color: var(--white); margin: 0 0 8px; letter-spacing: -.01em; }
    .cc-intro-sub { color: var(--muted); font-size: 14px; margin: 0 0 22px; }
    .cc-suggestions {
      display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px;
      max-width: 680px; margin: 0 auto; text-align: left;
    }
    @media (max-width: 560px) { .cc-suggestions { grid-template-columns: 1fr; } }
    .cc-suggestions .deep-chat-suggestion-button {
      font: inherit; font-size: 13.5px; line-height: 1.45; color: var(--text); background: ${tint(0.03)};
      border: 1px solid var(--border); border-radius: 14px; padding: 12px 14px; cursor: pointer;
      text-align: left; transition: background .15s, border-color .15s;
    }
    .cc-suggestions .deep-chat-suggestion-button:hover { background: ${tint(0.09)}; border-color: var(--border-strong); }
  `;

  const apply = (el) => {
    el.chatStyle = {
      width: "100%", height: "100%", border: "none", borderRadius: "0",
      backgroundColor: "var(--bg)", color: "var(--text)", fontFamily: "var(--ui-font)",
    };
    el.auxiliaryStyle = AUX;
    el.messageStyles = {
      default: {
        shared: {
          outerContainer: { padding: "0 16px", boxSizing: "border-box" },
          innerContainer: { width: "100%", maxWidth: COLUMN },
          bubble: {
            maxWidth: "100%", backgroundColor: "transparent", color: "var(--text)",
            padding: "0", marginTop: "10px", marginBottom: "10px", borderRadius: "0",
          },
        },
        user: {
          bubble: {
            maxWidth: "78%", backgroundColor: tint(0.08), color: "var(--text)",
            padding: "10px 16px", borderRadius: "20px", marginLeft: "auto",
          },
        },
        ai: { bubble: { maxWidth: "100%", padding: "2px 0" } },
      },
      html: { shared: { bubble: { maxWidth: "100%", backgroundColor: "transparent", padding: "0" } } },
      intro: { bubble: { maxWidth: "100%", backgroundColor: "transparent", padding: "0", width: "100%" } },
      error: {
        bubble: {
          backgroundColor: "rgb(var(--danger-rgb) / 0.12)", color: "var(--text)",
          border: "1px solid rgb(var(--danger-rgb) / 0.35)", borderRadius: "12px", padding: "10px 14px",
        },
      },
      loading: { message: { styles: { bubble: { backgroundColor: "transparent", padding: "8px 0" } } } },
    };
    el.inputAreaStyle = {
      // border-box: #input is `width: 100%`, so content-box + this padding
      // overflows the host by 32px and clips the send button on a phone.
      backgroundColor: "var(--bg)", padding: "8px 16px 18px", borderTop: "none", boxSizing: "border-box",
      display: "flex", justifyContent: "center",
    };
    el.textInput = {
      placeholder: { text: "Ask about any concept…", style: { color: "var(--muted)" } },
      styles: {
        container: {
          width: `min(${COLUMN}, 100%)`, maxWidth: COLUMN, boxSizing: "border-box",
          backgroundColor: "var(--input-bg)", border: "1px solid var(--border-strong)",
          borderRadius: "24px", boxShadow: "var(--shadow-soft)", margin: "0",
        },
        text: {
          color: "var(--text)", fontSize: "15px", lineHeight: "1.5",
          padding: "13px 52px 13px 18px", maxHeight: "40vh",
        },
        focus: { border: "1px solid var(--muted-dim)" },
      },
    };
    el.submitButtonStyles = {
      position: "inside-end",
      submit: { ...roundButton("var(--accent)", "var(--on-accent)"), svg: icon(SEND_SVG) },
      loading: { ...roundButton(tint(0.12), "var(--muted)"), svg: icon(STOP_SVG) },
      stop: { ...roundButton("var(--text)", "var(--bg)"), svg: icon(STOP_SVG) },
      disabled: {
        container: { default: { ...roundButton(tint(0.12), "var(--muted)").container.default, cursor: "default" } },
        svg: icon(SEND_SVG),
      },
    };
  };

  window.DDConceptualChatTheme = { apply };
})();
