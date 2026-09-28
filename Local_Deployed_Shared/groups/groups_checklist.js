/* ================================================================
   THE CHECKLIST EDITOR — your own column, and only your own.

   A Tiptap task list whose checkbox cycles open → checked → X → open,
   ported from Delta Note's sub-goal editor
   (`shared/web-components/js/subgoals/subgoal_editor.js` +
   `subgoal_taskitem.js`) so the two apps write the SAME `{v, doc}`
   document. The pure half of that port — the document, the counting,
   the read-only renderer — is `groups_checklist_doc.js`; this file is
   the lifecycle around it.

   ── 🔴 THE BUNDLE IS LOADED ON FIRST USE, NEVER AT BOOT ──────────
   `vendor/tiptap/tiptap.bundle.esm.js` is 400 KB. `import()` is a
   runtime expression and works perfectly well inside a classic script
   (this file is one — the whole app is), so nobody who never opens
   the Groups tab pays for it, and the twelve member rows of a group
   share ONE import: the promise is cached module-wide.

   🔴 THE SPECIFIER IS RELATIVE TO THIS FILE, NOT TO THE PAGE. Unlike
   `<script src>`, `fetch()` and every other URL this app writes —
   all of which resolve against the DOCUMENT — a dynamic `import()`
   resolves against the base URL of the script that ran it. This file
   is `groups/groups_checklist.js`, so `./vendor/...` asks for
   `/groups/vendor/...` and 404s. It has to climb out: `../vendor/`.
   Written the other way first, and the page said so — the column read
   "the checklist editor could not load" while every other part of the
   tab worked.

   ── 🔴 ONE EDITOR ON THE PAGE, AND IT IS YOURS ───────────────────
   Every other member's list is drawn by `DDChecklistDoc.renderDoc`.
   Mounting an editor per member would load node views and plugins
   twelve times for documents nobody may type into. `mount()` is only
   ever called for the row `is-you`.

   ── 🔴 A LOAD MUST NEVER SAVE ────────────────────────────────────
   Every `setContent` here passes `{ emitUpdate: false }` — an OPTIONS
   OBJECT. Tiptap 2 took `setContent(content, emitUpdate)`; Tiptap 3
   (the version vendored here) takes options, silently ignores a stray
   boolean, and defaults `emitUpdate` to TRUE. Delta Note shipped the
   positional form: every call site read as suppressed and emitted
   anyway, so opening a day wrote the row it had just read.
   ================================================================ */

const DDChecklist = (() => {
  const BUNDLE_URL = "../vendor/tiptap/tiptap.bundle.esm.js";
  const SAVE_DEBOUNCE_MS = 500;

  const docs = () => window.DDChecklistDoc;

  /* One import for the whole page. Held as the PROMISE, not the module,
     so twelve rows mounting in the same frame queue on one network read
     rather than starting twelve. */
  let bundle = null;
  const loadTiptap = () => {
    if (!bundle) {
      bundle = import(BUNDLE_URL).catch((err) => {
        /* Let the next mount try again: a failed import is usually a
           deploy that dropped the file, but it is sometimes a flaky
           network, and a permanently-poisoned promise would mean the
           column stays dead until a reload. */
        bundle = null;
        throw err;
      });
    }
    return bundle;
  };

  /* ---- the three-state checkbox -------------------------------------- */

  /**
   * Tiptap's TaskItem, extended with the third state.
   *
   * `checked` stays the boolean Tiptap itself knows (true only in the
   * 'checked' state), so TaskList's parse rules, the `[x] ` input rule and
   * any document written before this attr existed keep working. The X
   * lives in `completion`, serialized as `data-completion` on the <li> and
   * round-tripped in the stored JSON — it has to be IN the document to
   * survive a save.
   *
   * The node view mirrors the stock one's DOM (li > label(input+span) +
   * div contentDOM) because the CSS keys on that shape, with two changes:
   * it stamps `data-completion` on the li (the stock view never re-renders
   * custom attrs, so the X would go stale in the DOM), and a click cycles
   * three states in one transaction instead of toggling two.
   */
  const buildThreeStateTaskItem = (mod) =>
    mod.TaskItem.extend({
      addAttributes() {
        return {
          ...(this.parent ? this.parent() || {} : {}),
          completion: {
            default: "open",
            parseHTML: (el) =>
              el.getAttribute("data-completion") ||
              (el.getAttribute("data-checked") === "true" ? "checked" : "open"),
            renderHTML: (attrs) => ({ "data-completion": attrs.completion || "open" }),
          },
        };
      },
      addNodeView() {
        return ({ node, getPos, editor }) => {
          const li = document.createElement("li");
          li.dataset.type = "taskItem";
          const label = document.createElement("label");
          label.contentEditable = "false";
          const input = document.createElement("input");
          input.type = "checkbox";
          const span = document.createElement("span");
          const content = document.createElement("div");
          label.append(input, span);
          li.append(label, content);

          const paint = (n) => {
            const state = docs().effectiveCompletion(n.attrs);
            li.dataset.checked = state === "checked" ? "true" : "false";
            li.dataset.completion = state;
            /* Shared with the read-only renderer so both columns announce the
               third state the same way — `indeterminate` for the X, and the
               item's own text as the box's accessible name. */
            docs().paintCheckboxState(input, state, n.textContent);
          };
          paint(node);

          input.addEventListener("click", (event) => {
            /* Suppress the native binary toggle (and with it the stock
               change handler). Space fires a click too, so the keyboard
               cycles as well. */
            event.preventDefault();
            /* preventDefault reverts `input.checked` AFTER handlers finish,
               clobbering paint()'s assignment — re-assert it from the
               painted li once that has landed. */
            setTimeout(() => {
              const state = li.dataset.completion;
              input.checked = state === "checked";
              input.indeterminate = state === "x";
            }, 0);
            if (!editor.isEditable || typeof getPos !== "function") return;
            const pos = getPos();
            if (typeof pos !== "number") return;
            const cur = editor.state.doc.nodeAt(pos);
            if (!cur) return;
            const next = docs().nextCompletionState(docs().effectiveCompletion(cur.attrs));
            editor.view.dispatch(
              editor.state.tr.setNodeMarkup(pos, undefined, {
                ...cur.attrs,
                checked: next === "checked",
                completion: next,
              })
            );
          });

          return {
            dom: li,
            contentDOM: content,
            update: (updated) => {
              if (updated.type.name !== "taskItem") return false;
              paint(updated);
              return true;
            },
          };
        };
      },
    }).configure({ nested: true });

  /* ---- the editor ----------------------------------------------------- */

  const buildEditor = (mod, element, onUpdate) => {
    const { Editor, Extension, StarterKit, TaskList } = mod;

    /* Backspace at the START of a line removes the line and its marker
       (merging into the line above) instead of Tiptap's default, which
       lifts it out into a bare markerless paragraph — a line that is
       suddenly not a task, in a list that is only tasks. High priority so
       it beats StarterKit's list keymap; returns false when the caret is
       not at a line start, so ordinary delete is untouched. */
    const ListBackspace = Extension.create({
      name: "ddChecklistBackspace",
      priority: 1000,
      addKeyboardShortcuts() {
        return {
          Backspace: () => {
            const ed = this.editor;
            const { selection } = ed.state;
            if (!selection.empty) return false;
            const { $from } = selection;
            if ($from.parentOffset !== 0) return false;
            const parent = $from.node(-1);
            const name = parent && parent.type && parent.type.name;
            if (name !== "listItem" && name !== "taskItem") return false;
            if (ed.commands.joinBackward()) return true;
            /* The first item with nothing above it: swallow, so the default
               does not lift it into a markerless paragraph. */
            return true;
          },
        };
      },
    });

    return new Editor({
      element,
      extensions: [
        StarterKit.configure({ trailingNode: false }),
        TaskList,
        buildThreeStateTaskItem(mod),
        ListBackspace,
      ],
      content: docs().EMPTY_TASK_DOC,
      /* 🔴 NO PLACEHOLDER. A ProseMirror placeholder is drawn by the empty
         paragraph's own ::before, and inside a task item that paragraph sits
         BESIDE the checkbox — so the prompt text landed on top of the box
         instead of after it. The column is captioned already; an empty
         checklist needs no second label. */
      editorProps: {
        attributes: {
          "aria-label": "Your checklist for this day",
        },
      },
      onUpdate,
    });
  };

  /**
   * Mount the editor on `element`.
   *
   * @param {{element: HTMLElement, payload: string,
   *          onSave: (payload: string) => void,
   *          onCounts?: (counts: {checked: number, total: number}) => void}} deps
   * @returns {{ready: Promise, flush: Function, destroy: Function}}
   */
  const mount = ({ element, payload = "", onSave, onCounts = null }) => {
    const state = { editor: null, timer: null, destroyed: false };
    const noop = () => {};
    if (!element || !docs()) {
      return { ready: Promise.resolve(null), flush: noop, destroy: noop };
    }

    const save = () => {
      clearTimeout(state.timer);
      state.timer = null;
      if (!state.editor || !onSave) return;
      onSave(docs().payloadFromDoc(state.editor.getJSON()));
    };
    const scheduleSave = () => {
      clearTimeout(state.timer);
      state.timer = setTimeout(save, SAVE_DEBOUNCE_MS);
    };

    const ready = loadTiptap()
      .then((mod) => {
        /* Torn down while the bundle was in flight — a person clicking
           through days faster than a 400 KB import. Do not mount into an
           element that is no longer on the page. */
        if (state.destroyed) return null;
        const editor = buildEditor(mod, element, () => {
          if (onCounts) onCounts(docs().countTaskItems(editor.getJSON()));
          scheduleSave();
        });
        state.editor = editor;

        const doc = docs().docFromStored(payload);
        if (doc) {
          editor.commands.setContent(doc, { emitUpdate: false });
          /* Bullets typed with `- `, and anything written before the
             checkbox list existed, become checkboxes. Persist only when
             that actually changed something — `normalizeDoc` is idempotent
             precisely so this comparison is honest. */
          const normalized = docs().normalizeDoc(editor.getJSON());
          if (JSON.stringify(normalized) !== JSON.stringify(editor.getJSON())) {
            editor.commands.setContent(normalized, { emitUpdate: false });
            scheduleSave();
          }
        } else {
          editor.commands.setContent(docs().EMPTY_TASK_DOC, { emitUpdate: false });
        }
        if (onCounts) onCounts(docs().countTaskItems(editor.getJSON()));
        return editor;
      })
      .catch((err) => {
        console.error("[groups] the checklist editor could not load", err);
        /* Say so on the page. A column that is silently missing reads as
           "this person wrote nothing today", which is a different and
           wrong statement about somebody's day. */
        element.textContent = "The checklist editor could not load. Reload the page to try again.";
        element.classList.add("dd-checklist-broken");
        return null;
      });

    return {
      ready,
      flush: () => {
        if (state.timer) save();
      },
      destroy: () => {
        state.destroyed = true;
        /* 🔴 Flush BEFORE tearing down. The debounce is half a second and
           changing the day destroys this editor — without the flush, the
           last thing typed before a click on ▶ is the thing that is lost,
           which is exactly the edit a person remembers making. */
        if (state.timer) save();
        try {
          if (state.editor) state.editor.destroy();
        } catch (_) {
          /* already torn down */
        }
        state.editor = null;
      },
    };
  };

  return { mount };
})();

window.DDChecklist = DDChecklist;
