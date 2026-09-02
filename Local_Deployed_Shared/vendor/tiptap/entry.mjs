// Vendored Tiptap entry — bundled into a single deduped ESM file so the web
// UI loads ONE copy of prosemirror-* (separate per-package bundles would each
// ship their own prosemirror-state and break the editor).
//
// Surface kept intentionally small: enough to build the brain-dump rich editor
// and, later, the unified nested-checkbox outliner (TaskList/TaskItem).
//
// NOTE: @tiptap/extensions Placeholder is deliberately NOT bundled. Its v3
// viewport-tracking plugin relies on layout APIs that can't be smoke-verified
// under jsdom; the empty-state placeholder is done in the app layer via
// editor.isEmpty + a CSS class instead. Keep this bundle jsdom-verifiable.
export { Editor, Node, Mark, Extension } from '@tiptap/core';
export { default as StarterKit } from '@tiptap/starter-kit';
export { TaskList, TaskItem } from '@tiptap/extension-list';
