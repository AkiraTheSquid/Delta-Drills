/* Setup orchestration only. No grading, rung selection, or answer replay.
   One queue across notebooks prevents late setup from changing a new context. */
window.ArenaNotebookSession = (() => {
  let queue = Promise.resolve();
  const fingerprint = (text) => {
    let hash = 5381;
    for (const c of text) hash = ((hash * 33) ^ c.charCodeAt(0)) >>> 0;
    return `${text.length}-${hash}`;
  };
  const create = ({ context, setup, run, onSetup, onFresh, isCurrent }) => {
    const checked = async (source, options = {}) => {
      if (!isCurrent()) throw new Error("Notebook changed; run cancelled.");
      const result = await run(source, { context, skipOnFresh: true, ...options });
      if (!result) throw new Error("Python unavailable. Retry setup when connected.");
      if (result.busy) throw new Error(result.text);
      return result;
    };
    const ensure = async () => {
      const cells = setup();
      const key = fingerprint(JSON.stringify(cells));
      const probe = await checked(`globals().get('__dd_arena_setup_key') == '${key}'`);
      if (probe.fresh) onFresh();
      if (!probe.fresh && !probe.failed && probe.text.trim() === "True") return;
      onSetup("running");
      for (const cell of cells) {
        const result = await checked(cell.source, { name: `<${cell.id}>`, timeout: 300 });
        if (result.fresh) {
          onFresh();
          throw new Error("Python restarted during setup. Retry setup.");
        }
        onSetup("cell", cell, result);
        if (result.failed) throw new Error(`Setup stopped at ${cell.id}. See its output above; edit the cell if needed, then retry.`);
      }
      const marker = await checked(`__dd_arena_setup_key = '${key}'`, { echo: false });
      if (marker.failed || marker.fresh) throw new Error("Python restarted before setup finished. Retry setup.");
      onSetup("ready");
    };
    const enqueue = (fn) => {
      const next = queue.catch(() => {}).then(fn);
      queue = next.catch(() => {});
      return next;
    };
    return {
      prepare: () => enqueue(ensure),
      execute: (source, options) => enqueue(async () => {
        await ensure();
        let result = await checked(source, options);
        if (result.fresh) {
          onFresh();
          await ensure();
          result = await checked(source, options);
          if (result.fresh) throw new Error("Python keeps restarting. Retry once connected.");
        }
        return result;
      }),
      restart: (reset) => enqueue(async () => {
        if (!isCurrent()) return;
        await reset();
        onFresh();
        await ensure();
      }),
    };
  };
  return { create };
})();
