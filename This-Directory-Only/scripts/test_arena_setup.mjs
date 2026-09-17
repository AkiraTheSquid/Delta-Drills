import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

const sandbox = { window: {} };
vm.runInNewContext(fs.readFileSync(new URL("../../Local_Deployed_Shared/practice/arena-notebook-session.js", import.meta.url), "utf8"), sandbox);
const { create } = sandbox.window.ArenaNotebookSession;

function fixture() {
  let marker = "", fresh = true, current = true, failSetup = false, expireAt = "", setupSource = "import torch";
  const calls = [], phases = [];
  const run = async (source, options) => {
    calls.push(source);
    assert.equal(options.skipOnFresh, true);
    if (source === expireAt) { fresh = true; expireAt = ""; }
    if (fresh) { fresh = false; marker = ""; return { fresh: true, failed: false, text: "" }; }
    if (source.startsWith("globals()")) return { text: String(marker === source.match(/'([^']+)'$/)[1]) === "true" ? "True" : "False" };
    if (source.startsWith("__dd_arena_setup_key =")) marker = source.match(/'([^']+)'/)[1];
    if (source === "import torch" && failSetup) return { failed: true, text: "No module named torch" };
    return { text: "ok", failed: false };
  };
  const session = create({ context: "arena:test", setup: () => [{ id: "setup", source: setupSource }],
    run, onFresh: () => phases.push("fresh"), onSetup: (phase) => phases.push(phase), isCurrent: () => current });
  return { session, calls, phases, expire: () => { fresh = true; }, leave: () => { current = false; },
    expireAt: (source) => { expireAt = source; }, editSetup: (source) => { setupSource = source; },
    fail: (value) => { failSetup = value; } };
}

const f = fixture();
await f.session.prepare();
assert.equal(f.calls.filter((s) => s === "import torch").length, 1);
await Promise.all([f.session.execute("answer1"), f.session.execute("answer2")]);
assert.equal(f.calls.filter((s) => s === "import torch").length, 1, "ready setup must not repeat on every answer");
assert.ok(f.calls.indexOf("answer1") < f.calls.indexOf("answer2"));
f.expire();
await f.session.execute("answer3");
assert.equal(f.calls.filter((s) => s === "import torch").length, 2);
assert.ok(f.calls.lastIndexOf("import torch") < f.calls.indexOf("answer3"));
assert.equal(f.calls.filter((s) => s === "answer1").length, 1, "never replay previous experiments");
f.expireAt("answer4");
await f.session.execute("answer4");
assert.equal(f.calls.filter((s) => s === "answer4").length, 2, "skipped expired request is retried once after setup");
assert.equal(f.calls.filter((s) => s === "import torch").length, 3);
f.editSetup("import math");
await f.session.execute("answer5");
assert.ok(f.calls.lastIndexOf("import math") < f.calls.indexOf("answer5"));
f.leave();
await assert.rejects(f.session.execute("wrong notebook"), /cancelled/);
assert.ok(!f.calls.includes("wrong notebook"));

const bad = fixture();
bad.fail(true);
await assert.rejects(bad.session.execute("must not run"), /Setup stopped/);
assert.ok(!bad.calls.includes("must not run"));
assert.ok(!bad.phases.includes("ready"));
bad.fail(false);
await bad.session.execute("retry answer");
assert.ok(bad.calls.includes("retry answer"), "failure must not poison the queue");
await bad.session.restart(async () => bad.expire());
assert.equal(bad.calls.filter((s) => s === "import torch").length, 3);
console.log("PASS: automatic setup, serialization, expiry, failure/retry, navigation cancellation, restart");
