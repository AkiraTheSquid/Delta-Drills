// Dump the drills catalog to JSON so the python ERE exporter can map
// atomId -> notebook paths without re-parsing the E()/C() builder DSL.
// Output: ere/catalog_dump.json — [{id, atomId, compositeAtomIds, notebookPath,
// subtopics, heading, exerciseIndex, isComposite}]
const fs = require("fs");
const path = require("path");

const REPO = path.resolve(__dirname, "..", "..", "..");
const CAT = path.join(REPO, "Local_Deployed_Shared", "practice", "drills-catalog.js");
const OUT = path.join(__dirname, "catalog_dump.json");

global.window = {};
eval(fs.readFileSync(CAT, "utf8"));
const cat =
  global.window.DRILLS_CATALOG ||
  global.window.drillsCatalog ||
  global.window.__drillsCatalog;
if (!cat) {
  console.error("catalog not found on window; keys:", Object.keys(global.window));
  process.exit(1);
}
const rows = cat.map((e) => ({
  id: e.id,
  atomId: e.atomId || null,
  compositeAtomIds: e.compositeAtomIds || null,
  notebookPath: e.notebookPath || null,
  subtopics: e.subtopics || [],
  heading: e.heading || e.title || "",
  exerciseIndex: e.exerciseIndex || null,
  isComposite: !!e.isComposite,
}));
fs.writeFileSync(OUT, JSON.stringify(rows, null, 1));
console.log(`catalog dump: ${rows.length} entries -> ${path.relative(REPO, OUT)}`);
