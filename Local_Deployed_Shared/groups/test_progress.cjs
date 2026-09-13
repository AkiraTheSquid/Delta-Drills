/* Browser regression harness. Run with Playwright available on NODE_PATH.
 * Uses real group scripts and editor, synthetic API responses, no accounts.
 */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const http = require("node:http");
const { chromium } = require("playwright");
const root = path.resolve(__dirname, "..");
const scripts = ["practice/activity-bars.js", "practice/placement-results.js", "groups/groups_store.js",
  "groups/groups_checklist_doc.js", "groups/groups_checklist.js", "groups/groups_day.js",
  "groups/groups_join.js", "groups/groups_progress.js", "groups/groups_lane.js", "groups/groups_view.js"];
const styles = ["styles/variables.css", "styles/base.css", "styles/components.css", "styles/practice/diagnostic.css", "styles/groups.css"];
const harness = `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">${styles.map(p => `<link rel="stylesheet" href="/${p}">`).join("")}</head><body><main class="groups-page"><h1>Study group</h1><div id="groups-root"></div></main><script>window.DDIdentity={isSignedIn:()=>true};window.apiFetch=(url,opts)=>fetch(url,opts);</script>${scripts.map(p => `<script src="/${p}"></script>`).join("")}<script>DDGroups.refresh()</script></body></html>`;

(async () => {
  const server = http.createServer((req, res) => {
    const pathname = new URL(req.url, "http://localhost").pathname;
    if (pathname === "/") { res.setHeader("Content-Type", "text/html"); res.end(harness); return; }
    const file = path.resolve(root, "." + pathname);
    if (!file.startsWith(root + path.sep) || !fs.existsSync(file)) { res.writeHead(404); res.end(); return; }
    res.setHeader("Content-Type", file.endsWith(".js") ? "text/javascript" : file.endsWith(".css") ? "text/css" : "application/octet-stream");
    fs.createReadStream(file).pipe(res);
  });
  await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 1100 }, timezoneId: "America/Chicago" });
    const errors = [];
    page.on("pageerror", error => errors.push(error.message));
    const savedDays = {}, targets = {}, requests = [];
    let failProgress = false, failSave = false, slowDay = "";
    await page.route("**/api/practice/groups/**", async route => {
      const req = route.request(), url = new URL(req.url());
      requests.push(url.pathname + url.search);
      let payload;
      if (url.pathname.endsWith("/mine")) payload = { group: { id: "group-a", name: "ARENA study group", member_id: "me", visibility: "private", members: [
        { member_id: "me", display_name: "Alex", initials: "AL", areas: [{ topic: "PyTorch", theta: 0, sd: .2, probes: 5 }], probes: 5 },
        { member_id: "peer", display_name: "Jordan", initials: "JO", areas: [] },
      ] } };
      if (url.pathname.endsWith("/day")) {
        if (req.method() === "PUT") { const body = req.postDataJSON(); savedDays[body.date] = body.payload; payload = body; }
        else payload = { entries: { me: savedDays[url.searchParams.get("date")] || "", peer: "" } };
      }
      if (url.pathname.endsWith("/target")) {
        if (failSave) { await route.fulfill({ status: 500, json: { detail: "Test save failure" } }); return; }
        const body = req.postDataJSON();
        targets[body.area] = { date: body.date, level: body.level };
        payload = body;
      }
      if (url.pathname.endsWith("/progress")) {
        if (failProgress) { await route.fulfill({ status: 503, json: { detail: "Test read failure" } }); return; }
        const key = url.searchParams.get("date"), horizon = url.searchParams.get("horizon");
        const anchor = new Date(key + "T12:00:00Z");
        let start = new Date(anchor), end = new Date(anchor);
        if (horizon === "weekly") { start.setUTCDate(start.getUTCDate() - ((start.getUTCDay() + 6) % 7)); end = new Date(start); end.setUTCDate(end.getUTCDate() + 6); }
        if (horizon === "monthly") { start.setUTCDate(1); end = new Date(start); end.setUTCMonth(end.getUTCMonth() + 1); end.setUTCDate(0); }
        const days = [];
        for (let d = new Date(start); d <= end; d.setUTCDate(d.getUTCDate() + 1)) days.push(d.toISOString().slice(0, 10));
        const today = new Date().toISOString().slice(0, 10);
        const entry = peer => ({ activity: { today, days: days.map((date, i) => ({ date, count: i % 6, practice: i % 6, placement: 0 })), total: days.reduce((n, _, i) => n + i % 6, 0) },
          history: days.map((date, i) => ({ date, scores: Object.fromEntries(["aggregate", "0.0", "0.1"].map((key, j) => [key, date > today ? null : { score: 40 + i + j * 10 + (peer ? 5 : 0), coverage: 70, proxies: 2, concepts: 8 }])) })),
          targets: peer ? {} : structuredClone(targets) });
        payload = { start: days[0], end: days.at(-1), horizon, today, benchmark: 80,
          areas: [{ id: "0.0", label: "0.0 · Prerequisites", color: "#4f9fe0", concepts: 8 }, { id: "0.1", label: "0.1 · Ray tracing", color: "#bb7de8", concepts: 8 }],
          entries: { me: entry(false), peer: entry(true) } };
        if (key === slowDay) await new Promise(resolve => setTimeout(resolve, 200));
      }
      await route.fulfill({ json: payload || {} });
    });
    await page.goto(`http://127.0.0.1:${server.address().port}`);
    await page.locator(".dd-checklist--mine .ProseMirror").waitFor();
    assert.equal(await page.locator('[contenteditable="true"]').count(), 1);
    const originalDay = await page.locator(".dd-day-input").inputValue();
    await page.locator(".dd-checklist--mine .ProseMirror").pressSequentially("Read tensor chapter");
    await page.getByLabel("View", { exact: true }).selectOption("graph");
    await page.locator(".dd-competency-chart").first().waitFor();
    assert.equal(await page.locator('[contenteditable="true"]').count(), 0);
    assert.ok(savedDays[originalDay].includes("Read tensor chapter"), "switching views flushes the checklist");
    assert.equal(await page.locator(".dd-competency-chart").count(), 2);
    assert.equal(await page.locator(".dd-target-form").count(), 1, "only your target is editable");
    const own = page.locator(".dd-member.is-you");
    await own.getByRole("button", { name: /0\.1 · Ray tracing/ }).click();
    const chart = own.locator("svg");
    const box = await chart.boundingBox();
    await page.mouse.move(box.x + box.width * .6, box.y + box.height * .3);
    assert.equal(await own.locator(".dd-graph-crosshair line").count(), 2);
    await page.mouse.click(box.x + box.width * .6, box.y + box.height * .3);
    await own.getByLabel("Mastery / 100", { exact: true }).fill("82");
    await own.getByRole("button", { name: "Save target" }).click();
    await own.locator(".dd-target-saved").waitFor();
    assert.equal(targets["0.1"].level, 82);
    assert.equal(await own.locator(".dd-graph-target line").count(), 2);
    await own.getByRole("button", { name: /All areas.*compare/ }).click();
    assert.equal(await own.locator(".dd-target-form").count(), 0);
    assert.equal(await own.locator("svg path").count(), 2);
    assert.equal(await own.locator("svg[tabindex]").count(), 0);
    assert.equal(await own.locator(".dd-graph-target").count(), 0);
    await own.getByRole("button", { name: /All scores · aggregate/ }).click();
    await own.locator("svg").focus();
    await page.keyboard.press("ArrowUp");
    await page.keyboard.press("Enter");
    assert.equal(await own.getByLabel("Mastery / 100", { exact: true }).inputValue(), "81");
    failSave = true;
    await own.getByRole("button", { name: "Save target" }).click();
    await own.getByText("Not saved: Test save failure").waitFor();
    failSave = false;
    await page.getByLabel("Time horizon", { exact: true }).selectOption("monthly");
    await page.locator(".dd-day-input").fill("2024-01-31");
    await page.locator(".dd-day-input").dispatchEvent("change");
    await page.getByRole("button", { name: "Next month", exact: true }).click();
    await page.waitForFunction(() => document.querySelector(".dd-day-input").value === "2024-02-01" && document.querySelector(".dd-target-form"));
    await page.getByLabel("View", { exact: true }).selectOption("activity");
    await page.waitForFunction(() => document.querySelectorAll(".dd-member.is-you .activity-day").length === 29);
    await page.getByLabel("Time horizon", { exact: true }).selectOption("daily");
    await page.waitForFunction(() => document.querySelectorAll(".dd-member.is-you .activity-day").length === 1);
    await page.getByLabel("Time horizon", { exact: true }).selectOption("weekly");
    await page.waitForFunction(() => document.querySelectorAll(".dd-member.is-you .activity-day").length === 7);
    slowDay = "2024-02-08";
    await page.getByRole("button", { name: "Next week", exact: true }).click();
    await page.getByRole("button", { name: "Next week", exact: true }).click();
    await page.waitForFunction(() => document.querySelector(".dd-member.is-you .activity-day")?.title.startsWith("2024-02-12"));
    await page.waitForTimeout(250);
    assert.ok((await own.locator(".activity-day").first().getAttribute("title")).startsWith("2024-02-12"));
    failProgress = true;
    await page.getByRole("button", { name: "Next week", exact: true }).click();
    await page.getByRole("button", { name: "Retry", exact: true }).waitFor();
    assert.equal(await page.getByText("Progress could not be read. Retry above.").count(), 2);
    failProgress = false;
    await page.getByRole("button", { name: "Retry", exact: true }).click();
    await own.locator(".activity-day").first().waitFor();
    await page.getByLabel("View", { exact: true }).selectOption("graph");
    await own.locator("svg").waitFor();
    await page.screenshot({ path: process.env.GROUP_SCREENSHOT || "/tmp/dd-group-progress-desktop.png", fullPage: true });
    await page.setViewportSize({ width: 390, height: 844 });
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), "mobile does not overflow the page");
    await page.screenshot({ path: "/tmp/dd-group-progress-mobile.png", fullPage: true });
    await page.getByLabel("View", { exact: true }).selectOption("goals");
    await page.locator(".dd-day-input").fill(originalDay);
    await page.locator(".dd-day-input").dispatchEvent("change");
    await page.locator(".dd-checklist--mine .ProseMirror").getByText("Read tensor chapter").waitFor();
    assert.equal(await page.locator('[contenteditable="true"]').count(), 1);
    assert.deepEqual(errors, []);
    console.log("PASS: checklist flush, graph modes, hover/keyboard targets, save failure, leap month, all horizons, stale reads, retry, persistence, mobile layout.");
  } finally {
    await browser.close();
    await new Promise(resolve => server.close(resolve));
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
