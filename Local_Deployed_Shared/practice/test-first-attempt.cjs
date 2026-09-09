/* Run: node --test Local_Deployed_Shared/practice/test-first-attempt.cjs */
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { chromium } = require('playwright');
const source = name => fs.readFileSync(path.join(__dirname, name), 'utf8');
const bank = JSON.parse(source('../questions.json'));
const q391 = bank.find(q => q.id === 391);

function planner(attemptFirst, quota = 8) {
  const ctx = vm.createContext({ window: {}, localStorage: { getItem: () => null, setItem() {} } });
  vm.runInContext(source('exercise-planner.js'), ctx);
  const Planner = ctx.window.ExercisePlanner.Planner;
  const item = (kind, id, kc = 'target') => ({ kind, questionId: id, kc });
  const cfg = {
    attemptFirst, quota,
    target: { kc: 'target', m: .2, pools: {
      faded: [item('faded', 1)], guided: [item('guided', 2)], independent: [item('independent', 3)],
    }, variants: [item('integrated', 10), item('integrated', 11)] },
    prereqs: [{ kc: 'pre', m: .1, pools: { faded: [item('faded', 4, 'pre')] } }],
  };
  return { p: new Planner(cfg), Planner };
}

test('first attempt uses one slot, restores, then returns to adaptive prep after a miss', () => {
  const { p, Planner } = planner(true);
  assert.equal(p.peek().choice, 'attempt');
  assert.equal(p.used, 0);
  const restoredBefore = Planner.restore(JSON.parse(JSON.stringify(p.serialize())));
  assert.equal(restoredBefore.peek().choice, 'attempt');
  const item = p.next();
  assert.equal(item.questionId, 10);
  assert.equal(p.used, 1);
  assert.equal(p.attempts, 1);
  p.observe(item, false);
  const restored = Planner.restore(JSON.parse(JSON.stringify(p.serialize())));
  const adaptive = Planner.restore({ ...p.serialize(), attemptFirst: false });
  assert.equal(restored.peek().choice, adaptive.peek().choice);
  assert.equal(restored.peek().choice, 'prep');
  assert.equal(restored.peek().R, 7);
  assert.equal(planner(false).p.peek().choice, 'prep');
});

test('success ends block; one-question cap and missing variants remain bounded', () => {
  const { p } = planner(true);
  p.observe(p.next(), true);
  assert.equal(p.next(), null);
  assert.equal(p.used, 1);
  const one = planner(true, 1).p;
  one.observe(one.next(), false);
  assert.equal(one.next(), null);
  const missing = planner(true).p;
  missing.drop(missing.peek());
  assert.equal(missing.used, 0);
  assert.equal(missing.next().questionId, 11);
  const none = planner(true).p;
  none.target.variants = [];
  assert.equal(none.peek().choice, 'prep');
});

test('q391 prompt and fresh starters do not contain the answer pattern', () => {
  const structured = JSON.parse(source('../questions_structured.json')).find(q => q.id === 391).exercise;
  const correction = JSON.parse(source('../pipeline/question_content_corrections.json'))['391'];
  for (const q of [q391, structured, correction]) {
    assert.doesNotMatch(q.question_text + q.starter_code, /c h w\s*->\s*c \(h w\)/);
    assert.match(q.starter_code, /return None/);
    assert.equal(q.question_text, q391.question_text);
  }
  assert.equal(q391.test_cases.length, 4);
  const ctx = vm.createContext({});
  vm.runInContext(source('questions.js'), ctx);
  const old = { question_id: 391, question_text: "Flatten via 'c h w -> c (h w)'.", starter_code: "def solve(img):\n    \"\"\"Return shape (c, h*w) via 'c h w -> c (h w)'.\"\"\"\n    return None" };
  ctx.repairKnownQuestionContent(old);
  assert.doesNotMatch(old.question_text + old.starter_code, /c h w ->/);
  const scaffold = { question_id: 391, starter_code: "return einops.rearrange(img, '_____')" };
  ctx.repairKnownQuestionContent(scaffold);
  assert.match(scaffold.starter_code, /_____/);
});

async function browserPage(t) {
  const browser = await chromium.launch({ headless: true });
  t.after(() => browser.close());
  const page = await browser.newPage();
  await page.route('http://drills.test/**', route => route.fulfill({ contentType: 'text/html', body: '<main id="page-practice" class="session-idle"><div id="question-number"></div><div id="question-text"></div><textarea id="code-editor"></textarea><pre id="output-area"></pre></main>' }));
  await page.goto('http://drills.test/');
  return page;
}

async function lessonPage(t) {
  const page = await browserPage(t);
  await page.evaluate(q => {
    window.DEFAULT_EDITOR_CODE = '# Write your solution here';
    window.practiceMode = 'backend';
    window.getPracticeStorageKey = () => 'test';
    window.calls = { credit: 0, exposure: 0, xp: 0, mount: 0, done: 0 };
    window.fetch = async () => ({ ok: true, json: async () => ({ lessons: [{ topic: 'Einops', kps: [{ kc: 'merge', title: 'Merge axes', concept_markdown: 'LESSON SECRET', worked_example_markdown: '```python\nprint("ANSWER SECRET")\n```' }] }] }) });
    window.apiFetch = async () => { calls.exposure++; return { ok: true }; };
    window.loadQuestionsBank = async () => [q];
    window.getQuestionFromBank = () => q;
    window.LadderUI = { creditTaught: async () => { calls.credit++; } };
    window.LessonNotebook = { mount: () => { calls.mount++; } };
    window.addEventListener('delta:xp', () => calls.xp++);
    window.q = { question_id: 391, ladder_kc: 'merge', ladder_stage: 'worked', starter_code: 'SCAFFOLD SECRET', lesson_gate: [{ kc: 'merge' }] };
    window.done = () => { calls.done++; document.querySelector('#code-editor').value = window.q.starter_code; };
  }, q391);
  await page.addScriptTag({ content: source('lessons.js') });
  assert.equal(await page.evaluate(() => LessonGate.maybeShow(q, done)), true);
  return page;
}

test('ordinary practice opens the lesson directly without a starting-point chooser', async t => {
  const page = await lessonPage(t);
  await page.locator('#lesson-continue-btn').waitFor();
  assert.equal(await page.locator('#lesson-attempt-btn').count(), 0);
  assert.equal(await page.locator('#lesson-review-btn').count(), 0);
  assert.match(await page.locator('body').innerText(), /LESSON SECRET/);
  assert.deepEqual(await page.evaluate(() => calls), { credit: 0, exposure: 0, xp: 0, mount: 1, done: 0 });
});

test('completing the lesson credits exposure once and renders the pending question', async t => {
  const page = await lessonPage(t);
  await page.locator('#lesson-continue-btn').waitFor();
  assert.match(await page.locator('body').innerText(), /LESSON SECRET/);
  assert.match(await page.locator('#code-editor').inputValue(), /ANSWER SECRET/);
  await page.locator('#lesson-continue-btn').click();
  assert.deepEqual(await page.evaluate(() => calls), { credit: 1, exposure: 1, xp: 1, mount: 1, done: 1 });
  assert.equal(await page.evaluate(() => !!q.attempt_first), false);
});

test('ARENA drills start a scoped ladder with valid independent clock options', async t => {
  const page = await browserPage(t);
  await page.evaluate(() => {
    window.getPracticeStorageKey = () => 'test';
    window.SessionClock = { answerSecs: () => 1200 };
    window.PracticeSession = { isActive: () => false, hasPausedSession: () => false, configure(cfg) { window.cfg = cfg; }, start() { window.started = true; } };
    window.KcPractice = { startScoped: async kc => { window.scoped = kc; return true; }, startPlanned() { throw Error('obsolete planner'); } };
  });
  await page.addScriptTag({ content: source('exercise-session.js') });
  await page.evaluate(() => ExerciseSession.open({ kc: 'merge', title: 'Flatten', variants: [391] }));
  assert.equal(await page.locator('input[value="attempt"]').count(), 0);
  assert.equal(await page.locator('input[name="answer"][value="5m"]').isChecked(), true);
  await page.locator('.dd-ex-start-btn').click();
  assert.deepEqual(await page.evaluate(() => [started, scoped, cfg.answer, cfg.review, cfg.quota]), [true, 'merge', 300, 120, 8]);
  await page.evaluate(() => ExerciseSession.open({ kc: 'other', title: 'No variants', variants: [] }));
  await page.locator('.dd-ex-advanced > summary').click();
  await page.locator('label').filter({ has: page.locator('input[name="answer"][value="off"]') }).click();
  await page.locator('.dd-ex-start-btn').click();
  assert.deepEqual(await page.evaluate(() => [scoped, cfg.answer]), ['other', null]);
});
