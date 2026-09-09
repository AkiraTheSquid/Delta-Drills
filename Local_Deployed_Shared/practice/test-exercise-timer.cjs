const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { chromium } = require('playwright');
const source = name => fs.readFileSync(path.join(__dirname, name), 'utf8');

async function fixture(t, original = 391) {
  const browser = await chromium.launch({ headless: true });
  t.after(() => browser.close());
  const page = await browser.newPage();
  await page.route('http://drills.test/**', route => route.fulfill({ contentType: 'text/html', body: `
    <style>.hidden {display:none}</style>
    <main id="page-arena-notebook"><aside class="anb-toc">Contents</aside><div id="cells">
      <section class="nbv-cell nbv-code" id="setup"><textarea>import torch</textarea></section>
      <section class="nbv-cell nbv-md" id="heading"><h3>Exercise - implement rays</h3></section>
      <div class="dd-ex-block" id="block"><div class="dd-ex-row"></div></div>
      <section class="nbv-cell nbv-md" id="instructions">You should spend up to 10-20 minutes on this exercise.</section>
      <section class="nbv-cell nbv-code" id="answer"><textarea>def rays(): pass</textarea></section>
      <section class="nbv-cell nbv-code" id="tests">tests.test_rays(rays)</section>
      <details class="nbv-cell" data-role="details" id="solution">Solution</details>
      <section class="nbv-cell nbv-md" id="next"><h3>Exercise - implement next_problem</h3></section>
      <div class="dd-ex-block" id="other"><div class="dd-ex-row"></div></div>
      <section class="nbv-cell nbv-code" id="later">next_problem()</section>
    </div></main>` }));
  await page.goto('http://drills.test');
  await page.evaluate(original => {
    window.getPracticeStorageKey = () => 'account_a';
    window.practiceMode = 'backend';
    window.recorded = [];
    window.PracticeAPI = { recordLocalEval: async (...args) => { recorded.push(args); return { finalized: true }; } };
    window.ex = { fn: 'rays', nb: 'test', title: 'Rays', kc: 'rays', original };
    document.querySelector('#block')._exercise = ex;
    document.querySelector('#block')._sourceCell = document.querySelector('#heading');
    document.querySelector('#other')._sourceCell = document.querySelector('#next');
  }, original);
  await page.addStyleTag({ content: source('../styles/practice/exercise-timer.css') });
  await page.addScriptTag({ content: source('exercise-timer.js') });
  return page;
}

test('focus keeps setup, current problem, editor and tests; hides later exercise and solutions', async t => {
  const page = await fixture(t);
  assert.equal(await page.evaluate(() => ExerciseTimer.budgetSecs(document.querySelector('#block'))), 1200);
  await page.evaluate(() => ExerciseTimer.start(ex, document.querySelector('#block')));
  for (const id of ['setup', 'heading', 'instructions', 'answer', 'tests']) assert.equal(await page.locator('#' + id).isVisible(), true, id);
  for (const sel of ['#next', '#later', '#solution', '.anb-toc']) assert.equal(await page.locator(sel).isVisible(), false, sel);
  await page.evaluate(() => ExerciseTimer.stop('done'));
  assert.equal(await page.locator('#later').isVisible(), true);
  assert.deepEqual(await page.evaluate(() => recorded), []);
});

test('timeout records the linked problem once; missing IDs and failed writes never claim success', async t => {
  const page = await fixture(t);
  await page.evaluate(() => { ExerciseTimer.start(ex, document.querySelector('#block')); ExerciseTimer.stop('expired'); ExerciseTimer.stop('expired'); });
  await page.waitForFunction(() => document.querySelector('.dd-ex-verdict').textContent.includes('Recorded as a miss'));
  assert.deepEqual(await page.evaluate(() => recorded), [[391, false]]);
  await page.evaluate(() => { ex.original = null; ExerciseTimer.start(ex, document.querySelector('#block')); ExerciseTimer.stop('expired'); });
  await page.waitForFunction(() => document.querySelector('.dd-ex-verdict').textContent.includes('No result recorded'));
  assert.deepEqual(await page.evaluate(() => recorded), [[391, false]]);
  await page.evaluate(() => { ex.original = 391; PracticeAPI.recordLocalEval = async () => null; ExerciseTimer.start(ex, document.querySelector('#block')); ExerciseTimer.stop('expired'); });
  await page.waitForFunction(() => document.querySelector('.dd-ex-verdict').textContent.includes('No result recorded'));
});

test('leaving and returning restores a paused timer; another account cannot restore it', async t => {
  const page = await fixture(t);
  await page.evaluate(() => ExerciseTimer.start(ex, document.querySelector('#block'), 100));
  await page.evaluate(() => document.querySelector('main').classList.add('hidden'));
  await page.waitForFunction(() => !ExerciseTimer.isRunning());
  assert.equal(await page.evaluate(() => JSON.parse(localStorage.getItem('account_a_nb_timer')).paused), true);
  await page.evaluate(() => { window.getPracticeStorageKey = () => 'account_b'; document.querySelector('main').classList.remove('hidden'); });
  assert.equal(await page.evaluate(() => ExerciseTimer.isRunning()), false);
  await page.evaluate(() => { window.getPracticeStorageKey = () => 'account_a'; document.querySelector('main').classList.add('hidden'); });
  await page.evaluate(() => document.querySelector('main').classList.remove('hidden'));
  await page.waitForFunction(() => ExerciseTimer.isRunning());
  assert.equal(await page.locator('.dd-nbt-pause').textContent(), 'Resume');
  await page.evaluate(() => ExerciseTimer.stop('left'));
});

test('answer pace excludes grading waits, skips and account switches', () => {
  let now = 0, user = 'a';
  const saved = new Map();
  const ctx = vm.createContext({ window: {}, Date: { now: () => now }, getPracticeStorageKey: () => user,
    localStorage: { getItem: key => saved.get(key) || null, setItem: (key, value) => saved.set(key, value) } });
  vm.runInContext(source('answer-history.js'), ctx);
  const h = ctx.window.AnswerHistory;
  for (const secs of [10, 20, 30]) {
    h.begin(); now += secs * 1000; h.pause(); now += 60000; h.settle(); h.settle();
  }
  assert.equal(h.samples(), 3);
  assert.equal(h.secondsPerProblem(), 20);
  h.begin(); now += 90000; h.abandon(); h.settle();
  assert.equal(h.samples(), 3);
  h.begin(); now += 10000; user = 'b'; h.settle();
  assert.equal(h.samples(), 0);
});

test('switching notebook sections suspends the old timer before its cells disappear', async t => {
  const page = await fixture(t);
  await page.evaluate(() => {
    ExerciseTimer.start(ex, document.querySelector('#block'), 100);
    document.querySelector('#cells').replaceChildren();
    document.dispatchEvent(new CustomEvent('arena-notebook:rendered', { detail: { id: 'other' } }));
  });
  assert.equal(await page.evaluate(() => ExerciseTimer.isRunning()), false);
  assert.equal(await page.locator('.dd-nbt-bar').isVisible(), false);
  assert.equal(await page.evaluate(() => JSON.parse(localStorage.getItem('account_a_nb_timer')).paused), true);
  assert.deepEqual(await page.evaluate(() => recorded), []);
});

test('release entry point loads timer, pace history and focus CSS in dependency order', () => {
  const html = source('../index.html');
  for (const asset of ['practice/exercise-timer.js', 'practice/answer-history.js', 'styles/practice/exercise-timer.css']) assert.ok(html.includes(asset), asset);
  assert.ok(html.indexOf('practice/answer-history.js') < html.indexOf('practice/timer.js?v='));
  assert.ok(html.indexOf('src="practice/exercise-timer.js') < html.indexOf('src="practice/exercise-session.js'));
});
