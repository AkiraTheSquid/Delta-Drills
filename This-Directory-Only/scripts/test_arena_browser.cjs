// Run against a local static server. Python RPC is mocked; no accounts or remote experiments.
const assert = require('node:assert/strict');
const {chromium} = require('playwright');
(async () => {
 const browser = await chromium.launch({headless:true,args:['--no-sandbox']});
 const page = await browser.newPage({viewport:{width:1440,height:1000}});
 const errors=[];page.on('pageerror', e=>errors.push(e.message));
 await page.goto(process.env.ARENA_TEST_URL || 'http://127.0.0.1:8874', {waitUntil:'networkidle'});
 await page.evaluate(() => {
  window.testRuns=[]; let marker='';
  window.DeltaKernel.available=()=>true;
  window.LessonNotebook.runSource=async(source, options)=>{
   window.testRuns.push({source,options});
   if(source.startsWith('globals()')) return {text:marker===source.match(/'([^']+)'$/)[1]?'True':'False'};
   if(source.startsWith('__dd_arena_setup_key =')) marker=source.match(/'([^']+)'/)[1];
   return {text:'mock output', failed:false,outputs:[]};
  };
 });
 const nb=await page.evaluate(async()=>{
  const nb=await (await fetch('lessons/notebooks/arena-0-0.json')).json();
  await window.ArenaNotebook.open('0-0',nb.exercises[3].id);return nb;
 });
 await page.waitForFunction(()=>document.querySelector('.nbv-banner')?.textContent.includes('Setup ready'));
 for (const id of nb.setup_cells) assert.equal(await page.locator(`[data-cell-id="${id}"].has-run`).count(),1);
 const answer=nb.exercises[3].answer_cell;
 const editor=page.locator(`[data-cell-id="${answer}"] .nbv-src code`);
 assert.equal(await editor.evaluate(el=>el.isContentEditable),true);
 await editor.fill('print(42)');
 await page.locator(`[data-cell-id="${answer}"] .nbv-run`).click();
 await page.waitForFunction(id=>document.querySelector(`[data-cell-id="${id}"]`)?.classList.contains('has-run'),answer);
 const before=await page.evaluate(()=>window.testRuns.length);
 await page.locator('[data-exercise-step="1"]').click();
 assert.equal(await page.locator('.arena-exercise-select').inputValue(),nb.exercises[4].id);
 await page.locator('[data-exercise-step="-1"]').click();
 assert.equal(await editor.innerText(),'print(42)');
 assert.equal(await page.evaluate(()=>window.testRuns.length),before,'navigation must not replay answers');
 assert.equal(await page.locator('#arena-notebook-host .nbv-cell').count(),nb.cells.length);
 await page.evaluate(()=>window.ArenaNotebook.open('0-1'));
 await page.evaluate(id=>window.ArenaNotebook.open('0-0',id),nb.exercises[3].id);
 await page.waitForTimeout(850);
 assert.equal(await editor.innerText(),'print(42)','answer survives section switch');
 const top=await page.locator(`[data-cell-id="${nb.exercises[3].prompt_cell}"]`).evaluate(el=>el.getBoundingClientRect().top);
 assert.ok(Math.abs(top)<150,`explicit exercise link must win over scroll restore: ${top}`);
 await page.evaluate(()=>{document.querySelectorAll('main.page').forEach(el=>el.classList.add('hidden'));document.querySelector('#page-knowledge-graph').classList.remove('hidden');});
 await page.getByRole('button',{name:'ARENA curriculum',exact:true}).click();
 await page.waitForSelector('.arena-curriculum-canvas canvas');
 const chapters=await page.locator('.arena-curriculum select option').count();assert.equal(chapters,5);
 for(let i=0;i<chapters;i++){
  await page.locator('.arena-curriculum select').selectOption({index:i});
  assert.ok(await page.locator('.arena-curriculum-detail button').count()>0);
 }
 await page.locator('.arena-curriculum select').selectOption({index:1});
 await page.getByRole('button',{name:'1.1 Transformers from Scratch',exact:true}).click();
 await page.getByRole('button',{name:'Explore exercises',exact:true}).click();
 await page.locator('.arena-curriculum-detail li button').first().click();
 assert.equal(await page.locator('.arena-curriculum-detail h3').first().innerText(),'Lesson context');
 await page.locator('.arena-curriculum-detail h3').first().locator('xpath=following-sibling::button[1]').click();
 assert.ok(await page.locator('.arena-curriculum-detail button').count()>0,'topic must open its exercises');
 await page.setViewportSize({width:390,height:844});
 await page.screenshot({path:'/tmp/arena-graph-mobile.png'});
 assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+2),'no horizontal overflow on mobile graph');
 assert.deepEqual(errors,[]);
 console.log('PASS browser: 5 chapters, full topic context, writable answers, retained prior cells/edits, automatic setup/output, explicit anchor across section switches, mobile layout');
 await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
