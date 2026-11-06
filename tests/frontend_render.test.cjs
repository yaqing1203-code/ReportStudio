// No browser or npm dependencies: node --test tests/frontend_render.test.cjs
const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const context=vm.createContext({document:{addEventListener(){}},location:{search:''},URLSearchParams});
vm.runInContext(fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8'),context);
const render=source=>context.renderMarkdown(source);

test('tables render headings, inline spans and alignment',()=>{
 const html=render('| 指标 | 结果 |\n| :--- | ---: |\n| **循环** | 100 |\n| 温度 | 25 |');
 assert.match(html,/<thead><tr><th class="align-left">指标<\/th><th class="align-right">结果<\/th>/);
 assert.match(html,/<td class="align-left"><strong>循环<\/strong><\/td>/);
 assert.equal((html.match(/<tr>/g)||[]).length,3);
 assert.match(html,/class="markdown-table".*tabindex="0"/);
});

test('escaped pipes stay in a cell and missing cells are empty',()=>{
 const html=render('| A | B |\n| --- | --- |\n| a\\|b | c |\n| only |');
 assert.match(html,/<td class="align-left">a\|b<\/td><td class="align-left">c<\/td>/);
 assert.match(html,/<td class="align-left">only<\/td><td class="align-left"><\/td>/);
});

test('non-table pipe text stays as prose',()=>{
 assert.doesNotMatch(render('A | B\nnot | a divider'),/<table>/);
 assert.doesNotMatch(render('A | B\n---'),/<table>/);
});

test('table and quote content cannot inject raw HTML or event attributes',()=>{
 const html=render('| A | B |\n| --- | --- |\n| <img src=x onerror="alert(1)"> | <script>alert(1)</script> |\n\n> <svg onload="alert(1)">');
 assert.doesNotMatch(html,/<(?:img|script|svg)\b/);
 assert.match(html,/&lt;img src=x onerror=&quot;alert\(1\)&quot;&gt;/);
 assert.match(html,/<blockquote>&lt;svg/);
});

test('fenced tables remain code and later headings still render',()=>{
 const html=render('```text\n| A | B |\n| --- | --- |\n```\n\n## 后续章节');
 assert.doesNotMatch(html,/<table>/);
 assert.match(html,/<pre class="code-block"><code>\| A \| B \|/);
 assert.match(html,/<h2>后续章节<\/h2>/);
});
