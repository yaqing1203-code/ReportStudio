// No browser or npm dependencies: node --test tests/frontend_theme.test.cjs
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../web/theme.js'), 'utf8');

function workspace({saved = null, dark = false, blocked = false, media = true, ready = false} = {}) {
  const root = {dataset: {}, style: {}};
  const attributes = {};
  const label = {textContent: ''};
  const buttonEvents = {}, documentEvents = {}, windowEvents = {};
  const button = {
    title: '',
    setAttribute(key, value) { attributes[key] = value; },
    querySelector(selector) { return selector === '.theme-label' ? label : null; },
    addEventListener(name, callback) { buttonEvents[name] = callback; },
  };
  let stored = saved;
  const storage = {
    getItem(key) {
      assert.equal(key, 'rs-theme');
      if (blocked) throw new Error('storage blocked');
      return stored;
    },
    setItem(key, value) {
      assert.equal(key, 'rs-theme');
      if (blocked) throw new Error('storage blocked');
      stored = value;
    },
  };
  let mediaListener = null;
  const query = {
    matches: dark,
    addEventListener(name, callback) { assert.equal(name, 'change'); mediaListener = callback; },
  };
  const document = {
    documentElement: root,
    readyState: ready ? 'complete' : 'loading',
    getElementById(id) { return id === 'theme-toggle' ? button : null; },
    addEventListener(name, callback) { documentEvents[name] = callback; },
  };
  const window = {
    matchMedia() { if (!media) throw new Error('unsupported'); return query; },
    addEventListener(name, callback) { windowEvents[name] = callback; },
  };
  vm.runInNewContext(source, {window, document, localStorage: storage});
  return {
    root, attributes, label, button, storage,
    get saved() { return stored; },
    ready() { documentEvents.DOMContentLoaded?.(); },
    click() { buttonEvents.click(); },
    system(dark) { query.matches = dark; mediaListener?.({matches: dark}); },
    otherTab(value, key = 'rs-theme', storageArea = storage) {
      windowEvents.storage({key, newValue: value, storageArea});
    },
  };
}

test('initial theme follows the system before DOM readiness', () => {
  for (const dark of [false, true]) {
    const app = workspace({dark});
    assert.equal(app.root.dataset.theme, dark ? 'dark' : 'light');
    assert.equal(app.root.style.colorScheme, dark ? 'dark' : 'light');
    assert.equal(app.label.textContent, '');
    app.ready();
    assert.equal(app.label.textContent, dark ? '暗色' : '亮色');
    assert.equal(app.attributes['aria-pressed'], String(dark));
  }
});

test('explicit preference wins over system and survives a reload', () => {
  const app = workspace({saved: 'dark', dark: false});
  app.ready();
  assert.equal(app.root.dataset.theme, 'dark');
  assert.equal(app.button.title, '切换到亮色模式');
  app.click();
  assert.equal(app.saved, 'light');
  assert.equal(app.root.dataset.theme, 'light');
  assert.equal(app.attributes['aria-label'], '切换到暗色模式');
  const reloaded = workspace({saved: app.saved, dark: true});
  assert.equal(reloaded.root.dataset.theme, 'light');
});

test('system changes apply until the user makes an explicit choice', () => {
  const app = workspace();
  app.ready();
  app.system(true);
  assert.equal(app.root.dataset.theme, 'dark');
  app.click();
  app.system(true);
  assert.equal(app.root.dataset.theme, 'light');
  assert.equal(app.saved, 'light');
});

test('cross-tab preference changes and removal update an open workspace', () => {
  const app = workspace({dark: true, saved: 'light', ready: true});
  app.otherTab('dark');
  assert.equal(app.root.dataset.theme, 'dark');
  assert.equal(app.label.textContent, '暗色');
  app.otherTab('light', 'unrelated-key');
  assert.equal(app.root.dataset.theme, 'dark');
  app.otherTab('light');
  assert.equal(app.root.dataset.theme, 'light');
  app.otherTab(null);
  assert.equal(app.root.dataset.theme, 'dark');
  app.system(false);
  assert.equal(app.root.dataset.theme, 'light');
  app.otherTab('dark');
  app.otherTab(null, null);
  assert.equal(app.root.dataset.theme, 'light');
});

test('sessionStorage events do not replace the theme preference', () => {
  const app = workspace({saved: 'dark'});
  app.otherTab('light', 'rs-theme', {});
  assert.equal(app.root.dataset.theme, 'dark');
});

test('blocked storage still allows toggling and keeps the choice for this window', () => {
  const app = workspace({blocked: true, dark: true, ready: true});
  assert.equal(app.root.dataset.theme, 'dark');
  app.click();
  assert.equal(app.root.dataset.theme, 'light');
  assert.equal(app.attributes['aria-pressed'], 'false');
  app.system(true);
  assert.equal(app.root.dataset.theme, 'light');
});

test('invalid saved values and unavailable media queries fall back safely', () => {
  assert.equal(workspace({saved: 'unexpected', dark: true}).root.dataset.theme, 'dark');
  const app = workspace({media: false, ready: true});
  assert.equal(app.root.dataset.theme, 'light');
  app.click();
  assert.equal(app.root.dataset.theme, 'dark');
});
