'use strict';
(() => {
  const data = window.RECAP;
  const text = (id, value) => { document.getElementById(id).textContent = value; };
  const percent = value => (100 * value).toFixed(1);
  const examples = data.examples;

  for (const model of examples.audio.models) {
    const predictions = model.predictions;
    const first = predictions[0];
    if (!predictions.every(p => p.native_prediction === first.native_prediction && p.reader_prediction === first.reader_prediction)) {
      throw new Error('This illustration requires consistent predictions across seeds.');
    }
    const row = document.createElement('tr');
    for (const value of [model.name, first.native_prediction, first.reader_prediction]) {
      const cell = document.createElement('td');
      cell.textContent = value;
      row.appendChild(cell);
    }
    document.getElementById('audio-predictions').appendChild(row);
  }
  text('audio-label', examples.audio.truth);
  const audioScores = data.audio_scores.gemma2;
  text('audio-population', `On all 2,000 ESC-50 clips, Gemma 2 rises from ${percent(audioScores.native_prompt_selected_inner)}% with selected prompts to ${percent(audioScores.logistic_nested)}% with nested logistic regression. The clip above is one selected illustration.`);

  let target = 'video';
  let reader = 'video';
  function renderTransfer() {
    const example = examples.transfer.find(e => e.target === target);
    const matched = target === reader;
    document.getElementById('target-video').hidden = target !== 'video';
    document.getElementById('target-audio').hidden = target !== 'audio';
    document.querySelectorAll('[data-target]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.target === target)));
    document.querySelectorAll('[data-reader]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.reader === reader)));
    text('target-label', example.truth);
    text('target-id', example.id);
    text('transfer-prediction', matched ? example.matched : example.cross);
    document.getElementById('transfer-decision').classList.toggle('wrong', !matched);
    text('transfer-status', matched ? 'Correct on this example' : 'Wrong on this example');
    text('transfer-explanation', `A ${reader}-trained reader applied to the fixed ${target} vector ${matched ? 'recovers the event label.' : 'changes the decision to a different event class.'}`);
    const j = target === 'audio' ? 0 : 1;
    const matrix = data.transfer.balanced_accuracy_matrix;
    text('transfer-population', `On the same held-out ${target} inputs, the target-trained reader scores ${percent(matrix[j][j])}% balanced accuracy; the cross-trained reader scores ${percent(matrix[1-j][j])}%. Both see the same 402 test events.`);
  }
  document.querySelectorAll('[data-target]').forEach(button => button.addEventListener('click', () => {
    document.getElementById('horse-video').pause();
    document.getElementById('guitar-audio').pause();
    target = button.dataset.target;
    // Keep the chosen reader fixed so changing the target exposes both directions.
    renderTransfer();
  }));
  document.querySelectorAll('[data-reader]').forEach(button => button.addEventListener('click', () => {
    reader = button.dataset.reader;
    renderTransfer();
  }));
  renderTransfer();

  function renderWidth(endpoint) {
    document.querySelectorAll('[data-width]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.width === endpoint)));
    const prompt = data.width[`native_prompt_ba_${endpoint}`];
    const ridge = data.width[`ridge_alpha10_ba_${endpoint}`];
    document.getElementById('prompt-bar').style.width = `${100 * prompt}%`;
    document.getElementById('ridge-bar').style.width = `${100 * ridge}%`;
    text('prompt-score', percent(prompt));
    text('ridge-score', percent(ridge));
    text('width-verdict', ridge > prompt ? 'At 32D, the reader ranking reverses.' : 'At full width, text prompts win.');
    text('width-gap', `${ridge > prompt ? 'Ridge' : 'Prompt'} advantage: ${(100 * Math.abs(ridge-prompt)).toFixed(1)} points`);
    document.getElementById('width-chart').setAttribute('aria-label', `Balanced accuracy: text prompt ${percent(prompt)} percent, fixed Ridge ${percent(ridge)} percent at ${endpoint === 'native' ? '1024' : '32'} dimensions.`);
  }
  document.querySelectorAll('[data-width]').forEach(button => button.addEventListener('click', () => renderWidth(button.dataset.width)));
  renderWidth('native');

  for (const example of examples.images) {
    const card = document.createElement('article');
    card.className = `image-card ${example.kind}`;
    const figure = document.createElement('figure');
    const image = document.createElement('img');
    image.src = example.asset;
    image.alt = `${example.dataset} image with released label ${example.truth}`;
    image.loading = 'lazy';
    figure.appendChild(image);
    const body = document.createElement('div');
    body.className = 'image-body';
    const outcome = document.createElement('span');
    outcome.className = 'outcome';
    outcome.textContent = example.kind === 'rescue' ? 'Reader rescue' : 'Reader regression';
    const title = document.createElement('h3');
    title.textContent = example.truth;
    body.append(outcome, title);
    for (const [name, value] of [['Text prompt', example.prompt], ['Nested Ridge', example.ridge]]) {
      const row = document.createElement('div');
      row.className = 'image-result';
      const label = document.createElement('span');
      label.textContent = name;
      const prediction = document.createElement('strong');
      prediction.textContent = value;
      row.append(label, prediction);
      body.appendChild(row);
    }
    card.append(figure, body);
    document.getElementById('image-examples').appendChild(card);
  }
  text('budget-summary', `On ESC-50, a 512-byte vector cap and eight labeled source groups per class give ${percent(data.budget.best_feasible_native_only)}% balanced accuracy with native-only selection and ${percent(data.budget.selector)}% with joint selection. Same storage cap. Same labeled support.`);

  // Avoid overlapping examples when a visitor starts another player.
  document.querySelectorAll('audio,video').forEach(player => player.addEventListener('play', () => {
    document.querySelectorAll('audio,video').forEach(other => { if (other !== player) other.pause(); });
  }));
})();
