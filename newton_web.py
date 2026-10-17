"""
Newton's Method — Web App (Flask)
Запускается на компьютере, открывается с iPhone через Wi-Fi.
"""

from flask import Flask, request, jsonify, render_template_string
import numpy as np
import json

app = Flask(__name__)


# ─── Ядро метода Ньютона ──────────────────────────────────────────

def numerical_jacobian(F, x, eps=1e-7):
    n = len(x)
    J = np.zeros((n, n))
    f0 = F(x)
    for j in range(n):
        xp = x.copy(); xp[j] += eps
        J[:, j] = (F(xp) - f0) / eps
    return J


def newton_method(F, x0, tol=1e-10, max_iter=100):
    x = np.array(x0, dtype=float)
    history = []
    for k in range(max_iter):
        Fx = F(x)
        res = float(np.linalg.norm(Fx))
        history.append({
            "iter": k + 1,
            "x": x.tolist(),
            "residual": res,
        })
        J = numerical_jacobian(F, x)
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            return x.tolist(), {"converged": False, "iterations": k,
                                "residual": res, "history": history,
                                "error": "Якобиан вырожден"}
        x = x + dx
        step = float(np.linalg.norm(dx))
        history[-1]["step"] = step
        if step < tol:
            final_res = float(np.linalg.norm(F(x)))
            history.append({"iter": k + 2, "x": x.tolist(),
                             "residual": final_res, "step": 0.0})
            return x.tolist(), {"converged": True, "iterations": k + 1,
                                "residual": final_res, "history": history,
                                "error": None}
    return x.tolist(), {"converged": False, "iterations": max_iter,
                        "residual": float(np.linalg.norm(F(x))),
                        "history": history, "error": "Достигнут лимит итераций"}


# ─── HTML страница ────────────────────────────────────────────────

HTML = r"""
<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
  <meta name="apple-mobile-web-app-capable" content="yes"/>
  <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent"/>
  <meta name="theme-color" content="#1e1e2e"/>
  <title>Метод Ньютона</title>
  <style>
    :root {
      --bg:        #1e1e2e;
      --surface:   #2a2a3e;
      --surface2:  #313145;
      --accent:    #7c6af7;
      --accent2:   #56cfb2;
      --success:   #a6e3a1;
      --error:     #f38ba8;
      --warn:      #fab387;
      --text:      #cdd6f4;
      --muted:     #6c7086;
      --border:    #45475a;
      --entry:     #181825;
      --radius:    14px;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    html, body { background: var(--bg); color: var(--text); font-family: -apple-system, "SF Pro Display", Helvetica, sans-serif; min-height: 100dvh; }

    /* ── Header ── */
    header {
      background: var(--surface);
      padding: 18px 20px 14px;
      padding-top: max(18px, env(safe-area-inset-top));
      text-align: center;
      position: sticky; top: 0; z-index: 100;
      border-bottom: 1px solid var(--border);
    }
    header h1 { font-size: 20px; font-weight: 700; color: var(--accent); }
    header p  { font-size: 12px; color: var(--muted); margin-top: 2px; }

    /* ── Layout ── */
    main { padding: 16px; max-width: 700px; margin: 0 auto; padding-bottom: 40px; }

    /* ── Cards ── */
    .card {
      background: var(--surface);
      border-radius: var(--radius);
      padding: 16px;
      margin-bottom: 14px;
    }
    .card-title {
      font-size: 13px; font-weight: 600;
      color: var(--accent); text-transform: uppercase;
      letter-spacing: .06em; margin-bottom: 12px;
    }

    /* ── Examples ── */
    .examples { display: flex; flex-direction: column; gap: 8px; }
    .ex-btn {
      background: var(--surface2); color: var(--text);
      border: 1px solid var(--border); border-radius: 10px;
      padding: 10px 14px; font-size: 14px; cursor: pointer;
      text-align: left; transition: background .15s;
    }
    .ex-btn:active { background: var(--border); }

    /* ── n-selector ── */
    .n-selector { display: flex; gap: 8px; }
    .n-btn {
      flex: 1; padding: 10px; border-radius: 10px;
      border: 1.5px solid var(--border);
      background: var(--surface2); color: var(--text);
      font-size: 16px; font-weight: 600; cursor: pointer;
      transition: all .15s;
    }
    .n-btn.active { background: var(--accent); border-color: var(--accent); color: #fff; }

    /* ── Equations ── */
    .eq-row { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
    .eq-label { color: var(--accent2); font-weight: 700; font-size: 15px; min-width: 28px; }
    .eq-row input, .param-row input, #x0-input {
      flex: 1; background: var(--entry); color: var(--text);
      border: 1.5px solid var(--border); border-radius: 10px;
      padding: 10px 12px; font-size: 14px; font-family: "SF Mono", Menlo, monospace;
      outline: none; transition: border-color .2s;
      -webkit-appearance: none;
    }
    .eq-row input:focus, .param-row input:focus, #x0-input:focus {
      border-color: var(--accent);
    }

    /* ── Params ── */
    .param-row { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
    .param-label { color: var(--muted); font-size: 13px; min-width: 120px; }

    /* ── Buttons ── */
    .solve-btn {
      width: 100%; padding: 15px;
      background: var(--accent); color: #fff;
      border: none; border-radius: var(--radius);
      font-size: 17px; font-weight: 700; cursor: pointer;
      transition: opacity .15s;
      -webkit-tap-highlight-color: transparent;
    }
    .solve-btn:active { opacity: .8; }
    .solve-btn:disabled { opacity: .5; }

    /* ── Status badge ── */
    #status-badge {
      display: none; text-align: center;
      padding: 10px 16px; border-radius: 10px;
      font-size: 14px; font-weight: 600; margin-bottom: 14px;
    }
    #status-badge.success { background: #1d3a1d; color: var(--success); }
    #status-badge.error   { background: #3a1d1d; color: var(--error);   }

    /* ── Solution box ── */
    #solution-box { display: none; }
    .solution-text {
      background: var(--entry); border-radius: 10px;
      padding: 14px; font-family: "SF Mono", Menlo, monospace;
      font-size: 14px; line-height: 1.7; white-space: pre-wrap;
    }
    .copy-btn {
      margin-top: 10px; padding: 9px 18px;
      background: var(--surface2); color: var(--text);
      border: 1px solid var(--border); border-radius: 8px;
      font-size: 13px; cursor: pointer;
    }
    .copy-btn:active { background: var(--border); }

    /* ── Table ── */
    #table-box { display: none; overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th {
      background: var(--surface2); color: var(--accent);
      padding: 9px 10px; text-align: center;
      font-size: 11px; text-transform: uppercase; letter-spacing: .05em;
      position: sticky; top: 0;
    }
    td { padding: 8px 10px; text-align: center; border-bottom: 1px solid var(--border); }
    tr:last-child td { color: var(--success); font-weight: 600; }
    tr:nth-child(even) td { background: rgba(255,255,255,.02); }

    /* ── Convergence chart ── */
    #chart-box { display: none; }
    canvas#chart { width: 100% !important; border-radius: 10px; }

    /* ── Spinner ── */
    .spinner {
      display: inline-block; width: 18px; height: 18px;
      border: 2px solid rgba(255,255,255,.3);
      border-top-color: #fff; border-radius: 50%;
      animation: spin .7s linear infinite; vertical-align: middle; margin-right: 6px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* ── Math Keypad ── */
    #math-keypad {
      position: fixed;
      bottom: 0; left: 0; right: 0;
      background: #242438;
      border-top: 2px solid var(--accent);
      box-shadow: 0 -8px 28px rgba(0,0,0,.6);
      z-index: 1000;
      display: none;
      padding: 8px 10px max(10px, env(safe-area-inset-bottom));
      user-select: none;
      -webkit-user-select: none;
    }
    .keypad-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      max-width: 600px;
      margin: 0 auto 6px;
      padding: 0 4px;
    }
    .keypad-title {
      font-size: 13px;
      font-weight: 700;
      color: var(--text);
    }
    .keypad-title span#keypad-field-name {
      color: var(--accent2);
    }
    .keypad-nav {
      display: flex;
      gap: 6px;
    }
    .keypad-nav-btn {
      background: var(--surface2);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 4px 10px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
    }
    .keypad-nav-btn:active {
      background: var(--border);
    }
    .keypad-close-btn {
      background: var(--accent);
      color: #fff;
      border: none;
      border-radius: 6px;
      padding: 4px 12px;
      font-size: 12px;
      font-weight: 700;
      cursor: pointer;
    }
    .keypad-grid {
      display: grid;
      grid-template-columns: repeat(7, 1fr);
      gap: 5px;
      max-width: 600px;
      margin: 0 auto;
    }
    .k-btn {
      background: var(--surface2);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 8px;
      font-family: "SF Mono", Menlo, monospace;
      font-size: 14px;
      font-weight: 600;
      height: 40px;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      touch-action: manipulation;
      transition: background 0.1s, transform 0.05s;
      -webkit-tap-highlight-color: transparent;
    }
    .k-btn:active {
      transform: scale(0.92);
      background: var(--border);
    }
    .k-var {
      color: var(--accent2);
      background: rgba(86, 207, 178, 0.14);
      border-color: rgba(86, 207, 178, 0.35);
      font-weight: 700;
      font-size: 16px;
    }
    .k-fn {
      color: #b4befe;
      background: rgba(180, 190, 254, 0.12);
      border-color: rgba(180, 190, 254, 0.25);
      font-size: 12px;
    }
    .k-op {
      color: var(--warn);
      background: rgba(250, 179, 135, 0.12);
      border-color: rgba(250, 179, 135, 0.3);
      font-weight: 700;
    }
    .k-num {
      font-weight: 600;
      font-size: 15px;
    }
    .k-ctrl {
      color: var(--error);
      background: rgba(243, 139, 168, 0.12);
      border-color: rgba(243, 139, 168, 0.3);
    }
    .k-nav {
      color: var(--accent);
      background: rgba(124, 106, 247, 0.15);
      border-color: rgba(124, 106, 247, 0.3);
    }
    .eq-kb-btn {
      background: var(--surface2);
      border: 1.5px solid var(--border);
      color: var(--text);
      border-radius: 10px;
      padding: 10px 12px;
      font-size: 14px;
      cursor: pointer;
      line-height: 1;
    }
    .eq-kb-btn:active {
      background: var(--accent);
    }
    .active-input {
      border-color: var(--accent) !important;
      box-shadow: 0 0 0 2px rgba(124, 106, 247, 0.3) !important;
    }
  </style>
</head>
<body>

<header>
  <h1>Newton's Method</h1>
  <p>Системы нелинейных уравнений</p>
</header>

<main>

  <!-- Примеры -->
  <div class="card">
    <div class="card-title">📚 Примеры</div>
    <div class="examples">
      <button class="ex-btn" onclick="loadExample(0)">🔵 Окружность ∩ Парабола (2×2)</button>
      <button class="ex-btn" onclick="loadExample(1)">🟣 Классическая Burden & Faires (3×3)</button>
      <button class="ex-btn" onclick="loadExample(2)">🟢 Тригонометрическая (2×2)</button>
    </div>
  </div>

  <!-- Размер системы -->
  <div class="card">
    <div class="card-title">🔢 Количество уравнений</div>
    <div class="n-selector">
      <button class="n-btn active" id="n-2" onclick="setN(2)">2</button>
      <button class="n-btn"        id="n-3" onclick="setN(3)">3</button>
      <button class="n-btn"        id="n-4" onclick="setN(4)">4</button>
      <button class="n-btn"        id="n-5" onclick="setN(5)">5</button>
    </div>
  </div>

  <!-- Уравнения -->
  <div class="card">
    <div class="card-title">📝 Уравнения (F<sub>i</sub> = 0)</div>
    <div style="font-size:12px;color:var(--muted);margin-bottom:12px;line-height:1.5;">
      Переменные: <b style="color:var(--accent2);">x, y, z</b> (или w, u) &nbsp;|&nbsp; Функции: <b>sin, cos, exp, log, sqrt, pi</b><br/>
      Степень: <b>^</b> или <b>**</b> &nbsp;|&nbsp; Можно писать уравнения с <b>=</b> (напр. <code>x^2 + y^2 = 4</code>)
    </div>
    <div id="eq-container"></div>
  </div>

  <!-- Начальное приближение -->
  <div class="card">
    <div class="card-title">🎯 Начальное приближение (<span id="x0-vars" style="color:var(--accent2);">x, y</span>)₀</div>
    <div style="font-size:12px;color:var(--muted);margin-bottom:10px;">Числа через пробел (по порядку переменных)</div>
    <div style="display:flex;gap:10px;">
      <input id="x0-input" type="text" placeholder="напр. 1.0  0.5" inputmode="decimal"
             onfocus="openKeypadFor(this, 'x₀', -1)" onclick="openKeypadFor(this, 'x₀', -1)"/>
      <button type="button" class="eq-kb-btn" onmousedown="event.preventDefault()" onclick="openKeypadFor(document.getElementById('x0-input'), 'x₀', -1)" title="Математическая клавиатура">⌨️</button>
    </div>
  </div>

  <!-- Параметры -->
  <div class="card">
    <div class="card-title">⚙️ Параметры</div>
    <div class="param-row">
      <span class="param-label">Точность (tol)</span>
      <input id="tol-input" type="text" value="1e-10" style="max-width:150px;"/>
    </div>
    <div class="param-row">
      <span class="param-label">Макс. итераций</span>
      <input id="iter-input" type="number" value="100" style="max-width:150px;"/>
    </div>
  </div>

  <!-- Кнопка -->
  <button class="solve-btn" id="solve-btn" onclick="solve()">▶ &nbsp;Решить</button>

  <!-- Статус -->
  <div id="status-badge"></div>

  <!-- Решение -->
  <div class="card" id="solution-box">
    <div class="card-title">✅ Решение</div>
    <div class="solution-text" id="solution-text"></div>
    <button class="copy-btn" onclick="copySolution()">📋 Копировать</button>
  </div>

  <!-- Таблица -->
  <div class="card" id="table-box">
    <div class="card-title">📊 Таблица итераций</div>
    <div style="overflow-x:auto;">
      <table id="iter-table">
        <thead>
          <tr><th>Итер.</th><th>‖F(x)‖</th><th>‖Δx‖</th><th>Значения x</th></tr>
        </thead>
        <tbody id="table-body"></tbody>
      </table>
    </div>
  </div>

  <!-- График -->
  <div class="card" id="chart-box">
    <div class="card-title">📈 Сходимость</div>
    <canvas id="chart"></canvas>
  </div>

</main>

<!-- Математическая клавиатура -->
<div id="math-keypad">
  <div class="keypad-header">
    <div class="keypad-title">
      <span style="color:var(--muted);font-weight:400;">Ввод: </span><span id="keypad-field-name">F1</span>
    </div>
    <div class="keypad-nav">
      <button type="button" class="keypad-nav-btn" onmousedown="event.preventDefault()" onclick="navKeypad(-1)">◀ Пред</button>
      <button type="button" class="keypad-nav-btn" onmousedown="event.preventDefault()" onclick="navKeypad(1)">След ▶</button>
      <button type="button" class="keypad-close-btn" onmousedown="event.preventDefault()" onclick="hideKeypad()">Готово ✕</button>
    </div>
  </div>
  <div class="keypad-grid">
    <!-- Ряд 1 -->
    <button type="button" class="k-btn k-var" onmousedown="event.preventDefault()" onclick="kInsert('x')">x</button>
    <button type="button" class="k-btn k-var" onmousedown="event.preventDefault()" onclick="kInsert('y')">y</button>
    <button type="button" class="k-btn k-var" onmousedown="event.preventDefault()" onclick="kInsert('z')">z</button>
    <button type="button" class="k-btn k-op"  onmousedown="event.preventDefault()" onclick="kInsert('^')">^</button>
    <button type="button" class="k-btn k-op"  onmousedown="event.preventDefault()" onclick="kInsert('(', -1, ')')">(</button>
    <button type="button" class="k-btn k-op"  onmousedown="event.preventDefault()" onclick="kInsertCloseParen()">)</button>
    <button type="button" class="k-btn k-ctrl" onmousedown="event.preventDefault()" onclick="kBackspace()">⌫</button>

    <!-- Ряд 2 -->
    <button type="button" class="k-btn k-fn" onmousedown="event.preventDefault()" onclick="kInsert('sin(', -1, ')')">sin</button>
    <button type="button" class="k-btn k-fn" onmousedown="event.preventDefault()" onclick="kInsert('cos(', -1, ')')">cos</button>
    <button type="button" class="k-btn k-fn" onmousedown="event.preventDefault()" onclick="kInsert('tan(', -1, ')')">tan</button>
    <button type="button" class="k-btn k-num" onmousedown="event.preventDefault()" onclick="kInsert('7')">7</button>
    <button type="button" class="k-btn k-num" onmousedown="event.preventDefault()" onclick="kInsert('8')">8</button>
    <button type="button" class="k-btn k-num" onmousedown="event.preventDefault()" onclick="kInsert('9')">9</button>
    <button type="button" class="k-btn k-op"  onmousedown="event.preventDefault()" onclick="kInsert('/')">/</button>

    <!-- Ряд 3 -->
    <button type="button" class="k-btn k-fn" onmousedown="event.preventDefault()" onclick="kInsert('exp(', -1, ')')">exp</button>
    <button type="button" class="k-btn k-fn" onmousedown="event.preventDefault()" onclick="kInsert('log(', -1, ')')">ln</button>
    <button type="button" class="k-btn k-fn" onmousedown="event.preventDefault()" onclick="kInsert('sqrt(', -1, ')')">√</button>
    <button type="button" class="k-btn k-num" onmousedown="event.preventDefault()" onclick="kInsert('4')">4</button>
    <button type="button" class="k-btn k-num" onmousedown="event.preventDefault()" onclick="kInsert('5')">5</button>
    <button type="button" class="k-btn k-num" onmousedown="event.preventDefault()" onclick="kInsert('6')">6</button>
    <button type="button" class="k-btn k-op"  onmousedown="event.preventDefault()" onclick="kInsert('*')">×</button>

    <!-- Ряд 4 -->
    <button type="button" class="k-btn k-fn" onmousedown="event.preventDefault()" onclick="kInsert('pi')">π</button>
    <button type="button" class="k-btn k-fn" onmousedown="event.preventDefault()" onclick="kInsert('e')">e</button>
    <button type="button" class="k-btn k-op"  onmousedown="event.preventDefault()" onclick="kInsert(' = ')">=</button>
    <button type="button" class="k-btn k-num" onmousedown="event.preventDefault()" onclick="kInsert('1')">1</button>
    <button type="button" class="k-btn k-num" onmousedown="event.preventDefault()" onclick="kInsert('2')">2</button>
    <button type="button" class="k-btn k-num" onmousedown="event.preventDefault()" onclick="kInsert('3')">3</button>
    <button type="button" class="k-btn k-op"  onmousedown="event.preventDefault()" onclick="kInsert('-')">−</button>

    <!-- Ряд 5 -->
    <button type="button" class="k-btn k-ctrl" onmousedown="event.preventDefault()" onclick="kClear()">AC</button>
    <button type="button" class="k-btn k-var"  onmousedown="event.preventDefault()" onclick="kInsert('w')">w</button>
    <button type="button" class="k-btn"        onmousedown="event.preventDefault()" onclick="kInsert(' ')" title="Пробел">␣</button>
    <button type="button" class="k-btn k-num"  onmousedown="event.preventDefault()" onclick="kInsert('0')">0</button>
    <button type="button" class="k-btn k-num"  onmousedown="event.preventDefault()" onclick="kInsert('.')">.</button>
    <button type="button" class="k-btn k-op"   onmousedown="event.preventDefault()" onclick="kInsert('+')">+</button>
    <button type="button" class="k-btn k-nav"  onmousedown="event.preventDefault()" onclick="navKeypad(1)">↵ F↓</button>
  </div>
</div>

<script>
// ── Переменные ───────────────────────────────────────────────────
const VAR_NAMES = ['x', 'y', 'z', 'w', 'u', 'v'];
const getVarName = (i) => (i < VAR_NAMES.length ? VAR_NAMES[i] : `x${i+1}`);

// ── Данные примеров ──────────────────────────────────────────────
const EXAMPLES = [
  {
    n: 2,
    eqs: ["x**2 + y**2 - 4", "x**2 - y - 1"],
    x0: "1.0  0.5"
  },
  {
    n: 3,
    eqs: [
      "3*x - cos(y*z) - 0.5",
      "x**2 - 81*(y+0.1)**2 + sin(z) + 1.06",
      "exp(-x*y) + 20*z + (10*pi-3)/3"
    ],
    x0: "0.1  0.1  -0.1"
  },
  {
    n: 2,
    eqs: ["sin(x) + y - 1.2", "x + cos(y) - 0.8"],
    x0: "0.5  0.5"
  }
];

let currentN = 2;
let chartInstance = null;

// ── Инициализация ────────────────────────────────────────────────
window.onload = () => { setN(2); loadExample(0); };

// ── Математическая клавиатура (логика) ────────────────────────────
let currentActiveInput = null;
let currentActiveFieldIdx = 0;

function openKeypadFor(inputEl, label, idx) {
  currentActiveInput = inputEl;
  currentActiveFieldIdx = idx;
  document.querySelectorAll('.active-input').forEach(el => el.classList.remove('active-input'));
  inputEl.classList.add('active-input');

  const nameEl = document.getElementById('keypad-field-name');
  if (nameEl) nameEl.textContent = label;

  const keypad = document.getElementById('math-keypad');
  keypad.style.display = 'block';
  document.body.style.paddingBottom = '290px';

  setTimeout(() => {
    inputEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, 80);
}

function hideKeypad() {
  const keypad = document.getElementById('math-keypad');
  keypad.style.display = 'none';
  document.body.style.paddingBottom = '0px';
  document.querySelectorAll('.active-input').forEach(el => el.classList.remove('active-input'));
  currentActiveInput = null;
}

function navKeypad(direction) {
  let fields = [];
  for (let i = 0; i < currentN; i++) {
    fields.push({ id: `eq-${i}`, label: `F${i+1}`, idx: i });
  }
  fields.push({ id: 'x0-input', label: 'x₀', idx: -1 });

  let curIdx = fields.findIndex(f => f.idx === currentActiveFieldIdx);
  if (curIdx === -1) curIdx = 0;
  let nextIdx = curIdx + direction;
  if (nextIdx >= fields.length) {
    hideKeypad();
    return;
  }
  if (nextIdx < 0) nextIdx = 0;

  const target = fields[nextIdx];
  const el = document.getElementById(target.id);
  if (el) {
    el.focus();
    openKeypadFor(el, target.label, target.idx);
  }
}

function kInsert(text, cursorOffset = 0, closeChar = '') {
  if (!currentActiveInput) {
    const el = document.getElementById('eq-0');
    if (el) openKeypadFor(el, 'F1', 0);
    else return;
  }
  const el = currentActiveInput;
  const start = el.selectionStart ?? el.value.length;
  const end = el.selectionEnd ?? el.value.length;
  const val = el.value;

  let insertText = text;
  if (closeChar) {
    insertText = text + closeChar;
  }

  el.value = val.substring(0, start) + insertText + val.substring(end);
  const newPos = start + insertText.length + cursorOffset;
  el.focus();
  el.setSelectionRange(newPos, newPos);
  el.dispatchEvent(new Event('input', { bubbles: true }));
}

function kInsertCloseParen() {
  if (!currentActiveInput) return;
  const el = currentActiveInput;
  const start = el.selectionStart ?? el.value.length;
  const val = el.value;
  if (val.charAt(start) === ')') {
    el.setSelectionRange(start + 1, start + 1);
  } else {
    kInsert(')');
  }
}

function kBackspace() {
  if (!currentActiveInput) return;
  const el = currentActiveInput;
  const start = el.selectionStart ?? el.value.length;
  const end = el.selectionEnd ?? el.value.length;
  const val = el.value;

  if (start !== end) {
    el.value = val.substring(0, start) + val.substring(end);
    el.setSelectionRange(start, start);
  } else if (start > 0) {
    let deleteCount = 1;
    const prefixes = ['sin(', 'cos(', 'tan(', 'exp(', 'log(', 'sqrt('];
    for (const p of prefixes) {
      if (val.substring(0, start).endsWith(p)) {
        deleteCount = p.length;
        break;
      }
    }
    el.value = val.substring(0, start - deleteCount) + val.substring(start);
    const newPos = start - deleteCount;
    el.setSelectionRange(newPos, newPos);
  }
  el.focus();
  el.dispatchEvent(new Event('input', { bubbles: true }));
}

function kClear() {
  if (!currentActiveInput) return;
  currentActiveInput.value = '';
  currentActiveInput.focus();
  currentActiveInput.dispatchEvent(new Event('input', { bubbles: true }));
}

function setN(n) {
  currentN = n;
  [2,3,4,5].forEach(i => {
    document.getElementById('n-'+i).classList.toggle('active', i===n);
  });

  const activeVars = VAR_NAMES.slice(0, n).join(', ');
  const x0VarsEl = document.getElementById('x0-vars');
  if (x0VarsEl) x0VarsEl.textContent = activeVars;
  const x0Input = document.getElementById('x0-input');
  if (x0Input) {
    x0Input.placeholder = `напр. ${Array(n).fill('0.5').join('  ')} (для ${activeVars})`;
  }

  const container = document.getElementById('eq-container');
  container.innerHTML = '';
  for (let i = 0; i < n; i++) {
    const row = document.createElement('div');
    row.className = 'eq-row';
    row.innerHTML = `<span class="eq-label">F${i+1} =</span>
      <input type="text" id="eq-${i}" placeholder="выражение (переменные: ${activeVars})" 
             autocorrect="off" autocapitalize="off" spellcheck="false"
             onfocus="openKeypadFor(this, 'F${i+1}', ${i})"
             onclick="openKeypadFor(this, 'F${i+1}', ${i})"/>
      <button type="button" class="eq-kb-btn" onmousedown="event.preventDefault()" onclick="openKeypadFor(document.getElementById('eq-${i}'), 'F${i+1}', ${i})" title="Математическая клавиатура">⌨️</button>`;
    container.appendChild(row);
  }
}

function loadExample(idx) {
  const ex = EXAMPLES[idx];
  setN(ex.n);
  ex.eqs.forEach((eq, i) => {
    const el = document.getElementById('eq-'+i);
    if (el) el.value = eq;
  });
  document.getElementById('x0-input').value = ex.x0;
  hideResults();
}

// ── Решить ───────────────────────────────────────────────────────
async function solve() {
  const equations = [];
  for (let i = 0; i < currentN; i++) {
    const el = document.getElementById('eq-'+i);
    if (!el || !el.value.trim()) { alert(`Уравнение F${i+1} не заполнено!`); return; }
    equations.push(el.value.trim());
  }

  const x0raw = document.getElementById('x0-input').value.trim().split(/\s+/).map(Number);
  if (x0raw.some(isNaN) || x0raw.length !== currentN) {
    const activeVars = VAR_NAMES.slice(0, currentN).join(', ');
    alert(`Нужно ${currentN} начальных значений через пробел (для переменных: ${activeVars}).`); return;
  }

  const tol = parseFloat(document.getElementById('tol-input').value) || 1e-10;
  const max_iter = parseInt(document.getElementById('iter-input').value) || 100;

  const btn = document.getElementById('solve-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Вычисление...';
  hideResults();

  try {
    const resp = await fetch('/solve', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ equations, x0: x0raw, tol, max_iter })
    });
    const data = await resp.json();
    if (data.error_msg) { showError(data.error_msg); }
    else { showResults(data); }
  } catch(e) {
    showError('Ошибка соединения: ' + e.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = '▶ &nbsp;Решить';
  }
}

// ── Отображение результатов ──────────────────────────────────────
function showResults(data) {
  const info = data.info;
  const sol  = data.solution;

  // Статус
  const badge = document.getElementById('status-badge');
  badge.style.display = 'block';
  if (info.converged) {
    badge.className = 'success';
    badge.textContent = `✓ Сошлось за ${info.iterations} итераций  |  ‖F(x)‖ = ${info.residual.toExponential(3)}`;
  } else {
    badge.className = 'error';
    badge.textContent = `✗ ${info.error || 'Не сошлось'}  |  ‖F(x)‖ = ${info.residual.toExponential(3)}`;
  }

  // Решение
  let solText = sol.map((v,i) => `  ${getVarName(i)} = ${v.toPrecision(12)}`).join('\n');
  solText += `\n\n  ‖F(x)‖ = ${info.residual.toExponential(6)}`;
  solText += `\n  Итераций: ${info.iterations}`;
  solText += info.converged ? '\n  Статус: ✓ Сошлось' : '\n  Статус: ✗ Не сошлось';
  document.getElementById('solution-text').textContent = solText;
  document.getElementById('solution-box').style.display = 'block';

  // Таблица
  const tbody = document.getElementById('table-body');
  tbody.innerHTML = '';
  info.history.forEach((row, idx) => {
    const xStr = row.x.map((v,i) => `${getVarName(i)}=${v.toFixed(6)}`).join('  ');
    const step = row.step !== undefined ? row.step.toExponential(3) : '—';
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${row.iter}</td><td>${row.residual.toExponential(3)}</td><td>${step}</td><td style="font-family:monospace;font-size:11px;text-align:left;">${xStr}</td>`;
    tbody.appendChild(tr);
  });
  document.getElementById('table-box').style.display = 'block';

  // График (Canvas)
  drawChart(info.history);
}

function drawChart(history) {
  document.getElementById('chart-box').style.display = 'block';
  const canvas = document.getElementById('chart');
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.parentElement.clientWidth - 32;
  const H = Math.min(220, W * 0.5);
  canvas.width  = W * dpr;
  canvas.height = H * dpr;
  canvas.style.width  = W + 'px';
  canvas.style.height = H + 'px';

  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const PAD = { top: 20, right: 20, bottom: 36, left: 60 };
  const pw = W - PAD.left - PAD.right;
  const ph = H - PAD.top  - PAD.bottom;

  const residuals = history.map(h => h.residual).filter(r => r > 0);
  const iters     = history.map(h => h.iter);
  const logMin = Math.log10(Math.min(...residuals) * 0.5);
  const logMax = Math.log10(Math.max(...residuals) * 2);

  const xScale = i => PAD.left + (i / (history.length - 1 || 1)) * pw;
  const yScale = r => PAD.top + ph - ((Math.log10(Math.max(r, 1e-20)) - logMin) / (logMax - logMin)) * ph;

  // Background
  ctx.fillStyle = '#181825';
  ctx.beginPath();
  ctx.roundRect(0, 0, W, H, 10);
  ctx.fill();

  // Grid lines
  ctx.strokeStyle = '#45475a'; ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {
    const y = PAD.top + (ph / 4) * i;
    ctx.beginPath(); ctx.moveTo(PAD.left, y); ctx.lineTo(PAD.left + pw, y); ctx.stroke();
  }

  // Tol line
  const tol = parseFloat(document.getElementById('tol-input').value) || 1e-10;
  const tolY = yScale(tol);
  if (tolY > PAD.top && tolY < PAD.top + ph) {
    ctx.strokeStyle = '#fab387'; ctx.lineWidth = 1.2; ctx.setLineDash([5, 4]);
    ctx.beginPath(); ctx.moveTo(PAD.left, tolY); ctx.lineTo(PAD.left + pw, tolY); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#fab387'; ctx.font = '10px -apple-system';
    ctx.fillText('tol', PAD.left + pw - 22, tolY - 4);
  }

  // Fill under curve
  ctx.beginPath();
  history.forEach((h, i) => {
    const x = xScale(i), y = yScale(Math.max(h.residual, 1e-20));
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.lineTo(xScale(history.length - 1), PAD.top + ph);
  ctx.lineTo(xScale(0), PAD.top + ph);
  ctx.closePath();
  ctx.fillStyle = 'rgba(124,106,247,0.15)';
  ctx.fill();

  // Line
  ctx.beginPath();
  ctx.strokeStyle = '#7c6af7'; ctx.lineWidth = 2.2;
  history.forEach((h, i) => {
    const x = xScale(i), y = yScale(Math.max(h.residual, 1e-20));
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Dots
  history.forEach((h, i) => {
    const x = xScale(i), y = yScale(Math.max(h.residual, 1e-20));
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fillStyle = '#56cfb2'; ctx.fill();
    ctx.strokeStyle = '#7c6af7'; ctx.lineWidth = 1.5; ctx.stroke();
  });

  // X axis labels
  ctx.fillStyle = '#6c7086'; ctx.font = '10px -apple-system';
  ctx.textAlign = 'center';
  history.forEach((h, i) => {
    if (history.length <= 10 || i % Math.ceil(history.length / 8) === 0) {
      ctx.fillText(h.iter, xScale(i), H - PAD.bottom + 16);
    }
  });

  // Y axis labels
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const logVal = logMin + ((logMax - logMin) / 4) * (4 - i);
    ctx.fillText(`1e${Math.round(logVal)}`, PAD.left - 6, PAD.top + (ph / 4) * i + 4);
  }

  // Axis labels
  ctx.fillStyle = '#6c7086'; ctx.font = '11px -apple-system';
  ctx.textAlign = 'center';
  ctx.fillText('Итерация', PAD.left + pw / 2, H - 4);
}

// ── Утилиты ──────────────────────────────────────────────────────
function hideResults() {
  document.getElementById('status-badge').style.display  = 'none';
  document.getElementById('solution-box').style.display  = 'none';
  document.getElementById('table-box').style.display     = 'none';
  document.getElementById('chart-box').style.display     = 'none';
}

function showError(msg) {
  const badge = document.getElementById('status-badge');
  badge.style.display = 'block';
  badge.className = 'error';
  badge.textContent = '✗ ' + msg;
}

function copySolution() {
  const text = document.getElementById('solution-text').textContent;
  if (navigator.clipboard) {
    navigator.clipboard.writeText(text).then(() => alert('Скопировано!'));
  } else {
    const ta = document.createElement('textarea');
    ta.value = text; document.body.appendChild(ta);
    ta.select(); document.execCommand('copy');
    document.body.removeChild(ta); alert('Скопировано!');
  }
}
</script>
</body>
</html>
"""


# ─── API маршруты ─────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)


VAR_NAMES = ["x", "y", "z", "w", "u", "v"]


def preprocess_equation(expr):
    expr = expr.strip()
    if not expr:
        return expr
    # Если передано равенство вида 'f(...) = g(...)', приводим к '(f(...)) - (g(...))'
    if "=" in expr:
        parts = expr.split("=", 1)
        expr = f"({parts[0]}) - ({parts[1]})"
    # Степень: заменяем ^ на **
    expr = expr.replace("^", "**")
    # Обратная совместимость с форматом x[0], x[1]...
    legacy_map = {
        "x[0]": "x", "x[1]": "y", "x[2]": "z",
        "x[3]": "w", "x[4]": "u", "x[5]": "v"
    }
    for old_v, new_v in legacy_map.items():
        expr = expr.replace(old_v, new_v)
    return expr


@app.route("/solve", methods=["POST"])
def solve():
    data = request.get_json()
    equations = data.get("equations", [])
    x0        = data.get("x0", [])
    tol       = float(data.get("tol", 1e-10))
    max_iter  = int(data.get("max_iter", 100))
    n         = len(equations)

    clean_equations = [preprocess_equation(eq) for eq in equations]

    def F(x_vec):
        result = np.zeros(n)
        scope = {
            "np": np,
            "sin": np.sin, "cos": np.cos, "tan": np.tan,
            "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
            "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
            "exp": np.exp, "log": np.log, "log10": np.log10,
            "sqrt": np.sqrt, "abs": np.abs,
            "pi": np.pi, "e": np.e,
            "__builtins__": {}
        }
        for idx, val in enumerate(x_vec):
            if idx < len(VAR_NAMES):
                scope[VAR_NAMES[idx]] = float(val)
            scope[f"x{idx+1}"] = float(val)  # поддержка x1, x2, ...

        for i, expr in enumerate(clean_equations):
            try:
                result[i] = eval(expr, scope)
            except Exception as e:
                raise ValueError(f"Ошибка в уравнении F{i+1} ('{equations[i]}'): {e}")
        return result

    try:
        solution, info = newton_method(F, x0, tol=tol, max_iter=max_iter)
        return jsonify({"solution": solution, "info": info})
    except Exception as ex:
        return jsonify({"error_msg": str(ex)}), 400


# ─── Запуск ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import socket

    # Определяем локальный IP-адрес
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
    except Exception:
        local_ip = "127.0.0.1"
    finally:
        s.close()

    import os
    PORT = int(os.environ.get("PORT", 5050))
    print("\n" + "═" * 55)
    print("  🚀  Newton's Method — Web App запущен!")
    print("═" * 55)
    print(f"\n  💻  На этом компьютере:")
    print(f"      http://localhost:{PORT}")
    print(f"\n  📱  На iPhone (та же Wi-Fi сеть):")
    print(f"      http://{local_ip}:{PORT}")
    print(f"\n  ➡  Совет: в Safari на iPhone нажми")
    print(f"     «Поделиться» → «На экран Домой» —")
    print(f"     и получишь иконку как у настоящего приложения!")
    print("\n  Для остановки нажмите Ctrl+C")
    print("═" * 55 + "\n")

    app.run(host="0.0.0.0", port=PORT, debug=False)
