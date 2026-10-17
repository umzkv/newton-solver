"""
Newton's Method — Graphical User Interface
==========================================
GUI-приложение для решения систем нелинейных уравнений
методом Ньютона. Использует tkinter + matplotlib.
"""

import tkinter as tk
from tkinter import ttk, messagebox, font as tkfont
import numpy as np
import threading

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ─── Цветовая схема ───────────────────────────────────────────────
COLORS = {
    "bg":          "#1e1e2e",   # основной фон
    "surface":     "#2a2a3e",   # карточки / панели
    "surface2":    "#313145",   # вложенные панели
    "accent":      "#7c6af7",   # фиолетовый акцент
    "accent2":     "#56cfb2",   # бирюзовый акцент
    "success":     "#a6e3a1",   # зелёный — успех
    "error":       "#f38ba8",   # красный — ошибка
    "warn":        "#fab387",   # оранжевый
    "text":        "#cdd6f4",   # основной текст
    "text_muted":  "#6c7086",   # приглушённый текст
    "border":      "#45475a",   # рамки
    "row_even":    "#2a2a3e",
    "row_odd":     "#252535",
    "entry_bg":    "#181825",
    "button":      "#7c6af7",
    "button_hover":"#9580ff",
}

FONT_FAMILY = "Helvetica"  # will be updated after Tk root is created

EXAMPLES = [
    {
        "name": "🔵  Пример 1 — Окружность ∩ Парабола (2×2)",
        "equations": ["x**2 + y**2 - 4",
                      "x**2 - y - 1"],
        "x0": "1.0  0.5",
        "desc": "x² + y² − 4 = 0\nx² − y − 1 = 0",
    },
    {
        "name": "🟣  Пример 2 — Классическая 3×3 (Burden & Faires)",
        "equations": ["3*x - cos(y*z) - 0.5",
                      "x**2 - 81*(y+0.1)**2 + sin(z) + 1.06",
                      "exp(-x*y) + 20*z + (10*pi-3)/3"],
        "x0": "0.1  0.1  -0.1",
        "desc": "3x − cos(yz) − 0.5 = 0\nx² − 81(y+0.1)² + sin(z) + 1.06 = 0\ne^(−xy) + 20z + (10π−3)/3 = 0",
    },
    {
        "name": "🟢  Пример 3 — Тригонометрическая (2×2)",
        "equations": ["sin(x) + y - 1.2",
                      "x + cos(y) - 0.8"],
        "x0": "0.5  0.5",
        "desc": "sin(x) + y − 1.2 = 0\nx + cos(y) − 0.8 = 0",
    },
]


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
        res = np.linalg.norm(Fx)
        history.append((k + 1, x.copy(), res))
        J = numerical_jacobian(F, x)
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            return x, {"iterations": k, "residual": res,
                       "converged": False, "history": history,
                       "error": "Якобиан вырожден"}
        x = x + dx
        if np.linalg.norm(dx) < tol:
            final_res = np.linalg.norm(F(x))
            history.append((k + 2, x.copy(), final_res))
            return x, {"iterations": k + 1, "residual": final_res,
                       "converged": True, "history": history, "error": None}
    return x, {"iterations": max_iter, "residual": np.linalg.norm(F(x)),
                "converged": False, "history": history, "error": "Лимит итераций"}


# ─── Вспомогательные виджеты ──────────────────────────────────────

class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)
        self.tip = None

    def show(self, _=None):
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        lbl = tk.Label(self.tip, text=self.text, background="#45475a",
                       foreground=COLORS["text"], relief="flat", padx=8, pady=4,
                       font=(FONT_FAMILY, 10), justify="left", wraplength=300)
        lbl.pack()

    def hide(self, _=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


def styled_button(parent, text, command, color=None, **kwargs):
    bg = color or COLORS["button"]
    btn = tk.Button(parent, text=text, command=command,
                    bg=bg, fg="white", activebackground=COLORS["button_hover"],
                    activeforeground="white", relief="flat", bd=0,
                    padx=14, pady=7, cursor="hand2",
                    font=(FONT_FAMILY, 11, "bold"), **kwargs)
    btn.bind("<Enter>", lambda e: btn.config(bg=COLORS["button_hover"]))
    btn.bind("<Leave>", lambda e: btn.config(bg=bg))
    return btn


def section_label(parent, text):
    return tk.Label(parent, text=text, bg=COLORS["surface"],
                    fg=COLORS["accent"], font=(FONT_FAMILY, 11, "bold"))


# ─── Главное окно ─────────────────────────────────────────────────

class NewtonApp(tk.Tk):
    def __init__(self):
        super().__init__()
        # Update font family now that the Tk root exists
        global FONT_FAMILY
        families = tkfont.families()
        for candidate in ("SF Pro Display", "Helvetica Neue", "Segoe UI", "Arial", "Helvetica"):
            if candidate in families:
                FONT_FAMILY = candidate
                break

        self.title("Newton's Method — Системы нелинейных уравнений")
        self.configure(bg=COLORS["bg"])
        self.geometry("1200x800")
        self.minsize(900, 650)
        self._eq_entries = []
        self._x0_entry = None
        self._n_var = tk.IntVar(value=2)
        self._tol_var = tk.StringVar(value="1e-10")
        self._iter_var = tk.StringVar(value="100")
        self._status_var = tk.StringVar(value="Ожидание ввода...")
        self._build_ui()
        self._load_example(0)

    # ── Построение интерфейса ──────────────────────────────────────

    def _build_ui(self):
        # ── Заголовок ──
        hdr = tk.Frame(self, bg=COLORS["surface"], pady=14)
        hdr.pack(fill="x")
        tk.Label(hdr, text="Newton's Method", bg=COLORS["surface"],
                 fg=COLORS["accent"], font=(FONT_FAMILY, 20, "bold")).pack()
        tk.Label(hdr, text="Метод Ньютона для систем нелинейных уравнений",
                 bg=COLORS["surface"], fg=COLORS["text_muted"],
                 font=(FONT_FAMILY, 11)).pack()

        # ── Основная область: левая + правая панели ──
        main = tk.Frame(self, bg=COLORS["bg"])
        main.pack(fill="both", expand=True, padx=10, pady=8)
        main.columnconfigure(0, weight=2, minsize=340)
        main.columnconfigure(1, weight=3)
        main.rowconfigure(0, weight=1)

        self._build_left(main)
        self._build_right(main)

        # ── Статус-бар ──
        bar = tk.Frame(self, bg=COLORS["surface2"], pady=4)
        bar.pack(fill="x", side="bottom")
        self._status_lbl = tk.Label(bar, textvariable=self._status_var,
                                    bg=COLORS["surface2"], fg=COLORS["text_muted"],
                                    font=(FONT_FAMILY, 10), anchor="w", padx=10)
        self._status_lbl.pack(fill="x")

    # ── Левая панель (ввод) ────────────────────────────────────────

    def _build_left(self, parent):
        left = tk.Frame(parent, bg=COLORS["surface"], bd=0)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        # Примеры
        ex_frame = tk.Frame(left, bg=COLORS["surface"], padx=12, pady=10)
        ex_frame.pack(fill="x")
        section_label(ex_frame, "📚  Примеры").pack(anchor="w", pady=(0, 6))
        for i, ex in enumerate(EXAMPLES):
            btn = tk.Button(ex_frame, text=ex["name"],
                            command=lambda idx=i: self._load_example(idx),
                            bg=COLORS["surface2"], fg=COLORS["text"],
                            activebackground=COLORS["border"],
                            relief="flat", bd=0, padx=8, pady=5,
                            anchor="w", cursor="hand2",
                            font=(FONT_FAMILY, 10))
            btn.pack(fill="x", pady=2)
            btn.bind("<Enter>", lambda e, b=btn: b.config(bg=COLORS["border"]))
            btn.bind("<Leave>", lambda e, b=btn: b.config(bg=COLORS["surface2"]))

        ttk.Separator(left, orient="horizontal").pack(fill="x", padx=12, pady=6)

        # Количество уравнений
        n_frame = tk.Frame(left, bg=COLORS["surface"], padx=12)
        n_frame.pack(fill="x")
        section_label(n_frame, "🔢  Количество уравнений").pack(anchor="w", pady=(0, 6))
        n_row = tk.Frame(n_frame, bg=COLORS["surface"])
        n_row.pack(anchor="w")
        for n in (2, 3, 4, 5):
            rb = tk.Radiobutton(n_row, text=str(n), variable=self._n_var, value=n,
                                command=self._rebuild_eq_fields,
                                bg=COLORS["surface"], fg=COLORS["text"],
                                selectcolor=COLORS["surface2"],
                                activebackground=COLORS["surface"],
                                activeforeground=COLORS["accent"],
                                font=(FONT_FAMILY, 11))
            rb.pack(side="left", padx=6)

        ttk.Separator(left, orient="horizontal").pack(fill="x", padx=12, pady=6)

        # Уравнения
        self._eq_section = tk.Frame(left, bg=COLORS["surface"], padx=12)
        self._eq_section.pack(fill="x")
        section_label(self._eq_section, "📝  Уравнения  (F_i(x) = 0)").pack(anchor="w", pady=(0, 4))
        tk.Label(self._eq_section, text="Используйте: x[0], x[1], ... | np.sin, np.cos, np.exp, np.log, np.pi",
                 bg=COLORS["surface"], fg=COLORS["text_muted"],
                 font=(FONT_FAMILY, 9), wraplength=310, justify="left").pack(anchor="w", pady=(0, 6))
        self._eq_container = tk.Frame(self._eq_section, bg=COLORS["surface"])
        self._eq_container.pack(fill="x")

        ttk.Separator(left, orient="horizontal").pack(fill="x", padx=12, pady=6)

        # Начальное приближение
        x0_frame = tk.Frame(left, bg=COLORS["surface"], padx=12)
        x0_frame.pack(fill="x")
        row = tk.Frame(x0_frame, bg=COLORS["surface"])
        row.pack(fill="x")
        section_label(row, "🎯  Начальное приближение x₀").pack(side="left", anchor="w")
        ToolTip(row, "Введите числа через пробел, по одному на каждую переменную")
        self._x0_entry = tk.Entry(x0_frame, bg=COLORS["entry_bg"], fg=COLORS["text"],
                                  insertbackground=COLORS["text"], relief="flat",
                                  font=(FONT_FAMILY, 11), bd=0)
        self._x0_entry.pack(fill="x", pady=(4, 0), ipady=6, padx=2)
        self._add_entry_border(x0_frame, self._x0_entry)

        ttk.Separator(left, orient="horizontal").pack(fill="x", padx=12, pady=6)

        # Параметры
        params = tk.Frame(left, bg=COLORS["surface"], padx=12)
        params.pack(fill="x")
        section_label(params, "⚙️  Параметры").pack(anchor="w", pady=(0, 6))

        def param_row(parent, label_text, var, placeholder=""):
            row = tk.Frame(parent, bg=COLORS["surface"])
            row.pack(fill="x", pady=3)
            tk.Label(row, text=label_text, bg=COLORS["surface"],
                     fg=COLORS["text"], font=(FONT_FAMILY, 10),
                     width=16, anchor="w").pack(side="left")
            e = tk.Entry(row, textvariable=var, bg=COLORS["entry_bg"],
                         fg=COLORS["text"], insertbackground=COLORS["text"],
                         relief="flat", font=(FONT_FAMILY, 11), bd=0, width=14)
            e.pack(side="left", fill="x", expand=True, ipady=5)
            return e

        param_row(params, "Точность (tol):", self._tol_var)
        param_row(params, "Макс. итерации:", self._iter_var)



        # Кнопки
        btn_frame = tk.Frame(left, bg=COLORS["surface"], padx=12)
        btn_frame.pack(fill="x", pady=(0, 10))
        solve_btn = styled_button(btn_frame, "▶  Решить", self._solve)
        solve_btn.pack(fill="x", pady=(0, 6))
        clear_btn = styled_button(btn_frame, "🗑  Очистить", self._clear,
                                  color=COLORS["surface2"])
        clear_btn.pack(fill="x")

        self._rebuild_eq_fields()

    def _add_entry_border(self, parent, entry):
        """Рисует тонкую цветную рамку под полем ввода."""
        border = tk.Frame(parent, bg=COLORS["accent"], height=1)
        border.pack(fill="x", padx=2)
        entry.bind("<FocusIn>",  lambda e: border.config(bg=COLORS["accent2"]))
        entry.bind("<FocusOut>", lambda e: border.config(bg=COLORS["accent"]))

    def _rebuild_eq_fields(self):
        for w in self._eq_container.winfo_children():
            w.destroy()
        self._eq_entries = []
        n = self._n_var.get()
        for i in range(n):
            row = tk.Frame(self._eq_container, bg=COLORS["surface"])
            row.pack(fill="x", pady=3)
            tk.Label(row, text=f"F{i+1} =", bg=COLORS["surface"],
                     fg=COLORS["accent2"], font=(FONT_FAMILY, 11, "bold"),
                     width=4).pack(side="left")
            entry = tk.Entry(row, bg=COLORS["entry_bg"], fg=COLORS["text"],
                             insertbackground=COLORS["text"], relief="flat",
                             font=(FONT_FAMILY, 11), bd=0)
            entry.pack(side="left", fill="x", expand=True, ipady=6, padx=(4, 2))
            self._eq_entries.append(entry)
            sep = tk.Frame(row, bg=COLORS["accent"], height=1)

    # ── Правая панель (результаты + график) ───────────────────────

    def _build_right(self, parent):
        right = tk.Frame(parent, bg=COLORS["bg"])
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.rowconfigure(1, weight=2)
        right.columnconfigure(0, weight=1)

        # ── Таблица итераций ──
        tbl_frame = tk.Frame(right, bg=COLORS["surface"])
        tbl_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 6))

        hdr = tk.Frame(tbl_frame, bg=COLORS["surface"], padx=12, pady=8)
        hdr.pack(fill="x")
        section_label(hdr, "📊  Таблица итераций").pack(side="left")
        self._result_label = tk.Label(hdr, text="", bg=COLORS["surface"],
                                      fg=COLORS["success"], font=(FONT_FAMILY, 11, "bold"))
        self._result_label.pack(side="right")

        # Treeview
        cols = ("iter", "residual", "step", "x_values")
        self._tree = ttk.Treeview(tbl_frame, columns=cols, show="headings",
                                  height=8, selectmode="browse")
        col_cfg = [
            ("iter",     "Итер.",     60,  "center"),
            ("residual", "||F(x)||",  130, "center"),
            ("step",     "||Δx||",    130, "center"),
            ("x_values", "Значения x",  0, "w"),
        ]
        for cid, label, width, anchor in col_cfg:
            self._tree.heading(cid, text=label)
            self._tree.column(cid, width=width, anchor=anchor,
                              stretch=(cid == "x_values"))

        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview",
                        background=COLORS["row_even"],
                        foreground=COLORS["text"],
                        rowheight=26,
                        fieldbackground=COLORS["row_even"],
                        borderwidth=0,
                        font=(FONT_FAMILY, 10))
        style.configure("Treeview.Heading",
                        background=COLORS["surface2"],
                        foreground=COLORS["accent"],
                        relief="flat",
                        font=(FONT_FAMILY, 10, "bold"))
        style.map("Treeview", background=[("selected", COLORS["accent"])])
        self._tree.tag_configure("odd",  background=COLORS["row_odd"])
        self._tree.tag_configure("even", background=COLORS["row_even"])
        self._tree.tag_configure("last", background="#2d3f2d", foreground=COLORS["success"])

        vsb = ttk.Scrollbar(tbl_frame, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.pack(side="left", fill="both", expand=True, padx=(12, 0), pady=(0, 10))
        vsb.pack(side="right", fill="y", pady=(0, 10), padx=(0, 4))

        # Панель решения
        sol_frame = tk.Frame(right, bg=COLORS["surface2"], padx=14, pady=10)
        sol_frame.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        sol_hdr = tk.Frame(sol_frame, bg=COLORS["surface2"])
        sol_hdr.pack(fill="x")
        section_label(sol_hdr, "✅  Решение").pack(side="left")
        styled_button(sol_hdr, "📋 Копировать", self._copy_solution,
                      color=COLORS["surface"]).pack(side="right")
        self._solution_text = tk.Text(sol_frame, bg=COLORS["entry_bg"], fg=COLORS["text"],
                                      font=("Menlo" if "Menlo" in tkfont.families() else "Courier", 11),
                                      relief="flat", height=4, padx=10, pady=8,
                                      state="disabled", wrap="word",
                                      insertbackground=COLORS["text"])
        self._solution_text.pack(fill="x", pady=(6, 0))

        # ── График ──
        plot_frame = tk.Frame(right, bg=COLORS["surface"])
        plot_frame.grid(row=2, column=0, sticky="nsew", pady=(0, 0))
        right.rowconfigure(2, weight=3)

        plot_hdr = tk.Frame(plot_frame, bg=COLORS["surface"], padx=12, pady=6)
        plot_hdr.pack(fill="x")
        section_label(plot_hdr, "📈  График сходимости").pack(side="left")

        self._fig = Figure(figsize=(6, 3), facecolor=COLORS["surface"])
        self._ax = self._fig.add_subplot(111)
        self._style_axes()
        self._canvas = FigureCanvasTkAgg(self._fig, master=plot_frame)
        self._canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def _style_axes(self):
        ax = self._ax
        ax.set_facecolor(COLORS["bg"])
        ax.tick_params(colors=COLORS["text_muted"], labelsize=9)
        ax.xaxis.label.set_color(COLORS["text_muted"])
        ax.yaxis.label.set_color(COLORS["text_muted"])
        for spine in ax.spines.values():
            spine.set_edgecolor(COLORS["border"])
        ax.set_xlabel("Итерация")
        ax.set_ylabel("‖F(x)‖  (log)")
        ax.set_title("Норма невязки по итерациям", color=COLORS["text"], fontsize=11)
        self._fig.tight_layout(pad=1.5)

    # ── Загрузка примеров ─────────────────────────────────────────

    def _load_example(self, idx):
        ex = EXAMPLES[idx]
        n = len(ex["equations"])
        self._n_var.set(n)
        self._rebuild_eq_fields()
        for i, eq in enumerate(ex["equations"]):
            self._eq_entries[i].delete(0, "end")
            self._eq_entries[i].insert(0, eq)
        self._x0_entry.delete(0, "end")
        self._x0_entry.insert(0, ex["x0"])
        self._set_status(f"Загружен: {ex['name']}")

    # ── Решение ───────────────────────────────────────────────────

    def _solve(self):
        # Валидация
        n = self._n_var.get()
        equations = []
        for i, e in enumerate(self._eq_entries):
            expr = e.get().strip()
            if not expr:
                messagebox.showerror("Ошибка", f"Уравнение F{i+1} не заполнено!")
                return
            equations.append(expr)

        try:
            x0 = list(map(float, self._x0_entry.get().split()))
        except ValueError:
            messagebox.showerror("Ошибка", "Некорректное начальное приближение!\nВведите числа через пробел.")
            return
        if len(x0) != n:
            messagebox.showerror("Ошибка", f"Нужно {n} начальных значений, получено {len(x0)}.")
            return

        try:
            tol = float(self._tol_var.get())
            max_iter = int(self._iter_var.get())
        except ValueError:
            messagebox.showerror("Ошибка", "Некорректные параметры tol / max_iter.")
            return

        var_names = ["x", "y", "z", "w", "u", "v"]
        clean_equations = []
        for eq in equations:
            s = eq.strip()
            if "=" in s:
                parts = s.split("=", 1)
                s = f"({parts[0]}) - ({parts[1]})"
            s = s.replace("^", "**")
            for old_v, new_v in {"x[0]": "x", "x[1]": "y", "x[2]": "z", "x[3]": "w", "x[4]": "u"}.items():
                s = s.replace(old_v, new_v)
            clean_equations.append(s)

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
                if idx < len(var_names):
                    scope[var_names[idx]] = float(val)
                scope[f"x{idx+1}"] = float(val)

            for i, expr in enumerate(clean_equations):
                result[i] = eval(expr, scope)
            return result

        self._set_status("Вычисление...")
        self._result_label.config(text="")
        threading.Thread(target=self._run_solver, args=(F, x0, tol, max_iter),
                         daemon=True).start()

    def _run_solver(self, F, x0, tol, max_iter):
        try:
            solution, info = newton_method(F, x0, tol=tol, max_iter=max_iter)
        except Exception as exc:
            self.after(0, lambda: messagebox.showerror("Ошибка вычисления", str(exc)))
            self.after(0, lambda: self._set_status(f"Ошибка: {exc}"))
            return
        self.after(0, lambda: self._display_results(solution, info))

    def _display_results(self, solution, info):
        # Таблица
        self._tree.delete(*self._tree.get_children())
        history = info["history"]
        prev_res = None
        for idx, (it, xv, res) in enumerate(history):
            step = abs(res - prev_res) if prev_res is not None else 0.0
            prev_res = res
            x_str = "  ".join(f"x{j+1}={v:.8g}" for j, v in enumerate(xv))
            tag = "last" if idx == len(history) - 1 else ("odd" if idx % 2 else "even")
            self._tree.insert("", "end", values=(it, f"{res:.6e}", f"{step:.6e}", x_str),
                              tags=(tag,))

        # Статус
        converged = info["converged"]
        iters = info["iterations"]
        residual = info["residual"]
        if converged:
            status_text = f"✓ Сошлось за {iters} итераций  |  ‖F(x)‖ = {residual:.3e}"
            self._result_label.config(text=status_text, fg=COLORS["success"])
        else:
            err = info.get("error", "")
            status_text = f"✗ {err}  |  ‖F(x)‖ = {residual:.3e}"
            self._result_label.config(text=status_text, fg=COLORS["error"])
        self._set_status(status_text)

        # Панель решения
        sol_lines = [f"  x{i+1} = {v:.12g}" for i, v in enumerate(solution)]
        sol_lines.append(f"\n  ‖F(x)‖ = {residual:.6e}")
        sol_lines.append(f"  Итераций: {iters}  |  {'Сошлось ✓' if converged else 'Не сошлось ✗'}")
        sol_text = "\n".join(sol_lines)
        self._solution_text.config(state="normal")
        self._solution_text.delete("1.0", "end")
        self._solution_text.insert("end", sol_text)
        self._solution_text.config(state="disabled",
                                   fg=COLORS["success"] if converged else COLORS["error"])

        # График
        self._ax.cla()
        self._style_axes()
        iters_list = [h[0] for h in history]
        res_list   = [max(h[2], 1e-20) for h in history]
        self._ax.semilogy(iters_list, res_list,
                          color=COLORS["accent"], linewidth=2.2,
                          marker="o", markersize=6,
                          markerfacecolor=COLORS["accent2"],
                          markeredgecolor=COLORS["accent"])
        self._ax.fill_between(iters_list, res_list,
                              alpha=0.12, color=COLORS["accent"])
        self._ax.axhline(y=float(self._tol_var.get()),
                         color=COLORS["warn"], linestyle="--",
                         linewidth=1.2, label=f"tol = {self._tol_var.get()}")
        self._ax.legend(facecolor=COLORS["surface2"], edgecolor=COLORS["border"],
                        labelcolor=COLORS["text"], fontsize=9)
        self._ax.set_xticks(iters_list)
        self._fig.tight_layout(pad=1.5)
        self._canvas.draw()

    # ── Утилиты ───────────────────────────────────────────────────

    def _clear(self):
        for e in self._eq_entries:
            e.delete(0, "end")
        if self._x0_entry:
            self._x0_entry.delete(0, "end")
        self._tree.delete(*self._tree.get_children())
        self._result_label.config(text="")
        self._solution_text.config(state="normal")
        self._solution_text.delete("1.0", "end")
        self._solution_text.config(state="disabled")
        self._ax.cla()
        self._style_axes()
        self._canvas.draw()
        self._set_status("Очищено. Ожидание ввода...")

    def _copy_solution(self):
        text = self._solution_text.get("1.0", "end").strip()
        if text:
            self.clipboard_clear()
            self.clipboard_append(text)
            self._set_status("Скопировано в буфер обмена ✓")

    def _set_status(self, msg):
        self._status_var.set(f"  {msg}")


# ─── Запуск ───────────────────────────────────────────────────────

if __name__ == "__main__":
    app = NewtonApp()
    app.mainloop()
