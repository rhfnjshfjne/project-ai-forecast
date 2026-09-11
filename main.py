import tkinter as tk
from tkinter import ttk, messagebox
import random
import math
import json
import os
from datetime import datetime


DATA_FILE = "projects.json"


DEFAULT_PROJECTS = [
    [50, 120000, 2, 300000, 25, 15],
    [60, 140000, 2, 340000, 28, 18],
    [70, 160000, 3, 380000, 31, 22],
    [80, 180000, 3, 430000, 35, 25],
    [90, 200000, 4, 470000, 39, 28],
    [100, 230000, 4, 550000, 42, 32],
    [110, 250000, 5, 610000, 46, 37],
    [120, 270000, 5, 670000, 50, 42],
    [140, 310000, 6, 760000, 56, 48],
    [150, 330000, 6, 820000, 60, 52],
    [160, 350000, 7, 880000, 65, 57],
    [180, 390000, 7, 980000, 72, 63],
    [200, 440000, 8, 1100000, 82, 70],
    [220, 490000, 8, 1200000, 90, 76],
    [250, 550000, 9, 1350000, 100, 84],
    [280, 620000, 9, 1500000, 112, 90],
    [300, 680000, 10, 1650000, 125, 96]
]


def load_projects():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as file:
                data = json.load(file)
                if isinstance(data, list) and len(data) >= 5:
                    return data
        except:
            pass

    return DEFAULT_PROJECTS.copy()


def save_projects(data):
    with open(DATA_FILE, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def sigmoid(x):
    if x < -60:
        return 0.0
    if x > 60:
        return 1.0
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def normalize(value, minimum, maximum):
    if maximum == minimum:
        return 0.5
    return (value - minimum) / (maximum - minimum)


def denormalize(value, minimum, maximum):
    return value * (maximum - minimum) + minimum


class NeuralNetwork:

    def __init__(self, input_size=3, hidden1=10, hidden2=8, output_size=3):

        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.output_size = output_size

        self.w1 = [
            [random.uniform(-1, 1) for _ in range(hidden1)]
            for _ in range(input_size)
        ]

        self.w2 = [
            [random.uniform(-1, 1) for _ in range(hidden2)]
            for _ in range(hidden1)
        ]

        self.w3 = [
            [random.uniform(-1, 1) for _ in range(output_size)]
            for _ in range(hidden2)
        ]

        self.b1 = [random.uniform(-1, 1) for _ in range(hidden1)]
        self.b2 = [random.uniform(-1, 1) for _ in range(hidden2)]
        self.b3 = [random.uniform(-1, 1) for _ in range(output_size)]

    def forward(self, x):

        h1 = []

        for j in range(self.hidden1):
            total = self.b1[j]

            for i in range(self.input_size):
                total += x[i] * self.w1[i][j]

            h1.append(sigmoid(total))

        h2 = []

        for j in range(self.hidden2):
            total = self.b2[j]

            for i in range(self.hidden1):
                total += h1[i] * self.w2[i][j]

            h2.append(sigmoid(total))

        output = []

        for j in range(self.output_size):
            total = self.b3[j]

            for i in range(self.hidden2):
                total += h2[i] * self.w3[i][j]

            output.append(sigmoid(total))

        return h1, h2, output

    def train(self, x, target, learning_rate):

        h1, h2, output = self.forward(x)

        error_output = [
            target[i] - output[i]
            for i in range(self.output_size)
        ]

        gradient3 = [
            error_output[i] * sigmoid_derivative(output[i])
            for i in range(self.output_size)
        ]

        error_h2 = []

        for i in range(self.hidden2):
            total = 0

            for j in range(self.output_size):
                total += gradient3[j] * self.w3[i][j]

            error_h2.append(total)

        gradient2 = [
            error_h2[i] * sigmoid_derivative(h2[i])
            for i in range(self.hidden2)
        ]

        error_h1 = []

        for i in range(self.hidden1):
            total = 0

            for j in range(self.hidden2):
                total += gradient2[j] * self.w2[i][j]

            error_h1.append(total)

        gradient1 = [
            error_h1[i] * sigmoid_derivative(h1[i])
            for i in range(self.hidden1)
        ]

        for i in range(self.hidden2):
            for j in range(self.output_size):
                self.w3[i][j] += (
                    learning_rate *
                    gradient3[j] *
                    h2[i]
                )

        for j in range(self.output_size):
            self.b3[j] += learning_rate * gradient3[j]

        for i in range(self.hidden1):
            for j in range(self.hidden2):
                self.w2[i][j] += (
                    learning_rate *
                    gradient2[j] *
                    h1[i]
                )

        for j in range(self.hidden2):
            self.b2[j] += learning_rate * gradient2[j]

        for i in range(self.input_size):
            for j in range(self.hidden1):
                self.w1[i][j] += (
                    learning_rate *
                    gradient1[j] *
                    x[i]
                )

        for j in range(self.hidden1):
            self.b1[j] += learning_rate * gradient1[j]

        return sum(e * e for e in error_output) / len(error_output)


class ForecastApp:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "Project AI — прогнозирование стоимости и сроков"
        )

        self.root.geometry("1200x760")
        self.root.minsize(1050, 680)

        self.bg = "#111318"
        self.panel = "#191c23"
        self.panel2 = "#20242d"
        self.accent = "#6c63ff"
        self.accent2 = "#8178ff"
        self.text = "#f1f3f5"
        self.muted = "#9298a6"
        self.green = "#35c98a"
        self.yellow = "#f2c94c"
        self.red = "#eb5757"
        self.border = "#2b303b"

        self.root.configure(bg=self.bg)

        self.projects = load_projects()

        self.network = NeuralNetwork()

        self.trained = False
        self.training_error = 0
        self.test_mse = 0
        self.test_mae = 0
        self.test_r2 = 0

        self.prepare_data()
        self.create_styles()
        self.create_interface()

    def create_styles(self):

        style = ttk.Style()

        style.theme_use("clam")

        style.configure(
            "TNotebook",
            background=self.bg,
            borderwidth=0
        )

        style.configure(
            "TNotebook.Tab",
            background=self.panel,
            foreground=self.muted,
            padding=(18, 10),
            font=("Segoe UI", 10)
        )

        style.map(
            "TNotebook.Tab",
            background=[
                ("selected", self.accent)
            ],
            foreground=[
                ("selected", "white")
            ]
        )

        style.configure(
            "Treeview",
            background=self.panel,
            foreground=self.text,
            fieldbackground=self.panel,
            rowheight=34,
            borderwidth=0,
            font=("Segoe UI", 10)
        )

        style.configure(
            "Treeview.Heading",
            background=self.panel2,
            foreground=self.text,
            font=("Segoe UI Semibold", 10),
            padding=8
        )

        style.map(
            "Treeview",
            background=[
                ("selected", self.accent)
            ]
        )

        style.configure(
            "TEntry",
            fieldbackground=self.panel2,
            foreground=self.text,
            borderwidth=0,
            padding=10
        )

        style.configure(
            "TButton",
            background=self.accent,
            foreground="white",
            padding=(14, 9),
            borderwidth=0,
            font=("Segoe UI Semibold", 10)
        )

        style.map(
            "TButton",
            background=[
                ("active", self.accent2)
            ]
        )

    def prepare_data(self):

        self.min_values = []
        self.max_values = []

        for column in range(6):
            values = [p[column] for p in self.projects]
            self.min_values.append(min(values))
            self.max_values.append(max(values))

        self.inputs = []
        self.outputs = []

        for p in self.projects:

            x = [
                normalize(
                    p[0],
                    self.min_values[0],
                    self.max_values[0]
                ),
                normalize(
                    p[1],
                    self.min_values[1],
                    self.max_values[1]
                ),
                normalize(
                    p[2],
                    self.min_values[2],
                    self.max_values[2]
                )
            ]

            y = [
                normalize(
                    p[3],
                    self.min_values[3],
                    self.max_values[3]
                ),
                normalize(
                    p[4],
                    self.min_values[4],
                    self.max_values[4]
                ),
                normalize(
                    p[5],
                    self.min_values[5],
                    self.max_values[5]
                )
            ]

            self.inputs.append(x)
            self.outputs.append(y)

    def create_interface(self):

        main = tk.Frame(
            self.root,
            bg=self.bg
        )

        main.pack(
            fill="both",
            expand=True
        )

        self.create_sidebar(main)

        content = tk.Frame(
            main,
            bg=self.bg
        )

        content.pack(
            side="left",
            fill="both",
            expand=True
        )

        self.create_header(content)

        self.notebook = ttk.Notebook(content)

        self.notebook.pack(
            fill="both",
            expand=True,
            padx=20,
            pady=(0, 20)
        )

        self.create_forecast_tab()
        self.create_history_tab()
        self.create_network_tab()

    def create_sidebar(self, parent):

        sidebar = tk.Frame(
            parent,
            bg=self.panel,
            width=220
        )

        sidebar.pack(
            side="left",
            fill="y"
        )

        sidebar.pack_propagate(False)

        logo = tk.Label(
            sidebar,
            text="PROJECT AI",
            bg=self.panel,
            fg="white",
            font=("Segoe UI", 20, "bold")
        )

        logo.pack(
            pady=(30, 5)
        )

        subtitle = tk.Label(
            sidebar,
            text="Система прогнозирования",
            bg=self.panel,
            fg=self.muted,
            font=("Segoe UI", 9)
        )

        subtitle.pack(
            pady=(0, 35)
        )

        self.sidebar_button(
            sidebar,
            "¦   Прогноз",
            lambda: self.notebook.select(0)
        )

        self.sidebar_button(
            sidebar,
            "?   История",
            lambda: self.notebook.select(1)
        )

        self.sidebar_button(
            sidebar,
            "?   Нейросеть",
            lambda: self.notebook.select(2)
        )

        separator = tk.Frame(
            sidebar,
            bg=self.border,
            height=1
        )

        separator.pack(
            fill="x",
            padx=25,
            pady=30
        )

        info = tk.Label(
            sidebar,
            text="AI FORECAST\n\n3 входных параметра\n2 скрытых слоя\n3 результата\n\nЧистый Python",
            bg=self.panel,
            fg=self.muted,
            justify="left",
            font=("Segoe UI", 9),
            padx=25
        )

        info.pack(
            anchor="w"
        )

        version = tk.Label(
            sidebar,
            text="v1.0",
            bg=self.panel,
            fg=self.muted,
            font=("Segoe UI", 8)
        )

        version.pack(
            side="bottom",
            pady=20
        )

    def sidebar_button(self, parent, text, command):

        button = tk.Button(
            parent,
            text=text,
            command=command,
            bg=self.panel,
            fg=self.text,
            activebackground=self.panel2,
            activeforeground="white",
            relief="flat",
            bd=0,
            anchor="w",
            padx=25,
            pady=12,
            font=("Segoe UI", 10)
        )

        button.pack(
            fill="x",
            padx=10,
            pady=2
        )

    def create_header(self, parent):

        header = tk.Frame(
            parent,
            bg=self.bg,
            height=90
        )

        header.pack(
            fill="x",
            padx=25
        )

        title = tk.Label(
            header,
            text="Прогнозирование проекта",
            bg=self.bg,
            fg=self.text,
            font=("Segoe UI", 23, "bold")
        )

        title.pack(
            side="left",
            pady=25
        )

        self.status_label = tk.Label(
            header,
            text="? Модель не обучена",
            bg=self.bg,
            fg=self.yellow,
            font=("Segoe UI", 10)
        )

        self.status_label.pack(
            side="right",
            pady=30
        )

    def create_forecast_tab(self):

        tab = tk.Frame(
            self.notebook,
            bg=self.bg
        )

        self.notebook.add(
            tab,
            text="  Прогноз  "
        )

        left = tk.Frame(
            tab,
            bg=self.panel,
            width=400
        )

        left.pack(
            side="left",
            fill="y",
            padx=(10, 10),
            pady=10
        )

        left.pack_propagate(False)

        title = tk.Label(
            left,
            text="Параметры проекта",
            bg=self.panel,
            fg=self.text,
            font=("Segoe UI", 15, "bold")
        )

        title.pack(
            anchor="w",
            padx=25,
            pady=(25, 5)
        )

        desc = tk.Label(
            left,
            text="Введите характеристики нового проекта",
            bg=self.panel,
            fg=self.muted,
            font=("Segoe UI", 9)
        )

        desc.pack(
            anchor="w",
            padx=25,
            pady=(0, 25)
        )

        self.area_entry = self.create_input(
            left,
            "Площадь проекта",
            "м?"
        )

        self.materials_entry = self.create_input(
            left,
            "Стоимость материалов",
            "руб."
        )

        self.complexity_entry = self.create_input(
            left,
            "Сложность проекта",
            "1 — 10"
        )

        button_frame = tk.Frame(
            left,
            bg=self.panel
        )

        button_frame.pack(
            fill="x",
            padx=25,
            pady=20
        )

        predict_button = tk.Button(
            button_frame,
            text="ПОЛУЧИТЬ ПРОГНОЗ",
            command=self.predict,
            bg=self.accent,
            fg="white",
            activebackground=self.accent2,
            activeforeground="white",
            relief="flat",
            bd=0,
            pady=12,
            font=("Segoe UI Semibold", 10)
        )

        predict_button.pack(
            fill="x"
        )

        random_button = tk.Button(
            button_frame,
            text="Случайный проект",
            command=self.random_project,
            bg=self.panel2,
            fg=self.text,
            activebackground=self.border,
            activeforeground="white",
            relief="flat",
            bd=0,
            pady=10,
            font=("Segoe UI", 9)
        )

        random_button.pack(
            fill="x",
            pady=(8, 0)
        )

        train_button = tk.Button(
            button_frame,
            text="Переобучить нейросеть",
            command=self.train_network,
            bg=self.panel2,
            fg=self.text,
            activebackground=self.border,
            activeforeground="white",
            relief="flat",
            bd=0,
            pady=10,
            font=("Segoe UI", 9)
        )

        train_button.pack(
            fill="x",
            pady=(8, 0)
        )

        right = tk.Frame(
            tab,
            bg=self.bg
        )

        right.pack(
            side="left",
            fill="both",
            expand=True,
            pady=10
        )

        result_title = tk.Label(
            right,
            text="Результаты",
            bg=self.bg,
            fg=self.text,
            font=("Segoe UI", 15, "bold")
        )

        result_title.pack(
            anchor="w",
            padx=15,
            pady=(5, 15)
        )

        cards = tk.Frame(
            right,
            bg=self.bg
        )

        cards.pack(
            fill="x",
            padx=5
        )

        self.cost_card = self.create_result_card(
            cards,
            "СТОИМОСТЬ",
            "—",
            "руб.",
            0
        )

        self.time_card = self.create_result_card(
            cards,
            "СРОК",
            "—",
            "дней",
            1
        )

        self.risk_card = self.create_result_card(
            cards,
            "РИСК",
            "—",
            "%",
            2
        )

        self.recommendation = tk.Frame(
            right,
            bg=self.panel
        )

        self.recommendation.pack(
            fill="x",
            padx=15,
            pady=20
        )

        self.recommendation_label = tk.Label(
            self.recommendation,
            text="Введите параметры проекта и запустите прогноз.",
            bg=self.panel,
            fg=self.muted,
            justify="left",
            font=("Segoe UI", 10),
            padx=20,
            pady=20
        )

        self.recommendation_label.pack(
            anchor="w"
        )

        self.chart_frame = tk.Frame(
            right,
            bg=self.panel
        )

        self.chart_frame.pack(
            fill="both",
            expand=True,
            padx=15,
            pady=(0, 10)
        )

        chart_title = tk.Label(
            self.chart_frame,
            text="Историческая динамика стоимости",
            bg=self.panel,
            fg=self.text,
            font=("Segoe UI Semibold", 10)
        )

        chart_title.pack(
            anchor="w",
            padx=15,
            pady=10
        )

        self.canvas = tk.Canvas(
            self.chart_frame,
            bg=self.panel,
            highlightthickness=0
        )

        self.canvas.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=5
        )

        self.canvas.bind(
            "<Configure>",
            lambda event: self.draw_chart()
        )

        self.draw_chart()

    def create_input(self, parent, title, unit):

        frame = tk.Frame(
            parent,
            bg=self.panel
        )

        frame.pack(
            fill="x",
            padx=25,
            pady=8
        )

        label = tk.Label(
            frame,
            text=title,
            bg=self.panel,
            fg=self.text,
            font=("Segoe UI", 9)
        )

        label.pack(
            anchor="w"
        )

        row = tk.Frame(
            frame,
            bg=self.panel
        )

        row.pack(
            fill="x",
            pady=(5, 0)
        )

        entry = ttk.Entry(
            row,
            font=("Segoe UI", 11)
        )

        entry.pack(
            side="left",
            fill="x",
            expand=True
        )

        unit_label = tk.Label(
            row,
            text=unit,
            bg=self.panel,
            fg=self.muted,
            font=("Segoe UI", 9)
        )

        unit_label.pack(
            side="right",
            padx=(8, 0)
        )

        return entry

    def create_result_card(self, parent, title, value, unit, column):

        card = tk.Frame(
            parent,
            bg=self.panel,
            height=120
        )

        card.grid(
            row=0,
            column=column,
            sticky="nsew",
            padx=5
        )

        parent.grid_columnconfigure(
            column,
            weight=1
        )

        title_label = tk.Label(
            card,
            text=title,
            bg=self.panel,
            fg=self.muted,
            font=("Segoe UI", 8, "bold")
        )

        title_label.pack(
            anchor="w",
            padx=18,
            pady=(18, 3)
        )

        value_label = tk.Label(
            card,
            text=value,
            bg=self.panel,
            fg=self.text,
            font=("Segoe UI", 19, "bold")
        )

        value_label.pack(
            anchor="w",
            padx=18
        )

        unit_label = tk.Label(
            card,
            text=unit,
            bg=self.panel,
            fg=self.muted,
            font=("Segoe UI", 8)
        )

        unit_label.pack(
            anchor="w",
            padx=18
        )

        return value_label

    def create_history_tab(self):

        tab = tk.Frame(
            self.notebook,
            bg=self.bg
        )

        self.notebook.add(
            tab,
            text="  История проектов  "
        )

        top = tk.Frame(
            tab,
            bg=self.bg
        )

        top.pack(
            fill="x",
            padx=15,
            pady=15
        )

        title = tk.Label(
            top,
            text="Исторические проекты",
            bg=self.bg,
            fg=self.text,
            font=("Segoe UI", 16, "bold")
        )

        title.pack(
            side="left"
        )

        add_button = tk.Button(
            top,
            text="+ Добавить проект",
            command=self.add_project,
            bg=self.accent,
            fg="white",
            activebackground=self.accent2,
            relief="flat",
            bd=0,
            padx=15,
            pady=8,
            font=("Segoe UI Semibold", 9)
        )

        add_button.pack(
            side="right"
        )

        delete_button = tk.Button(
            top,
            text="Удалить",
            command=self.delete_project,
            bg=self.panel2,
            fg=self.text,
            activebackground=self.border,
            relief="flat",
            bd=0,
            padx=15,
            pady=8,
            font=("Segoe UI", 9)
        )

        delete_button.pack(
            side="right",
            padx=8
        )

        table_frame = tk.Frame(
            tab,
            bg=self.panel
        )

        table_frame.pack(
            fill="both",
            expand=True,
            padx=15,
            pady=(0, 15)
        )

        columns = (
            "area",
            "materials",
            "complexity",
            "cost",
            "time",
            "risk"
        )

        self.tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings"
        )

        headings = {
            "area": "Площадь, м?",
            "materials": "Материалы, руб.",
            "complexity": "Сложность",
            "cost": "Стоимость, руб.",
            "time": "Срок, дней",
            "risk": "Риск, %"
        }

        for column in columns:

            self.tree.heading(
                column,
                text=headings[column]
            )

            self.tree.column(
                column,
                anchor="center",
                width=140
            )

        scrollbar = ttk.Scrollbar(
            table_frame,
            orient="vertical",
            command=self.tree.yview
        )

        self.tree.configure(
            yscrollcommand=scrollbar.set
        )

        self.tree.pack(
            side="left",
            fill="both",
            expand=True
        )

        scrollbar.pack(
            side="right",
            fill="y"
        )

        self.refresh_history()

    def create_network_tab(self):

        tab = tk.Frame(
            self.notebook,
            bg=self.bg
        )

        self.notebook.add(
            tab,
            text="  Нейросеть  "
        )

        title = tk.Label(
            tab,
            text="Архитектура и качество модели",
            bg=self.bg,
            fg=self.text,
            font=("Segoe UI", 18, "bold")
        )

        title.pack(
            anchor="w",
            padx=20,
            pady=(20, 5)
        )

        subtitle = tk.Label(
            tab,
            text="Полносвязная нейронная сеть для многопараметрического прогнозирования",
            bg=self.bg,
            fg=self.muted,
            font=("Segoe UI", 9)
        )

        subtitle.pack(
            anchor="w",
            padx=20,
            pady=(0, 20)
        )

        architecture = tk.Frame(
            tab,
            bg=self.panel
        )

        architecture.pack(
            fill="x",
            padx=20
        )

        self.layer_box(
            architecture,
            "ВХОД",
            "3",
            "Площадь\nМатериалы\nСложность",
            0
        )

        self.arrow(
            architecture,
            1
        )

        self.layer_box(
            architecture,
            "СКРЫТЫЙ",
            "10",
            "Sigmoid\nнейронов",
            2
        )

        self.arrow(
            architecture,
            3
        )

        self.layer_box(
            architecture,
            "СКРЫТЫЙ",
            "8",
            "Sigmoid\nнейронов",
            4
        )

        self.arrow(
            architecture,
            5
        )

        self.layer_box(
            architecture,
            "ВЫХОД",
            "3",
            "Стоимость\nСрок\nРиск",
            6
        )

        metrics = tk.Frame(
            tab,
            bg=self.bg
        )

        metrics.pack(
            fill="x",
            padx=20,
            pady=20
        )

        self.metric_label(
            metrics,
            "ЭПОХИ",
            "10 000",
            0
        )

        self.metric_label(
            metrics,
            "СКРЫТЫХ СЛОЯ",
            "2",
            1
        )

        self.metric_label(
            metrics,
            "НЕЙРОНОВ",
            "21",
            2
        )

        self.metric_label(
            metrics,
            "MSE",
            "—",
            3
        )

        self.metric_label(
            metrics,
            "MAE",
            "—",
            4
        )

        self.metric_label(
            metrics,
            "R?",
            "—",
            5
        )

        info = tk.Frame(
            tab,
            bg=self.panel
        )

        info.pack(
            fill="both",
            expand=True,
            padx=20,
            pady=(0, 20)
        )

        text = (
            "Модель обучается методом обратного распространения ошибки.\n\n"
            "Входные параметры:\n"
            "• площадь проекта\n"
            "• стоимость материалов\n"
            "• сложность проекта\n\n"
            "Нейросеть одновременно прогнозирует три значения:\n"
            "• итоговую стоимость\n"
            "• длительность проекта\n"
            "• уровень риска\n\n"
            "Для обучения используются исторические проекты. "
            "Часть данных используется для проверки качества модели."
        )

        info_label = tk.Label(
            info,
            text=text,
            bg=self.panel,
            fg=self.muted,
            justify="left",
            anchor="nw",
            font=("Segoe UI", 10),
            padx=25,
            pady=25
        )

        info_label.pack(
            fill="both",
            expand=True
        )

    def layer_box(self, parent, title, number, description, column):

        box = tk.Frame(
            parent,
            bg=self.panel2,
            width=150,
            height=120
        )

        box.grid(
            row=0,
            column=column,
            padx=10,
            pady=20
        )

        box.grid_propagate(False)

        tk.Label(
            box,
            text=title,
            bg=self.panel2,
            fg=self.muted,
            font=("Segoe UI", 8, "bold")
        ).pack(
            pady=(12, 0)
        )

        tk.Label(
            box,
            text=number,
            bg=self.panel2,
            fg=self.accent2,
            font=("Segoe UI", 25, "bold")
        ).pack()

        tk.Label(
            box,
            text=description,
            bg=self.panel2,
            fg=self.text,
            justify="center",
            font=("Segoe UI", 8)
        ).pack()

    def arrow(self, parent, column):

        tk.Label(
            parent,
            text=">",
            bg=self.panel,
            fg=self.accent2,
            font=("Segoe UI", 24)
        ).grid(
            row=0,
            column=column,
            pady=40
        )

    def metric_label(self, parent, title, value, column):

        frame = tk.Frame(
            parent,
            bg=self.panel
        )

        frame.grid(
            row=0,
            column=column,
            sticky="nsew",
            padx=4
        )

        parent.grid_columnconfigure(
            column,
            weight=1
        )

        tk.Label(
            frame,
            text=title,
            bg=self.panel,
            fg=self.muted,
            font=("Segoe UI", 8, "bold")
        ).pack(
            pady=(12, 3)
        )

        label = tk.Label(
            frame,
            text=value,
            bg=self.panel,
            fg=self.text,
            font=("Segoe UI", 14, "bold")
        )

        label.pack(
            pady=(0, 12)
        )

        if title == "MSE":
            self.mse_label = label

        elif title == "MAE":
            self.mae_label = label

        elif title == "R?":
            self.r2_label = label

    def train_network(self):

        if len(self.projects) < 5:
            messagebox.showwarning(
                "Недостаточно данных",
                "Для обучения необходимо минимум 5 проектов."
            )
            return

        self.network = NeuralNetwork()

        indices = list(range(len(self.inputs)))
        random.seed(42)
        random.shuffle(indices)

        split = max(
            3,
            int(len(indices) * 0.8)
        )

        train_indices = indices[:split]
        test_indices = indices[split:]

        learning_rate = 0.5
        epochs = 10000

        for epoch in range(epochs):

            total_error = 0

            random.shuffle(train_indices)

            for index in train_indices:

                total_error += self.network.train(
                    self.inputs[index],
                    self.outputs[index],
                    learning_rate
                )

            self.training_error = (
                total_error / len(train_indices)
            )

        predictions = []

        for index in test_indices:

            _, _, prediction = self.network.forward(
                self.inputs[index]
            )

            predictions.append(
                (
                    self.outputs[index],
                    prediction
                )
            )

        mse = 0
        mae = 0

        actual_flat = []
        predicted_flat = []

        for actual, prediction in predictions:

            for i in range(3):

                error = actual[i] - prediction[i]

                mse += error ** 2
                mae += abs(error)

                actual_flat.append(actual[i])
                predicted_flat.append(prediction[i])

        count = len(actual_flat)

        if count > 0:

            mse /= count
            mae /= count

            actual_mean = (
                sum(actual_flat) / count
            )

            ss_total = sum(
                (v - actual_mean) ** 2
                for v in actual_flat
            )

            ss_error = sum(
                (actual_flat[i] - predicted_flat[i]) ** 2
                for i in range(count)
            )

            if ss_total != 0:

                r2 = 1 - ss_error / ss_total

            else:

                r2 = 0

        else:

            mse = 0
            mae = 0
            r2 = 0

        self.test_mse = mse
        self.test_mae = mae
        self.test_r2 = r2

        self.trained = True

        self.status_label.config(
            text="? Модель обучена",
            fg=self.green
        )

        self.mse_label.config(
            text=f"{mse:.5f}"
        )

        self.mae_label.config(
            text=f"{mae:.5f}"
        )

        self.r2_label.config(
            text=f"{r2:.3f}"
        )

        messagebox.showinfo(
            "Обучение завершено",
            f"Нейросеть успешно обучена.\n\n"
            f"Эпох: {epochs}\n"
            f"MSE: {mse:.5f}\n"
            f"MAE: {mae:.5f}\n"
            f"R?: {r2:.3f}"
        )

    def predict(self):

        if not self.trained:

            self.train_network()

        try:

            area = float(
                self.area_entry.get()
            )

            materials = float(
                self.materials_entry.get()
            )

            complexity = float(
                self.complexity_entry.get()
            )

        except ValueError:

            messagebox.showerror(
                "Ошибка",
                "Введите числовые значения."
            )

            return

        if area <= 0:

            messagebox.showerror(
                "Ошибка",
                "Площадь должна быть больше нуля."
            )

            return

        if materials < 0:

            messagebox.showerror(
                "Ошибка",
                "Стоимость материалов не может быть отрицательной."
            )

            return

        if not 1 <= complexity <= 10:

            messagebox.showerror(
                "Ошибка",
                "Сложность должна быть от 1 до 10."
            )

            return

        x = [
            normalize(
                area,
                self.min_values[0],
                self.max_values[0]
            ),
            normalize(
                materials,
                self.min_values[1],
                self.max_values[1]
            ),
            normalize(
                complexity,
                self.min_values[2],
                self.max_values[2]
            )
        ]

        _, _, prediction = self.network.forward(x)

        cost = denormalize(
            prediction[0],
            self.min_values[3],
            self.max_values[3]
        )

        time = denormalize(
            prediction[1],
            self.min_values[4],
            self.max_values[4]
        )

        risk = denormalize(
            prediction[2],
            self.min_values[5],
            self.max_values[5]
        )

        cost = max(0, cost)
        time = max(1, time)
        risk = max(0, min(100, risk))

        reserve_cost = cost * (
            1 + risk / 200
        )

        reserve_time = time * (
            1 + risk / 200
        )

        self.cost_card.config(
            text=f"{cost:,.0f}"
        )

        self.time_card.config(
            text=f"{time:.0f}"
        )

        self.risk_card.config(
            text=f"{risk:.1f}"
        )

        if risk < 30:

            risk_color = self.green
            level = "НИЗКИЙ РИСК"

            recommendation = (
                f"Проект имеет низкий уровень риска.\n\n"
                f"Рекомендуемый бюджет: {reserve_cost:,.0f} руб.\n"
                f"Рекомендуемый срок: {reserve_time:.0f} дней.\n\n"
                f"Дополнительный резерв может быть минимальным."
            )

        elif risk < 60:

            risk_color = self.yellow
            level = "СРЕДНИЙ РИСК"

            recommendation = (
                f"Проект имеет средний уровень риска.\n\n"
                f"Рекомендуемый бюджет: {reserve_cost:,.0f} руб.\n"
                f"Рекомендуемый срок: {reserve_time:.0f} дней.\n\n"
                f"Рекомендуется предусмотреть резерв бюджета и времени."
            )

        else:

            risk_color = self.red
            level = "ВЫСОКИЙ РИСК"

            recommendation = (
                f"Проект имеет высокий уровень риска.\n\n"
                f"Рекомендуемый бюджет: {reserve_cost:,.0f} руб.\n"
                f"Рекомендуемый срок: {reserve_time:.0f} дней.\n\n"
                f"Рекомендуется увеличить резерв и внимательно "
                f"контролировать выполнение проекта."
            )

        self.risk_card.config(
            fg=risk_color
        )

        self.recommendation_label.config(
            text=f"{level}\n\n{recommendation}",
            fg=risk_color
        )

    def random_project(self):

        area = random.randint(50, 300)
        materials = int(
            area * random.uniform(1900, 2400)
        )
        complexity = random.randint(1, 10)

        self.area_entry.delete(
            0,
            tk.END
        )

        self.materials_entry.delete(
            0,
            tk.END
        )

        self.complexity_entry.delete(
            0,
            tk.END
        )

        self.area_entry.insert(
            0,
            str(area)
        )

        self.materials_entry.insert(
            0,
            str(materials)
        )

        self.complexity_entry.insert(
            0,
            str(complexity)
        )

    def refresh_history(self):

        for item in self.tree.get_children():
            self.tree.delete(item)

        for project in self.projects:

            self.tree.insert(
                "",
                tk.END,
                values=(
                    f"{project[0]:.0f}",
                    f"{project[1]:,.0f}",
                    f"{project[2]:.0f}",
                    f"{project[3]:,.0f}",
                    f"{project[4]:.0f}",
                    f"{project[5]:.0f}"
                )
            )

        self.prepare_data()
        self.draw_chart()

    def add_project(self):

        window = tk.Toplevel(
            self.root
        )

        window.title(
            "Добавление проекта"
        )

        window.geometry(
            "400x430"
        )

        window.configure(
            bg=self.bg
        )

        window.resizable(
            False,
            False
        )

        tk.Label(
            window,
            text="Новый исторический проект",
            bg=self.bg,
            fg=self.text,
            font=("Segoe UI", 16, "bold")
        ).pack(
            pady=20
        )

        entries = []

        labels = [
            "Площадь, м?",
            "Материалы, руб.",
            "Сложность 1-10",
            "Фактическая стоимость, руб.",
            "Фактический срок, дней",
            "Фактический риск, %"
        ]

        for label_text in labels:

            frame = tk.Frame(
                window,
                bg=self.bg
            )

            frame.pack(
                fill="x",
                padx=30,
                pady=5
            )

            tk.Label(
                frame,
                text=label_text,
                bg=self.bg,
                fg=self.text,
                font=("Segoe UI", 9)
            ).pack(
                anchor="w"
            )

            entry = ttk.Entry(
                frame
            )

            entry.pack(
                fill="x"
            )

            entries.append(entry)

        def save():

            try:

                values = [
                    float(entry.get())
                    for entry in entries
                ]

                if not 1 <= values[2] <= 10:
                    raise ValueError

                if not 0 <= values[5] <= 100:
                    raise ValueError

                self.projects.append(
                    values
                )

                save_projects(
                    self.projects
                )

                self.refresh_history()

                self.trained = False

                self.status_label.config(
                    text="? Модель требует переобучения",
                    fg=self.yellow
                )

                window.destroy()

            except ValueError:

                messagebox.showerror(
                    "Ошибка",
                    "Проверьте введённые данные."
                )

        tk.Button(
            window,
            text="Добавить проект",
            command=save,
            bg=self.accent,
            fg="white",
            relief="flat",
            bd=0,
            pady=10
        ).pack(
            fill="x",
            padx=30,
            pady=20
        )

    def delete_project(self):

        selected = self.tree.selection()

        if not selected:
            messagebox.showwarning(
                "Удаление",
                "Выберите проект."
            )
            return

        index = self.tree.index(
            selected[0]
        )

        if len(self.projects) <= 5:

            messagebox.showwarning(
                "Удаление",
                "Нельзя оставить меньше 5 проектов."
            )

            return

        answer = messagebox.askyesno(
            "Удаление",
            "Удалить выбранный проект?"
        )

        if answer:

            self.projects.pop(index)

            save_projects(
                self.projects
            )

            self.refresh_history()

            self.trained = False

            self.status_label.config(
                text="? Модель требует переобучения",
                fg=self.yellow
            )

    def draw_chart(self):

        if not hasattr(self, "canvas"):
            return

        self.canvas.delete(
            "all"
        )

        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()

        if width < 100 or height < 100:
            return

        margin_left = 55
        margin_right = 20
        margin_top = 20
        margin_bottom = 35

        chart_width = (
            width
            - margin_left
            - margin_right
        )

        chart_height = (
            height
            - margin_top
            - margin_bottom
        )

        costs = [
            p[3]
            for p in self.projects
        ]

        if not costs:
            return

        maximum = max(costs)
        minimum = min(costs)

        if maximum == minimum:
            maximum += 1

        self.canvas.create_line(
            margin_left,
            margin_top,
            margin_left,
            margin_top + chart_height,
            fill=self.border
        )

        self.canvas.create_line(
            margin_left,
            margin_top + chart_height,
            width - margin_right,
            margin_top + chart_height,
            fill=self.border
        )

        points = []

        for i, cost in enumerate(costs):

            x = (
                margin_left
                + i * chart_width
                / max(1, len(costs) - 1)
            )

            y = (
                margin_top
                + chart_height
                - (
                    (cost - minimum)
                    / (maximum - minimum)
                    * chart_height
                )
            )

            points.append(
                (x, y)
            )

        if len(points) > 1:

            for i in range(
                len(points) - 1
            ):

                self.canvas.create_line(
                    points[i][0],
                    points[i][1],
                    points[i + 1][0],
                    points[i + 1][1],
                    fill=self.accent2,
                    width=2
                )

        for x, y in points:

            self.canvas.create_oval(
                x - 4,
                y - 4,
                x + 4,
                y + 4,
                fill=self.accent2,
                outline=""
            )

        self.canvas.create_text(
            10,
            margin_top,
            text=f"{maximum / 1000000:.1f}M",
            fill=self.muted,
            anchor="w",
            font=("Segoe UI", 8)
        )

        self.canvas.create_text(
            10,
            margin_top + chart_height,
            text=f"{minimum / 1000000:.1f}M",
            fill=self.muted,
            anchor="w",
            font=("Segoe UI", 8)
        )


def main():

    random.seed()

    root = tk.Tk()

    app = ForecastApp(
        root
    )

    root.mainloop()


if __name__ == "__main__":
    main()
