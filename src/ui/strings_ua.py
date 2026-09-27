"""Ukrainian localization string table for the GRNN Forecasting UI.

Locked UI language is Ukrainian.
Established scientific abbreviations (GRNN, sMAPE, MASE, OWA, M4, CSV, ZIP, ID)
stay as-is.
"""

from types import SimpleNamespace

# ==============================================================================
# Main App Header & Meta
# ==============================================================================
APP_TITLE = "GRNN Forecasting Demo"
APP_HEADER = "Прогнозування часових рядів на основі ансамблів GRNN"
APP_SUBTITLE = (
    "Стійкі ансамблі нейронних мереж узагальненої регресії (GRNN) "
    "для прогнозування коротких часових рядів"
)
BADGE_THESIS = "Магістерська кваліфікаційна робота"
BADGE_BENCHMARK = "M4 Yearly Benchmark"

# ==============================================================================
# Operational Modes
# ==============================================================================
MODE_LABEL = "Режим роботи"
MODE_EVALUATION = "Оцінка (Evaluation)"
MODE_FORECAST = "Прогноз (Forecast)"
MODE_EVALUATION_HELP = (
    "Утримання останніх h точок кожного ряду як тестової вибірки, "
    "побудова прогнозу та розрахунок метрик якості (sMAPE, MASE, OWA, MAE, RMSE)."
)
MODE_FORECAST_HELP = (
    "Прогнозування майбутніх h точок уперед без розрахунку метрик "
    "(майбутні значення невідомі)."
)
MODE_M4_LOCKED_NOTE = "Для вбудованої вибірки M4 режим зафіксовано: Оцінка (офіційний тест із 6 точок)."

# ==============================================================================
# Ensemble & Modeling Parameters
# ==============================================================================
PARAMS_HEADER = "Параметри моделювання"

PARAM_LAGS_LABEL = "Кількість лагів (L)"
PARAM_LAGS_HELP = "Розмір ковзного вікна лагової затримки для формування матриці ознак."

PARAM_MODELS_LABEL = "Кількість моделей в ансамблі (M)"
PARAM_MODELS_HELP = "Кількість підмоделей чергувального розкладання (M=1: одинарна GRNN)."

PARAM_AGG_LABEL = "Метод агрегування"
PARAM_AGG_HELP = "Спосіб об'єднання виходів базових моделей ансамблю."
PARAM_AGG_MEDIAN = "Медіана (median)"
PARAM_AGG_MEAN = "Середнє (mean)"

PARAM_OPTIMIZER_LABEL = "Оптимізатор параметра розмиття (σ)"
PARAM_OPTIMIZER_HELP = "Метод оптимізації гіперпараметра σ у нормалізованому просторі ознак."
OPTIMIZER_LABELS = {
    "brent": "brent",
    "gradient": "gradient (L-BFGS-B)",
    "coarse2fine": "coarse2fine",
    "silverman": "silverman",
    "grid": "grid",
}
OPTIMIZER_TOOLTIPS = {
    "brent": "Метод Брента одновимірної мінімізації валідаційної помилки.",
    "gradient": "Градієнтна оптимізація L-BFGS-B за точним градієнтом MSE.",
    "coarse2fine": "Двоетапний грубий і точний пошук на логарифмічній сітці.",
    "silverman": "Емпіричне правило Сільвермана (rule-of-thumb, без мінімізації цільової функції).",
    "grid": "Вичерпний сітковий пошук.",
}

PARAM_HORIZON_LABEL = "Горизонт прогнозування (h)"
PARAM_HORIZON_HELP = "Кількість майбутніх точок для прогнозування."
PARAM_HORIZON_M4_LOCKED_NOTE = "Для M4 Yearly горизонт зафіксовано: h=6."

RESET_DEFAULTS_BUTTON = "Скинути до замовчувань"
RESET_DEFAULTS_HELP = "Відновити базову конфігурацію: L=3, M=6, агрегування=median, оптимізатор=brent, h=6."

DIFFERENCING_NOTE = (
    "Диференціювання (взяття перших різниць): завжди увімкнено для усунення "
    "нестаціонарності тренду (згідно з протоколом дослідження)."
)

# ==============================================================================
# Data Sources & Ingestion
# ==============================================================================
DATA_SOURCE_LABEL = "Джерело даних"
DATA_SOURCE_M4 = "Вбудовані дані M4 (Yearly)"
DATA_SOURCE_UPLOAD = "Завантаження власного файлу (CSV)"

# M4 source controls
M4_SCOPE_LABEL = "Обсяг вибірки M4"
M4_SCOPE_100 = "100 рядів"
M4_SCOPE_1000 = "1 000 рядів"
M4_SCOPE_ALL = "Всі 23 000 рядів"
M4_SCOPE_ALL_WARNING = "Увага: повний запуск на 23 000 рядах вимагає значного часу обчислень."

M4_DOMAIN_LABEL = "Предметна область (Domain)"
M4_DOMAIN_ALL = "Всі області"
M4_SEED_LABEL = "Випадкове зерно (Seed)"

# M4 missing download panel
M4_MISSING_TITLE = "Локальні файли M4 не знайдено"
M4_MISSING_MSG = "Файли M4 Yearly відсутні в каталозі `data/raw/`. Ви можете завантажити їх за офіційним посиланням:"
M4_URL_LABEL = "URL архіву M4"
M4_DOWNLOAD_BTN = "Завантажити дані M4"
M4_DOWNLOAD_SUCCESS = "Дані M4 успішно завантажено та перевірено."
M4_DOWNLOAD_ERROR = "Не вдалося завантажити дані за вказаною адресою. Перевірте URL та з'єднання."

# Custom CSV upload
UPLOAD_LABEL = "Оберіть CSV файл для завантаження"
UPLOAD_FORMATS_HINT = (
    "Підтримувані структури CSV:\n"
    "1. Окремий ряд: колонки `t,y` або одна колонка `y`;\n"
    "2. Порядковий (M4-стиль): один ряд на рядок із довільною довжиною;\n"
    "3. Поколонковий: один ряд на стовпець із заголовками."
)
DOWNLOAD_EXAMPLE_BTN = "Завантажити приклад файлу"
AMBIGUOUS_LAYOUT_TITLE = "Уточнення структури даних"
AMBIGUOUS_LAYOUT_MSG = "Розмірність завантаженого файлу неоднозначна. Будь ласка, вкажіть орієнтацію рядів:"
AMBIGUOUS_CHOICE_ROWS = "Кожен рядок — це окремий часовий ряд"
AMBIGUOUS_CHOICE_COLS = "Кожен стовпець — це окремий часовий ряд"

# Parsing validation errors
ERR_NON_NUMERIC = "Нечислове значення у клітинці (рядок {row}, стовпець {col}): '{val}'."
ERR_INTERNAL_NAN = "Пропущені значення (NaN) всередині ряду '{series_id}' неприпустимі (дозволені лише у хвості)."
ERR_SERIES_TOO_SHORT = "Ряд '{series_id}' містить лише {count} точок. Мінімальна довжина для парсингу — 3 точки."
ERR_CONSTANT_SERIES = "Ряд '{series_id}' є константним: знаменник MASE дорівнює нулю. В режимі Оцінки запуск заблоковано."
WARN_ZERO_VALUES = "Ряд '{series_id}' містить нульові значення, що може спричинити нестабільність sMAPE."

# ==============================================================================
# Data Sufficiency Validation
# ==============================================================================
SUFFICIENCY_LIVE_TEMPLATE = (
    "Мінімально необхідна довжина ряду: {required} · найкоротший ряд у даних: {actual} (ID: {series_id})"
)
SUFFICIENCY_OK = "Довжина всіх рядів достатня для поточної конфігурації параметрів."
SUFFICIENCY_FAIL = "Довжина найкоротшого ряду менша за мінімально необхідну."

DIALOG_INSUFFICIENCY_TITLE = "Недостатня довжина даних"
DIALOG_INSUFFICIENCY_BODY = (
    "Для конфігурації параметрів (L = {L}, M = {M}, h = {h}, режим = {mode}) "
    "мінімально необхідна довжина ряду становить **{required}** точок.\n\n"
    "Найкоротший ряд у вибірці — **{series_id}** — має довжину **{actual}** точок "
    "(дефіцит: **{deficit}** точок).\n\n"
    "**Шляхи вирішення:**\n"
    "1. Зменшіть кількість лагів (L) або кількість моделей в ансамблі (M);\n"
    "2. {resolution_hint}"
)
RESOLUTION_UPLOAD_HINT = "Завантажте інший файл із довшими часовими рядами."
RESOLUTION_M4_HINT = "Змініть параметри моделювання (L або M)."
DIALOG_OK_BTN = "Зрозуміло (OK)"

# ==============================================================================
# Action Controls & Run Flow
# ==============================================================================
BTN_RUN = "Запустити прогнозування"
BTN_CANCEL = "Скасувати"
BTN_EDA = "Аналіз даних (EDA)"

STATUS_RUNNING_TEMPLATE = "Обчислення... Ряд {i} з {n} ({percent:.1f}%)"
STATUS_SUCCESS = "Прогнозування успішно завершено!"
STATUS_CANCELLED = "Обчислення скасовано користувачем. Попередні результати збережено."
STATUS_ERROR_TEMPLATE = "Помилка під час обчислень: {error}"

# ==============================================================================
# Exploratory Data Analysis (EDA)
# ==============================================================================
EDA_HEADER = "Попередній аналіз даних (EDA)"
EDA_DATASET_SUMMARY_TEMPLATE = (
    "Загальна кількість рядів: {n_series} · Довжина рядів: min={min_len}, median={median_len}, max={max_len}"
)
EDA_COLS = {
    "series_id": "ID ряду",
    "count": "N (кількість)",
    "min": "Мінімум",
    "max": "Максимум",
    "mean": "Середнє",
    "median": "Медіана",
    "mode": "Мода",
    "std": "Станд. відхилення (std)",
    "q1": "Q1 (25%)",
    "q3": "Q3 (75%)",
    "iqr": "IQR",
    "skewness": "Асиметрія (Skewness)",
    "kurtosis": "Ексцес (Kurtosis)",
    "outliers_count": "Викиди (|z| > 3)",
}
BTN_DOWNLOAD_EDA = "Завантажити таблицю EDA (CSV)"

# ==============================================================================
# Results View & Plots
# ==============================================================================
RESULTS_HEADER = "Результати моделювання"
PLOTS_HEADER = "Графіки прогнозів"

PLOT_TITLE_TEMPLATE = "Ряд {series_id} ({criterion})"
PLOT_CRITERION_BEST = "найкращий sMAPE"
PLOT_CRITERION_MEDIAN = "медіанний sMAPE"
PLOT_CRITERION_WORST = "найгірший sMAPE"
PLOT_CRITERION_MIN_LEN = "найкоротший ряд"
PLOT_CRITERION_MEDIAN_LEN = "медіанна довжина"
PLOT_CRITERION_MAX_LEN = "найдовший ряд"

PLOT_LEGEND_HISTORY = "Історія"
PLOT_LEGEND_FORECAST = "Прогноз GRNN"
PLOT_LEGEND_ACTUAL = "Фактичні значення"
PLOT_LEGEND_BOUNDARY = "Межа поділу"

FORECAST_NO_METRICS_NOTE = "Метрики недоступні: майбутні значення невідомі (режим прогнозування)."

METRICS_AGGREGATE_HEADER = "Зведені метрики якості (усереднені)"
METRICS_PER_SERIES_HEADER = "Детальні результати за рядами"

METRIC_LABELS = {
    "smape": "sMAPE (%)",
    "mase": "MASE",
    "owa": "OWA",
    "mae": "MAE",
    "rmse": "RMSE",
    "sigma": "Обрана σ",
}

BTN_DOWNLOAD_ALL = "Завантажити всі результати (ZIP)"
DOWNLOAD_ALL_HELP = (
    "Архів містить прогнози (forecasts.csv), метрики за рядами (metrics_per_series.csv), "
    "зведені метрики (metrics_aggregate.csv), параметри запуску (run_config.json) "
    "та дані EDA."
)
