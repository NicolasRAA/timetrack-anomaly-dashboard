"""Streamlit-дашборд: обнаружение аномалий в телеком-сети (TimeTrack, автоэнкодер)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

PROJECT_DIR = Path(__file__).resolve().parent
LOCAL_CSV = PROJECT_DIR / "compute_dataset.csv"
FALLBACK_CSV = Path.home() / "Downloads" / "compute_dataset.csv"

INCIDENT_START = pd.Timestamp("2024-07-18 15:18:55.300396")
INCIDENT_END = pd.Timestamp("2024-07-18 15:19:37.596428")

MACHINE_NAMES = [
    "machine01",
    "machine02",
    "machine03",
    "machine04",
    "machine05",
    "machine06",
    "machine07",
]
METRICS = ["AM", "UM", "CU", "CF", "DRT", "DWT"]
CLUSTER_COLS = [
    "cluster AM",
    "cluster UM",
    "cluster Available disk space",
    "cluster UD",
]


def _add_incident_zone(fig: go.Figure) -> go.Figure:
    fig.add_vrect(
        x0=INCIDENT_START,
        x1=INCIDENT_END,
        fillcolor="orange",
        opacity=0.25,
        layer="below",
        line_width=0,
        annotation_text="Инцидент",
        annotation_position="top left",
    )
    return fig


def _format_count(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}k"
    return f"{value:.2f}"


@st.cache_data(show_spinner="Загрузка датасета TimeTrack…")
def load_dataset() -> pd.DataFrame:
    if LOCAL_CSV.exists():
        csv_path = LOCAL_CSV
    elif FALLBACK_CSV.exists():
        csv_path = FALLBACK_CSV
    else:
        raise FileNotFoundError(
            "Файл compute_dataset.csv не найден. "
            f"Поместите его в папку проекта: {PROJECT_DIR}"
        )

    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_data(show_spinner="Подготовка признаков…")
def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, StandardScaler]:
    selected_cols = ["timestamp"]
    for machine in MACHINE_NAMES:
        for metric in METRICS:
            col = f"{machine} {metric}"
            if col in df.columns:
                selected_cols.append(col)
    for extra in CLUSTER_COLS:
        if extra in df.columns:
            selected_cols.append(extra)

    df_features = df[selected_cols].copy()
    missing_pct = df_features.drop("timestamp", axis=1).isnull().mean() * 100
    cols_to_drop = missing_pct[missing_pct > 50].index
    df_features = df_features.drop(columns=cols_to_drop)
    df_features = df_features.ffill().bfill()

    timestamps = df_features["timestamp"].values
    feature_df = df_features.drop("timestamp", axis=1)
    X = feature_df.values.astype(np.float64)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return feature_df, X_scaled, timestamps, scaler


def build_autoencoder(input_dim: int) -> keras.Model:
    input_layer = layers.Input(shape=(input_dim,))
    x = layers.Dense(32, activation="relu")(input_layer)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(16, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    bottleneck = layers.Dense(16, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(bottleneck)
    x = layers.Dense(32, activation="relu")(x)
    output_layer = layers.Dense(input_dim, activation="linear")(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model


class _EpochProgress(keras.callbacks.Callback):
    def __init__(self, progress_bar, status_text, total_epochs: int = 100):
        super().__init__()
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        progress = min((epoch + 1) / self.total_epochs, 1.0)
        self.progress_bar.progress(progress)
        val_loss = logs.get("val_loss", float("nan"))
        loss = logs.get("loss", float("nan"))
        self.status_text.text(
            f"Эпоха {epoch + 1}/{self.total_epochs} — "
            f"loss: {loss:.4f}, val_loss: {val_loss:.4f}"
        )


@st.cache_resource(show_spinner="Обучение автоэнкодера…")
def train_autoencoder(X_scaled: np.ndarray) -> keras.Model:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    progress_bar = st.progress(0)
    status_text = st.empty()
    model = build_autoencoder(X_scaled.shape[1])
    early_stop = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    progress_cb = _EpochProgress(progress_bar, status_text, total_epochs=100)
    model.fit(
        X_scaled,
        X_scaled,
        epochs=100,
        batch_size=256,
        validation_split=0.1,
        callbacks=[early_stop, progress_cb],
        verbose=0,
    )
    progress_bar.progress(1.0)
    status_text.text("Обучение завершено.")
    return model


def compute_reconstruction_error(
    model: keras.Model, X_scaled: np.ndarray
) -> np.ndarray:
    X_pred = model.predict(X_scaled, verbose=0)
    return np.mean(np.square(X_scaled - X_pred), axis=1)


def plot_reconstruction_error(
    timestamps: np.ndarray,
    reconstruction_error: np.ndarray,
    threshold: float,
    percentile: int,
) -> go.Figure:
    t = pd.to_datetime(timestamps)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t,
            y=reconstruction_error,
            mode="lines",
            name="Ошибка реконструкции (MSE)",
            line=dict(color="#1f77b4", width=0.8),
        )
    )
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Порог ({percentile}-й перцентиль): {threshold:.4f}",
        annotation_position="top right",
    )
    _add_incident_zone(fig)
    fig.update_layout(
        title="Ошибка реконструкции автоэнкодера во времени",
        xaxis_title="Время",
        yaxis_title="MSE",
        hovermode="x unified",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_kpi(
    timestamps: np.ndarray,
    values: np.ndarray,
    title: str,
    yaxis_title: str,
) -> go.Figure:
    t = pd.to_datetime(timestamps)
    fig = go.Figure(
        go.Scatter(
            x=t,
            y=values,
            mode="lines",
            name=title,
            line=dict(color="#ff7f0e", width=0.8),
        )
    )
    _add_incident_zone(fig)
    fig.update_layout(
        title=title,
        xaxis_title="Время",
        yaxis_title=yaxis_title,
        hovermode="x unified",
        height=320,
    )
    return fig


def _pct_change(before: float, during: float) -> str:
    if before == 0:
        if during == 0:
            return "без изменений"
        return "рост с нуля"
    change = (during - before) / before * 100
    sign = "+" if change >= 0 else ""
    return f"{sign}{change:.0f} %"


def machine_incident_stats(df: pd.DataFrame, machine: str) -> dict[str, float]:
    mask_incident = (df["timestamp"] >= INCIDENT_START) & (
        df["timestamp"] <= INCIDENT_END
    )
    mask_before = df["timestamp"] < INCIDENT_START

    stats = {}
    for metric in ("CU", "DRT", "DWT"):
        col = f"{machine} {metric}"
        stats[f"{metric}_before"] = float(df.loc[mask_before, col].median())
        stats[f"{metric}_during"] = float(df.loc[mask_incident, col].median())
    return stats


def render_incident_summary(
    machine: str, stats: dict[str, float], percentile: int, threshold: float
) -> str:
    duration_sec = int((INCIDENT_END - INCIDENT_START).total_seconds())
    cu_b, cu_d = stats["CU_before"], stats["CU_during"]
    drt_b, drt_d = stats["DRT_before"], stats["DRT_during"]
    dwt_b, dwt_d = stats["DWT_before"], stats["DWT_during"]

    return (
        f"**Ключевой инцидент:** {INCIDENT_START.strftime('%d.%m.%Y %H:%M:%S')} — "
        f"{INCIDENT_END.strftime('%H:%M:%S')}  \n"
        f"**Длительность:** {duration_sec} секунды  \n"
        f"**CPU ({machine} CU):** медиана до инцидента {cu_b:.2f} %, "
        f"во время инцидента {cu_d:.2f} % ({_pct_change(cu_b, cu_d)})  \n"
        f"**Дисковое чтение ({machine} DRT):** медиана до {_format_count(drt_b)}, "
        f"во время инцидента {_format_count(drt_d)}  \n"
        f"**Дисковая запись ({machine} DWT):** медиана до {_format_count(dwt_b)}, "
        f"во время инцидента {_format_count(dwt_d)}  \n\n"
        f"Метод обнаружения: безнадзорный автоэнкодер, порог MSE на {percentile}-м "
        f"перцентиле ({threshold:.4f})."
    )


def main() -> None:
    st.set_page_config(
        page_title="Обнаружение аномалий — TimeTrack",
        page_icon="📡",
        layout="wide",
    )

    st.title("Обнаружение аномалий в сети с помощью автоэнкодера")
    st.caption(
        "Датасет TimeTrack (Open Air Interface CI/CD): метрики вычислительного кластера "
        "и автоэнкодер для выявления аномалий по ошибке реконструкции."
    )

    with st.sidebar:
        st.header("Настройки")
        percentile = st.slider(
            "Чувствительность",
            min_value=90,
            max_value=99,
            value=95,
            help=(
                "Перцентиль порога аномалии. Чем ниже значение, тем ниже порог "
                "и больше точек будет помечено как аномалии."
            ),
        )

    df_raw = load_dataset()
    feature_df, X_scaled, timestamps, _scaler = preprocess_data(df_raw)

    col1, col2, col3 = st.columns(3)
    col1.metric("Записей", f"{len(df_raw):,}")
    col2.metric("Признаков", feature_df.shape[1])
    col3.metric(
        "Период",
        f"{pd.to_datetime(timestamps).min():%d.%m.%Y} — "
        f"{pd.to_datetime(timestamps).max():%d.%m.%Y}",
    )

    st.subheader("Обучение модели")
    with st.spinner("Подготовка к обучению…"):
        model = train_autoencoder(X_scaled)

    with st.spinner("Вычисление ошибки реконструкции…"):
        reconstruction_error = compute_reconstruction_error(model, X_scaled)

    threshold = float(np.percentile(reconstruction_error, percentile))
    anomalies_count = int(np.sum(reconstruction_error > threshold))

    col_a, col_b, col_c = st.columns(3)
    col_a.metric(f"Порог ({percentile}-й перцентиль)", f"{threshold:.6f}")
    col_b.metric("Аномалий", f"{anomalies_count:,}")
    col_c.metric(
        "Доля аномалий",
        f"{100 * anomalies_count / len(reconstruction_error):.2f} %",
    )

    st.subheader("Ошибка реконструкции")
    st.plotly_chart(
        plot_reconstruction_error(
            timestamps, reconstruction_error, threshold, percentile
        ),
        use_container_width=True,
    )

    machine = st.selectbox(
        "Машина",
        MACHINE_NAMES,
        index=0,
        format_func=lambda m: m.replace("machine", "machine "),
    )
    st.subheader(f"Метрики {machine} во время инцидента")
    ts_raw = df_raw["timestamp"].values
    cu = df_raw[f"{machine} CU"].values
    drt = df_raw[f"{machine} DRT"].values
    dwt = df_raw[f"{machine} DWT"].values

    st.plotly_chart(
        plot_kpi(ts_raw, cu, f"CPU {machine} (CU)", "Загрузка CPU, %"),
        use_container_width=True,
    )
    st.plotly_chart(
        plot_kpi(ts_raw, drt, f"Disk Read {machine} (DRT)", "Чтение с диска"),
        use_container_width=True,
    )
    st.plotly_chart(
        plot_kpi(ts_raw, dwt, f"Disk Write {machine} (DWT)", "Запись на диск"),
        use_container_width=True,
    )

    st.subheader("Сводка инцидента")
    incident_stats = machine_incident_stats(df_raw, machine)
    st.info(
        render_incident_summary(machine, incident_stats, percentile, threshold)
    )

    with st.expander("Что это значит?", expanded=False):
        st.markdown(
            "**Что произошло в сети.** "
            "18 июля 2024 года на одной из виртуальных машин кластера за короткое время "
            "резко выросла нагрузка на процессор и диск: сервер начал интенсивно читать "
            "и записывать данные. Такое поведение похоже на сбой приложения или "
            "перегрузку узла, которая могла ухудшить работу сервисов связи на этом узле."
        )
        st.markdown(
            "**Почему простой порог мог не заметить, а автоэнкодер — да.** "
            "Отдельные метрики (например, только CPU) иногда остаются в «нормальном» "
            "диапазоне, хотя сочетание признаков уже нетипично. Автоэнкодер учится "
            "на всей картине метрик сразу и сигнализирует, когда восстановить их "
            "совместное поведение не удаётся — то есть когда ситуация отличается "
            "от обычной, даже без заранее заданных правил."
        )
        st.markdown(
            "**Польза для оператора связи.** "
            "Система помогает быстрее увидеть нештатную ситуацию на инфраструктуре, "
            "не просматривая вручную десятки графиков. Оператор может сузить "
            "расследование до конкретной машины и временного окна и принять меры "
            "до того, как инцидент затронет пользователей сети."
        )


if __name__ == "__main__":
    main()
