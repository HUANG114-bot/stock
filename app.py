from pathlib import Path

code = r'''import datetime
import math
import warnings

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from FinMind.data import DataLoader

warnings.filterwarnings("ignore", category=FutureWarning)

# =========================================================
# 台股盤後分析站｜完整版本
# 執行：streamlit run app.py
# 套件：pip install streamlit yfinance pandas plotly FinMind
# =========================================================

st.set_page_config(
    page_title="台股盤後分析站",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------- CSS ----------------------------
st.markdown(
    """
    <style>
    #MainMenu, footer, header {visibility: hidden;}
    [data-testid="collapsedControl"] {display:none;}
    .block-container {
        padding-top: .75rem;
        padding-bottom: 1.5rem;
        padding-left: max(.65rem, 2vw);
        padding-right: max(.65rem, 2vw);
        max-width: 1600px;
    }
    [data-testid="stMetric"] {
        background: #111c2b;
        border: 1px solid #26364b;
        border-radius: 12px;
        padding: 12px 14px;
    }
    div[data-testid="stButton"] > button {
        min-height: 40px;
        border-radius: 9px;
        font-weight: 600;
    }
    div[data-testid="stTextInput"] input {
        min-height: 42px;
        border-radius: 9px;
    }
    .section-label {
        color: #93a4bb;
        font-size: 12px;
        margin-bottom: 4px;
    }
    .hint {
        color: #9aa9bc;
        font-size: 12px;
    }
    @media (max-width: 640px) {
        .block-container {
            padding-top: .4rem;
            padding-left: .45rem;
            padding-right: .45rem;
        }
        [data-testid="stMetric"] {
            padding: 8px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------- 常數 / 自選股 -----------------------
DEFAULT_FAVORITES = {
    "2330": "台積電",
    "1513": "中興電",
    "6257": "矽格",
}
STOCK_NAMES = {
    "2330": "台積電",
    "1513": "中興電",
    "6257": "矽格",
    "2454": "聯發科",
    "2317": "鴻海",
    "2308": "台達電",
    "2382": "廣達",
    "2881": "富邦金",
    "2882": "國泰金",
    "2603": "長榮",
    "2615": "萬海",
    "3481": "群創",
    "3037": "欣興",
    "3711": "日月光投控",
    "6223": "旺矽",
    "2408": "南亞科",
    "4919": "新唐",
    "6187": "萬潤",
}
RANGE_MAP = {
    "近 1 個月": 22,
    "近 3 個月": 66,
    "近半年": 130,
    "近 1 年": 250,
    "近 3 年": 750,
    "全部": 0,
}
CHIP_COLS = ["外資", "投信", "自營商"]
PRICE_COLS = ["Open", "High", "Low", "Close", "Volume"]


def init_state():
    if "favorites" not in st.session_state:
        st.session_state.favorites = DEFAULT_FAVORITES.copy()
    if "chart_bars" not in st.session_state:
        st.session_state.chart_bars = 130
    if "chart_offset" not in st.session_state:
        st.session_state.chart_offset = 0
    if "chart_view_id" not in st.session_state:
        st.session_state.chart_view_id = 0
    if "ticker_value" not in st.session_state:
        st.session_state.ticker_value = "2330"


init_state()


# ---------------------- 資料抓取 --------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_raw_data(ticker_num, start_d, end_d):
    stock_id = str(ticker_num).strip().upper()
    stock_id = stock_id.replace(".TW", "").replace(".TWO", "")
    if not stock_id:
        return None, "", ""

    # 先試上市，再試上櫃
    yf_symbol = f"{stock_id}.TW"
    try:
        df = yf.Ticker(yf_symbol).history(
            start=start_d.isoformat(),
            end=(end_d + datetime.timedelta(days=1)).isoformat(),
            auto_adjust=False,
        )
    except Exception:
        df = pd.DataFrame()

    if df.empty:
        yf_symbol = f"{stock_id}.TWO"
        try:
            df = yf.Ticker(yf_symbol).history(
                start=start_d.isoformat(),
                end=(end_d + datetime.timedelta(days=1)).isoformat(),
                auto_adjust=False,
            )
        except Exception:
            df = pd.DataFrame()

    if df.empty:
        return None, stock_id, ""

    df = df.copy()
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # yfinance 欄位可能因版本略有差異
    missing = [c for c in PRICE_COLS if c not in df.columns]
    if missing:
        return None, stock_id, yf_symbol
    df = df[PRICE_COLS].copy()

    for c in PRICE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    # 先建立法人欄位，即使 FinMind 失敗，圖表也能正常顯示
    for c in CHIP_COLS:
        df[c] = 0.0

    try:
        loader = DataLoader()
        inst_df = loader.taiwan_stock_institutional_investors(
            stock_id=stock_id,
            start_date=start_d.isoformat(),
            end_date=end_d.isoformat(),
        )
        if inst_df is not None and not inst_df.empty:
            inst_df = inst_df.copy()
            inst_df["buy"] = pd.to_numeric(inst_df["buy"], errors="coerce").fillna(0)
            inst_df["sell"] = pd.to_numeric(inst_df["sell"], errors="coerce").fillna(0)
            inst_df["net_buy"] = inst_df["buy"] - inst_df["sell"]

            def map_inst_name(name):
                name = str(name)
                if "外資" in name or "Foreign" in name:
                    return "外資"
                if "投信" in name or "Investment_Trust" in name:
                    return "投信"
                if "自營商" in name or "Dealer" in name:
                    return "自營商"
                return None

            inst_df["short_name"] = inst_df["name"].map(map_inst_name)
            inst_df = inst_df.dropna(subset=["short_name"])
            if not inst_df.empty:
                chip = (
                    inst_df.groupby(["date", "short_name"])["net_buy"]
                    .sum()
                    .unstack(fill_value=0)
                )
                chip.index = pd.to_datetime(chip.index)
                chip = chip.rename_axis(index=None, columns=None)
                for c in CHIP_COLS:
                    if c not in chip.columns:
                        chip[c] = 0.0
                chip = chip[CHIP_COLS]
                # 對齊交易日，不存在的日期維持 0
                for date_idx, row in chip.iterrows():
                    if date_idx in df.index:
                        for c in CHIP_COLS:
                            df.loc[date_idx, c] = float(row[c])
    except Exception:
        # FinMind 服務或欄位有問題時，不讓整個看盤站失效
        pass

    return df, stock_id, yf_symbol


# ----------------------- 指標計算 --------------------------
def calc_rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0)
    return rsi


def add_indicators(df):
    df = df.copy()
    for n in [5, 10, 20, 60]:
        df[f"SMA_{n}"] = df["Close"].rolling(n, min_periods=n).mean()

    std20 = df["Close"].rolling(20, min_periods=20).std()
    df["BBU_20_2.0"] = df["SMA_20"] + 2 * std20
    df["BBL_20_2.0"] = df["SMA_20"] - 2 * std20

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["OSC"] = df["MACD"] - df["Signal"]

    df["RSI_6"] = calc_rsi(df["Close"], 6)
    df["RSI_14"] = calc_rsi(df["Close"], 14)

    low9 = df["Low"].rolling(9, min_periods=9).min()
    high9 = df["High"].rolling(9, min_periods=9).max()
    denom = (high9 - low9).replace(0, float("nan"))
    df["RSV"] = ((df["Close"] - low9) / denom * 100).fillna(50)
    df["K"] = df["RSV"].ewm(com=2, adjust=False).mean()
    df["D"] = df["K"].ewm(com=2, adjust=False).mean()
    return df


def resample_weekly(df):
    agg = {
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum",
        "外資": "sum", "投信": "sum", "自營商": "sum",
    }
    return df.resample("W-FRI").agg(agg).dropna(subset=["Open", "High", "Low", "Close"])


def fmt_num(value, decimals=2):
    try:
        if pd.isna(value):
            return "—"
        return f"{value:,.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def consecutive_net_days(series):
    """從最新交易日往回計算連續買超或賣超天數；0 不算買賣超。"""
    vals = pd.to_numeric(series, errors="coerce").fillna(0).iloc[::-1]
    if vals.empty or vals.iloc[0] == 0:
        return "無連續買賣超"
    direction = "買超" if vals.iloc[0] > 0 else "賣超"
    count = 0
    for val in vals:
        if (val > 0 and direction == "買超") or (val < 0 and direction == "賣超"):
            count += 1
        else:
            break
    return f"連續{count}日{direction}"


def build_signals(df):
    signals = []
    if len(df) < 2:
        return [("資料不足", "目前資料筆數不足，無法判斷指標變化。")]

    now, prev = df.iloc[-1], df.iloc[-2]
    checks = [
        ("均線", "5MA 上穿 20MA",
         pd.notna(prev["SMA_5"]) and pd.notna(prev["SMA_20"])
         and prev["SMA_5"] <= prev["SMA_20"]
         and now["SMA_5"] > now["SMA_20"]),
        ("均線", "5MA 下穿 20MA",
         pd.notna(prev["SMA_5"]) and pd.notna(prev["SMA_20"])
         and prev["SMA_5"] >= prev["SMA_20"]
         and now["SMA_5"] < now["SMA_20"]),
        ("MACD", "MACD 上穿訊號線",
         prev["MACD"] <= prev["Signal"] and now["MACD"] > now["Signal"]),
        ("MACD", "MACD 下穿訊號線",
         prev["MACD"] >= prev["Signal"] and now["MACD"] < now["Signal"]),
        ("RSI", "RSI(14) 高於 70（偏熱區）",
         pd.notna(now["RSI_14"]) and now["RSI_14"] >= 70),
        ("RSI", "RSI(14) 低於 30（偏弱區）",
         pd.notna(now["RSI_14"]) and now["RSI_14"] <= 30),
        ("KD", "K 上穿 D",
         prev["K"] <= prev["D"] and now["K"] > now["D"]),
        ("KD", "K 下穿 D",
         prev["K"] >= prev["D"] and now["K"] < now["D"]),
    ]
    for category, message, passed in checks:
        if bool(passed):
            signals.append((category, message))
    if not signals:
        signals.append(("觀察", "最新一根 K 線未觸發上述條件。"))
    return signals


def find_support_resistance(df, window=60):
    recent = df.tail(window)
    if recent.empty:
        return None, None
    return float(recent["Low"].min()), float(recent["High"].max())


# ------------------------- 頁首 ----------------------------
st.title("📈 台股盤後分析站")
st.caption("K 線・技術指標・法人籌碼｜數據可能延遲，訊號僅供研究參考")

# --------------------- 股票控制區 --------------------------
with st.container(border=True):
    fav_items = list(st.session_state.favorites.items())
    fav_labels = [f"{sid}｜{name}" for sid, name in fav_items]
    control1, control2 = st.columns([2.2, 1.2])

    with control1:
        typed_ticker = st.text_input(
            "股票代號",
            value=st.session_state.ticker_value,
            placeholder="輸入股票代號，例如 2330",
            help="輸入上市或上櫃台股代號。",
        ).strip().upper().replace(".TW", "").replace(".TWO", "")
        if typed_ticker:
            st.session_state.ticker_value = typed_ticker

    with control2:
        period = st.radio(
            "K 線週期", ["日線", "週線"],
            horizontal=True, index=0,
        )

    f1, f2, f3 = st.columns([2, 1, 1])
    with f1:
        if fav_items:
            selected_fav = st.selectbox("自選股快速切換", fav_labels)
            selected_id = selected_fav.split("｜")[0]
            if st.button("切換到所選自選股", use_container_width=True):
                st.session_state.ticker_value = selected_id
                st.session_state.chart_offset = 0
                st.session_state.chart_view_id += 1
                st.rerun()
        else:
            selected_id = None
            st.caption("尚無自選股")

    with f2:
        if st.button("☆ 加入自選", use_container_width=True):
            if typed_ticker:
                st.session_state.favorites[typed_ticker] = STOCK_NAMES.get(
                    typed_ticker, typed_ticker
                )
                st.toast(f"已加入 {typed_ticker} 自選股")
                st.rerun()

    with f3:
        if st.button("管理自選股", use_container_width=True):
            st.session_state.show_fav_manager = not st.session_state.get(
                "show_fav_manager", False
            )

    if st.session_state.get("show_fav_manager", False):
        st.markdown("**自選股管理**")
        for sid, name in list(st.session_state.favorites.items()):
            mc1, mc2, mc3 = st.columns([1, 2, 1])
            mc1.write(sid)
            mc2.write(name)
            if mc3.button("移除", key=f"remove_{sid}"):
                st.session_state.favorites.pop(sid, None)
                st.rerun()

    range_label = st.radio(
        "資料區間",
        list(RANGE_MAP.keys()),
        index=3,
        horizontal=True,
    )

# ---------------------- 取得行情 ---------------------------
ticker = st.session_state.ticker_value
end_date = datetime.date.today()
days_back = 365 * 5 if range_label == "全部" else max(RANGE_MAP[range_label] * 2, 100)
start_date = end_date - datetime.timedelta(days=days_back)

with st.spinner(f"正在載入 {ticker} 行情與三大法人資料…"):
    raw_df, stock_id, yf_symbol = fetch_raw_data(ticker, start_date, end_date)

if raw_df is None or raw_df.empty:
    st.error(f"找不到股票代號「{ticker}」的行情資料，請確認代號或稍後再試。")
    st.stop()

df_daily = add_indicators(raw_df)
df = resample_weekly(df_daily) if period == "週線" else df_daily.copy()
df = add_indicators(df)

if df.empty:
    st.error("目前沒有足夠資料可以繪圖。")
    st.stop()

# ---------------------- 快速圖表操作 -----------------------
st.markdown("### 圖表操作")
q1, q2, q3, q4, q5, q6 = st.columns(6)
with q1:
    if st.button("放大 ＋", use_container_width=True):
        st.session_state.chart_bars = max(20, int(st.session_state.chart_bars * 0.7))
        st.session_state.chart_offset = 0
        st.session_state.chart_view_id += 1
        st.rerun()
with q2:
    if st.button("縮小 －", use_container_width=True):
        st.session_state.chart_bars = min(len(df), int(st.session_state.chart_bars * 1.4) + 1)
        st.session_state.chart_offset = 0
        st.session_state.chart_view_id += 1
        st.rerun()
with q3:
    if st.button("← 較早", use_container_width=True):
        st.session_state.chart_offset = min(
            max(0, len(df) - 20),
            st.session_state.chart_offset + max(5, st.session_state.chart_bars // 4),
        )
        st.session_state.chart_view_id += 1
        st.rerun()
with q4:
    if st.button("較新 →", use_container_width=True):
        st.session_state.chart_offset = max(
            0, st.session_state.chart_offset - max(5, st.session_state.chart_bars // 4)
        )
        st.session_state.chart_view_id += 1
        st.rerun()
with q5:
    if st.button("最近 60 根", use_container_width=True):
        st.session_state.chart_bars = min(60, len(df))
        st.session_state.chart_offset = 0
        st.session_state.chart_view_id += 1
        st.rerun()
with q6:
    if st.button("重設視圖", use_container_width=True):
        st.session_state.chart_bars = min(130, len(df))
        st.session_state.chart_offset = 0
        st.session_state.chart_view_id += 1
        st.rerun()

# 快速區間：選擇顯示最近多少根 K 線
quick1, quick2, quick3, quick4, quick5 = st.columns(5)
quick_options = [
    ("20 根", 20), ("60 根", 60), ("120 根", 120),
    ("250 根", 250), ("全部", len(df)),
]
for col, (label, count) in zip(
    [quick1, quick2, quick3, quick4, quick5], quick_options
):
    with col:
        if st.button(label, key=f"quick_{label}", use_container_width=True):
            st.session_state.chart_bars = min(count, len(df))
            st.session_state.chart_offset = 0
            st.session_state.chart_view_id += 1
            st.rerun()

# 以尾端為 0，offset 越大代表往歷史方向移動
bars = max(20, min(int(st.session_state.chart_bars), len(df)))
end_pos = max(1, len(df) - int(st.session_state.chart_offset))
start_pos = max(0, end_pos - bars)
view_df = df.iloc[start_pos:end_pos].copy()
if view_df.empty:
    view_df = df.tail(min(bars, len(df))).copy()

st.caption(
    f"目前顯示 {len(view_df)} 根 K 線｜"
    f"{view_df.index.min().strftime('%Y-%m-%d')} 至 "
    f"{view_df.index.max().strftime('%Y-%m-%d')}｜"
    "拖曳可平移；使用圖表工具列可框選縮放；滾輪縮放可在工具列切換。"
)

# ------------------------ 行情摘要 -------------------------
last = df.iloc[-1]
prev_close = df["Close"].iloc[-2] if len(df) >= 2 else last["Close"]
change = float(last["Close"] - prev_close)
change_pct = change / prev_close * 100 if prev_close else 0
stock_name = STOCK_NAMES.get(stock_id, stock_id)

st.subheader(f"{stock_id}｜{stock_name}")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("最新收盤", fmt_num(last["Close"]), f"{change:+.2f} ({change_pct:+.2f}%)")
m2.metric("開盤", fmt_num(last["Open"]))
m3.metric("最高 / 最低", f"{fmt_num(last['High'])} / {fmt_num(last['Low'])}")
m4.metric("成交量", f"{last['Volume']:,.0f}" if pd.notna(last["Volume"]) else "—")
m5.metric("資料日期", df.index[-1].strftime("%Y-%m-%d"))

# --------------------- 支撐壓力分析 ------------------------
support, resistance = find_support_resistance(df_daily, 60)
if support is not None and resistance is not None:
    support_gap = (last["Close"] - support) / support * 100 if support else 0
    resistance_gap = (resistance - last["Close"]) / last["Close"] * 100 if last["Close"] else 0
    sr1, sr2, sr3 = st.columns(3)
    sr1.metric("近 60 交易日區間低點（參考支撐）", fmt_num(support), f"距收盤 {support_gap:+.2f}%")
    sr2.metric("近 60 交易日區間高點（參考壓力）", fmt_num(resistance), f"距收盤 {resistance_gap:+.2f}%")
    if last["Close"] > resistance:
        breakout_text = "收盤高於近 60 日區間高點"
    elif last["Close"] < support:
        breakout_text = "收盤低於近 60 日區間低點"
    else:
        breakout_text = "收盤仍在近 60 日高低點區間內"
    sr3.metric("區間位置", breakout_text)
    st.caption("支撐／壓力以近 60 個日線交易日的最高、最低價估算，並非保證有效的價位。")

# -------------------------- 圖表 ---------------------------
fig = make_subplots(
    rows=6, cols=1, shared_xaxes=True,
    vertical_spacing=0.018,
    row_heights=[0.46, 0.12, 0.12, 0.10, 0.10, 0.10],
    subplot_titles=("K 線與均線", "成交量", "MACD", "RSI", "KD", "三大法人買賣超"),
)

x = view_df.index
hover_kline = (
    "日期：%{x|%Y-%m-%d}<br>"
    "開：%{open:.2f}<br>高：%{high:.2f}<br>"
    "低：%{low:.2f}<br>收：%{close:.2f}"
    "<extra></extra>"
)
fig.add_trace(
    go.Candlestick(
        x=x, open=view_df["Open"], high=view_df["High"],
        low=view_df["Low"], close=view_df["Close"],
        increasing_line_color="#ff4d5a",
        decreasing_line_color="#00c087",
        name="K 線", hovertemplate=hover_kline,
    ), row=1, col=1
)

ma_styles = {
    "SMA_5": ("#f4f7fb", "5MA"),
    "SMA_10": ("#f5c542", "10MA"),
    "SMA_20": ("#ff66cc", "20MA"),
    "SMA_60": ("#31d0aa", "60MA"),
}
for col_name, (color, label) in ma_styles.items():
    fig.add_trace(
        go.Scatter(
            x=x, y=view_df[col_name], mode="lines",
            line=dict(color=color, width=1.2), name=label,
            hovertemplate=f"{label}: %{{y:.2f}}<extra></extra>",
        ), row=1, col=1
    )

if st.checkbox("顯示布林通道", value=False):
    fig.add_trace(
        go.Scatter(
            x=x, y=view_df["BBU_20_2.0"], mode="lines",
            line=dict(color="rgba(100,170,255,.7)", width=1, dash="dot"),
            name="布林上軌",
        ), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=view_df["BBL_20_2.0"], mode="lines",
            line=dict(color="rgba(100,170,255,.7)", width=1, dash="dot"),
            name="布林下軌",
        ), row=1, col=1
    )

show_sr = st.checkbox("在 K 線圖標記近 60 日高低點", value=True)
if show_sr and support is not None and resistance is not None:
    fig.add_hline(
        y=support, row=1, col=1,
        line_dash="dash", line_color="#00c087",
        annotation_text=f"參考支撐 {support:.2f}",
        annotation_position="bottom right",
    )
    fig.add_hline(
        y=resistance, row=1, col=1,
        line_dash="dash", line_color="#ff5964",
        annotation_text=f"參考壓力 {resistance:.2f}",
        annotation_position="top right",
    )

vol_colors = [
    "#ff4d5a" if c >= o else "#00c087"
    for c, o in zip(view_df["Close"], view_df["Open"])
]
fig.add_trace(
    go.Bar(
        x=x, y=view_df["Volume"], marker_color=vol_colors,
        name="成交量", hovertemplate="成交量：%{y:,.0f}<extra></extra>",
    ), row=2, col=1
)

fig.add_trace(
    go.Scatter(x=x, y=view_df["MACD"], line=dict(color="#f4f7fb", width=1.2),
               name="MACD", hovertemplate="MACD：%{y:.3f}<extra></extra>"),
    row=3, col=1
)
fig.add_trace(
    go.Scatter(x=x, y=view_df["Signal"], line=dict(color="#f5c542", width=1.2),
               name="訊號線", hovertemplate="訊號線：%{y:.3f}<extra></extra>"),
    row=3, col=1
)
osc_colors = ["#ff4d5a" if v >= 0 else "#00c087" for v in view_df["OSC"].fillna(0)]
fig.add_trace(
    go.Bar(x=x, y=view_df["OSC"], marker_color=osc_colors, name="OSC",
           hovertemplate="OSC：%{y:.3f}<extra></extra>"),
    row=3, col=1
)

fig.add_trace(
    go.Scatter(x=x, y=view_df["RSI_6"], line=dict(color="#f4f7fb", width=1.2),
               name="RSI(6)", hovertemplate="RSI(6)：%{y:.1f}<extra></extra>"),
    row=4, col=1
)
fig.add_trace(
    go.Scatter(x=x, y=view_df["RSI_14"], line=dict(color="#f5c542", width=1.2),
               name="RSI(14)", hovertemplate="RSI(14)：%{y:.1f}<extra></extra>"),
    row=4, col=1
)
for yline in [30, 70]:
    fig.add_hline(y=yline, line_dash="dot", line_color="rgba(255,255,255,.25)", row=4, col=1)

fig.add_trace(
    go.Scatter(x=x, y=view_df["K"], line=dict(color="#f4f7fb", width=1.2),
               name="K", hovertemplate="K：%{y:.1f}<extra></extra>"),
    row=5, col=1
)
fig.add_trace(
    go.Scatter(x=x, y=view_df["D"], line=dict(color="#f5c542", width=1.2),
               name="D", hovertemplate="D：%{y:.1f}<extra></extra>"),
    row=5, col=1
)
for yline in [20, 80]:
    fig.add_hline(y=yline, line_dash="dot", line_color="rgba(255,255,255,.25)", row=5, col=1)

for chip_col, color in zip(CHIP_COLS, ["#ff4d5a", "#00c087", "#f5c542"]):
    if chip_col in view_df.columns:
        fig.add_trace(
            go.Bar(
                x=x, y=view_df[chip_col], name=chip_col,
                marker_color=color,
                hovertemplate=f"{chip_col}：%{{y:,.0f} 張/股（依來源單位）<extra></extra>",
            ), row=6, col=1
        )

fig.update_layout(
    height=1050,
    margin=dict(l=8, r=12, t=40, b=10),
    template="plotly_dark",
    hovermode="x unified",
    barmode="group",
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    dragmode="pan",
    uirevision=f"{stock_id}-{period}-{st.session_state.chart_view_id}",
)
fig.update_xaxes(
    showgrid=True, gridcolor="rgba(128,128,128,.12)",
    rangeslider_visible=False,
    showspikes=True, spikemode="across", spikesnap="cursor",
    showline=True, linecolor="#344154",
)
fig.update_yaxes(
    showgrid=True, gridcolor="rgba(128,128,128,.12)",
    side="right", fixedrange=False,
)
for annotation in fig["layout"]["annotations"]:
    annotation["font"] = dict(size=11, color="#9aa9bc")

# Plotly 操作：
# - dragmode pan：拖曳平移
# - 工具列可切換框選縮放 / 平移 / 重設
# - scrollZoom：桌面可按住 Ctrl 或依瀏覽器支援使用滾輪縮放
st.plotly_chart(
    fig,
    use_container_width=True,
    key=f"stock_chart_{stock_id}_{period}_{st.session_state.chart_view_id}",
    config={
        "displayModeBar": True,
        "displaylogo": False,
        "scrollZoom": True,
        "doubleClick": "reset",
        "responsive": True,
        "modeBarButtonsToAdd": ["drawline", "drawopenpath", "drawrect", "eraseshape"],
        "toImageButtonOptions": {
            "format": "png",
            "filename": f"{stock_id}_stock_chart",
            "height": 1200,
            "width": 1800,
            "scale": 2,
        },
    },
)

st.caption(
    "操作提示：目前預設拖曳為平移；框選縮放、畫線、下載圖片可使用圖表上方工具列。"
    "手機觸控效果會受 iOS/瀏覽器版本影響，若雙指縮放不穩，可使用上方快速區間與放大縮小按鈕。"
)

# --------------------- 技術指標訊號 ------------------------
st.subheader("🔎 技術指標條件提示")
signals = build_signals(df)
sig_rows = [{"類別": cat, "條件提示": message} for cat, message in signals]
st.dataframe(pd.DataFrame(sig_rows), use_container_width=True, hide_index=True)
st.caption(
    "提示只描述指標條件是否成立，不是投資建議。均線、MACD、RSI、KD 可能出現落後或假訊號，"
    "請搭配價格位置、成交量、基本面與個人風險管理判斷。"
)

# ---------------------- 三大法人籌碼 -----------------------
st.subheader("🏦 三大法人籌碼")
chip_df = df_daily[CHIP_COLS].copy()
chip_df = chip_df.loc[chip_df.index <= df_daily.index[-1]].tail(30)
chip_latest = chip_df.iloc[-1] if not chip_df.empty else pd.Series({c: 0 for c in CHIP_COLS})

cc1, cc2, cc3 = st.columns(3)
for col, name in zip([cc1, cc2, cc3], CHIP_COLS):
    with col:
        value = float(chip_latest.get(name, 0))
        col.metric(name, f"{value:+,.0f}", consecutive_net_days(chip_df[name]) if name in chip_df else "無資料")

if not chip_df.empty:
    chip_display = chip_df.tail(10).iloc[::-1].copy()
    chip_display.index = chip_display.index.strftime("%Y-%m-%d")
    chip_display.index.name = "日期"
    st.dataframe(chip_display, use_container_width=True)
else:
    st.info("目前沒有法人資料。")

st.caption(
    "法人資料來自 FinMind；若來源沒有回傳資料，圖表會以 0 顯示，這不代表法人當日一定沒有交易。"
    "請留意來源資料的更新時間與買賣超單位。"
)

# ------------------------- 頁尾 ----------------------------
with st.expander("資料來源與計算說明"):
    st.markdown(
        """
        - 行情：Yahoo Finance（yfinance），上市代號先查 `.TW`，無資料再查 `.TWO`。
        - 三大法人：FinMind Taiwan institutional investors 資料。
        - 均線：5、10、20、60 期簡單移動平均。
        - MACD：EMA(12) − EMA(26)，訊號線為 EMA(9)。
        - RSI：以 Wilder 風格的指數平滑近似計算 RSI(6)、RSI(14)。
        - KD：9 期 RSV，K、D 使用指數平滑近似計算。
        - 支撐／壓力：近 60 個日線交易日的最低價與最高價，僅作參考。
        - 自選股清單目前存在 Streamlit session state；重開工作階段可能重設。
        """
    )
'''
