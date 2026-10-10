import datetime
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
    if "ticker_input" not in st.session_state:
        st.session_state.ticker_input = st.session_state.ticker_value
    if "favorites_selector" not in st.session_state:
        st.session_state.favorites_selector = "2330｜台積電"


def sync_favorite_to_ticker():
    selected = st.session_state.get("favorites_selector", "")
    if "｜" in selected:
        selected_id = selected.split("｜", 1)[0].strip().upper()
        st.session_state.ticker_value = selected_id
        st.session_state.ticker_input = selected_id


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
st.markdown(
    """
    <style>
    .stApp { background: #0e1117; color: #e8edf5; }
    .block-container { max-width: 1800px; padding-top: .5rem; padding-bottom: .5rem; }
    div[data-testid="stHorizontalBlock"] { gap: .55rem; }
    div[data-testid="stTextInput"] input, div[data-testid="stSelectbox"] div[data-baseweb="select"] {
        min-height: 36px;
    }
    div[data-testid="stRadio"] > div { gap: .55rem; }
    div[data-testid="stCheckbox"] { margin-top: 0; }
    .quote-line { color:#c8d1df; font-size:13px; line-height:1.8; margin: 2px 0 5px 0; }
    .quote-up { color:#e53945; } .quote-down { color:#20a779; }
    </style>
    """,
    unsafe_allow_html=True,
)
# 第一列：所有主要操作集中在圖表上方
c1, c2, c3, c4, c5 = st.columns([1.1, 1.45, 1.5, 1.45, 1.35])
with c1:
    st.text_input(
        "股票代號", key="ticker_input", placeholder="輸入代碼後按 Enter",
        label_visibility="collapsed"
    )
    typed_ticker = str(st.session_state.ticker_input).strip().upper().replace(".TW", "").replace(".TWO", "")
    if typed_ticker and typed_ticker != st.session_state.ticker_value:
        st.session_state.ticker_value = typed_ticker
with c2:
    fav_items = list(st.session_state.favorites.items())
    fav_labels = [f"{sid}｜{name}" for sid, name in fav_items]
    current_label = next((f"{sid}｜{name}" for sid, name in fav_items
                          if sid == st.session_state.ticker_value), fav_labels[0] if fav_labels else "")
    if fav_items:
        if st.session_state.get("favorites_selector") not in fav_labels:
            st.session_state.favorites_selector = current_label
        st.selectbox(
            "自選股", fav_labels, key="favorites_selector",
            on_change=sync_favorite_to_ticker,
            label_visibility="collapsed"
        )
with c3:
    period = st.radio("週期", ["日線", "週線", "月線"], horizontal=True, index=0, label_visibility="collapsed")
with c4:
    range_label = st.selectbox(
        "顯示區間", ["近 1 個月", "近 3 個月", "近半年", "近 1 年", "近 3 年", "全部"],
        index=3, label_visibility="collapsed"
    )
with c5:
    indicator = st.selectbox(
        "副圖指標", ["成交量", "MACD", "RSI", "KD", "三大法人"],
        index=0, label_visibility="collapsed"
    )

# 第二列：均線直接開關；不再需要往下找控制項
m1, m2, m3, m4 = st.columns([.7, .8, .8, 4.7])
with m1:
    show_ma5 = st.checkbox("5MA", value=True)
with m2:
    show_ma20 = st.checkbox("20MA", value=True)
with m3:
    show_ma60 = st.checkbox("60MA", value=True)
with m4:
    if st.button("＋ 加入目前股票到自選股", type="secondary"):
        current = st.session_state.ticker_value.strip().upper()
        if current:
            st.session_state.favorites[current] = STOCK_NAMES.get(current, current)
            st.toast(f"已加入 {current}")
            st.rerun()

# ---------------------- 取得行情 ---------------------------
ticker = st.session_state.ticker_value
end_date = datetime.date.today()
days_back = 365 * 5 if range_label == "全部" else max(RANGE_MAP[range_label] * 2, 100)
start_date = end_date - datetime.timedelta(days=days_back)

with st.spinner(f"正在載入 {ticker} 行情…"):
    raw_df, stock_id, yf_symbol = fetch_raw_data(ticker, start_date, end_date)

if raw_df is None or raw_df.empty:
    st.error(f"找不到股票代號「{ticker}」的行情資料，請確認代號或稍後再試。")
    st.stop()

df_daily = add_indicators(raw_df)
if period == "週線":
    df = resample_weekly(df_daily)
elif period == "月線":
    df = df_daily.resample("ME").agg({
        "Open":"first", "High":"max", "Low":"min", "Close":"last",
        "Volume":"sum", "外資":"sum", "投信":"sum", "自營商":"sum"
    }).dropna(subset=["Open", "High", "Low", "Close"])
else:
    df = df_daily.copy()
df = add_indicators(df)
if df.empty:
    st.error("目前沒有足夠資料可以繪圖。")
    st.stop()

range_count = RANGE_MAP[range_label]
view_df = df.copy() if range_label == "全部" else df.tail(max(20, range_count)).copy()

last = df.iloc[-1]
prev_close = df["Close"].iloc[-2] if len(df) >= 2 else last["Close"]
change = float(last["Close"] - prev_close)
change_pct = change / prev_close * 100 if prev_close else 0
stock_name = STOCK_NAMES.get(stock_id, stock_id)
change_class = "quote-up" if change >= 0 else "quote-down"
st.markdown(
    f'<div class="quote-line"><b>{stock_id} {stock_name}</b>　'
    f'收 <b>{fmt_num(last["Close"])}</b>　'
    f'<span class="{change_class}">{change:+.2f} ({change_pct:+.2f}%)</span>　'
    f'開 {fmt_num(last["Open"])}　高 {fmt_num(last["High"])}　低 {fmt_num(last["Low"])}　'
    f'量 {last["Volume"]:,.0f}　<span style="color:#8492a6">{df.index[-1].strftime("%Y-%m-%d")}</span></div>',
    unsafe_allow_html=True
)

support, resistance = find_support_resistance(df_daily, 60)
fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.035,
    row_heights=[0.76, 0.24], subplot_titles=("K 線", indicator)
)
x = view_df.index
hover_kline = "日期：%{x|%Y-%m-%d}<br>開：%{open:.2f}<br>高：%{high:.2f}<br>低：%{low:.2f}<br>收：%{close:.2f}<extra></extra>"
fig.add_trace(go.Candlestick(
    x=x, open=view_df["Open"], high=view_df["High"], low=view_df["Low"], close=view_df["Close"],
    increasing_line_color="#e53945", decreasing_line_color="#20a779", name="K 線",
    hovertemplate=hover_kline
), row=1, col=1)

ma_styles = [
    ("SMA_5", "#f39c35", "5MA", show_ma5),
    ("SMA_20", "#3987ff", "20MA", show_ma20),
    ("SMA_60", "#38c6c9", "60MA", show_ma60),
]
for col_name, color, label, visible in ma_styles:
    if visible:
        fig.add_trace(go.Scatter(
            x=x, y=view_df[col_name], mode="lines", line=dict(color=color, width=1.25),
            name=label, hovertemplate=f"{label}: %{{y:.2f}}<extra></extra>"
        ), row=1, col=1)

if indicator == "成交量":
    vol_colors = ["#e53945" if c >= o else "#20a779" for c, o in zip(view_df["Close"], view_df["Open"])]
    fig.add_trace(go.Bar(x=x, y=view_df["Volume"], marker_color=vol_colors, name="成交量"), row=2, col=1)
elif indicator == "MACD":
    fig.add_trace(go.Scatter(x=x, y=view_df["MACD"], line=dict(color="#3987ff", width=1.4), name="MACD"), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=view_df["Signal"], line=dict(color="#f39c35", width=1.2), name="訊號線"), row=2, col=1)
    osc_colors = ["#e53945" if v >= 0 else "#20a779" for v in view_df["OSC"].fillna(0)]
    fig.add_trace(go.Bar(x=x, y=view_df["OSC"], marker_color=osc_colors, name="OSC"), row=2, col=1)
elif indicator == "RSI":
    fig.add_trace(go.Scatter(x=x, y=view_df["RSI_6"], line=dict(color="#3987ff", width=1.2), name="RSI(6)"), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=view_df["RSI_14"], line=dict(color="#f39c35", width=1.2), name="RSI(14)"), row=2, col=1)
    for yline in [30, 70]:
        fig.add_hline(y=yline, line_dash="dot", line_color="rgba(128,128,128,.5)", row=2, col=1)
elif indicator == "KD":
    fig.add_trace(go.Scatter(x=x, y=view_df["K"], line=dict(color="#3987ff", width=1.2), name="K"), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=view_df["D"], line=dict(color="#f39c35", width=1.2), name="D"), row=2, col=1)
    for yline in [20, 80]:
        fig.add_hline(y=yline, line_dash="dot", line_color="rgba(128,128,128,.5)", row=2, col=1)
else:
    for chip_col, color in zip(CHIP_COLS, ["#e53945", "#20a779", "#f39c35"]):
        fig.add_trace(go.Bar(x=x, y=view_df[chip_col], name=chip_col, marker_color=color), row=2, col=1)

if support is not None and resistance is not None:
    fig.add_hline(y=support, row=1, col=1, line_dash="dot", line_color="rgba(32,167,121,.7)",
                  annotation_text=f"低點 {support:.2f}", annotation_position="bottom right")
    fig.add_hline(y=resistance, row=1, col=1, line_dash="dot", line_color="rgba(229,57,69,.7)",
                  annotation_text=f"高點 {resistance:.2f}", annotation_position="top right")

fig.update_layout(
    height=790, margin=dict(l=8, r=12, t=36, b=8), template="plotly_dark",
    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font=dict(color="#e8edf5", size=12),
    hovermode="x unified", barmode="group", showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    dragmode="pan", uirevision=f"{stock_id}-{period}-{indicator}"
)
fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,.15)", rangeslider_visible=False,
                 showspikes=True, spikemode="across", spikesnap="cursor")
fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,.15)", side="right", fixedrange=False)
for annotation in fig["layout"]["annotations"]:
    annotation["font"] = dict(size=11, color="#7f9cc5")

st.plotly_chart(
    fig, use_container_width=True,
    config={
        "displayModeBar": True, "displaylogo": False, "scrollZoom": True,
        "doubleClick": "reset", "responsive": True,
        "modeBarButtonsToAdd": ["drawline", "eraseshape"],
        "toImageButtonOptions": {"format":"png", "filename":f"{stock_id}_stock_chart",
                                 "height":1000, "width":1600, "scale":2}
    }
)

# 其餘訊號與法人明細收進最底部折疊區，預設不佔畫面
with st.expander("技術訊號與法人明細（需要時再展開）", expanded=False):
    st.subheader("技術指標條件提示")
    signals = build_signals(df)
    st.dataframe(pd.DataFrame([{"類別": cat, "條件提示": message} for cat, message in signals]),
                 use_container_width=True, hide_index=True)
    st.subheader("三大法人籌碼")
    chip_df = df_daily[CHIP_COLS].copy().tail(30)
    if not chip_df.empty:
        chip_latest = chip_df.iloc[-1]
        cc1, cc2, cc3 = st.columns(3)
        for col, name in zip([cc1, cc2, cc3], CHIP_COLS):
            with col:
                col.metric(name, f"{float(chip_latest.get(name, 0)):+,.0f}",
                           consecutive_net_days(chip_df[name]))
        chip_display = chip_df.tail(10).iloc[::-1].copy()
        chip_display.index = chip_display.index.strftime("%Y-%m-%d")
        chip_display.index.name = "日期"
        st.dataframe(chip_display, use_container_width=True)
    st.caption("行情來源：Yahoo Finance；法人資料來源：FinMind。法人資料未回傳時，0 不代表當日一定沒有交易。")
