import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from FinMind.data import DataLoader

# === 1. iPhone 專屬全螢幕與手勢支援設定 ===
st.set_page_config(
    page_title="台股盤後分析站", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# 隱藏選單並透過 CSS 開放手機繪圖區的雙指縮放與觸控手勢
st.markdown("""
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        [data-testid="collapsedControl"] {display: none;}
        .block-container {padding-top: 1rem; padding-bottom: 0rem; padding-left: 0.5rem; padding-right: 0.5rem;}
        
        /* 關鍵：讓 Plotly 圖表區域完全接收手機縮放與滑動手勢 */
        .js-plotly-plot .plotly, .js-plotly-plot .plotly div {
            touch-action: auto !important;
        }
    </style>
""", unsafe_allow_html=True)

# === 2. 快取抓取資料 (自動判斷上市櫃) ===
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_raw_data(ticker_num, start_d, end_d):
    tw_stock_id = str(ticker_num).strip().upper().replace('.TW', '').replace('.TWO', '')
    
    yf_ticker = f"{tw_stock_id}.TW"
    stock = yf.Ticker(yf_ticker)
    df = stock.history(start=start_d, end=end_d)
    
    if df.empty:
        yf_ticker = f"{tw_stock_id}.TWO"
        stock = yf.Ticker(yf_ticker)
        df = stock.history(start=start_d, end=end_d)
        
    if df.empty: return None, tw_stock_id, None
    df.index = df.index.tz_localize(None)
    
    dl = DataLoader()
    inst_df = dl.taiwan_stock_institutional_investors(
        stock_id=tw_stock_id, start_date=start_d.strftime('%Y-%m-%d'), end_date=end_d.strftime('%Y-%m-%d')
    )
    
    if not inst_df.empty:
        inst_df['net_buy'] = inst_df['buy'] - inst_df['sell']
        def map_inst_name(name):
            if '外資' in name: return '外資'
            if '投信' in name: return '投信'
            if '自營商' in name: return '自營商'
            return '其他'
        inst_df['short_name'] = inst_df['name'].apply(map_inst_name)
        chip_df = inst_df.groupby(['date', 'short_name'])['net_buy'].sum().unstack(fill_value=0)
        chip_df.index = pd.to_datetime(chip_df.index)
        df = df.join(chip_df, how='left').fillna(0)
        
    return df, tw_stock_id, yf_ticker

# === 3. 單頁控制項 ===
col_stock, col_period, col_range = st.columns([1.2, 1, 1.2])

with col_stock:
    ticker_input = st.text_input("股票代號", value="2330", label_visibility="collapsed", placeholder="輸入代號(如2330)")

with col_period:
    period = st.radio("週期", ["日線", "周線"], horizontal=True, label_visibility="collapsed")

with col_range:
    time_range = st.selectbox("區間", ["近 3 個月", "近半年", "近 1 年", "近 3 年"], index=2, label_visibility="collapsed")

end_date = datetime.date.today()
if time_range == "近 3 個月": start_date = end_date - datetime.timedelta(days=90)
elif time_range == "近半年": start_date = end_date - datetime.timedelta(days=180)
elif time_range == "近 1 年": start_date = end_date - datetime.timedelta(days=365)
elif time_range == "近 3 年": start_date = end_date - datetime.timedelta(days=365*3)

raw_df, tw_stock_id, yf_ticker = fetch_raw_data(ticker_input, start_date, end_date)

if raw_df is None:
    st.error(f"⚠️ 找不到代號 `{ticker_input}` 資料，請確認是否正確。")
else:
    df = raw_df.copy()

    if period == "周線":
        agg_rules = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
        for chip_col in ['外資', '投信', '自營商']:
            if chip_col in df.columns: agg_rules[chip_col] = 'sum'
        df = df.resample('W-FRI').agg(agg_rules).dropna()

    # 技術指標運算
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_60'] = df['Close'].rolling(window=60).mean()
    
    std_20 = df['Close'].rolling(window=20).std()
    df['BBU_20_2.0'] = df['SMA_20'] + 2 * std_20
    df['BBL_20_2.0'] = df['SMA_20'] - 2 * std_20
    
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['OSC'] = df['MACD'] - df['Signal']
    
    def calc_rsi(data, periods):
        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        rs = (gain.ewm(alpha=1/periods, adjust=False).mean()) / (loss.ewm(alpha=1/periods, adjust=False).mean())
        return 100 - (100 / (1 + rs))
    
    df['RSI_6'] = calc_rsi(df['Close'], 6)
    df['RSI_14'] = calc_rsi(df['Close'], 14)

    min_low9 = df['Low'].rolling(window=9).min()
    max_high9 = df['High'].rolling(window=9).max()
    df['RSV'] = (df['Close'] - min_low9) / (max_high9 - min_low9) * 100
    df['RSV'] = df['RSV'].fillna(50)
    df['K'] = df['RSV'].ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()

    # === 4. 圖表繪製 ===
    fig = make_subplots(
        rows=6, cols=1, shared_xaxes=True, 
        vertical_spacing=0.015, 
        row_heights=[0.45, 0.11, 0.11, 0.11, 0.11, 0.11], 
        subplot_titles=("K線與均線", "成交量", "MACD", "RSI", "KD", "三大法人")
    )

    hover_kline = "日期: %{x|%m/%d}<br>開: %{open}<br>高: %{high}<br>低: %{low}<br>收: %{close}<extra></extra>"
    
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], increasing_line_color='#FF3333', decreasing_line_color='#00CC00', name='K線', hovertemplate=hover_kline), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_5'], line=dict(color='white', width=1), name='5MA', hovertemplate="5MA: %{y:.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_10'], line=dict(color='yellow', width=1), name='10MA', hovertemplate="10MA: %{y:.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], line=dict(color='#FF66FF', width=1), name='20MA', hovertemplate="20MA: %{y:.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_60'], line=dict(color='#00FF00', width=1), name='60MA', hovertemplate="60MA: %{y:.2f}<extra></extra>"), row=1, col=1)
    
    vol_colors = ['#FF3333' if c >= o else '#00CC00' for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=vol_colors, name='成交量', hovertemplate="量: %{y}<extra></extra>"), row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='white', width=1), name='MACD', hovertemplate="MACD: %{y:.2f}<extra></extra>"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], line=dict(color='yellow', width=1), name='Signal', hovertemplate="Signal: %{y:.2f}<extra></extra>"), row=3, col=1)
    macd_colors = ['#FF3333' if val >= 0 else '#00CC00' for val in df['OSC']]
    fig.add_trace(go.Bar(x=df.index, y=df['OSC'], marker_color=macd_colors, name='OSC', hovertemplate="OSC: %{y:.2f}<extra></extra>"), row=3, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['RSI_6'], line=dict(color='white', width=1), name='RSI(6)', hovertemplate="RSI(6): %{y:.2f}<extra></extra>"), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI_14'], line=dict(color='yellow', width=1), name='RSI(14)', hovertemplate="RSI(14): %{y:.2f}<extra></extra>"), row=4, col=1)
    fig.add_hline(y=80, line_dash="dot", line_color="rgba(255,255,255,0.3)", row=4, col=1)
    fig.add_hline(y=20, line_dash="dot", line_color="rgba(255,255,255,0.3)", row=4, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['K'], line=dict(color='white', width=1), name='K(9)', hovertemplate="K: %{y:.2f}<extra></extra>"), row=5, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['D'], line=dict(color='yellow', width=1), name='D(3)', hovertemplate="D: %{y:.2f}<extra></extra>"), row=5, col=1)
    fig.add_hline(y=80, line_dash="dot", line_color="rgba(255,255,255,0.3)", row=5, col=1)
    fig.add_hline(y=20, line_dash="dot", line_color="rgba(255,255,255,0.3)", row=5, col=1)

    if '外資' in df.columns: fig.add_trace(go.Bar(x=df.index, y=df['外資'], name='外資', marker_color='#FF3333'), row=6, col=1)
    if '投信' in df.columns: fig.add_trace(go.Bar(x=df.index, y=df['投信'], name='投信', marker_color='#00CC00'), row=6, col=1)
    if '自營商' in df.columns: fig.add_trace(go.Bar(x=df.index, y=df['自營商'], name='自營商', marker_color='#FFFF00'), row=6, col=1)

    fig.update_layout(
        height=820, 
        margin=dict(l=5, r=5, t=20, b=5), 
        xaxis_rangeslider_visible=False, 
        template="plotly_dark", 
        hovermode="x", 
        barmode='group',
        showlegend=False,
        dragmode="zoom" # 關鍵修正：設為 zoom 以支援手機畫面的捏合放縮手勢
    )
    
    if period == "日線":
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])], showgrid=True, gridcolor='rgba(128,128,128,0.1)')
    else:
        fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.1)')
        
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.1)', side="right")
    
    for annotation in fig['layout']['annotations']: annotation['font'] = dict(size=11, color='gray')
    
    # 關鍵修正：開啟 scrollZoom: True 讓雙指縮放功能生效
    st.plotly_chart(
        fig, 
        use_container_width=True, 
        config={
            'displayModeBar': False,
            'scrollZoom': True
        }
    )
