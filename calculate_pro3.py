import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta, time
import pytz

# ---------------------------------------------------------
# 1. 設定と関数定義
# ---------------------------------------------------------

def get_usdjpy_rate():
    try:
        ticker = yf.Ticker("USDJPY=X")
        data = ticker.history(period="1d")
        if not data.empty:
            return data['Close'].iloc[-1]
        return 150.0
    except:
        return 150.0

def calculate_technical_indicators(df):
    df = df.copy()
    df = df.ffill().dropna()
    
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['BB_upper'] = df['SMA_20'] + 2 * df['Close'].rolling(window=20).std()
    df['BB_lower'] = df['SMA_20'] - 2 * df['Close'].rolling(window=20).std()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['Momentum'] = df['Close'] - df['Close'].shift(10)

    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']

    df['SMA_20_Slope'] = df['SMA_20'].diff()

    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    
    df['UpMove'] = df['High'] - df['High'].shift(1)
    df['DownMove'] = df['Low'].shift(1) - df['Low']
    df['+DM'] = np.where((df['UpMove'] > df['DownMove']) & (df['UpMove'] > 0), df['UpMove'], 0.0)
    df['-DM'] = np.where((df['DownMove'] > df['UpMove']) & (df['DownMove'] > 0), df['DownMove'], 0.0)
    
    safe_atr = df['ATR'].replace(0, np.nan)
    df['+DI'] = 100 * (pd.Series(df['+DM']).ewm(alpha=1/14, adjust=False).mean() / safe_atr)
    df['-DI'] = 100 * (pd.Series(df['-DM']).ewm(alpha=1/14, adjust=False).mean() / safe_atr)
    di_sum = df['+DI'] + df['-DI']
    df['DX'] = 100 * np.abs(df['+DI'] - df['-DI']) / di_sum.replace(0, np.nan)
    df['ADX'] = df['DX'].ewm(alpha=1/14, adjust=False).mean()
    
    df['RSI_Lag1'] = df['RSI'].shift(1)
    df['Close_Pct_Change'] = df['Close'].pct_change()
    df['Close_Pct_Lag1'] = df['Close_Pct_Change'].shift(1)

    df.dropna(inplace=True)
    return df

def fetch_and_resample_data(ticker, timeframe, duration_days):
    end_date = datetime.now(pytz.utc) + timedelta(days=2)
    start_date = end_date - timedelta(days=duration_days + 60)
    yf_intervals = {"5m": "5m", "1h": "1h", "1d": "1d"}
    interval_to_fetch = yf_intervals[timeframe] if timeframe in yf_intervals else "1h"
    
    try:
        df = yf.download(ticker, start=start_date, end=end_date, interval=interval_to_fetch, progress=False)
    except:
        return None
        
    if df is None or df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
    if timeframe not in yf_intervals:
        logic = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
        if 'Volume' in df.columns: logic['Volume'] = 'sum'
        resample_rule = "6h" if timeframe == "6h" else "12h"
        df = df.resample(resample_rule).apply(logic).dropna()
        
    if df.empty: return None
    if df.index.tz is None: df.index = df.index.tz_localize('UTC')
    else: df.index = df.index.tz_convert('UTC')
        
    return calculate_technical_indicators(df)

def simulate_trade(df, start_time_utc, trade_type, entry_price, tp_price, sl_price, prediction_steps):
    future_candles = df[df.index > start_time_utc].head(prediction_steps)
    if future_candles.empty: return "NO_DATA", None, None, None
    hit_result, hit_price, close_time = "DRAW", future_candles.iloc[-1]['Close'], future_candles.index[-1]
    
    for idx, row in future_candles.iterrows():
        if trade_type == "BUY":
            if row['Low'] <= sl_price: hit_result, hit_price, close_time = "LOSS", sl_price, idx; break
            elif row['High'] >= tp_price: hit_result, hit_price, close_time = "WIN", tp_price, idx; break
        else:
            if row['High'] >= sl_price: hit_result, hit_price, close_time = "LOSS", sl_price, idx; break
            elif row['Low'] <= tp_price: hit_result, hit_price, close_time = "WIN", tp_price, idx; break
    return hit_result, hit_price, close_time, entry_price

def train_and_predict(df, target_dt_utc, prediction_steps=6):
    train_data = df[df.index < target_dt_utc].dropna().copy()
    try:
        target_idx = df.index.get_indexer([target_dt_utc], method='pad')[0]
        prediction_row = df.iloc[[target_idx]].copy()
    except: return None
    if len(train_data) < 50: return None

    df['Target_Price'] = df['Close'].shift(-prediction_steps)
    df['Target'] = (df['Target_Price'] > df['Close']).astype(int)
    features = ['Close', 'SMA_5', 'SMA_20', 'RSI', 'BB_upper', 'BB_lower', 'Momentum', 'MACD', 'Signal', 'MACD_Hist', 'SMA_20_Slope', 'ADX', 'RSI_Lag1', 'Close_Pct_Lag1']
    
    model = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_split=5, random_state=42)
    model.fit(train_data[features], df.loc[train_data.index, 'Target'])
    proba = model.predict_proba(prediction_row[features])[0]
    fi_df = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False)
    return proba, prediction_row['Close'].values[0], prediction_row['ATR'].values[0], prediction_row['ADX'].values[0], fi_df, prediction_row.index[0]

# --- 個別/グローバル最適化ロジック ---
def run_optimization(tickers, timeframe, duration_key):
    duration_map = {"1日": 1, "1週間": 7, "1ヶ月": 30, "1年": 365}
    days = duration_map[duration_key]
    best_r = -float('inf')
    best_combo = None
    usdjpy_rate = get_usdjpy_rate()
    jst = pytz.timezone('Asia/Tokyo')
    
    for tk in tickers:
        df = fetch_and_resample_data(tk, timeframe, days)
        if df is None or len(df) < 80: continue
        test_start = df.index[-1] - timedelta(days=days)
        train_df, test_df = df[df.index < test_start].dropna(), df[df.index >= test_start]
        if len(train_df) < 50 or len(test_df) < 5: continue
        
        features = ['Close', 'SMA_5', 'SMA_20', 'RSI', 'BB_upper', 'BB_lower', 'Momentum', 'MACD', 'Signal', 'MACD_Hist', 'SMA_20_Slope', 'ADX', 'RSI_Lag1', 'Close_Pct_Lag1']
        df_temp = df.copy()
        df_temp['Target'] = (df_temp['Close'].shift(-6) > df_temp['Close']).astype(int)
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(train_df[features], df_temp.loc[train_df.index, 'Target'])
        preds = model.predict_proba(test_df[features])
        
        for rr in [1.5, 2.0, 3.0]:
            for sl in [1.0, 1.5, 2.0]:
                total_r, total_units, next_entry, trade_logs = 0, 0, None, []
                for i in range(len(test_df)):
                    curr_t = test_df.index[i]
                    if next_entry and curr_t <= next_entry: continue
                    row = test_df.iloc[i]
                    direction = "BUY" if preds[i][1] > preds[i][0] else "SELL"
                    sl_dist = row['ATR'] * sl
                    tp_p = row['Close'] + (sl_dist * rr) if direction == "BUY" else row['Close'] - (sl_dist * rr)
                    sl_p = row['Close'] - sl_dist if direction == "BUY" else row['Close'] + sl_dist
                    res, exit_p, exit_t, entry_p = simulate_trade(df, curr_t, direction, row['Close'], tp_p, sl_p, 6)
                    if res != "NO_DATA":
                        r_score = rr if res == "WIN" else (-1.0 if res == "LOSS" else 0.0)
                        total_r += r_score
                        next_entry = exit_t
                        trade_logs.append({
                            "通貨": tk, "種別": direction, "Entry時間 (JST)": curr_t.astimezone(jst).strftime('%m/%d %H:%M'),
                            "Exit時間 (JST)": exit_t.astimezone(jst).strftime('%m/%d %H:%M'), "Entry価格": f"{entry_p:.5f}",
                            "Exit価格": f"{exit_p:.5f}", "結果": res, "Rスコア": r_score
                        })
                        if sl_dist > 0: total_units += 10000 / (sl_dist * (usdjpy_rate if "USD" in tk else 1.0))
                
                if total_r > best_r:
                    best_r = total_r
                    best_combo = { "asset": tk, "rr": rr, "sl": sl, "r_profit": total_r, "trades": len(trade_logs), "logs": trade_logs, "avg_u": total_units / (len(trade_logs) or 1)}
    return best_combo

# ---------------------------------------------------------
# 2. UI
# ---------------------------------------------------------

st.set_page_config(page_title="FX AI Pro", page_icon="📈", layout="wide")
st.title("💹 AI FX マルチタイムフレーム・トレーダー")

# --- サイドバー ---
st.sidebar.header("🎛️ グローバル設定")
tf_choice = st.sidebar.selectbox("使用する時間足", ["5m", "1h", "6h", "12h", "1d"], index=1)
bt_duration = st.sidebar.selectbox("バックテスト期間", ["1日", "1週間", "1ヶ月", "1年"], index=1)

if st.sidebar.button("🏆 全通貨から最強を探す"):
    with st.spinner("総当たり検証中..."):
        best = run_optimization(["USDJPY=X", "EURUSD=X", "GBPUSD=X", "GC=F", "BTC-USD", "ETH-USD", "SI=F"], tf_choice, bt_duration)
        if best:
            st.sidebar.success(f"**【最強設定判明】**\n\n👑 {best['asset']}\n⚖️ RR: {best['rr']} / 🛑 SL: {best['sl']}ATR\n📊 平均: {best['avg_u']:,.0f} Units\n💰 利益: +{best['r_profit']*10000:,.0f}円")
            st.session_state.best_logs = best['logs']
        else: st.sidebar.warning("有効な設定なし")

st.sidebar.markdown("---")
st.sidebar.header("🎯 特定ペアの個別最適化")
opt_ticker = st.sidebar.text_input("最適化する通貨ペア", "USDJPY=X")
if st.sidebar.button("🔍 このペアで最適設定を探す"):
    with st.spinner(f"{opt_ticker} を徹底分析中..."):
        best = run_optimization([opt_ticker], tf_choice, bt_duration)
        if best:
            st.sidebar.success(f"**【{opt_ticker} の最適設定】**\n\n⚖️ RR: {best['rr']} / 🛑 SL: {best['sl']}ATR\n📊 平均: {best['avg_u']:,.0f} Units\n💰 利益: +{best['r_profit']*10000:,.0f}円\n📈 トレード数: {best['trades']}回")
            st.session_state.best_logs = best['logs']
        else: st.sidebar.warning("データ不足または利益設定が見つかりません。")

st.sidebar.markdown("---")
ticker1 = st.sidebar.text_input("分析通貨 (メイン表示用)", "USDJPY=X")
pred_steps = st.sidebar.slider("予測本数 (Steps)", 1, 24, 6)

# 資金管理
st.sidebar.markdown("---")
trade_units = st.sidebar.number_input("手動取引量", 0.01, 1e7, 10000.0)
rr_input = st.sidebar.number_input("手動リスクリワード", 1.0, 10.0, 2.0)
sl_input = st.sidebar.number_input("手動損切りATR倍率", 0.01, 5.0, 1.5)

# --- 実行 ---
if st.button("🚀 予測実行"):
    jst = pytz.timezone('Asia/Tokyo')
    target_dt = datetime.now(jst)
    with st.spinner("解析中..."):
        df = fetch_and_resample_data(ticker1, tf_choice, 30)
        if df is not None:
            res = train_and_predict(df, target_dt.astimezone(pytz.utc), pred_steps)
            if res:
                proba, price_now, atr, adx, fi, used_t = res
                regime = "トレンド" if adx > 25 else "レンジ"
                st.info(f"🧭 相場環境: **{regime}** (ADX: {adx:.1f}) | 最終データ: {used_t.astimezone(jst)}")
                c1, c2, c3 = st.columns(3)
                c1.metric("開始価格", f"{price_now:.5f}"); c2.metric("AI予測", "UP ↗️" if proba[1]>proba[0] else "DOWN ↘️", f"{max(proba)*100:.1f}%")
                
                sl_d = atr * sl_input; tp_d = sl_d * rr_input; direction = "BUY" if proba[1]>proba[0] else "SELL"
                tp_p = price_now + tp_d if direction=="BUY" else price_now - tp_d
                sl_p = price_now - sl_d if direction=="BUY" else price_now + sl_d
                
                st.markdown("---"); st.subheader("🛡️ トレードプラン")
                p1, p2, p3 = st.columns(3)
                p1.warning(f"利確: {tp_p:.5f}"); p2.info(f"Entry: {price_now:.5f}"); p3.error(f"損切り: {sl_p:.5f}")
                st.line_chart(df.tail(50)['Close'])
                st.subheader("🧠 判断根拠"); st.bar_chart(fi.set_index('Feature'))

# --- ログ/グラフ ---
if 'best_logs' in st.session_state:
    st.markdown("---")
    st.subheader("📈 最適化バックテストの結果")
    h_jpy = [l['Rスコア'] * 10000 for l in st.session_state.best_logs]
    h_df = pd.DataFrame({"損益 (円)": h_jpy})
    h_df['累積損益'] = h_df['損益 (円)'].cumsum()
    g1, g2 = st.columns(2)
    with g1: st.caption("個別損益"); st.bar_chart(h_df['損益 (円)'])
    with g2: st.caption("累積資産"); st.line_chart(h_df['累積損益'])
    st.subheader("📑 詳細履歴")
    st.table(pd.DataFrame(st.session_state.best_logs))
