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
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval_to_fetch, progress=False)
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
    X_train = train_data[features]
    y_train = df.loc[X_train.index, 'Target']
    model = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_split=5, random_state=42)
    model.fit(X_train, y_train)
    proba = model.predict_proba(prediction_row[features])[0]
    fi_df = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False)
    return proba, prediction_row['Close'].values[0], prediction_row['ATR'].values[0], prediction_row['ADX'].values[0], fi_df, prediction_row.index[0]

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

# --- 最強設定探索エンジン (単一通貨・全通貨共通) ---
def run_backtest_engine(ticker, timeframe, duration_key, h_steps):
    duration_map = {"1日": 1, "1週間": 7, "1ヶ月": 30, "1年": 365}
    days = duration_map[duration_key]
    usdjpy_rate = get_usdjpy_rate()
    jst = pytz.timezone('Asia/Tokyo')
    
    df = fetch_and_resample_data(ticker, timeframe, days)
    if df is None or len(df) < 100: return None
    
    test_start = df.index[-1] - timedelta(days=days)
    train_df = df[df.index < test_start].dropna()
    test_df = df[df.index >= test_start]
    if len(train_df) < 50 or len(test_df) < 5: return None
    
    df_temp = df.copy()
    df_temp['Target'] = (df_temp['Close'].shift(-h_steps) > df_temp['Close']).astype(int)
    features = ['Close', 'SMA_5', 'SMA_20', 'RSI', 'BB_upper', 'BB_lower', 'Momentum', 'MACD', 'Signal', 'MACD_Hist', 'SMA_20_Slope', 'ADX', 'RSI_Lag1', 'Close_Pct_Lag1']
    
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(train_df[features], df_temp.loc[train_df.index, 'Target'])
    preds = model.predict_proba(test_df[features])
    
    best_r_in_tk = -float('inf')
    best_in_tk = None
    
    for rr in [1.0, 1.5, 2.0, 3.0]:
        for sl in [0.5, 1.0, 1.5, 2.0]:
            total_r, total_units, next_entry, trade_logs = 0, 0, None, []
            for i in range(len(test_df)):
                curr_time = test_df.index[i]
                if next_entry and curr_time <= next_entry: continue
                row = test_df.iloc[i]
                direction = "BUY" if preds[i][1] > preds[i][0] else "SELL"
                sl_dist, tp_dist = row['ATR'] * sl, row['ATR'] * sl * rr
                tp_p = row['Close'] + tp_dist if direction == "BUY" else row['Close'] - tp_dist
                sl_p = row['Close'] - sl_dist if direction == "BUY" else row['Close'] + sl_dist
                res, exit_p, exit_t, entry_p = simulate_trade(df, curr_time, direction, row['Close'], tp_p, sl_p, h_steps)
                if res != "NO_DATA":
                    r_score = rr if res == "WIN" else (-1.0 if res == "LOSS" else 0.0)
                    total_r, next_entry = total_r + r_score, exit_t
                    trade_logs.append({
                        "通貨": ticker, "種別": direction, "Entry時間 (JST)": curr_time.astimezone(jst).strftime('%m/%d %H:%M'),
                        "Exit時間 (JST)": exit_t.astimezone(jst).strftime('%m/%d %H:%M'), "Entry価格": f"{entry_p:.5f}",
                        "Exit価格": f"{exit_p:.5f}", "結果": res, "Rスコア": r_score
                    })
                    total_units += 10000 / (sl_dist * (usdjpy_rate if "USD" in ticker else 1.0)) if sl_dist > 0 else 0
            
            if total_r > best_r_in_tk:
                best_r_in_tk = total_r
                best_in_tk = {"asset": ticker, "rr": rr, "sl": sl, "r_profit": total_r, "trades": len(trade_logs), "logs": trade_logs, "avg_u": total_units / (len(trade_logs) or 1)}
    return best_in_tk

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
    tickers = ["USDJPY=X", "EURUSD=X", "GBPUSD=X", "GC=F", "BTC-USD", "ETH-USD", "SI=F"]
    best_overall = None
    best_r_overall = -float('inf')
    with st.spinner("マーケット全体を走査中..."):
        for tk in tickers:
            res = run_backtest_engine(tk, tf_choice, bt_duration, 6)
            if res and res['r_profit'] > best_r_overall:
                best_r_overall, best_overall = res['r_profit'], res
        if best_overall:
            st.session_state.best_settings = best_overall
            st.sidebar.success(f"👑 全体最強: **{best_overall['asset']}**")
        else: st.sidebar.warning("有効なデータが見つかりません。")

st.sidebar.markdown("---")
ticker1 = st.sidebar.text_input("分析通貨 1", "USDJPY=X")

# --- 新機能: 特定通貨ペアの最強設定検索 ---
if st.sidebar.button(f"🔍 {ticker1} の最強設定を検索"):
    with st.spinner(f"{ticker1} の過去データを深層検証中..."):
        res = run_backtest_engine(ticker1, tf_choice, bt_duration, 6)
        if res:
            st.session_state.best_settings = res
            # 設定をUIに自動反映するためのフラグ
            st.session_state.rr_suggest = res['rr']
            st.session_state.sl_suggest = res['sl']
            st.sidebar.success(f"✅ {ticker1} 用の最適解を発見！")
        else: st.sidebar.warning("この通貨ペアの十分なデータがありません。")

use_realtime = st.sidebar.checkbox("🔴 リアルタイム予測", value=True)
pred_steps = st.sidebar.slider("予測本数 (Steps)", 1, 24, 6)

st.sidebar.markdown("---")
trade_units = st.sidebar.number_input("取引量", 0.01, 1e7, 10000.0)

# 最強設定検索から戻った値があれば初期値に設定
rr_init = st.session_state.get('rr_suggest', 2.0)
sl_init = st.session_state.get('sl_suggest', 1.5)
rr_input = st.sidebar.number_input("リスクリワード", 1.0, 10.0, rr_init)
sl_input = st.sidebar.number_input("損切りATR倍率", 0.01, 5.0, sl_init)

# --- バックテスト詳細表示 (サイドバー) ---
if 'best_settings' in st.session_state:
    best = st.session_state.best_settings
    st.sidebar.markdown(f"### 🏆 検証結果: {best['asset']}")
    st.sidebar.write(f"RR: **{best['rr']}** / SL: **{best['sl']}**")
    st.sidebar.write(f"利益: **+{best['r_profit']*10000:,.0f}円** ({best['trades']}回)")
    
    if len(best['logs']) > 0:
        history_jpy = [log['Rスコア'] * 10000 for log in best['logs']]
        history_df = pd.DataFrame({"損益": history_jpy})
        history_df['累積'] = history_df['損益'].cumsum()
        st.sidebar.line_chart(history_df['累積'])

# --- メイン実行 ---
if st.button("🚀 予測実行"):
    jst = pytz.timezone('Asia/Tokyo')
    target_dt = datetime.now(jst)
    with st.spinner("AI解析中..."):
        df = fetch_and_resample_data(ticker1, tf_choice, 30)
        if df is not None:
            res = train_and_predict(df, target_dt.astimezone(pytz.utc), pred_steps)
            if res:
                proba, price_now, atr, adx, fi, used_t = res
                st.info(f"🧭 環境: **{'トレンド' if adx > 25 else 'レンジ'}** (ADX: {adx:.1f}) | 確定: {used_t.astimezone(jst)}")
                col1, col2, col3 = st.columns(3)
                col1.metric("開始価格", f"{price_now:.5f}")
                col2.metric("予測", "UP ↗️" if proba[1]>proba[0] else "DOWN ↘️", f"確信度: {max(proba)*100:.1f}%")
                sl_dist, tp_dist = atr * sl_input, atr * sl_input * rr_input
                direction = "BUY" if proba[1]>proba[0] else "SELL"
                tp_p = price_now + tp_dist if direction=="BUY" else price_now - tp_dist
                sl_p = price_now - sl_dist if direction=="BUY" else price_now + sl_dist
                st.markdown("---")
                st.subheader("🛡️ トレードプラン")
                c_tp, c_ent, c_sl = st.columns(3)
                c_tp.warning(f"利確: {tp_p:.5f}"); c_ent.info(f"Entry: {price_now:.5f}"); c_sl.error(f"損切り: {sl_p:.5f}")
                st.line_chart(df.tail(50)['Close'])
                st.subheader("🧠 判断根拠"); st.bar_chart(fi.set_index('Feature'))

if 'best_settings' in st.session_state:
    st.markdown("---")
    st.subheader("📑 バックテスト履歴ログ")
    st.table(pd.DataFrame(st.session_state.best_settings['logs']))
