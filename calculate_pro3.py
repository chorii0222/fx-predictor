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
    # 欠損値対策
    df = df.ffill().dropna()
    
    # 基本指標
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

    # MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']

    df['SMA_20_Slope'] = df['SMA_20'].diff()

    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    
    # ADX
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
    """
    指定された期間と時間足でデータを取得・リサンプルする
    """
    end_date = datetime.now(pytz.utc) + timedelta(days=2)
    start_date = end_date - timedelta(days=duration_days + 60)
    
    yf_intervals = {"5m": "5m", "1h": "1h", "1d": "1d"}
    
    # yfinanceからデータを取得
    interval_to_fetch = yf_intervals[timeframe] if timeframe in yf_intervals else "1h"
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval_to_fetch, progress=False)
        
    if df is None or df.empty: 
        return None
    
    # マルチインデックス対応をリサンプルの「前」に行う
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    # リサンプル処理
    if timeframe not in yf_intervals:
        logic = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
        if 'Volume' in df.columns:
            logic['Volume'] = 'sum'
            
        resample_rule = timeframe
        df = df.resample(resample_rule).apply(logic).dropna()
        
    if df.empty: 
        return None
        
    # タイムゾーン処理
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')
        
    return calculate_technical_indicators(df)

def train_and_predict(df, target_dt_utc, prediction_steps=6):
    train_data = df[df.index < target_dt_utc].dropna().copy()
    
    try:
        target_idx = df.index.get_indexer([target_dt_utc], method='pad')[0]
        prediction_row = df.iloc[[target_idx]].copy()
    except:
        return None

    if len(train_data) < 50: return None

    # ターゲット作成
    df['Target_Price'] = df['Close'].shift(-prediction_steps)
    df['Target'] = (df['Target_Price'] > df['Close']).astype(int)
    
    features = ['Close', 'SMA_5', 'SMA_20', 'RSI', 'BB_upper', 'BB_lower', 'Momentum', 'MACD', 'Signal', 'MACD_Hist', 'SMA_20_Slope', 'ADX', 'RSI_Lag1', 'Close_Pct_Lag1']
    X_train = train_data[features]
    y_train = df.loc[X_train.index, 'Target']
    
    model = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_split=5, random_state=42)
    model.fit(X_train, y_train)

    proba = model.predict_proba(prediction_row[features])[0]
    
    # 指標重要度
    fi_df = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False)
    
    return proba, prediction_row['Close'].values[0], prediction_row['ATR'].values[0], prediction_row['ADX'].values[0], fi_df, prediction_row.index[0]

def simulate_trade(df, start_time_utc, trade_type, entry_price, tp_price, sl_price, prediction_steps):
    """
    決済ログを詳細に返すシミュレータ
    """
    future_candles = df[df.index > start_time_utc].head(prediction_steps)
    if future_candles.empty: return "NO_DATA", None, None, None
    
    hit_result = "DRAW"
    hit_price = future_candles.iloc[-1]['Close']
    close_time = future_candles.index[-1]
    
    for idx, row in future_candles.iterrows():
        # 買いの場合
        if trade_type == "BUY":
            if row['Low'] <= sl_price:
                hit_result, hit_price, close_time = "LOSS", sl_price, idx
                break
            elif row['High'] >= tp_price:
                hit_result, hit_price, close_time = "WIN", tp_price, idx
                break
        # 売りの場合
        else:
            if row['High'] >= sl_price:
                hit_result, hit_price, close_time = "LOSS", sl_price, idx
                break
            elif row['Low'] <= tp_price:
                hit_result, hit_price, close_time = "WIN", tp_price, idx
                break
                
    return hit_result, hit_price, close_time, entry_price

# --- 最強設定探索 (改良版：DRAW時の正確な損益計算に対応) ---
@st.cache_data(show_spinner=False, ttl=3600)
def find_best_settings_dynamic(timeframe, duration_key, specific_ticker=None):
    duration_map = {"1日": 1, "1週間": 7, "1ヶ月": 30, "1年": 365}
    days = duration_map[duration_key]
    tickers = [specific_ticker] if specific_ticker else ["USDJPY=X", "EURUSD=X", "GBPUSD=X", "GC=F", "BTC-USD", "ETH-USD", "SI=F"]
    
    best_r = -float('inf')
    best_combo = None
    usdjpy_rate = get_usdjpy_rate()
    
    jst = pytz.timezone('Asia/Tokyo')
    
    for tk in tickers:
        df = fetch_and_resample_data(tk, timeframe, days)
        if df is None or len(df) < 100: continue
        
        test_start = df.index[-1] - timedelta(days=days)
        train_df = df[df.index < test_start].dropna()
        test_df = df[df.index >= test_start]
        
        if len(train_df) < 50 or len(test_df) < 10: continue
        
        step_grid = [1, 3, 6, 12] if specific_ticker else [6]
        
        for h in step_grid:
            df_temp = df.copy()
            df_temp['Target'] = (df_temp['Close'].shift(-h) > df_temp['Close']).astype(int)
            features = ['Close', 'SMA_5', 'SMA_20', 'RSI', 'BB_upper', 'BB_lower', 'Momentum', 'MACD', 'Signal', 'MACD_Hist', 'SMA_20_Slope', 'ADX', 'RSI_Lag1', 'Close_Pct_Lag1']
            
            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            model.fit(train_df[features], df_temp.loc[train_df.index, 'Target'])
            
            preds = model.predict_proba(test_df[features])
            
            for rr in [1.5, 2.0, 3.0]:
                for sl in [1.0, 1.5, 2.0]:
                    total_r, total_units, next_entry = 0, 0, None
                    trade_logs = []
                    
                    for i in range(len(test_df)):
                        curr_time = test_df.index[i]
                        if next_entry and curr_time <= next_entry: continue
                        
                        row = test_df.iloc[i]
                        direction = "BUY" if preds[i][1] > preds[i][0] else "SELL"
                        sl_dist = row['ATR'] * sl
                        tp_dist = sl_dist * rr
                        
                        tp_p = row['Close'] + tp_dist if direction == "BUY" else row['Close'] - tp_dist
                        sl_p = row['Close'] - sl_dist if direction == "BUY" else row['Close'] + sl_dist
                        
                        res, exit_p, exit_t, entry_p = simulate_trade(df, curr_time, direction, row['Close'], tp_p, sl_p, h)
                        
                        if res != "NO_DATA":
                            # ⚠️ 【修正箇所】DRAW時の正確な損益（Rスコア）の計算
                            if res == "WIN":
                                r_score = rr
                            elif res == "LOSS":
                                r_score = -1.0
                            else: # res == "DRAW"
                                if sl_dist > 0:
                                    p_dist = exit_p - entry_p if direction == "BUY" else entry_p - exit_p
                                    r_score = p_dist / sl_dist
                                else:
                                    r_score = 0.0

                            total_r += r_score
                            next_entry = exit_t
                            
                            entry_time_jst = curr_time.astimezone(jst).strftime('%m/%d %H:%M')
                            exit_time_jst = exit_t.astimezone(jst).strftime('%m/%d %H:%M')
                            
                            trade_logs.append({
                                "通貨": tk, "種別": direction, "Entry時間 (JST)": entry_time_jst,
                                "Exit時間 (JST)": exit_time_jst, "Entry価格": f"{entry_p:.5f}",
                                "Exit価格": f"{exit_p:.5f}", "結果": res, "Rスコア": r_score
                            })
                            
                            if sl_dist > 0:
                                u = 10000 / (sl_dist * (usdjpy_rate if "USD" in tk else 1.0))
                            else:
                                u = 0
                            total_units += u
                    
                    if total_r > best_r:
                        best_r = total_r
                        best_combo = {
                            "asset": tk, "rr": rr, "sl": sl, "steps": h, "r_profit": total_r,
                            "trades": len(trade_logs), "logs": trade_logs, "avg_u": total_units / (len(trade_logs) or 1)
                        }
    return best_combo

# ---------------------------------------------------------
# 2. UI
# ---------------------------------------------------------

st.set_page_config(page_title="FX AI Pro", page_icon="📈", layout="wide")
st.title("💹 AI FX マルチタイムフレーム・トレーダー")

# --- サイドバー ---
st.sidebar.header("🎛️ グローバル設定")
tf_choice = st.sidebar.selectbox("使用する時間足", ["5m", "1h", "2h", "3h", "6h", "12h", "1d"], index=1)
bt_duration = st.sidebar.selectbox("バックテスト期間", ["1日", "1週間", "1ヶ月", "1年"], index=1)

col_btn1, col_btn2 = st.sidebar.columns(2)
with col_btn1:
    btn_all = st.button("🏆 最強設定を探索")
with col_btn2:
    btn_specific = st.button("🎯 現在のペアで最適化")

def display_best_result(best):
    fixed_risk_jpy = 10000
    est_profit_jpy = best['r_profit'] * fixed_risk_jpy
    st.sidebar.success(f"**【最強設定が判明しました】**\n\n"
                       f"👑 通貨ペア: **{best['asset']}**\n\n"
                       f"⏳ 使用時間足: **{tf_choice}** (予測本数: {best['steps']}本)\n\n"
                       f"⚖️ RR比率: **{best['rr']}**\n\n"
                       f"🛑 損切り(ATR): **{best['sl']}**\n\n"
                       f"📊 平均取引量(目安): **約{best['avg_u']:,.2f} 通貨**\n\n"
                       f"💰 1回の損切りを1万円に固定した場合の利益:\n"
                       f"**+{est_profit_jpy:,.0f}円**\n\n"
                       f"※バックテスト回数: {best['trades']}回")
    st.session_state.best_logs = best['logs']
    if len(best['logs']) > 0:
        st.sidebar.markdown("### 📈 バックテストの損益推移")
        history_jpy = [log['Rスコア'] * fixed_risk_jpy for log in best['logs']]
        history_df = pd.DataFrame({"損益 (円)": history_jpy})
        history_df.index = history_df.index + 1 
        history_df['累積損益 (円)'] = history_df['損益 (円)'].cumsum()
        st.sidebar.caption("各トレードごとの損益 (棒グラフ)")
        st.sidebar.bar_chart(history_df['損益 (円)'])
        st.sidebar.caption("資産の推移 (折れ線グラフ)")
        st.sidebar.line_chart(history_df['累積損益 (円)'])

if btn_all:
    with st.spinner("全通貨・全設定を総当たり検証中..."):
        best = find_best_settings_dynamic(tf_choice, bt_duration)
        if best: display_best_result(best)
        else: st.sidebar.warning("有効な設定が見つかりませんでした。")

st.sidebar.markdown("---")
ticker1 = st.sidebar.text_input("分析通貨 1", "USDJPY=X")

if btn_specific:
    with st.spinner(f"{ticker1} の最適設定を算出中..."):
        best = find_best_settings_dynamic(tf_choice, bt_duration, specific_ticker=ticker1)
        if best: display_best_result(best)
        else: st.sidebar.warning(f"{ticker1} の有効な設定が見つかりませんでした。")

use_realtime = st.sidebar.checkbox("🔴 リアルタイム予測", value=True)
pred_steps = st.sidebar.slider("予測本数 (Steps)", 1, 24, 6, help="選択した時間足×本数 が未来の予測対象時間になります。")

st.sidebar.markdown("---")
trade_units = st.sidebar.number_input("取引量", 0.01, 1e7, 10000.0)
rr_input = st.sidebar.number_input("リスクリワード", 1.0, 10.0, 2.0)
sl_input = st.sidebar.number_input("損切りATR倍率", 0.01, 5.0, 1.5)

# --- メイン実行 ---
if st.button("🚀 予測実行"):
    jst = pytz.timezone('Asia/Tokyo')
    target_dt = datetime.now(jst)
    
    with st.spinner("解析中..."):
        df = fetch_and_resample_data(ticker1, tf_choice, 30) # 分析用
        if df is not None:
            res = train_and_predict(df, target_dt.astimezone(pytz.utc), pred_steps)
            if res:
                proba, price_now, atr, adx, fi, used_t = res
                
                regime = "トレンド" if adx > 25 else "レンジ"
                st.info(f"🧭 現在の相場環境: **{regime}** (ADX: {adx:.1f}) | 足確定時刻: {used_t.astimezone(jst)}")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("開始価格", f"{price_now:.5f}")
                col2.metric("AI予測", "UP ↗️" if proba[1]>proba[0] else "DOWN ↘️", f"確信度: {max(proba)*100:.1f}%")
                
                sl_dist = atr * sl_input
                tp_dist = sl_dist * rr_input
                direction = "BUY" if proba[1]>proba[0] else "SELL"
                tp_p = price_now + tp_dist if direction=="BUY" else price_now - tp_dist
                sl_p = price_now - sl_dist if direction=="BUY" else price_now + sl_dist
                
                st.markdown("---")
                st.subheader("🛡️ トレードプラン")
                c_tp, c_ent, c_sl = st.columns(3)
                c_tp.warning(f"利確: {tp_p:.5f}")
                c_ent.info(f"Entry: {price_now:.5f}")
                c_sl.error(f"損切り: {sl_p:.5f}")
                
                st.line_chart(df.tail(50)['Close'])
                
                st.subheader("🧠 AIの判断根拠")
                st.bar_chart(fi.set_index('Feature'))

# --- バックテストログの表示 ---
if 'best_logs' in st.session_state:
    st.markdown("---")
    st.subheader("📑 バックテスト詳細履歴 (最強設定のログ)")
    
    # Rスコアを小数点以下4桁で見やすくフォーマット
    formatted_logs = []
    for log in st.session_state.best_logs:
        formatted_log = log.copy()
        formatted_log['Rスコア'] = f"{log['Rスコア']:.4f}"
        formatted_logs.append(formatted_log)
        
    st.table(pd.DataFrame(formatted_logs))
