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
    """日本円換算用にUSDJPYの現在レートを取得する"""
    try:
        ticker = yf.Ticker("USDJPY=X")
        data = ticker.history(period="1d")
        if not data.empty:
            return data['Close'].iloc[-1]
        return 150.0
    except:
        return 150.0

def calculate_technical_indicators(df):
    """テクニカル指標を計算"""
    df = df.copy()
    
    # --- 基本指標 ---
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
    df['BB_upper'] = df['SMA_20'] + 2 * df['Close'].rolling(window=20).std()
    df['BB_lower'] = df['SMA_20'] - 2 * df['Close'].rolling(window=20).std()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['Momentum'] = df['Close'] - df['Close'].shift(10)

    # --- MACD ---
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']

    df['SMA_20_Slope'] = df['SMA_20'].diff()

    # --- ATR ---
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    
    # --- ADX ---
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
    
    # --- Lag Features ---
    df['RSI_Lag1'] = df['RSI'].shift(1)
    df['Close_Pct_Change'] = df['Close'].pct_change()
    df['Close_Pct_Lag1'] = df['Close_Pct_Change'].shift(1)

    df.dropna(inplace=True)
    return df

def fetch_and_process_data(ticker, target_dt_jst, prediction_hours=6):
    """データを取得し、UTC変換して処理 (予測時間幅を動的に変更可能に)"""
    target_dt_utc = target_dt_jst.astimezone(pytz.utc)
    
    start_date = target_dt_utc - timedelta(days=60)
    end_date = datetime.now(pytz.utc) + timedelta(days=2)
    
    try:
        df_1h = yf.download(ticker, start=start_date, end=end_date, interval="1h", progress=False)
    except Exception as e:
        st.error(f"データ取得エラー: {e}")
        return None, None, None

    if df_1h.empty:
        st.error(f"{ticker} の指定された期間のデータが見つかりませんでした。")
        return None, None, None

    if df_1h.index.tz is None:
        df_1h.index = df_1h.index.tz_localize('UTC')
    else:
        df_1h.index = df_1h.index.tz_convert('UTC')

    if isinstance(df_1h.columns, pd.MultiIndex):
        df_1h.columns = df_1h.columns.get_level_values(0)

    df_1h = calculate_technical_indicators(df_1h)
    
    # 予測時間幅に基づいてターゲットを作成
    target_col_name = f'Target_Price_{prediction_hours}h'
    df_1h[target_col_name] = df_1h['Close'].shift(-prediction_hours)
    df_1h['Target'] = (df_1h[target_col_name] > df_1h['Close']).astype(int)
    
    return df_1h, target_dt_utc, target_dt_jst

def train_and_predict(df, target_dt_utc, prediction_hours=6):
    """学習と予測を実行"""
    
    train_data = df[df.index < target_dt_utc].dropna().copy()
    
    try:
        target_idx = df.index.get_indexer([target_dt_utc], method='pad')[0]
        prediction_row = df.iloc[[target_idx]].copy()
        
        time_diff = abs(prediction_row.index[0] - target_dt_utc)
        if time_diff > timedelta(hours=4):
            st.warning(f"⚠️ データ日時({prediction_row.index[0]})が指定日時と離れています。")
            
    except:
        st.error("指定された日時のデータポイントが見つかりません。")
        return None

    if len(train_data) < 50:
        st.error("学習データが不足しています。")
        return None

    features = [
        'Close', 'SMA_5', 'SMA_20', 'RSI', 'BB_upper', 'BB_lower', 'Momentum',
        'MACD', 'Signal', 'MACD_Hist', 'SMA_20_Slope', 'ADX', 'RSI_Lag1', 'Close_Pct_Lag1'
    ]
    
    X_train = train_data[features]
    y_train = train_data['Target']
    X_target = prediction_row[features]

    model = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_split=5, random_state=42)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_target)[0]
    
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    current_price = prediction_row['Close'].values[0]
    target_col_name = f'Target_Price_{prediction_hours}h'
    future_price = prediction_row[target_col_name].values[0]
    atr_val = prediction_row['ATR'].values[0]
    adx_val = prediction_row['ADX'].values[0]
    used_time_utc = prediction_row.index[0]
    
    return proba, current_price, future_price, used_time_utc, atr_val, feature_importance_df, adx_val

def simulate_trade(df, start_time_utc, trade_type, entry_price, tp_price, sl_price, prediction_hours=6):
    """トレード結果シミュレーション"""
    # 予測時間幅に基づいてチェックする足の数を調整
    future_candles = df[df.index > start_time_utc].head(prediction_hours)
    
    if future_candles.empty or len(future_candles) < 1:
        return "NO_DATA", None
    
    hit_result = "DRAW"
    hit_price = future_candles.iloc[-1]['Close']
    
    for i, row in future_candles.iterrows():
        high = row['High']
        low = row['Low']
        
        if trade_type == "BUY":
            if low <= sl_price:
                hit_result = "LOSS"
                hit_price = sl_price
                break
            elif high >= tp_price:
                hit_result = "WIN"
                hit_price = tp_price
                break
        elif trade_type == "SELL":
            if high >= sl_price:
                hit_result = "LOSS"
                hit_price = sl_price
                break
            elif low <= tp_price:
                hit_result = "WIN"
                hit_price = tp_price
                break
                
    return hit_result, hit_price

# --- 新機能: 最強設定の探索 (時間幅対応) ---
@st.cache_data(show_spinner=False, ttl=3600)
def find_best_settings_last_week():
    """過去1週間のデータを使って全通貨ペア・設定・時間幅のバックテストを行う"""
    tickers_map = {
        "USDJPY": "USDJPY=X",
        "EURUSD": "EURUSD=X",
        "GBPUSD": "GBPUSD=X",
        "XAUUSD (金)": "GC=F",
        "BTCUSD (ビットコイン)": "BTC-USD",
        "ETHUSD (イーサリアム)": "ETH-USD",
        "XAGUSD (銀)": "SI=F"
    }
    
    # 探索するパラメータの組み合わせ
    rr_grid = [1.0, 1.5, 2.0, 3.0]
    sl_grid = [1.0, 1.5, 2.0]
    hours_grid = [1, 3, 6, 12, 24] # 探索する時間幅のバリエーション
    
    best_r = -float('inf')
    best_combo = None
    
    now_utc = datetime.now(pytz.utc)
    test_start = now_utc - timedelta(days=7)
    
    for name, ticker in tickers_map.items():
        start_date = test_start - timedelta(days=60)
        try:
            df = yf.download(ticker, start=start_date, end=now_utc + timedelta(days=1), interval="1h", progress=False)
            if df.empty: continue
            if df.index.tz is None: df.index = df.index.tz_localize('UTC')
            else: df.index = df.index.tz_convert('UTC')
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
            df = calculate_technical_indicators(df)
            
            for h in hours_grid:
                df_temp = df.copy()
                target_col = f'Target_Price_{h}h'
                df_temp[target_col] = df_temp['Close'].shift(-h)
                df_temp['Target'] = (df_temp[target_col] > df_temp['Close']).astype(int)
                
                train_df = df_temp[df_temp.index < test_start].dropna()
                test_df = df_temp[(df_temp.index >= test_start) & (df_temp.index <= now_utc - timedelta(hours=h))]
                
                if len(train_df) < 50 or len(test_df) < 1: continue
                
                features = ['Close', 'SMA_5', 'SMA_20', 'RSI', 'BB_upper', 'BB_lower', 'Momentum', 'MACD', 'Signal', 'MACD_Hist', 'SMA_20_Slope', 'ADX', 'RSI_Lag1', 'Close_Pct_Lag1']
                X_train = train_df[features]
                y_train = train_df['Target']
                
                model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
                model.fit(X_train, y_train)
                
                preds = model.predict_proba(test_df[features])
                
                for rr in rr_grid:
                    for sl in sl_grid:
                        total_r = 0 
                        for i in range(len(test_df)):
                            row = test_df.iloc[i]
                            prob_down, prob_up = preds[i]
                            direction = "BUY" if prob_up > prob_down else "SELL"
                            
                            atr = row['ATR']
                            price_now = row['Close']
                            sl_dist = atr * sl
                            tp_dist = sl_dist * rr
                            
                            tp_price = price_now + tp_dist if direction == "BUY" else price_now - tp_dist
                            sl_price = price_now - sl_dist if direction == "BUY" else price_now + sl_dist
                            
                            res, _ = simulate_trade(df_temp, test_df.index[i], direction, price_now, tp_price, sl_price, prediction_hours=h)
                            
                            if res == "WIN":
                                total_r += rr
                            elif res == "LOSS":
                                total_r -= 1.0
                                
                        if total_r > best_r:
                            best_r = total_r
                            best_combo = {
                                "asset": name,
                                "hours": h,
                                "rr": rr,
                                "sl": sl,
                                "r_profit": total_r,
                                "total_trades": len(test_df)
                            }
        except:
            pass
            
    return best_combo

# ---------------------------------------------------------
# 2. Streamlit UI
# ---------------------------------------------------------

st.set_page_config(
    page_title="FX AI予測",
    page_icon="📈",  
    layout="wide"
)

st.title("💹 AI FX トレンド予測ツール")

# --- サイドバー設定 ---
st.sidebar.header("設定")

if 'init_done' not in st.session_state:
    now = datetime.now()
    st.session_state.default_date = now.date()
    st.session_state.default_time = time(now.hour, 0)
    st.session_state.init_done = True

# --- 最強設定の自動探索 ---
st.sidebar.markdown("---")
st.sidebar.subheader("🏆 過去1週間の最強設定を探索")
st.sidebar.caption("直近1週間で最もAIが機能した通貨ペア・時間幅・設定を提案します。")

if st.sidebar.button("🔍 自動探索スタート"):
    with st.spinner("AIが全通貨ペアのバックテストを実行中... (約10〜30秒)"):
        best = find_best_settings_last_week()
        if best and best['r_profit'] > 0:
            fixed_risk_jpy = 10000
            est_profit_jpy = best['r_profit'] * fixed_risk_jpy
            st.sidebar.success(f"**【最強設定が判明しました】**\n\n"
                               f"👑 通貨ペア: **{best['asset']}**\n\n"
                               f"⏳ 予測時間幅: **{best['hours']}時間**\n\n"
                               f"⚖️ RR比率: **{best['rr']}**\n\n"
                               f"🛑 損切り(ATR): **{best['sl']}**\n\n"
                               f"💰 1回の損切りを1万円に固定した場合の週間利益:\n"
                               f"**+{est_profit_jpy:,.0f}円**\n\n"
                               f"<small>※バックテスト回数: {best['total_trades']}回</small>")
        elif best:
            st.sidebar.warning("過去1週間はどの設定でもマイナス、または十分なデータが得られませんでした。相場が荒れている可能性があります。")
        else:
            st.sidebar.error("データ取得に失敗しました。")

# --- 通貨ペア設定 ---
st.sidebar.markdown("---")
st.sidebar.subheader("通貨ペア設定")
ticker1 = st.sidebar.text_input("通貨ペア 1", "USDJPY=X")

use_second_ticker = st.sidebar.checkbox("➕ 2つ目の通貨ペアも分析する", value=False)
if use_second_ticker:
    ticker2 = st.sidebar.text_input("通貨ペア 2", "EURUSD=X")
else:
    ticker2 = None

# --- 日時・予測設定 (変更箇所: 時間幅の追加) ---
st.sidebar.markdown("---")
st.sidebar.subheader("日時・予測設定")
use_realtime = st.sidebar.checkbox("🔴 リアルタイム予測 (現在時刻)", value=False)

if use_realtime:
    st.sidebar.info("現在時刻の最新データを取得して予測します。")
    input_date = datetime.now().date()
    input_time = datetime.now().time()
else:
    input_date = st.sidebar.date_input("日付", value=st.session_state.default_date)
    input_time = st.sidebar.time_input("時間 (JST)", value=st.session_state.default_time)

prediction_hours = st.sidebar.slider("予測時間幅 (時間)", min_value=1, max_value=24, value=6, step=1, help="予測する対象の未来の時間を指定します。")

# --- 資金・リスク管理 ---
st.sidebar.markdown("---")
st.sidebar.subheader(f"資金・リスク管理 (1つ目: {ticker1})")

trade_units_1 = st.sidebar.number_input(
    f"取引通貨量 ({ticker1})", 
    min_value=0.01, max_value=10000000.0, value=10000.0, step=0.01, format="%.2f",
    key="units_1"
)
risk_reward_ratio_1 = st.sidebar.number_input(f"リスクリワード比 ({ticker1})", 1.0, 10.0, 2.0, 0.1, key="rr_1")
sl_atr_multiplier_1 = st.sidebar.number_input(f"損切り幅 (ATR倍率) ({ticker1})", 0.01, 10.0, 1.5, 0.01, key="sl_1")

if use_second_ticker and ticker2:
    st.sidebar.markdown("---")
    st.sidebar.subheader(f"資金・リスク管理 (2つ目: {ticker2})")
    trade_units_2 = st.sidebar.number_input(
        f"取引通貨量 ({ticker2})", 
        min_value=0.01, max_value=10000000.0, value=10000.0, step=0.01, format="%.2f",
        key="units_2"
    )
    risk_reward_ratio_2 = st.sidebar.number_input(f"リスクリワード比 ({ticker2})", 1.0, 10.0, 2.0, 0.1, key="rr_2")
    sl_atr_multiplier_2 = st.sidebar.number_input(f"損切り幅 (ATR倍率) ({ticker2})", 0.01, 10.0, 1.5, 0.01, key="sl_2")

jst = pytz.timezone('Asia/Tokyo')

if use_realtime:
    target_dt_jst = datetime.now(jst)
else:
    target_dt_naive = datetime.combine(input_date, input_time)
    target_dt_jst = jst.localize(target_dt_naive)

st.sidebar.markdown("---")

if st.sidebar.button("予測を実行"):
    st.caption(f"基準日時 (JST): {target_dt_jst.strftime('%Y-%m-%d %H:%M:%S')}")
    st.caption(f"予測時間幅: {prediction_hours}時間")
    if use_realtime:
        st.markdown("**🔴 リアルタイム・モード** で実行中")
        
    tickers_to_process = [
        {'tk': ticker1, 'units': trade_units_1, 'rr': risk_reward_ratio_1, 'sl_atr': sl_atr_multiplier_1}
    ]
    if use_second_ticker and ticker2:
        tickers_to_process.append(
            {'tk': ticker2, 'units': trade_units_2, 'rr': risk_reward_ratio_2, 'sl_atr': sl_atr_multiplier_2}
        )
        
    analysis_results = []
    total_final_profit = 0
    total_est_profit = 0
    total_est_loss = 0
    is_future_global = False

    for item in tickers_to_process:
        tk = item['tk']
        trade_units = item['units']
        risk_reward_ratio = item['rr']
        sl_atr_multiplier = item['sl_atr']
        
        with st.spinner(f'{tk} のデータを取得・AI解析中...'):
            df, target_dt_utc, _ = fetch_and_process_data(tk, target_dt_jst, prediction_hours=prediction_hours)
            
            if df is not None:
                result = train_and_predict(df, target_dt_utc, prediction_hours=prediction_hours)
                
                if result:
                    proba, price_now, price_6h, used_time_utc, atr_val, fi_df, adx_val = result
                    
                    used_time_jst = used_time_utc.astimezone(jst)
                    down_prob = proba[0] * 100
                    up_prob = proba[1] * 100
                    
                    usdjpy_rate = 1.0
                    currency_label = "pips/通貨"
                    conversion_note = ""
                    
                    if "JPY" in tk:
                        currency_label = "円"
                    elif "USD" in tk:
                        usdjpy_rate = get_usdjpy_rate()
                        currency_label = "円 (概算)"
                        conversion_note = f"(USDJPYレート @ {usdjpy_rate:.2f} で換算)"
                    
                    p_fmt = ".3f" if "JPY" in tk else ".5f"

                    is_future = np.isnan(price_6h)
                    is_future_global = is_future
                    diff = 0 if is_future else price_6h - price_now
                    
                    ai_direction = "UP ↗️" if up_prob > down_prob else "DOWN ↘️"
                    ai_confidence = max(up_prob, down_prob)
                    
                    # 損益計算
                    final_profit = 0
                    if not is_future:
                        raw_profit = (price_6h - price_now) * trade_units if up_prob > down_prob else (price_now - price_6h) * trade_units
                        final_profit = raw_profit * usdjpy_rate
                    
                    sl_distance = atr_val * sl_atr_multiplier
                    tp_distance = sl_distance * risk_reward_ratio
                    
                    if up_prob > down_prob:
                        trade_type = "BUY"
                        tp_price = price_now + tp_distance
                        sl_price = price_now - sl_distance
                        sl_color = "red"
                        tp_color = "green"
                    else:
                        trade_type = "SELL"
                        tp_price = price_now - tp_distance
                        sl_price = price_now + sl_distance
                        sl_color = "red"
                        tp_color = "green"

                    est_profit = (tp_distance * trade_units) * usdjpy_rate
                    est_loss = (sl_distance * trade_units) * usdjpy_rate

                    sim_result, _ = simulate_trade(df, used_time_utc, trade_type, price_now, tp_price, sl_price, prediction_hours=prediction_hours)

                    total_final_profit += final_profit
                    total_est_profit += est_profit
                    total_est_loss += est_loss

                    analysis_results.append({
                        'tk': tk, 'trade_units': trade_units, 'risk_reward_ratio': risk_reward_ratio, 
                        'df': df, 'price_now': price_now, 'price_6h': price_6h,
                        'used_time_jst': used_time_jst, 'down_prob': down_prob, 'up_prob': up_prob,
                        'usdjpy_rate': usdjpy_rate, 'currency_label': currency_label, 'conversion_note': conversion_note,
                        'p_fmt': p_fmt, 'is_future': is_future, 'diff': diff,
                        'ai_direction': ai_direction, 'ai_confidence': ai_confidence,
                        'final_profit': final_profit, 'sl_distance': sl_distance, 'tp_distance': tp_distance,
                        'trade_type': trade_type, 'tp_price': tp_price, 'sl_price': sl_price,
                        'sl_color': sl_color, 'tp_color': tp_color, 'est_profit': est_profit,
                        'est_loss': est_loss, 'sim_result': sim_result, 'atr_val': atr_val, 'fi_df': fi_df,
                        'adx_val': adx_val
                    })

    # ==========================================
    # 結果の描画
    # ==========================================
    if len(analysis_results) > 0:
        
        if len(analysis_results) > 1:
            st.markdown("---")
            st.markdown("## 🌐 総合シミュレーション結果 (2通貨ペア合計)")
            
            if not is_future_global:
                bg_color = "#d4edda" if total_final_profit > 0 else "#f8d7da"
                sign_str = "+" if total_final_profit > 0 else ""
                st.markdown(f"""
                <div style="background-color:{bg_color}; padding:15px; border-radius:10px; margin-top:10px; text-align:center; border: 2px solid {'green' if total_final_profit>0 else 'red'};">
                    <h4 style="margin:0;">💰 合計 確定損益</h4>
                    <h1 style="margin:0; color:{'green' if total_final_profit>0 else 'red'}">{sign_str}{total_final_profit:,.2f} 円</h1>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("🕒 現在進行中のため、確定損益はまだありません。")
                st.markdown(f"""
                <div style="padding:15px; border-radius:10px; margin-top:10px; text-align:center; border:2px solid #ddd; background-color:#f8f9fa;">
                    <h4 style="margin:0;">📊 合計 予定損益</h4>
                    <p style="margin:5px 0; font-size:22px;">予定利益合計: <span style="color:green; font-weight:bold;">+{total_est_profit:,.0f} 円</span></p>
                    <p style="margin:5px 0; font-size:22px;">予定損失合計: <span style="color:red; font-weight:bold;">-{total_est_loss:,.0f} 円</span></p>
                </div>
                """, unsafe_allow_html=True)

        for res in analysis_results:
            st.markdown("---")
            st.markdown(f"## 📌 詳細分析: {res['tk']}")
            
            adx = res['adx_val']
            regime_text = ""
            regime_color = ""
            regime_desc = ""
            
            if adx < 25:
                regime_text = "レンジ相場 (Range)"
                regime_color = "#f39c12"
                regime_desc = "⚠️ 現在は明確なトレンドがありません。AIは順張り傾向があるため、ダマシに遭う確率が高くなります。シグナルの確信度が低い場合は見送りを検討してください。"
            else:
                regime_text = "トレンド相場 (Trend)"
                regime_color = "#27ae60"
                regime_desc = "✅ 明確なトレンドが発生しています。AIの順張り予測が機能しやすい環境です。"

            st.markdown(f"""
            <div style="border-left: 5px solid {regime_color}; padding: 10px; background-color: #f9f9f9; margin-bottom: 20px;">
                <h4 style="margin: 0; color: {regime_color};">🧭 相場環境認識: {regime_text}</h4>
                <p style="margin: 5px 0 0 0; font-size: 14px;">(ADX値: {adx:.1f}) {regime_desc}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader("📊 予測結果と損益シミュレーション")
            
            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric(label="🏁 開始価格", value=f"{res['price_now']:{res['p_fmt']}}")

            if res['is_future']:
                kpi2.metric(label=f"🏁 {prediction_hours}時間後の価格", value="未確定 (未来)", delta="Waiting...")
            else:
                kpi2.metric(
                    label=f"🏁 {prediction_hours}時間後の価格 (実際)",
                    value=f"{res['price_6h']:{res['p_fmt']}}",
                    delta=f"{res['diff']:{res['p_fmt']}}",
                    delta_color="inverse" if "JPY" in res['tk'] and res['diff'] < 0 else "normal"
                )

            kpi3.metric(
                label="🤖 AIの予測",
                value=f"{res['ai_direction']}",
                delta=f"確信度: {res['ai_confidence']:.1f}%"
            )

            if not res['is_future']:
                bg_color = "#d4edda" if res['final_profit'] > 0 else "#f8d7da"
                sign_str = "+" if res['final_profit'] > 0 else ""
                
                st.markdown(f"""
                <div style="background-color:{bg_color}; padding:15px; border-radius:10px; margin-top:10px; text-align:center;">
                    <h4 style="margin:0;">💰 もしAIに従って {res['trade_units']:,.2f} 通貨取引していたら...</h4>
                    <h2 style="margin:0; color:{'green' if res['final_profit']>0 else 'red'}">{sign_str}{res['final_profit']:,.2f} {res['currency_label']}</h2>
                    <small>{res['conversion_note']}</small>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("🛡️ トレードシナリオ (TP/SL)")
            
            col_tp, col_entry, col_sl = st.columns(3)
            
            tp_bg = "background-color:#d4edda;" if res['sim_result'] == "WIN" else ""
            sl_bg = "background-color:#f8d7da;" if res['sim_result'] == "LOSS" else ""
            
            with col_tp:
                st.markdown(f"<div style='{tp_bg} padding:10px; border-radius:10px; border:1px solid #ddd;'>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='color:{res['tp_color']}; text-align: center;'>🎯 利確 (TP)</h3>", unsafe_allow_html=True)
                st.markdown(f"<h2 style='text-align: center;'>{res['tp_price']:{res['p_fmt']}}</h2>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center;'>予定利益: <b>+{res['est_profit']:,.0f} {res['currency_label']}</b></p>", unsafe_allow_html=True)
                if res['sim_result'] == "WIN": st.markdown(f"<p style='text-align: center; color:green; font-weight:bold; background:white;'>✅ 達成</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col_entry:
                st.markdown(f"<h3 style='text-align: center;'>Entry</h3>", unsafe_allow_html=True)
                st.markdown(f"<h2 style='text-align: center;'>{res['price_now']:{res['p_fmt']}}</h2>", unsafe_allow_html=True)
                st.markdown(f"<div style='text-align: center; font-weight:bold; padding:5px; background-color:#333; color:white; border-radius:5px;'>{res['trade_type']}</div>", unsafe_allow_html=True)
                if res['conversion_note']:
                    st.caption(f"※{res['conversion_note']}")

            with col_sl:
                st.markdown(f"<div style='{sl_bg} padding:10px; border-radius:10px; border:1px solid #ddd;'>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='color:{res['sl_color']}; text-align: center;'>🛑 損切り (SL)</h3>", unsafe_allow_html=True)
                st.markdown(f"<h2 style='text-align: center;'>{res['sl_price']:{res['p_fmt']}}</h2>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center;'>予定損失: <b>-{res['est_loss']:,.0f} {res['currency_label']}</b></p>", unsafe_allow_html=True)
                if res['sim_result'] == "LOSS": st.markdown(f"<p style='text-align: center; color:red; font-weight:bold; background:white;'>❌ 損切り</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            st.caption(f"※ ライン計算基準: ATR={res['atr_val']:{res['p_fmt']}} / RR比=1:{res['risk_reward_ratio']}")

            st.markdown("---")
            st.subheader("🧠 なぜこの予測になったのか？")
            res['fi_df']['Importance'] = res['fi_df']['Importance'] / res['fi_df']['Importance'].sum()
            st.bar_chart(res['fi_df'].set_index('Feature'))
            
            st.markdown("---")
            col_up, col_down = st.columns(2)
            with col_up:
                st.write(f"📈 上昇確率: {res['up_prob']:.1f}%")
                st.progress(int(res['up_prob']))
            with col_down:
                st.write(f"📉 下落確率: {res['down_prob']:.1f}%")
                st.progress(int(res['down_prob']))

            chart_df = res['df'].copy()
            chart_df.index = chart_df.index.tz_convert(jst)
            plot_start = target_dt_jst - timedelta(hours=24)
            plot_end = target_dt_jst + timedelta(hours=prediction_hours)
            st.line_chart(chart_df.loc[plot_start:plot_end]['Close'])