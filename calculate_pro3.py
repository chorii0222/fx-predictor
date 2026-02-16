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
        # 直近1日のデータを取得
        data = ticker.history(period="1d")
        if not data.empty:
            return data['Close'].iloc[-1]
        return 150.0 # 取得失敗時のフェイルセーフ
    except:
        return 150.0

def calculate_technical_indicators(df):
    """テクニカル指標を計算"""
    df = df.copy()
    
    # --- 基本指標 ---
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
    # ボリンジャーバンド
    df['BB_upper'] = df['SMA_20'] + 2 * df['Close'].rolling(window=20).std()
    df['BB_lower'] = df['SMA_20'] - 2 * df['Close'].rolling(window=20).std()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # モメンタム
    df['Momentum'] = df['Close'] - df['Close'].shift(10)

    # --- 追加: MACD ---
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']

    # --- 追加: Slope ---
    df['SMA_20_Slope'] = df['SMA_20'].diff()

    # --- 追加: ATR ---
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    
    # --- 追加: Lag Features ---
    df['RSI_Lag1'] = df['RSI'].shift(1)
    df['Close_Pct_Change'] = df['Close'].pct_change()
    df['Close_Pct_Lag1'] = df['Close_Pct_Change'].shift(1)

    df.dropna(inplace=True)
    return df

def fetch_and_process_data(ticker, target_dt_jst):
    """データを取得し、UTC変換して処理"""
    target_dt_utc = target_dt_jst.astimezone(pytz.utc)
    
    start_date = target_dt_utc - timedelta(days=60)
    end_date = datetime.now(pytz.utc) + timedelta(days=2)
    
    try:
        df_1h = yf.download(ticker, start=start_date, end=end_date, interval="1h", progress=False)
    except Exception as e:
        st.error(f"データ取得エラー: {e}")
        return None, None, None

    if df_1h.empty:
        st.error("指定された期間のデータが見つかりませんでした。")
        return None, None, None

    if df_1h.index.tz is None:
        df_1h.index = df_1h.index.tz_localize('UTC')
    else:
        df_1h.index = df_1h.index.tz_convert('UTC')

    if isinstance(df_1h.columns, pd.MultiIndex):
        df_1h.columns = df_1h.columns.get_level_values(0)

    df_1h = calculate_technical_indicators(df_1h)
    
    df_1h['Target_Price_6h'] = df_1h['Close'].shift(-6)
    df_1h['Target'] = (df_1h['Target_Price_6h'] > df_1h['Close']).astype(int)
    
    return df_1h, target_dt_utc, target_dt_jst

def train_and_predict(df, target_dt_utc):
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
        'MACD', 'Signal', 'MACD_Hist', 'SMA_20_Slope', 'RSI_Lag1', 'Close_Pct_Lag1'
    ]
    
    X_train = train_data[features]
    y_train = train_data['Target']
    X_target = prediction_row[features]

    # モデル構築
    model = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_split=5, random_state=42)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_target)[0]
    
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    current_price = prediction_row['Close'].values[0]
    future_price = prediction_row['Target_Price_6h'].values[0]
    atr_val = prediction_row['ATR'].values[0]
    used_time_utc = prediction_row.index[0]
    
    return proba, current_price, future_price, used_time_utc, atr_val, feature_importance_df

def simulate_trade(df, start_time_utc, trade_type, entry_price, tp_price, sl_price):
    """トレード結果シミュレーション"""
    future_candles = df[df.index > start_time_utc].head(6)
    
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

# ---------------------------------------------------------
# 2. Streamlit UI
# ---------------------------------------------------------

st.set_page_config(
    page_title="FX AI予測",
    page_icon="📈",  
    layout="wide"
)

st.title("💹 AI FX 6時間後トレンド予測ツール")

# --- サイドバー設定 ---
st.sidebar.header("設定")

if 'init_done' not in st.session_state:
    now = datetime.now()
    st.session_state.default_date = now.date()
    st.session_state.default_time = time(now.hour, 0)
    st.session_state.init_done = True

ticker = st.sidebar.text_input("通貨ペア", "USDJPY=X")

st.sidebar.markdown("---")
st.sidebar.subheader("日時設定")
use_realtime = st.sidebar.checkbox("🔴 リアルタイム予測 (現在時刻)", value=False)

if use_realtime:
    st.sidebar.info("現在時刻の最新データを取得して予測します。")
    input_date = datetime.now().date()
    input_time = datetime.now().time()
else:
    input_date = st.sidebar.date_input("日付", value=st.session_state.default_date)
    input_time = st.sidebar.time_input("時間 (JST)", value=st.session_state.default_time)

st.sidebar.markdown("---")
st.sidebar.subheader("資金・リスク管理")

# --- 変更箇所: 最小値を0.01、型をfloatに変更 ---
trade_units = st.sidebar.number_input(
    "取引通貨量 (Units)", 
    min_value=0.01, 
    max_value=10000000.0, 
    value=10000.0, 
    step=0.01,
    format="%.2f",
    help="0.01から入力可能です。"
)

risk_reward_ratio = st.sidebar.number_input("リスクリワード比", 1.0, 10.0, 2.0, 0.1)
sl_atr_multiplier = st.sidebar.slider("損切り幅 (ATR倍率)", 1.0, 3.0, 1.5, 0.1)

jst = pytz.timezone('Asia/Tokyo')

if use_realtime:
    target_dt_jst = datetime.now(jst)
else:
    target_dt_naive = datetime.combine(input_date, input_time)
    target_dt_jst = jst.localize(target_dt_naive)

st.sidebar.markdown("---")

if st.sidebar.button("予測を実行"):
    st.write(f"### 分析対象: {ticker}")
    st.caption(f"基準日時 (JST): {target_dt_jst.strftime('%Y-%m-%d %H:%M:%S')}")
    if use_realtime:
        st.markdown("**🔴 リアルタイム・モード** で実行中")
    
    with st.spinner('データを取得・AI解析中...'):
        df, target_dt_utc, _ = fetch_and_process_data(ticker, target_dt_jst)
        
        if df is not None:
            result = train_and_predict(df, target_dt_utc)
            
            if result:
                proba, price_now, price_6h, used_time_utc, atr_val, fi_df = result
                
                used_time_jst = used_time_utc.astimezone(jst)
                down_prob = proba[0] * 100
                up_prob = proba[1] * 100
                
                # --- 通貨換算準備 ---
                usdjpy_rate = 1.0
                currency_label = "pips/通貨"
                conversion_note = ""
                
                if "JPY" in ticker:
                    currency_label = "円"
                elif "USD" in ticker:
                    usdjpy_rate = get_usdjpy_rate()
                    currency_label = "円 (概算)"
                    conversion_note = f"(USDJPYレート @ {usdjpy_rate:.2f} で換算)"
                
                # --- 結果表示レイアウト ---
                st.markdown("---")
                st.subheader("📊 予測結果と損益シミュレーション")

                is_future = np.isnan(price_6h)
                diff = 0 if is_future else price_6h - price_now
                
                ai_direction = "UP ↗️" if up_prob > down_prob else "DOWN ↘️"
                ai_confidence = max(up_prob, down_prob)
                
                kpi1, kpi2, kpi3 = st.columns(3)

                kpi1.metric(label="🏁 開始価格", value=f"{price_now:.3f}")

                if is_future:
                    kpi2.metric(label="🏁 6時間後の価格", value="未確定 (未来)", delta="Waiting...")
                else:
                    kpi2.metric(
                        label="🏁 6時間後の価格 (実際)",
                        value=f"{price_6h:.3f}",
                        delta=f"{diff:.3f}",
                        delta_color="inverse" if "JPY" in ticker and diff < 0 else "normal"
                    )

                kpi3.metric(
                    label="🤖 AIの予測",
                    value=f"{ai_direction}",
                    delta=f"確信度: {ai_confidence:.1f}%"
                )

                # 損益計算 (日本円対応)
                if not is_future:
                    raw_profit = (price_6h - price_now) * trade_units if up_prob > down_prob else (price_now - price_6h) * trade_units
                    final_profit = raw_profit * usdjpy_rate
                    
                    bg_color = "#d4edda" if final_profit > 0 else "#f8d7da"
                    sign_str = "+" if final_profit > 0 else ""
                    
                    # 小数点以下も表示するようにフォーマット調整（0.01単位の取引に対応するため）
                    st.markdown(f"""
                    <div style="background-color:{bg_color}; padding:15px; border-radius:10px; margin-top:10px; text-align:center;">
                        <h4 style="margin:0;">💰 もしAIに従って {trade_units:,.2f} 通貨取引していたら...</h4>
                        <h2 style="margin:0; color:{'green' if final_profit>0 else 'red'}">{sign_str}{final_profit:,.2f} {currency_label}</h2>
                        <small>{conversion_note}</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("🕒 現在進行中のため、確定損益はまだありません。")

                # --- トレードシナリオ ---
                st.markdown("---")
                st.subheader("🛡️ トレードシナリオ (TP/SL)")
                
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

                # 予定損益の計算
                est_profit = (tp_distance * trade_units) * usdjpy_rate
                est_loss = (sl_distance * trade_units) * usdjpy_rate

                sim_result, _ = simulate_trade(df, used_time_utc, trade_type, price_now, tp_price, sl_price)

                col_tp, col_entry, col_sl = st.columns(3)
                
                tp_bg = "background-color:#d4edda;" if sim_result == "WIN" else ""
                sl_bg = "background-color:#f8d7da;" if sim_result == "LOSS" else ""
                
                with col_tp:
                    st.markdown(f"<div style='{tp_bg} padding:10px; border-radius:10px; border:1px solid #ddd;'>", unsafe_allow_html=True)
                    st.markdown(f"<h3 style='color:{tp_color}; text-align: center;'>🎯 利確 (TP)</h3>", unsafe_allow_html=True)
                    st.markdown(f"<h2 style='text-align: center;'>{tp_price:.3f}</h2>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align: center;'>予定利益: <b>+{est_profit:,.0f} {currency_label}</b></p>", unsafe_allow_html=True)
                    if sim_result == "WIN": st.markdown(f"<p style='text-align: center; color:green; font-weight:bold; background:white;'>✅ 達成</p>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                with col_entry:
                    st.markdown(f"<h3 style='text-align: center;'>Entry</h3>", unsafe_allow_html=True)
                    st.markdown(f"<h2 style='text-align: center;'>{price_now:.3f}</h2>", unsafe_allow_html=True)
                    st.markdown(f"<div style='text-align: center; font-weight:bold; padding:5px; background-color:#333; color:white; border-radius:5px;'>{trade_type}</div>", unsafe_allow_html=True)
                    if conversion_note:
                        st.caption(f"※{conversion_note}")

                with col_sl:
                    st.markdown(f"<div style='{sl_bg} padding:10px; border-radius:10px; border:1px solid #ddd;'>", unsafe_allow_html=True)
                    st.markdown(f"<h3 style='color:{sl_color}; text-align: center;'>🛑 損切り (SL)</h3>", unsafe_allow_html=True)
                    st.markdown(f"<h2 style='text-align: center;'>{sl_price:.3f}</h2>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align: center;'>予定損失: <b>-{est_loss:,.0f} {currency_label}</b></p>", unsafe_allow_html=True)
                    if sim_result == "LOSS": st.markdown(f"<p style='text-align: center; color:red; font-weight:bold; background:white;'>❌ 損切り</p>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                st.caption(f"※ ライン計算基準: ATR={atr_val:.3f} / RR比=1:{risk_reward_ratio}")

                # --- 根拠の可視化 ---
                st.markdown("---")
                st.subheader("🧠 なぜこの予測になったのか？")
                fi_df['Importance'] = fi_df['Importance'] / fi_df['Importance'].sum()
                st.bar_chart(fi_df.set_index('Feature'))
                
                # --- チャートと確率 ---
                st.markdown("---")
                col_up, col_down = st.columns(2)
                with col_up:
                    st.write(f"📈 上昇確率: {up_prob:.1f}%")
                    st.progress(int(up_prob))
                with col_down:
                    st.write(f"📉 下落確率: {down_prob:.1f}%")
                    st.progress(int(down_prob))

                chart_df = df.copy()
                chart_df.index = chart_df.index.tz_convert(jst)
                plot_start = target_dt_jst - timedelta(hours=24)
                plot_end = target_dt_jst + timedelta(hours=6)
                st.line_chart(chart_df.loc[plot_start:plot_end]['Close'])