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

def calculate_technical_indicators(df):
    """テクニカル指標を計算 (ATRを追加)"""
    df = df.copy()
    
    # SMA
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # ボリンジャーバンド
    df['BB_upper'] = df['SMA_20'] + 2 * df['Close'].rolling(window=20).std()
    df['BB_lower'] = df['SMA_20'] - 2 * df['Close'].rolling(window=20).std()
    
    # モメンタム
    df['Momentum'] = df['Close'] - df['Close'].shift(10)

    # --- 追加: ATR (Average True Range) ---
    # 損切りラインの計算にボラティリティを使用するため
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    
    df.dropna(inplace=True)
    return df

def fetch_and_process_data(ticker, target_dt_jst):
    """データを取得し、指定されたJST日時をUTCに変換して処理する"""
    target_dt_utc = target_dt_jst.astimezone(pytz.utc)
    
    start_date = target_dt_utc - timedelta(days=60)
    end_date = target_dt_utc + timedelta(days=5)
    
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
    
    train_data = df[df.index < target_dt_utc].copy()
    
    try:
        target_idx = df.index.get_indexer([target_dt_utc], method='pad')[0]
        prediction_row = df.iloc[[target_idx]].copy()
        
        time_diff = abs(prediction_row.index[0] - target_dt_utc)
        if time_diff > timedelta(hours=2):
            st.warning(f"⚠️ 指定時間のデータが存在しないため、直近データ({prediction_row.index[0]})を使用します。")
            
    except:
        st.error("指定された日時のデータポイントが見つかりません。")
        return None

    if len(train_data) < 50:
        st.error("学習データが不足しています。")
        return None

    features = ['Close', 'SMA_5', 'SMA_20', 'RSI', 'BB_upper', 'BB_lower', 'Momentum']
    X_train = train_data[features]
    y_train = train_data['Target']
    X_target = prediction_row[features]

    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_target)[0]
    
    current_price = prediction_row['Close'].values[0]
    future_price = prediction_row['Target_Price_6h'].values[0]
    atr_val = prediction_row['ATR'].values[0] # ATRを取得
    used_time_utc = prediction_row.index[0]
    
    return proba, current_price, future_price, used_time_utc, atr_val

# ---------------------------------------------------------
# 2. Streamlit UI
# ---------------------------------------------------------

st.set_page_config(
    page_title="FX AI予測",     # ここがアプリ名になります
    page_icon="📈",            # ここがアイコンになります（絵文字が一番確実です）
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
input_date = st.sidebar.date_input("日付", value=st.session_state.default_date)
input_time = st.sidebar.time_input("時間 (JST)", value=st.session_state.default_time)

st.sidebar.markdown("---")
st.sidebar.subheader("リスク管理設定")
# 追加: リスクリワード比率の入力
risk_reward_ratio = st.sidebar.number_input(
    "リスクリワード比 (損失1に対して利益は?)", 
    min_value=0.5, 
    max_value=10.0, 
    value=2.0, 
    step=0.1,
    help="例: 2.0 に設定すると、損切り幅の2倍を利確幅に設定します。"
)
# 追加: ストップロスの広さ係数
sl_atr_multiplier = st.sidebar.slider(
    "損切り幅の余裕 (ATR倍率)",
    min_value=1.0,
    max_value=3.0,
    value=1.5,
    step=0.1,
    help="値を大きくすると損切りされにくくなりますが、損失額も増えます。通常は1.5〜2.0が推奨です。"
)

jst = pytz.timezone('Asia/Tokyo')
target_dt_naive = datetime.combine(input_date, input_time)
target_dt_jst = jst.localize(target_dt_naive)

st.sidebar.markdown("---")

if st.sidebar.button("予測を実行"):
    st.write(f"### 分析対象: {ticker}")
    st.caption(f"指定日時 (JST): {target_dt_jst.strftime('%Y-%m-%d %H:%M:%S')}")
    
    with st.spinner('データを取得・AI解析中...'):
        df, target_dt_utc, _ = fetch_and_process_data(ticker, target_dt_jst)
        
        if df is not None:
            result = train_and_predict(df, target_dt_utc)
            
            if result:
                # 戻り値にatr_valを追加
                proba, price_now, price_6h, used_time_utc, atr_val = result
                
                used_time_jst = used_time_utc.astimezone(jst)
                down_prob = proba[0] * 100
                up_prob = proba[1] * 100
                
                # --- 結果表示レイアウト ---
                st.markdown("---")
                st.subheader("📊 価格比較と予測評価")

                diff = price_6h - price_now
                ai_direction = "UP ↗️" if up_prob > down_prob else "DOWN ↘️"
                ai_confidence = max(up_prob, down_prob)
                actual_direction = "UP ↗️" if diff > 0 else "DOWN ↘️"
                
                kpi1, kpi2, kpi3 = st.columns(3)

                kpi1.metric(
                    label="🏁 開始価格 (Start)",
                    value=f"{price_now:.3f}",
                    help=f"データ取得時刻: {used_time_jst.strftime('%H:%M')}"
                )

                if np.isnan(price_6h):
                    kpi2.metric(label="🏁 6時間後の価格 (Actual)", value="N/A", delta="データなし")
                else:
                    kpi2.metric(
                        label="🏁 6時間後の価格 (Actual)",
                        value=f"{price_6h:.3f}",
                        delta=f"{diff:.3f} ({actual_direction})",
                        delta_color="inverse" if ticker.endswith("JPY=X") and diff < 0 else "normal" 
                    )

                kpi3.metric(
                    label="🤖 AIの予測方向",
                    value=f"{ai_direction}",
                    delta=f"確信度: {ai_confidence:.1f}%"
                )

                # 勝敗メッセージ
                is_correct = (up_prob > 50 and diff > 0) or (down_prob > 50 and diff < 0)
                if not np.isnan(price_6h):
                    st.write("")
                    if is_correct:
                        st.success(f"✅ **予測的中!** AIは「{ai_direction}」と予測し、実際に価格は {diff:+.3f} 変動しました。")
                    else:
                        st.error(f"❌ **予測ハズレ...** AIは「{ai_direction}」と予測しましたが、実際は逆方向に {diff:+.3f} 変動しました。")

                # --- 追加機能: トレードシナリオ提案 ---
                st.markdown("---")
                st.subheader("🛡️ トレードシナリオ提案 (リスクリワード計算)")
                
                # ATRを用いたライン計算
                sl_distance = atr_val * sl_atr_multiplier  # 損切り幅
                tp_distance = sl_distance * risk_reward_ratio  # 利確幅
                
                # AIの予測方向に基づいてラインを決定
                if up_prob > down_prob:
                    # 買い (LONG) の場合
                    trade_type = "BUY / LONG"
                    tp_price = price_now + tp_distance
                    sl_price = price_now - sl_distance
                    sl_color = "red"
                    tp_color = "green"
                else:
                    # 売り (SHORT) の場合
                    trade_type = "SELL / SHORT"
                    tp_price = price_now - tp_distance
                    sl_price = price_now + sl_distance
                    sl_color = "red"
                    tp_color = "green"

                # シナリオ表示
                st.info(f"あなたの設定したリスクリワード **1 : {risk_reward_ratio}** に基づく推奨ラインです。")
                
                col_tp, col_entry, col_sl = st.columns(3)
                
                with col_tp:
                    st.markdown(f"<h3 style='color:{tp_color}; text-align: center;'>🎯 Take Profit</h3>", unsafe_allow_html=True)
                    st.markdown(f"<h2 style='text-align: center;'>{tp_price:.3f}</h2>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align: center;'>変動幅: {tp_distance:.3f}</p>", unsafe_allow_html=True)
                    
                with col_entry:
                    st.markdown(f"<h3 style='text-align: center;'>Entry</h3>", unsafe_allow_html=True)
                    st.markdown(f"<h2 style='text-align: center;'>{price_now:.3f}</h2>", unsafe_allow_html=True)
                    st.markdown(f"<div style='text-align: center; font-weight:bold; padding:5px; background-color:#eee; border-radius:5px;'>{trade_type}</div>", unsafe_allow_html=True)

                with col_sl:
                    st.markdown(f"<h3 style='color:{sl_color}; text-align: center;'>🛑 Stop Loss</h3>", unsafe_allow_html=True)
                    st.markdown(f"<h2 style='text-align: center;'>{sl_price:.3f}</h2>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align: center;'>変動幅: {sl_distance:.3f}</p>", unsafe_allow_html=True)

                st.caption(f"※ ライン計算基準: 現在のATR(ボラティリティ) = {atr_val:.4f} / SL設定倍率 = {sl_atr_multiplier}x")

                # --- 確率バーとチャート ---
                st.markdown("---")
                st.subheader("📉 チャートと確率詳細")
                
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
                plot_end = target_dt_jst + timedelta(hours=12)
                st.line_chart(chart_df.loc[plot_start:plot_end]['Close'])