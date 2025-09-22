# ichimoku_streamlit_v14_optimized.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from io import BytesIO
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Ichimoku Multi-Ticker Equity Overlay", layout="wide")

# -------------------------
# Ichimoku calculation
# -------------------------
def compute_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Ichimoku indicators with error handling"""
    if df is None or df.empty:
        return pd.DataFrame()
        
    df = df.copy().sort_index()
    
    # Ensure required columns exist
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c not in df.columns:
            st.warning(f"Column {c} not found in data")
            return pd.DataFrame()
    
    try:
        # Calculate Ichimoku components
        df["Tenkan"] = (df["High"].rolling(9, min_periods=1).max() + df["Low"].rolling(9, min_periods=1).min()) / 2
        df["Kijun"] = (df["High"].rolling(26, min_periods=1).max() + df["Low"].rolling(26, min_periods=1).min()) / 2
        df["SenkouA"] = ((df["Tenkan"] + df["Kijun"]) / 2).shift(26)
        df["SenkouB"] = ((df["High"].rolling(52, min_periods=1).max() + df["Low"].rolling(52, min_periods=1).min()) / 2).shift(26)
        df["Chikou"] = df["Close"].shift(-26)
        df["Vol_SMA20"] = df["Volume"].rolling(20, min_periods=1).mean()
        
        return df
    except Exception as e:
        st.error(f"Error calculating Ichimoku indicators: {str(e)}")
        return pd.DataFrame()

# -------------------------
# Backtest simulation
# -------------------------
def backtest_sim(df: pd.DataFrame, stop_loss: float, take_profit: float):
    """Run backtest simulation with improved logic"""
    if df is None or df.empty:
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float)
        
    df = df.copy().sort_index()
    n = len(df)
    
    if n == 0:
        return df, pd.Series(dtype=float), pd.Series(dtype=float)

    try:
        # Ensure 1D arrays
        close = df["Close"].to_numpy(dtype=float).ravel()
        tenkan = df["Tenkan"].to_numpy(dtype=float).ravel()
        kijun = df["Kijun"].to_numpy(dtype=float).ravel()
        senkouA = df["SenkouA"].to_numpy(dtype=float).ravel()
        senkouB = df["SenkouB"].to_numpy(dtype=float).ravel()
        vol = df["Volume"].to_numpy(dtype=float).ravel()
        vol_sma = df["Vol_SMA20"].to_numpy(dtype=float).ravel()

        # Create conditions with NaN handling
        cond_close_gt_kijun = (~np.isnan(close)) & (~np.isnan(kijun)) & (close > kijun)
        cond_close_gt_kumo = (~np.isnan(close)) & (~np.isnan(senkouA)) & (~np.isnan(senkouB)) & (close > np.maximum(senkouA, senkouB))
        cond_vol = (~np.isnan(vol)) & (~np.isnan(vol_sma)) & (vol > vol_sma)
        cond_tenkan_gt_kijun = (~np.isnan(tenkan)) & (~np.isnan(kijun)) & (tenkan > kijun)

        buy_mask = cond_close_gt_kijun & cond_close_gt_kumo & cond_vol & cond_tenkan_gt_kijun

        positions = np.zeros(n, dtype=int)
        in_trade = False
        entry_price = np.nan

        # Trading logic
        for i in range(n):
            if np.isnan(close[i]):
                positions[i] = 0
                if in_trade:
                    in_trade = False
                    entry_price = np.nan
                continue
                
            if not in_trade and buy_mask[i]:
                in_trade = True
                entry_price = close[i]
                positions[i] = 1
            elif in_trade:
                change = (close[i] - entry_price) / entry_price
                if change <= -stop_loss or change >= take_profit:
                    positions[i] = 0
                    in_trade = False
                    entry_price = np.nan
                else:
                    positions[i] = 1

        # Calculate returns
        market_ret = pd.Series(close).pct_change().fillna(0).to_numpy()
        pos_prev = np.concatenate(([0], positions[:-1]))
        strat_ret = market_ret * pos_prev

        cum_market = pd.Series((1 + market_ret).cumprod(), index=df.index)
        cum_strategy = pd.Series((1 + strat_ret).cumprod(), index=df.index)

        # Add signals to dataframe - FIXED: Use iloc for position-based indexing
        df = df.copy()
        df["BuySignal"] = False
        df["Position"] = 0
        df["MarketReturn"] = 0.0
        df["StrategyReturn"] = 0.0
        
        # Use iloc for safe indexing
        for i in range(len(df)):
            if i < len(buy_mask):
                df.iloc[i, df.columns.get_loc("BuySignal")] = bool(buy_mask[i])
            if i < len(positions):
                df.iloc[i, df.columns.get_loc("Position")] = int(positions[i])
            if i < len(market_ret):
                df.iloc[i, df.columns.get_loc("MarketReturn")] = float(market_ret[i])
            if i < len(strat_ret):
                df.iloc[i, df.columns.get_loc("StrategyReturn")] = float(strat_ret[i])

        return df, cum_market, cum_strategy
        
    except Exception as e:
        st.error(f"Error in backtest simulation: {str(e)}")
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float)

# -------------------------
# UI
# -------------------------
st.title("📈 Ichimoku by JU")


# Sidebar for inputs
with st.sidebar:
    st.header("Settings")
    tickers_input = st.text_input("Tickers (comma-separated)", "BBCA.JK,BBRI.JK")
    default_start = datetime.now() - timedelta(days=365*2)  # 2 years back
    start = st.date_input("Start date", default_start)
    end = st.date_input("End date", datetime.now())
    
    st.subheader("Trading Parameters")
    stop_loss_pct = st.number_input("Stop Loss (%)", min_value=0.1, value=5.0, step=0.1)/100
    take_profit_pct = st.number_input("Take Profit (%)", min_value=0.1, value=10.0, step=0.1)/100
    
    st.subheader("Info")
    st.info("""
    This app analyzes stocks using Ichimoku Kinko Hyo indicators and runs a backtest simulation.
    Enter tickers (comma-separated) and adjust parameters as needed.
    """)

# Parse tickers
tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]

if st.button("Run Backtest", type="primary"):
    if len(tickers) == 0:
        st.error("Please enter at least one ticker.")
    else:
        summary = {}
        sheet_data = {}
        equity_curves = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, ticker in enumerate(tickers):
            status_text.text(f"Processing: {ticker} ({idx+1}/{len(tickers)})")
            progress_bar.progress((idx + 1) / len(tickers))
            
            st.write("---")
            st.subheader(f"Ticker: {ticker}")
            
            try:
                # Download data
                df_raw = yf.download(ticker, start=start, end=end, progress=False)
                if df_raw is None or df_raw.empty:
                    st.warning(f"No data available for {ticker}")
                    continue
                
                # Check for MultiIndex columns and fix if needed
                if isinstance(df_raw.columns, pd.MultiIndex):
                    st.warning(f"MultiIndex columns detected for {ticker}, flattening...")
                    df_raw.columns = ['_'.join(col).strip() for col in df_raw.columns.values]
                    # Rename essential columns to standard names
                    column_mapping = {
                        'Open': 'Open',
                        'High': 'High', 
                        'Low': 'Low',
                        'Close': 'Close',
                        'Volume': 'Volume',
                        'Adj Close': 'Adj Close'
                    }
                    
                    for col in df_raw.columns:
                        for standard_name, pattern in column_mapping.items():
                            if pattern in col:
                                df_raw = df_raw.rename(columns={col: standard_name})
                                break
                
                # Ensure we have the required columns
                required_cols = ["Open", "High", "Low", "Close", "Volume"]
                missing_cols = [col for col in required_cols if col not in df_raw.columns]
                if missing_cols:
                    st.warning(f"Missing columns for {ticker}: {missing_cols}")
                    continue
                
                # Calculate indicators
                ich = compute_ichimoku(df_raw)
                if ich.empty:
                    st.warning(f"Could not calculate indicators for {ticker}")
                    continue
                
                # Run backtest
                bt, cum_market, cum_strategy = backtest_sim(ich, stop_loss_pct, take_profit_pct)
                
                if bt.empty:
                    st.warning(f"Backtest failed for {ticker}")
                    continue
                
                # Store results
                sheet_name = ticker.replace(":", "_").replace("/", "_")[:31]  # Excel sheet name limitations
                sheet_data[sheet_name] = bt
                
                if len(cum_market) > 0 and len(cum_strategy) > 0:
                    market_return = float(cum_market.iloc[-1] - 1) if not pd.isna(cum_market.iloc[-1]) else np.nan
                    strategy_return = float(cum_strategy.iloc[-1] - 1) if not pd.isna(cum_strategy.iloc[-1]) else np.nan
                    
                    summary[ticker] = {
                        "market_return": market_return,
                        "strategy_return": strategy_return,
                        "outperformance": strategy_return - market_return if not (np.isnan(strategy_return) or np.isnan(market_return)) else np.nan
                    }
                    
                    equity_curves[ticker] = {
                        "Market": cum_market,
                        "Strategy": cum_strategy
                    }
                
                # Individual candlestick chart with signals
                fig = go.Figure()
                
                # Ensure we have valid data for plotting
                plot_data = bt.dropna(subset=["Open", "High", "Low", "Close"])
                if plot_data.empty:
                    st.warning(f"No valid data for plotting {ticker}")
                    continue
                
                fig.add_trace(go.Candlestick(
                    x=plot_data.index, 
                    open=plot_data["Open"], 
                    high=plot_data["High"], 
                    low=plot_data["Low"], 
                    close=plot_data["Close"], 
                    name="Price", 
                    increasing_line_color='#2E86AB', 
                    decreasing_line_color='#A23B72'
                ))
                
                # Add Ichimoku components if available
                for col, color, name in [
                    ("Tenkan", "#F9C80E", "Tenkan"),
                    ("Kijun", "#FF4365", "Kijun"),
                    ("SenkouA", "#43AA8B", "SenkouA"),
                    ("SenkouB", "#5E60CE", "SenkouB")
                ]:
                    if col in bt.columns:
                        # Filter out NaN values for plotting
                        plot_col = bt[[col]].dropna()
                        if not plot_col.empty:
                            fig.add_trace(go.Scatter(
                                x=plot_col.index, y=plot_col[col], mode="lines", 
                                name=name, line=dict(color=color, width=1)
                            ))
                
                # Add buy signals
                if "BuySignal" in bt.columns:
                    buy_mask = bt["BuySignal"].fillna(False)
                    close_mask = (~bt["Close"].isna())
                    buys = bt.loc[buy_mask & close_mask]
                    
                    if not buys.empty:
                        fig.add_trace(go.Scatter(
                            x=buys.index, y=buys["Close"], mode="markers",
                            name="Buy signal",
                            marker=dict(color="green", size=8, symbol="triangle-up", line=dict(width=1, color="DarkGreen"))
                        ))
                
                fig.update_layout(
                    title=f"{ticker} Candlestick + Ichimoku", 
                    xaxis_rangeslider_visible=False,
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error processing {ticker}: {str(e)}")
                import traceback
                st.code(f"Error details: {traceback.format_exc()}")
                continue
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Summary table
        if summary:
            st.subheader("📊 Performance Summary")
            df_summary = pd.DataFrame(summary).T
            df_summary["market_return"] = (df_summary["market_return"] * 100).round(2)
            df_summary["strategy_return"] = (df_summary["strategy_return"] * 100).round(2)
            df_summary["outperformance"] = (df_summary["outperformance"] * 100).round(2)
            
            df_summary.columns = ["Market Return (%)", "Strategy Return (%)", "Outperformance (%)"]
            st.dataframe(df_summary.style.format("{:.2f}%"))
            
            # Best and worst performers
            if not df_summary.empty:
                best_ticker = df_summary["Strategy Return (%)"].idxmax()
                worst_ticker = df_summary["Strategy Return (%)"].idxmin()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Best Performer", best_ticker, 
                             f"{df_summary.loc[best_ticker, 'Strategy Return (%)']:.2f}%")
                with col2:
                    st.metric("Worst Performer", worst_ticker, 
                             f"{df_summary.loc[worst_ticker, 'Strategy Return (%)']:.2f}%")
        
        # Equity curve overlay chart
        if equity_curves:
            st.subheader("📈 Equity Curve Comparison")
            fig_eq = go.Figure()
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            
            for i, (ticker, curves) in enumerate(equity_curves.items()):
                color_idx = i % len(colors)
                # Filter out NaN values
                market_curve = curves["Market"].dropna()
                strategy_curve = curves["Strategy"].dropna()
                
                if not market_curve.empty:
                    fig_eq.add_trace(go.Scatter(
                        x=market_curve.index, 
                        y=market_curve.values,
                        mode="lines", 
                        name=f"{ticker} Market",
                        line=dict(color=colors[color_idx], dash="dash")
                    ))
                if not strategy_curve.empty:
                    fig_eq.add_trace(go.Scatter(
                        x=strategy_curve.index, 
                        y=strategy_curve.values,
                        mode="lines", 
                        name=f"{ticker} Strategy",
                        line=dict(color=colors[color_idx])
                    ))
            
            fig_eq.update_layout(
                title="Equity Curves", 
                xaxis_title="Date", 
                yaxis_title="Equity (Normalized)",
                xaxis_rangeslider_visible=False,
                hovermode='x unified'
            )
            st.plotly_chart(fig_eq, use_container_width=True)
        
        # Excel export
        if sheet_data:
            try:
                excel_buf = BytesIO()
                with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
                    for sheet_name, df_sheet in sheet_data.items():
                        if not df_sheet.empty:
                            # Ensure the sheet name is valid for Excel
                            safe_sheet_name = ''.join(c for c in sheet_name if c not in r'[]:*?/\\')
                            safe_sheet_name = safe_sheet_name[:31]  # Excel limit
                            df_sheet.to_excel(writer, sheet_name=safe_sheet_name)
                    
                    if summary:
                        perf = pd.DataFrame(summary).T
                        perf.to_excel(writer, sheet_name="Performance_Summary")
                
                excel_buf.seek(0)
                
                st.download_button(
                    "💾 Download Results (Excel)",
                    excel_buf.getvalue(),
                    file_name="ichimoku_multi_equity.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                st.error(f"Error creating Excel file: {str(e)}")
        else:
            st.warning("No data available to export")

# Add some instructions when the app first loads
else:
    st.info("👈 Enter tickers and adjust parameters in the sidebar, then click 'Run Backtest' to analyze.")
    
    # Example tickers
    st.subheader("Example Tickers")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("US Stocks:")
        st.code("AAPL,MSFT,GOOGL,AMZN,TSLA")
    with col2:
        st.write("Indonesian Stocks:")
        st.code("BBCA.JK,BBRI.JK,TLKM.JK,ASII.JK")
    with col3:
        st.write("ETFs:")
        st.code("SPY,QQQ,IWM,GLD")
    
    # Add some information about Ichimoku
    with st.expander("About Ichimoku Kinko Hyo"):
        st.markdown("""
        **Ichimoku Kinko Hyo** is a comprehensive technical analysis system that provides:
        
        - **Tenkan-sen (Conversion Line)**: (9-period high + 9-period low)/2
        - **Kijun-sen (Base Line)**: (26-period high + 26-period low)/2
        - **Senkou Span A (Leading Span A)**: (Tenkan-sen + Kijun-sen)/2 shifted 26 periods forward
        - **Senkou Span B (Leading Span B)**: (52-period high + 52-period low)/2 shifted 26 periods forward
        - **Chikou Span (Lagging Span)**: Current closing price shifted 26 periods backward
        
        The system helps identify support/resistance, trend direction, momentum, and provides trading signals.
        """)
