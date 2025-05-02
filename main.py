# main.py (with backtesting)
import os
import requests
import pandas as pd
import numpy as np # For calculations
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, HTTPException, Query, Body
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from dotenv import load_dotenv

# Import backtesting components
from backtest_engine import run_backtest
from strategies import SimpleMACrossStrategy, BaseStrategy # Import BaseStrategy too

# --- Configuration & Setup ---
load_dotenv()
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

if not FINNHUB_API_KEY:
    print("WARNING: FINNHUB_API_KEY environment variable not set.")
    # raise ValueError("FINNHUB_API_KEY environment variable is required.")

# --- Finnhub Data Fetching Function (Unchanged from v1) ---
def get_finnhub_candles(symbol: str, resolution: str, start_timestamp: int, end_timestamp: int) -> pd.DataFrame | None:
    """ Fetches historical candle data from the Finnhub API. """
    if not FINNHUB_API_KEY:
        print("Error: Finnhub API Key is not configured.")
        return None
    FINNHUB_URL = "https://finnhub.io/api/v1/stock/candle"
    params = {
        'symbol': symbol.upper(),'resolution': resolution,'from': start_timestamp,
        'to': end_timestamp,'token': FINNHUB_API_KEY,'format': 'json'
    }
    try:
        # print(f"Requesting Finnhub data: {symbol} ({resolution}) from {datetime.fromtimestamp(start_timestamp)} to {datetime.fromtimestamp(end_timestamp)}")
        response = requests.get(FINNHUB_URL, params=params, timeout=20) # Increased timeout slightly
        response.raise_for_status()
        data = response.json()
        if data.get('s') == 'no_data' or 't' not in data or not data['t']:
            # print(f"No data found for {symbol} in the specified range.")
            return pd.DataFrame()
        if data.get('s') == 'ok':
            df = pd.DataFrame(data)
            df['datetime'] = pd.to_datetime(df['t'], unit='s', utc=True)
            df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
            df = df.set_index('datetime')
            df = df[['open', 'high', 'low', 'close', 'volume']]
            df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': int})
            # print(f"Successfully fetched {len(df)} candles for {symbol}.")
            return df
        else:
            print(f"Finnhub returned status '{data.get('s', 'Unknown')}' for {symbol}. Response: {data}")
            return None
    except requests.exceptions.Timeout:
        print(f"Error fetching Finnhub data for {symbol}: Request timed out.")
        return None
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred fetching Finnhub data for {symbol}: {http_err} - Status Code: {response.status_code}")
        return None
    except requests.exceptions.RequestException as req_err:
        print(f"Request error occurred fetching Finnhub data for {symbol}: {req_err}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred processing Finnhub data for {symbol}: {e}")
        return None

# --- Pydantic Models for Backtesting ---
class BacktestParams(BaseModel):
    """ Parameters specific to the strategy being backtested. """
    # Example for SimpleMACrossStrategy
    fast_ma_period: int = Field(20, gt=0)
    slow_ma_period: int = Field(50, gt=0)
    # Add other parameters for different strategies here
    # vhf_period: Optional[int] = None
    # fdi_period: Optional[int] = None

class BacktestRequest(BaseModel):
    """ Request model for the backtest endpoint. """
    strategy_template: str # Identifier for the strategy (e.g., 'simple_ma_cross')
    symbol: str
    resolution: str # e.g., 'D', '60', '15'
    start_date: str # YYYY-MM-DD
    end_date: str # YYYY-MM-DD
    initial_capital: float = Field(100000.0, gt=0)
    parameters: BacktestParams # Strategy-specific parameters

class EquityPoint(BaseModel):
    """ Represents a single point in the equity curve. """
    date: str # ISO format date string
    value: float

class BacktestMetrics(BaseModel):
    """ Performance metrics calculated from the backtest. """
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float | None = None # Optional as needs risk-free rate
    profit_factor: float | None = None
    win_rate_pct: float | None = None
    total_trades: int

class BacktestResponse(BaseModel):
    """ Response model for the backtest endpoint. """
    message: str
    symbol: str
    strategy: str
    equity_curve: List[EquityPoint]
    metrics: BacktestMetrics

# --- Strategy Mapping ---
# Map strategy template names (from frontend) to backend Strategy classes
STRATEGY_MAP: Dict[str, type[BaseStrategy]] = {
    "simple_ma_cross": SimpleMACrossStrategy,
    # Add other strategies here as they are implemented
    # "vhf_filter": VhfFilteredStrategy,
    # "fdi_adaptive": FdiAdaptiveStrategy,
}

# --- FastAPI Application ---
app = FastAPI(
    title="Simplified AlgoTrading Backend",
    description="Provides basic access to Finnhub market data and backtesting.",
    version="0.2.0" # Incremented version
)

@app.get("/")
async def read_root():
    """ Basic welcome endpoint. """
    return {"message": "Welcome to the Simplified AlgoTrading Backend. Use /docs for API details."}

@app.get("/data/candles/{symbol}")
async def get_candles_api(
    symbol: str,
    resolution: str = Query("D", description="Candle resolution (e.g., '1', '5', 'D', 'W')"),
    days_history: int = Query(30, description="Number of past days of data to fetch", ge=1, le=365*5),
):
    """ API endpoint to fetch historical candle data for a given symbol. (Unchanged) """
    if not FINNHUB_API_KEY:
         raise HTTPException(status_code=500, detail="Server configuration error: Finnhub API Key not set.")
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days_history)
    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())
    df = get_finnhub_candles(symbol, resolution, start_ts, end_ts)
    if df is None:
        raise HTTPException(status_code=502, detail=f"Failed to fetch data from Finnhub for {symbol}.")
    elif df.empty:
         return {"symbol": symbol, "resolution": resolution, "data": [], "message": "No data found for the specified parameters."}
    else:
        df_reset = df.reset_index()
        df_reset['datetime'] = df_reset['datetime'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        data_json = df_reset.to_dict(orient='records')
        return {"symbol": symbol, "resolution": resolution, "data": data_json}


@app.post("/backtest", response_model=BacktestResponse)
async def perform_backtest(request: BacktestRequest = Body(...)):
    """
    Performs a backtest for a given strategy and symbol.
    """
    print(f"Received backtest request: {request.dict()}")

    if not FINNHUB_API_KEY:
         raise HTTPException(status_code=500, detail="Server configuration error: Finnhub API Key not set.")

    # --- 1. Validate Strategy ---
    StrategyClass = STRATEGY_MAP.get(request.strategy_template)
    if not StrategyClass:
        raise HTTPException(status_code=400, detail=f"Unknown strategy template: '{request.strategy_template}'")

    # --- 2. Fetch Data ---
    try:
        # Convert date strings to timestamps
        start_dt = datetime.strptime(request.start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(request.end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        # Add extra time to end_dt to ensure the last day's candle is included if resolution is daily
        end_dt += timedelta(hours=23, minutes=59) # Adjust as needed based on resolution
        start_ts = int(start_dt.timestamp())
        end_ts = int(end_dt.timestamp())

        if start_ts >= end_ts:
             raise HTTPException(status_code=400, detail="Start date must be before end date.")

    except ValueError:
         raise HTTPException(status_code=400, detail="Invalid date format. Please use YYYY-MM-DD.")

    print(f"Fetching data for backtest: {request.symbol} ({request.resolution}) from {start_dt} to {end_dt}")
    historical_data = get_finnhub_candles(request.symbol, request.resolution, start_ts, end_ts)

    if historical_data is None:
        raise HTTPException(status_code=502, detail=f"Failed to fetch historical data from Finnhub for {request.symbol}.")
    if historical_data.empty:
        raise HTTPException(status_code=404, detail=f"No historical data found for {request.symbol} in the specified date range and resolution.")

    # --- 3. Instantiate Strategy ---
    try:
        # Pass only the relevant parameters defined in BacktestParams
        strategy_instance = StrategyClass(
            parameters=request.parameters.dict(),
            symbols=[request.symbol], # Simple backtester handles one symbol
            timeframe=request.resolution # Pass resolution as timeframe
        )
    except Exception as e:
        # Catch potential errors during strategy initialization (e.g., bad params)
        print(f"Error initializing strategy: {e}")
        raise HTTPException(status_code=400, detail=f"Error initializing strategy '{request.strategy_template}': {e}")


    # --- 4. Run Backtest Engine ---
    print(f"Running backtest for {request.symbol} with strategy {request.strategy_template}...")
    try:
        results = run_backtest(
            data=historical_data,
            strategy=strategy_instance,
            initial_capital=request.initial_capital
            # Add commission/slippage models here later
        )
        print("Backtest finished.")
    except Exception as e:
        # Catch errors during the backtest run itself
        print(f"Error during backtest execution: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        raise HTTPException(status_code=500, detail=f"An error occurred during backtest execution: {e}")


    # --- 5. Format Response ---
    equity_curve_response = [
        EquityPoint(date=dt.strftime('%Y-%m-%dT%H:%M:%SZ'), value=val)
        for dt, val in results['equity_curve'].items()
    ]

    response_data = BacktestResponse(
        message="Backtest completed successfully.",
        symbol=request.symbol,
        strategy=request.strategy_template,
        equity_curve=equity_curve_response,
        metrics=BacktestMetrics(**results['metrics']) # Unpack metrics dict
    )

    return response_data

# --- Helper Files (Create these in the same directory) ---
# File: strategies.py
# File: backtest_engine.py

# --- How to Run ---
# 1. Install dependencies:
#    pip install fastapi uvicorn requests pandas python-dotenv numpy
# 2. Create `.env` file with FINNHUB_API_KEY.
# 3. Create `strategies.py` and `backtest_engine.py` (code provided below).
# 4. Run the server:
#    uvicorn main:app --reload
# 5. Access the API docs at http://127.0.0.1:8000/docs

```

**Create `strategies.py`:**

```python
# strategies.py
import pandas as pd
import numpy as np
# Import algorithm implementations if needed later
# from algorithms import calculate_vhf, calculate_fdi

class BaseStrategy:
    """ Base class for trading strategies. """
    def __init__(self, parameters: dict, symbols: list[str], timeframe: str):
        self.parameters = parameters
        self.symbols = symbols # Should contain only one symbol for this simple backtester
        self.timeframe = timeframe
        self.symbol = symbols[0] if symbols else None
        # --- Internal State ---
        self._position = 0 # Current holding quantity (+ve long, -ve short, 0 flat)
        self._data = pd.DataFrame() # Holds the historical/live data processed so far
        self.signals = [] # List to store generated signals/orders (for analysis)

    @property
    def position(self):
        return self._position

    def initialize(self, initial_data: pd.DataFrame):
        """ Called once with the initial historical data chunk. """
        self._data = initial_data
        print(f"Initialized {self.__class__.__name__} for {self.symbol} with {len(initial_data)} bars.")
        # Pre-calculate indicators if needed

    def on_bar(self, current_bar: pd.Series):
        """
        Called for each new bar. Implement strategy logic here.
        Should return order signals (e.g., 'BUY', 'SELL', 'FLAT').
        """
        # Append the new bar to internal data
        # Important: Ensure index alignment if concatenating
        # Using pd.concat might be safer depending on how current_bar is structured
        # For simplicity here, we assume the backtester passes bars sequentially
        # In a real system, robust data handling is crucial.
        self._data = pd.concat([self._data, current_bar.to_frame().T])

        # --- Strategy logic implemented by subclasses ---
        raise NotImplementedError("Subclasses should implement the on_bar logic.")

    def _generate_signal(self, timestamp, signal_type: str, quantity: int | None = None):
        """ Helper to record signals/orders. """
        # Simple signal recording
        self.signals.append({
            "timestamp": timestamp,
            "symbol": self.symbol,
            "type": signal_type, # e.g., 'ENTER_LONG', 'EXIT_LONG', 'ENTER_SHORT', 'EXIT_SHORT'
            "quantity": quantity, # Optional: For systems tracking quantity
            "position_before": self._position
        })
        print(f"{timestamp} - Signal: {signal_type} {self.symbol}")


class SimpleMACrossStrategy(BaseStrategy):
    """ Simple Moving Average Crossover Strategy. """
    def on_bar(self, current_bar: pd.Series):
        """ Generates buy/sell signals based on MA crossover. """
        super().on_bar(current_bar) # Appends data

        fast_period = self.parameters.get('fast_ma_period', 20)
        slow_period = self.parameters.get('slow_ma_period', 50)
        timestamp = current_bar.name # Get the datetime index of the current bar

        # Ensure enough data for the longer MA
        if len(self._data) < slow_period + 1: # Need +1 to calculate previous bar's MA
            return 'FLAT' # Not enough data yet

        # Calculate MAs for the current and previous bar
        try:
            current_fast_ma = self._data['close'].rolling(window=fast_period).mean().iloc[-1]
            current_slow_ma = self._data['close'].rolling(window=slow_period).mean().iloc[-1]
            prev_fast_ma = self._data['close'].rolling(window=fast_period).mean().iloc[-2]
            prev_slow_ma = self._data['close'].rolling(window=slow_period).mean().iloc[-2]
        except IndexError:
             return 'FLAT' # Should not happen if length check is correct, but safety first

        # --- Crossover Logic ---
        signal = 'FLAT' # Default action is to do nothing or maintain position

        # Bullish Crossover (Fast MA crosses above Slow MA)
        if prev_fast_ma <= prev_slow_ma and current_fast_ma > current_slow_ma:
            if self._position <= 0: # Enter long only if flat or short
                signal = 'ENTER_LONG'
                self._generate_signal(timestamp, signal)
                self._position = 1 # Go fully long (simplified: position = 1)

        # Bearish Crossover (Fast MA crosses below Slow MA)
        elif prev_fast_ma >= prev_slow_ma and current_fast_ma < current_slow_ma:
             if self._position >= 0: # Exit long or enter short if flat or long
                signal = 'ENTER_SHORT' # Or 'EXIT_LONG' if not shorting
                self._generate_signal(timestamp, signal)
                self._position = -1 # Go fully short (simplified: position = -1)

        # Maintain position if no crossover
        # Note: This simple version doesn't explicitly exit positions unless
        # an opposite crossover occurs. More robust strategies need exit logic.

        return signal # Although signal isn't directly used by this simple backtester yet

```

**Create `backtest_engine.py`:**

```python
# backtest_engine.py
import pandas as pd
import numpy as np
from strategies import BaseStrategy # Import the base class

def run_backtest(data: pd.DataFrame, strategy: BaseStrategy, initial_capital: float) -> dict:
    """
    Runs a simplified vectorized/event-driven backtest.

    Args:
        data: DataFrame with OHLCV data, indexed by datetime.
        strategy: An instantiated strategy object inheriting from BaseStrategy.
        initial_capital: Starting capital for the backtest.

    Returns:
        A dictionary containing 'equity_curve' (dict) and 'metrics' (dict).
    """
    print(f"Starting backtest run. Initial Capital: ${initial_capital:,.2f}")
    if data.empty:
        raise ValueError("Input data for backtest is empty.")

    # --- Initialization ---
    capital = initial_capital
    position_size = 0 # Shares/Contracts currently held
    portfolio_value = initial_capital
    equity_curve = {} # Dictionary to store portfolio value at each timestamp
    trades = [] # List to store trade details
    peak_equity = initial_capital
    max_drawdown_pct = 0.0

    # --- Strategy Initialization ---
    # Pass all data initially (some strategies might precompute)
    strategy.initialize(data.iloc[0:0]) # Pass empty df initially, strategy.on_bar will build it up

    # --- Event Loop (Iterate through each bar) ---
    for timestamp, current_bar in data.iterrows():
        # 1. Update Portfolio Value (Mark-to-Market)
        # Simplification: Value position based on the current closing price
        current_price = current_bar['close']
        portfolio_value = capital + (position_size * current_price)
        equity_curve[timestamp] = portfolio_value

        # 2. Update Max Drawdown
        if portfolio_value > peak_equity:
            peak_equity = portfolio_value
        drawdown = (peak_equity - portfolio_value) / peak_equity if peak_equity > 0 else 0
        max_drawdown_pct = max(max_drawdown_pct, drawdown)

        # 3. Get Signal from Strategy
        # The strategy updates its internal data and position state
        signal = strategy.on_bar(current_bar)
        # In this simple engine, we directly use the strategy's internal position state
        # A more complex engine would use the returned signal ('BUY'/'SELL')
        # to simulate order fills based on next bar's open etc.

        target_position = strategy.position # Get desired position state (-1, 0, 1)

        # --- 4. Simplified Execution Simulation ---
        # Adjust position based on target state (very basic execution model)
        # Assumes we can trade at the current closing price without slippage/commission
        # Assumes fixed position sizing (e.g., invest full capital) - HIGHLY SIMPLIFIED

        if target_position == 1 and position_size <= 0: # Enter Long or reverse Short
            # Liquidate short position first if necessary
            if position_size < 0:
                capital += abs(position_size) * current_price # Close short
                trades.append({"timestamp": timestamp, "type": "Close Short", "price": current_price, "size": abs(position_size)})
                position_size = 0

            # Calculate size to buy (invest available capital)
            size_to_buy = int(capital / current_price)
            if size_to_buy > 0:
                capital -= size_to_buy * current_price
                position_size = size_to_buy
                trades.append({"timestamp": timestamp, "type": "Enter Long", "price": current_price, "size": size_to_buy})

        elif target_position == -1 and position_size >= 0: # Enter Short or reverse Long
             # Liquidate long position first if necessary
            if position_size > 0:
                capital += position_size * current_price # Close long
                trades.append({"timestamp": timestamp, "type": "Close Long", "price": current_price, "size": position_size})
                position_size = 0

            # Calculate size to short (invest available capital notionally)
            size_to_short = int(capital / current_price) # Simplified sizing
            if size_to_short > 0:
                capital += size_to_short * current_price # Add proceeds from shorting (simplistic)
                position_size = -size_to_short
                trades.append({"timestamp": timestamp, "type": "Enter Short", "price": current_price, "size": size_to_short})

        elif target_position == 0 and position_size != 0: # Exit position to Flat
            if position_size > 0: # Close long
                 capital += position_size * current_price
                 trades.append({"timestamp": timestamp, "type": "Exit Long", "price": current_price, "size": position_size})
                 position_size = 0
            elif position_size < 0: # Close short
                 capital += abs(position_size) * current_price # Cover short
                 trades.append({"timestamp": timestamp, "type": "Exit Short", "price": current_price, "size": abs(position_size)})
                 position_size = 0

        # Ensure capital doesn't go negative (basic check)
        if capital < 0 and position_size == 0:
            print(f"Warning: Capital became negative at {timestamp}. Resetting to 0.")
            capital = 0 # Basic floor

    # --- Final Portfolio Value ---
    final_portfolio_value = equity_curve[data.index[-1]] if equity_curve else initial_capital

    # --- Calculate Metrics ---
    total_return_pct = ((final_portfolio_value / initial_capital) - 1) * 100 if initial_capital > 0 else 0
    total_trades = len([t for t in trades if 'Enter' in t['type']]) # Count entry trades

    # Basic win rate / profit factor (needs proper trade P/L tracking)
    # These are placeholders as the simple engine doesn't track P/L per trade easily
    win_rate_pct = None
    profit_factor = None

    print(f"Backtest finished. Final Portfolio Value: ${final_portfolio_value:,.2f}")
    print(f"Total Return: {total_return_pct:.2f}%")
    print(f"Max Drawdown: {max_drawdown_pct*100:.2f}%")
    print(f"Total Trades: {total_trades}")

    # --- Return Results ---
    results = {
        "equity_curve": equity_curve,
        "metrics": {
            "total_return_pct": round(total_return_pct, 2),
            "max_drawdown_pct": round(max_drawdown_pct * 100, 2),
            "sharpe_ratio": None, # Requires more complex calculation
            "profit_factor": profit_factor,
            "win_rate_pct": win_rate_pct,
            "total_trades": total_trades,
        },
        "trades": trades # Include trades list for potential analysis
    }
    return results

```

**Part 2: Updated Frontend Code**

This version updates the backtesting section to call the new `/backtest` endpoint and display the results.


```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AlgoTrading Pro - Prototype (Backend Backtest)</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://unpkg.com/lucide-react@0.292.0/dist/umd/lucide.min.js"></script>
    <style>
        /* Use Inter font */
        body {
            font-family: 'Inter', sans-serif;
        }
        /* Custom scrollbar */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: #f1f1f1; border-radius: 10px; }
        ::-webkit-scrollbar-thumb { background: #888; border-radius: 10px; }
        ::-webkit-scrollbar-thumb:hover { background: #555; }
        /* Active sidebar link */
        .sidebar-link.active { background-color: #3b82f6; color: white; }
        .sidebar-link.active svg { stroke: white; }
        /* Responsive canvas */
        canvas { max-width: 100%; height: auto !important; }
        /* Loader */
        .loader {
            border: 4px solid #f3f3f3; border-top: 4px solid #3498db;
            border-radius: 50%; width: 20px; height: 20px;
            animation: spin 1s linear infinite; display: inline-block;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
    <link rel="preconnect" href="https://rsms.me/">
     <link rel="stylesheet" href="https://rsms.me/inter/inter.css">
     <script>
        tailwind.config = { theme: { extend: { fontFamily: { sans: ['Inter', 'sans-serif'], }, } } }
      </script>
</head>
<body class="bg-gray-100 flex h-screen overflow-hidden">

    <aside class="w-64 bg-gray-800 text-gray-300 flex flex-col fixed inset-y-0 left-0 z-30">
        <div class="p-4 border-b border-gray-700 flex items-center space-x-2">
             <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-candlestick-chart text-blue-400"><path d="M9 5v4"/><rect width="4" height="6" x="7" y="9" rx="1"/><path d="M9 15v2"/><path d="M17 3v2"/><rect width="4" height="8" x="15" y="5" rx="1"/><path d="M17 13v3"/><path d="M3 3v18h18"/></svg>
            <h1 class="text-xl font-semibold text-white">AlgoTrading Pro</h1>
        </div>
        <nav class="flex-1 overflow-y-auto p-4 space-y-2">
            <a href="#dashboard" class="sidebar-link flex items-center space-x-3 px-3 py-2 rounded-md hover:bg-gray-700 hover:text-white transition-colors duration-150 active" data-section="dashboard">
                <i data-lucide="layout-dashboard" class="w-5 h-5"></i><span>Dashboard</span>
            </a>
            <a href="#strategies" class="sidebar-link flex items-center space-x-3 px-3 py-2 rounded-md hover:bg-gray-700 hover:text-white transition-colors duration-150" data-section="strategies">
                <i data-lucide="sliders-horizontal" class="w-5 h-5"></i><span>Strategies</span>
            </a>
            <a href="#backtesting" class="sidebar-link flex items-center space-x-3 px-3 py-2 rounded-md hover:bg-gray-700 hover:text-white transition-colors duration-150" data-section="backtesting">
                <i data-lucide="history" class="w-5 h-5"></i><span>Backtesting</span>
            </a>
             <a href="#monitoring" class="sidebar-link flex items-center space-x-3 px-3 py-2 rounded-md hover:bg-gray-700 hover:text-white transition-colors duration-150" data-section="monitoring">
                <i data-lucide="activity" class="w-5 h-5"></i><span>Monitoring</span>
            </a>
            <a href="#alerts" class="sidebar-link flex items-center space-x-3 px-3 py-2 rounded-md hover:bg-gray-700 hover:text-white transition-colors duration-150" data-section="alerts">
                <i data-lucide="bell" class="w-5 h-5"></i><span>Alerts</span>
                 <span class="ml-auto inline-block py-0.5 px-2 text-xs font-medium bg-red-500 text-white rounded-full">3</span>
            </a>
        </nav>
        <div class="p-4 border-t border-gray-700">
            <a href="#settings" class="flex items-center space-x-3 px-3 py-2 rounded-md hover:bg-gray-700 hover:text-white transition-colors duration-150" data-section="settings">
                <i data-lucide="settings" class="w-5 h-5"></i><span>Settings</span>
            </a>
             <a href="#logout" class="flex items-center space-x-3 px-3 py-2 rounded-md hover:bg-gray-700 hover:text-white transition-colors duration-150" data-section="logout">
                <i data-lucide="log-out" class="w-5 h-5"></i><span>Logout</span>
            </a>
        </div>
    </aside>

    <main class="flex-1 ml-64 overflow-y-auto p-6 md:p-8">
        <section id="dashboard" class="content-section">
             <h2 class="text-2xl font-semibold text-gray-800 mb-6">Dashboard</h2>
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
                <div class="bg-white p-4 rounded-lg shadow flex items-center justify-between">
                    <div><p class="text-sm text-gray-500">Total P&L (Today)</p><p id="kpi-pnl" class="text-2xl font-semibold text-green-600">$0.00</p></div>
                     <i data-lucide="trending-up" class="w-8 h-8 text-green-500"></i>
                </div>
                 <div class="bg-white p-4 rounded-lg shadow flex items-center justify-between">
                    <div><p class="text-sm text-gray-500">Active Strategies</p><p id="kpi-active-strats" class="text-2xl font-semibold text-gray-700">5</p></div>
                    <i data-lucide="bot" class="w-8 h-8 text-blue-500"></i>
                </div>
                <div class="bg-white p-4 rounded-lg shadow flex items-center justify-between">
                    <div><p class="text-sm text-gray-500">Open Positions</p><p id="kpi-open-positions" class="text-2xl font-semibold text-gray-700">12</p></div>
                    <i data-lucide="layers" class="w-8 h-8 text-purple-500"></i>
                </div>
                 <div class="bg-white p-4 rounded-lg shadow flex items-center justify-between">
                    <div><p class="text-sm text-gray-500">System Status</p><p id="kpi-system-status" class="text-2xl font-semibold text-green-600 flex items-center"><span class="w-3 h-3 bg-green-500 rounded-full mr-2 inline-block"></span> Normal</p></div>
                     <i data-lucide="server" class="w-8 h-8 text-gray-500"></i>
                </div>
            </div>
            <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div class="lg:col-span-2 bg-white p-4 rounded-lg shadow"><h3 class="text-lg font-semibold text-gray-700 mb-4">Portfolio Equity Curve (Simulated)</h3><canvas id="equityCurveChart"></canvas></div>
                <div class="bg-white p-4 rounded-lg shadow"><h3 class="text-lg font-semibold text-gray-700 mb-4">Active Positions</h3><div class="overflow-x-auto max-h-96"><table class="min-w-full divide-y divide-gray-200 text-sm"><thead class="bg-gray-50"><tr><th class="px-4 py-2 text-left font-medium text-gray-500 uppercase tracking-wider">Symbol</th><th class="px-4 py-2 text-right font-medium text-gray-500 uppercase tracking-wider">Qty</th><th class="px-4 py-2 text-right font-medium text-gray-500 uppercase tracking-wider">Avg Price</th><th class="px-4 py-2 text-right font-medium text-gray-500 uppercase tracking-wider">P&L</th></tr></thead><tbody id="positions-table" class="bg-white divide-y divide-gray-200"></tbody></table></div></div>
            </div>
        </section>

        <section id="strategies" class="content-section hidden">
             <h2 class="text-2xl font-semibold text-gray-800 mb-6">Strategy Configuration</h2>
            <div class="bg-white p-6 rounded-lg shadow">
                <h3 class="text-lg font-semibold text-gray-700 mb-4">Configure New Strategy Instance</h3>
                 <form id="strategy-form">
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-4">
                        <div><label for="strategy-name" class="block text-sm font-medium text-gray-700 mb-1">Instance Name</label><input type="text" id="strategy-name" name="strategy-name" value="My Simple MA Strategy" class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"></div>
                        <div><label for="strategy-template" class="block text-sm font-medium text-gray-700 mb-1">Select Strategy Template</label><select id="strategy-template" name="strategy-template" class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"><option value="simple_ma_cross">Simple MA Crossover</option><option value="vhf_filter" disabled>Trend Filter (VHF Based) - Backend NYI</option><option value="fdi_adaptive" disabled>FDI Adaptive Oscillator - Backend NYI</option></select></div>
                    </div>
                    <div id="strategy-parameters" class="mb-6 border-t pt-6"><h4 class="text-md font-semibold text-gray-600 mb-3">Parameters</h4><div class="grid grid-cols-1 md:grid-cols-3 gap-4"><div><label for="param-fast_ma_period" class="block text-xs font-medium text-gray-600 mb-1">Fast MA Period</label><input type="number" id="param-fast_ma_period" name="fast_ma_period" value="20" class="w-full px-3 py-1.5 border border-gray-300 rounded-md shadow-sm text-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"></div><div><label for="param-slow_ma_period" class="block text-xs font-medium text-gray-600 mb-1">Slow MA Period</label><input type="number" id="param-slow_ma_period" name="slow_ma_period" value="50" class="w-full px-3 py-1.5 border border-gray-300 rounded-md shadow-sm text-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"></div></div></div>
                     <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                        <div><label for="strategy-symbol" class="block text-sm font-medium text-gray-700 mb-1">Symbol(s)</label><input type="text" id="strategy-symbol" name="strategy-symbol" value="AAPL" placeholder="Enter one symbol for backtest" class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"></div>
                         <div><label for="strategy-timeframe" class="block text-sm font-medium text-gray-700 mb-1">Timeframe</label><select id="strategy-timeframe" name="strategy-timeframe" class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"><option value="1">1 Minute</option><option value="5">5 Minutes</option><option value="15">15 Minutes</option><option value="60">1 Hour</option><option value="D" selected>1 Day</option><option value="W">1 Week</option></select></div>
                     </div>
                     <div class="flex justify-end space-x-3"><button type="button" class="px-4 py-2 bg-gray-200 text-gray-700 rounded-md hover:bg-gray-300 transition-colors">Cancel</button><button type="submit" class="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors">Save Config</button></div>
                </form>
            </div>
        </section>

        <section id="backtesting" class="content-section hidden">
            <h2 class="text-2xl font-semibold text-gray-800 mb-6">Backtesting</h2>
            <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div class="lg:col-span-1 bg-white p-6 rounded-lg shadow">
                    <h3 class="text-lg font-semibold text-gray-700 mb-4">Setup Backtest</h3>
                    <form id="backtest-form">
                         <div class="mb-4">
                            <label for="backtest-strategy-template" class="block text-sm font-medium text-gray-700 mb-1">Select Strategy Template</label>
                            <select id="backtest-strategy-template" name="strategy_template" class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500">
                                <option value="simple_ma_cross">Simple MA Crossover</option>
                                <option value="vhf_filter" disabled>Trend Filter (VHF Based) - Backend NYI</option>
                                <option value="fdi_adaptive" disabled>FDI Adaptive Oscillator - Backend NYI</option>
                            </select>
                        </div>
                        <div id="backtest-parameters" class="mb-4 border rounded-md p-3 bg-gray-50">
                             <h4 class="text-sm font-medium text-gray-600 mb-2">Strategy Parameters</h4>
                             <p class="text-xs text-gray-500">Parameters loaded based on selected template.</p>
                         </div>
                         <div class="mb-4">
                            <label for="backtest-symbol" class="block text-sm font-medium text-gray-700 mb-1">Symbol</label>
                            <input type="text" id="backtest-symbol" name="symbol" value="AAPL" class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500">
                        </div>
                         <div class="grid grid-cols-2 gap-4 mb-4">
                            <div>
                                <label for="backtest-start" class="block text-sm font-medium text-gray-700 mb-1">Start Date</label>
                                <input type="date" id="backtest-start" name="start_date" value="2023-01-01" class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500">
                            </div>
                            <div>
                                <label for="backtest-end" class="block text-sm font-medium text-gray-700 mb-1">End Date</label>
                                <input type="date" id="backtest-end" name="end_date" value="2023-12-31" class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500">
                            </div>
                         </div>
                         <div class="mb-4">
                             <label for="backtest-resolution" class="block text-sm font-medium text-gray-700 mb-1">Resolution</label>
                             <select id="backtest-resolution" name="resolution" class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500">
                                 <option value="1">1 Minute</option>
                                 <option value="5">5 Minutes</option>
                                 <option value="15">15 Minutes</option>
                                 <option value="60">1 Hour</option>
                                 <option value="D" selected>1 Day</option>
                                 <option value="W">1 Week</option>
                             </select>
                         </div>
                         <div class="mb-4">
                            <label for="backtest-capital" class="block text-sm font-medium text-gray-700 mb-1">Initial Capital</label>
                            <input type="number" id="backtest-capital" name="initial_capital" value="100000" step="1000" class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500">
                        </div>
                         <div class="mb-4 opacity-50">
                            <label class="block text-sm font-medium text-gray-500 mb-1">Commission (per share/contract) <span class="text-xs">(NYI)</span></label>
                            <input type="number" value="0.005" step="0.001" class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm bg-gray-100" disabled>
                        </div>
                         <div class="mb-6 opacity-50">
                            <label class="block text-sm font-medium text-gray-500 mb-1">Slippage (ticks/points) <span class="text-xs">(NYI)</span></label>
                            <input type="number" value="1" step="1" class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm bg-gray-100" disabled>
                        </div>
                        <button type="button" id="run-backtest-btn" class="w-full px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors flex items-center justify-center space-x-2">
                            <i data-lucide="play" class="w-5 h-5"></i>
                            <span>Run Backtest</span>
                        </button>
                        <div id="backtest-status" class="mt-4 text-sm"></div>
                    </form>
                </div>

                <div class="lg:col-span-2 bg-white p-6 rounded-lg shadow">
                    <h3 class="text-lg font-semibold text-gray-700 mb-4">Backtest Results</h3>
                    <div id="backtest-results-content" class="text-gray-500 text-center py-10">
                        Run a backtest to see results here.
                    </div>
                    <div id="backtest-results-details" class="hidden">
                        <div class="mb-6">
                            <h4 class="text-md font-semibold text-gray-600 mb-2">Equity Curve</h4>
                            <canvas id="backtestEquityCurveChart"></canvas>
                        </div>
                        <div>
                            <h4 class="text-md font-semibold text-gray-600 mb-3">Performance Metrics</h4>
                             <div class="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
                                <div class="bg-gray-50 p-3 rounded">
                                    <p class="text-gray-500">Total Return %</p>
                                    <p id="metric-return" class="font-semibold text-lg">0%</p>
                                </div>
                                <div class="bg-gray-50 p-3 rounded">
                                    <p class="text-gray-500">Max Drawdown %</p>
                                    <p id="metric-drawdown" class="font-semibold text-lg text-red-600">0%</p>
                                </div>
                                <div class="bg-gray-50 p-3 rounded">
                                    <p class="text-gray-500">Sharpe Ratio</p>
                                    <p id="metric-sharpe" class="font-semibold text-lg text-gray-700">N/A</p>
                                </div>
                                 <div class="bg-gray-50 p-3 rounded">
                                    <p class="text-gray-500">Profit Factor</p>
                                    <p id="metric-profit-factor" class="font-semibold text-lg text-gray-700">N/A</p>
                                </div>
                                <div class="bg-gray-50 p-3 rounded">
                                    <p class="text-gray-500">Win Rate %</p>
                                    <p id="metric-winrate" class="font-semibold text-lg text-gray-700">N/A</p>
                                </div>
                                 <div class="bg-gray-50 p-3 rounded">
                                    <p class="text-gray-500">Total Trades</p>
                                    <p id="metric-trades" class="font-semibold text-lg text-gray-700">0</p>
                                </div>
                            </div>
                        </div>
                         </div>
                </div>
            </div>
        </section>

        <section id="monitoring" class="content-section hidden"><h2 class="text-2xl font-semibold text-gray-800 mb-6">Live Monitoring</h2><div class="bg-white p-6 rounded-lg shadow text-center text-gray-500">Placeholder...<div class="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-md text-blue-700 text-sm text-left">**Unique Feature Idea:** Adaptive Strategy Dashboard...</div></div></section>
         <section id="alerts" class="content-section hidden"><h2 class="text-2xl font-semibold text-gray-800 mb-6">Alerts</h2><div class="bg-white p-6 rounded-lg shadow"><h3 class="text-lg font-semibold text-gray-700 mb-4">Active Alerts</h3><div class="space-y-3"><div class="p-3 bg-red-50 border border-red-200 rounded-md flex justify-between items-center"><div><p class="font-medium text-red-700">Risk Limit Breach...</p></div><button>Acknowledge</button></div><div class="p-3 bg-yellow-50 border border-yellow-200 rounded-md flex justify-between items-center"><div><p class="font-medium text-yellow-700">Connectivity Issue...</p></div><button>Acknowledge</button></div><div class="p-3 bg-blue-50 border border-blue-200 rounded-md flex justify-between items-center"><div><p class="font-medium text-blue-700">Info: Strategy Deployed...</p></div><button>Acknowledge</button></div></div><div class="mt-6 text-right"><button>Configure Alert Rules</button></div></div></section>
         <section id="settings" class="content-section hidden"><h2 class="text-2xl font-semibold text-gray-800 mb-6">Settings</h2><div class="bg-white p-6 rounded-lg shadow text-center text-gray-500">Placeholder...</div></section>
         <section id="logout" class="content-section hidden"><h2 class="text-2xl font-semibold text-gray-800 mb-6">Logout</h2><div class="bg-white p-6 rounded-lg shadow text-center text-gray-500">User would be logged out here.</div></section>

    </main>

    <script>
        // --- Configuration ---
        const BACKEND_URL = 'http://127.0.0.1:8000';

        // --- Globals ---
        let equityChart = null;
        let backtestEquityChart = null;
        let pnlValue = 1250.75; // Initial mock P&L for dashboard
        let intervalId = null; // For dashboard updates

        // --- Mock Data (Dashboard only) ---
        const mockPositions = [ { symbol: 'AAPL', qty: 100, avgPrice: 175.20, pnl: 230.50 }, /* ... more mock positions */ ];
        const generateMockEquityData = (numPoints = 50, startValue = 100000) => { /* ... unchanged from v2 */
             const data = []; const labels = []; let value = startValue;
             const startDate = new Date(); startDate.setDate(startDate.getDate() - numPoints);
             for (let i = 0; i < numPoints; i++) {
                 value += (Math.random() - 0.48) * (startValue * 0.005);
                 const date = new Date(startDate); date.setDate(startDate.getDate() + i);
                 labels.push(date.toLocaleDateString('en-CA')); data.push(value.toFixed(2));
             } return { labels, data };
        };

        // --- Charting Functions ---
        const createChart = (ctx, labels, data, label) => {
            const existingChart = Chart.getChart(ctx);
            if (existingChart) { existingChart.destroy(); }
            return new Chart(ctx, {
                type: 'line', data: { labels: labels, datasets: [{ label: label, data: data, borderColor: 'rgb(59, 130, 246)', backgroundColor: 'rgba(59, 130, 246, 0.1)', borderWidth: 2, pointRadius: 0, tension: 0.1, fill: true }] },
                options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: false, ticks: { callback: (v) => '$' + v.toLocaleString() } }, x: { ticks: { maxTicksLimit: 10, autoSkip: true, } } }, plugins: { legend: { display: false } } }
            });
        };

        // --- UI Update Functions (Dashboard - unchanged) ---
        const updatePositionsTable = () => { /* ... unchanged from v2 */
             const tbody = document.getElementById('positions-table'); tbody.innerHTML = '';
             mockPositions.forEach(pos => { const row = document.createElement('tr'); const pnlClass = pos.pnl >= 0 ? 'text-green-600' : 'text-red-600'; row.innerHTML = `<td class="px-4 py-2 ...">${pos.symbol}</td><td class="px-4 py-2 ... text-right">${pos.qty}</td><td class="px-4 py-2 ... text-right">$${pos.avgPrice.toFixed(2)}</td><td class="px-4 py-2 ... text-right ${pnlClass}">$${pos.pnl.toFixed(2)}</td>`; tbody.appendChild(row); });
        };
        const updateKPIs = () => { /* ... unchanged from v2 */
             pnlValue += (Math.random() - 0.5) * 50; const pnlEl = document.getElementById('kpi-pnl'); pnlEl.textContent = `$${pnlValue.toFixed(2)}`; pnlEl.className = `... ${pnlValue >= 0 ? 'text-green-600' : 'text-red-600'}`; document.getElementById('kpi-active-strats').textContent = '5'; document.getElementById('kpi-open-positions').textContent = mockPositions.length; if (Math.random() < 0.05) { const statusEl = document.getElementById('kpi-system-status'); const isNormal = Math.random() < 0.8; statusEl.innerHTML = `<span class="w-3 h-3 ${isNormal ? 'bg-green-500' : 'bg-red-500'} ..."></span> ${isNormal ? 'Normal' : 'Error'}`; statusEl.className = `... ${isNormal ? 'text-green-600' : 'text-red-600'}`; }
        };
        const updateDashboardChart = () => { /* ... unchanged from v2 */
             if (equityChart) { const currentData = equityChart.data.datasets[0].data; const lastValue = currentData.length > 0 ? parseFloat(currentData[currentData.length - 1]) : 100000; const newDataPoint = lastValue + (Math.random() - 0.48) * 500; const newLabel = new Date().toLocaleTimeString(); equityChart.data.labels.push(newLabel); equityChart.data.datasets[0].data.push(newDataPoint.toFixed(2)); const maxDataPoints = 50; if (equityChart.data.labels.length > maxDataPoints) { equityChart.data.labels.shift(); equityChart.data.datasets[0].data.shift(); } equityChart.update('none'); }
        };

        // --- Backtest Result Display (MODIFIED to use backend data) ---
        const displayBacktestResults = (results) => {
            // Hide placeholder, show results area
            document.getElementById('backtest-results-content').classList.add('hidden');
            document.getElementById('backtest-results-details').classList.remove('hidden');

            const metrics = results.metrics;
            const equityCurve = results.equity_curve;

            // Update metrics display using data from backend
            const returnEl = document.getElementById('metric-return');
            returnEl.textContent = `${metrics.total_return_pct}%`;
            returnEl.className = `font-semibold text-lg ${metrics.total_return_pct >= 0 ? 'text-green-600' : 'text-red-600'}`;

            document.getElementById('metric-drawdown').textContent = `${metrics.max_drawdown_pct}%`; // Already red text
            document.getElementById('metric-sharpe').textContent = metrics.sharpe_ratio !== null ? metrics.sharpe_ratio.toFixed(2) : 'N/A';
            document.getElementById('metric-profit-factor').textContent = metrics.profit_factor !== null ? metrics.profit_factor.toFixed(2) : 'N/A';
            document.getElementById('metric-winrate').textContent = metrics.win_rate_pct !== null ? `${metrics.win_rate_pct.toFixed(1)}%` : 'N/A';
            document.getElementById('metric-trades').textContent = metrics.total_trades;

            // Update chart using equity curve data from backend
            const ctx = document.getElementById('backtestEquityCurveChart').getContext('2d');
            const labels = equityCurve.map(point => new Date(point.date).toLocaleDateString('en-CA')); // Format date for display
            const data = equityCurve.map(point => point.value);
            backtestEquityChart = createChart(ctx, labels, data, 'Backtest Equity');
        };

        // Function to display status/error messages
        const showBacktestStatus = (message, isError = false) => {
            const statusEl = document.getElementById('backtest-status');
            statusEl.textContent = message;
            statusEl.className = `mt-4 text-sm ${isError ? 'text-red-600' : 'text-blue-600'}`;
        };

        // --- Navigation (Identical to v2) ---
        const sections = document.querySelectorAll('.content-section');
        const sidebarLinks = document.querySelectorAll('.sidebar-link');
        const showSection = (sectionId) => { /* ... unchanged from v2 */
             sections.forEach(section => { section.id === sectionId ? section.classList.remove('hidden') : section.classList.add('hidden'); }); sidebarLinks.forEach(link => { link.dataset.section === sectionId ? link.classList.add('active') : link.classList.remove('active'); }); if (sectionId === 'dashboard') { if (!intervalId) { updateKPIs(); updateDashboardChart(); intervalId = setInterval(() => { updateKPIs(); updateDashboardChart(); }, 3000); } } else { if (intervalId) { clearInterval(intervalId); intervalId = null; } }
        };
        sidebarLinks.forEach(link => { link.addEventListener('click', (e) => { e.preventDefault(); showSection(link.dataset.section); }); });

        // --- Dynamic Parameter Loading for Backtest Form ---
        const loadBacktestParameters = (strategyTemplate) => {
            const paramsContainer = document.getElementById('backtest-parameters');
            paramsContainer.innerHTML = '<h4 class="text-sm font-medium text-gray-600 mb-2">Strategy Parameters</h4>'; // Clear previous, add title

            let paramsHtml = '<div class="space-y-2">';
            if (strategyTemplate === 'simple_ma_cross') {
                paramsHtml += `
                    <div><label for="bt-param-fast_ma_period" class="block text-xs font-medium text-gray-500 mb-0.5">Fast MA Period</label><input type="number" id="bt-param-fast_ma_period" name="fast_ma_period" value="20" class="w-full px-2 py-1 border border-gray-300 rounded-md shadow-sm text-xs focus:outline-none focus:ring-blue-500 focus:border-blue-500"></div>
                    <div><label for="bt-param-slow_ma_period" class="block text-xs font-medium text-gray-500 mb-0.5">Slow MA Period</label><input type="number" id="bt-param-slow_ma_period" name="slow_ma_period" value="50" class="w-full px-2 py-1 border border-gray-300 rounded-md shadow-sm text-xs focus:outline-none focus:ring-blue-500 focus:border-blue-500"></div>
                `;
            } else {
                paramsHtml += '<p class="text-xs text-gray-500">Parameters for this strategy are not defined in the frontend prototype.</p>';
            }
            paramsHtml += '</div>';
            paramsContainer.innerHTML += paramsHtml;
        };

        // --- Event Listeners ---

        // Backtest Strategy Template Change Handler
         document.getElementById('backtest-strategy-template').addEventListener('change', (e) => {
             loadBacktestParameters(e.target.value);
         });


        // Backtest Button Click Handler (MODIFIED)
        document.getElementById('run-backtest-btn').addEventListener('click', async () => {
            const btn = document.getElementById('run-backtest-btn');
            const statusEl = document.getElementById('backtest-status');
            const form = document.getElementById('backtest-form');

            // --- 1. Gather Form Data ---
            const formData = new FormData(form);
            const requestPayload = {
                strategy_template: formData.get('strategy_template'),
                symbol: formData.get('symbol').toUpperCase(),
                resolution: formData.get('resolution'),
                start_date: formData.get('start_date'),
                end_date: formData.get('end_date'),
                initial_capital: parseFloat(formData.get('initial_capital')),
                parameters: {}
            };

            // Gather parameters dynamically based on selected strategy
            const paramsContainer = document.getElementById('backtest-parameters');
            const paramInputs = paramsContainer.querySelectorAll('input');
            paramInputs.forEach(input => {
                // Convert parameter name if needed (e.g., bt-param-fast_ma_period -> fast_ma_period)
                const paramName = input.name; // Assumes name attribute matches backend Pydantic model
                if (paramName) {
                    requestPayload.parameters[paramName] = input.type === 'number' ? parseFloat(input.value) : input.value;
                }
            });


            // --- 2. Basic Validation ---
            if (!requestPayload.symbol || !requestPayload.start_date || !requestPayload.end_date || !requestPayload.strategy_template) {
                showBacktestStatus('Please fill in Symbol, Dates, and select a Strategy Template.', true);
                return;
            }
             if (new Date(requestPayload.start_date) >= new Date(requestPayload.end_date)) {
                 showBacktestStatus('End Date must be after Start Date.', true);
                return;
            }
            if (isNaN(requestPayload.initial_capital) || requestPayload.initial_capital <= 0) {
                 showBacktestStatus('Initial Capital must be a positive number.', true);
                return;
            }
             // Validate parameters (basic check for NaN)
             for (const key in requestPayload.parameters) {
                 if (isNaN(requestPayload.parameters[key])) {
                     showBacktestStatus(`Invalid value for parameter: ${key}. Please enter a number.`, true);
                     return;
                 }
             }


            // --- 3. Update UI State (Loading) ---
            btn.disabled = true;
            btn.innerHTML = `<div class="loader mr-2"></div> Running Backtest...`;
            showBacktestStatus('Sending request to backend...', false);
            document.getElementById('backtest-results-details').classList.add('hidden');
            document.getElementById('backtest-results-content').classList.remove('hidden');
            document.getElementById('backtest-results-content').textContent = 'Running backtest on server...';

            // --- 4. API Call to Backend /backtest Endpoint ---
            try {
                const apiUrl = `${BACKEND_URL}/backtest`;
                console.log(`Calling backend API: POST ${apiUrl}`);
                console.log("Request Payload:", JSON.stringify(requestPayload, null, 2));

                const response = await fetch(apiUrl, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Accept': 'application/json' // Explicitly accept JSON
                    },
                    body: JSON.stringify(requestPayload) // Send data as JSON
                });

                // Check if response is ok and content type is JSON
                if (!response.ok) {
                    let errorMsg = `Error running backtest: ${response.status} ${response.statusText}`;
                    try {
                        // Try to parse error detail from backend JSON response
                        const errorData = await response.json();
                        errorMsg = errorData.detail || errorMsg;
                    } catch (e) { /* Ignore if response is not JSON */ }
                    throw new Error(errorMsg);
                }

                 // Check content type before parsing
                const contentType = response.headers.get("content-type");
                if (!contentType || !contentType.includes("application/json")) {
                    throw new Error(`Received non-JSON response from server: ${contentType}`);
                }

                const results = await response.json(); // Parse the JSON response from the backend

                console.log("Received backtest results:", results);

                if (results.equity_curve && results.metrics) {
                    showBacktestStatus(results.message || 'Backtest completed successfully.', false);
                    // --- Display REAL Results ---
                    displayBacktestResults(results);
                } else {
                     // Should not happen if backend validation/response model is correct
                     throw new Error("Invalid response format received from backend.");
                }

            } catch (error) {
                console.error("Backtest API call failed:", error);
                showBacktestStatus(`Failed to run backtest: ${error.message}`, true);
                document.getElementById('backtest-results-content').textContent = 'Error running backtest.';
            } finally {
                // Reset button state
                btn.disabled = false;
                 btn.innerHTML = `
                    <i data-lucide="play" class="w-5 h-5"></i>
                    <span>Run Backtest</span>
                 `;
                 lucide.createIcons(); // Re-render icons if needed
            }
        });

         // Strategy Config Form Parameter Loading (Identical to v2)
        document.getElementById('strategy-template').addEventListener('change', (e) => {
            const selectedStrategy = e.target.value;
            const paramsContainer = document.getElementById('strategy-parameters');
            paramsContainer.innerHTML = '<h4 class="text-md font-semibold text-gray-600 mb-3">Parameters</h4>'; // Clear previous
            let paramsHtml = '<div class="grid grid-cols-1 md:grid-cols-3 gap-4">';
            if (selectedStrategy === 'simple_ma_cross') {
                 paramsHtml += `
                    <div><label for="param-fast_ma_period" class="block text-xs font-medium text-gray-600 mb-1">Fast MA Period</label><input type="number" id="param-fast_ma_period" name="fast_ma_period" value="20" class="w-full px-3 py-1.5 border border-gray-300 rounded-md shadow-sm text-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"></div>
                    <div><label for="param-slow_ma_period" class="block text-xs font-medium text-gray-600 mb-1">Slow MA Period</label><input type="number" id="param-slow_ma_period" name="slow_ma_period" value="50" class="w-full px-3 py-1.5 border border-gray-300 rounded-md shadow-sm text-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"></div>
                `;
            } else { paramsHtml += '<p class="text-sm text-gray-500 md:col-span-3">Parameters NYI</p>'; }
             paramsHtml += '</div>'; paramsContainer.innerHTML += paramsHtml;
        });

        // --- Initialization ---
        document.addEventListener('DOMContentLoaded', () => {
            lucide.createIcons();

            // Init Dashboard (uses mock data)
            const equityCtx = document.getElementById('equityCurveChart').getContext('2d');
            const initialEquity = generateMockEquityData();
            equityChart = createChart(equityCtx, initialEquity.labels, initialEquity.data, 'Portfolio Equity');
            updatePositionsTable();

            showSection('dashboard'); // Show dashboard initially

            // Load initial parameters for both strategy config and backtest forms
            document.getElementById('strategy-template').dispatchEvent(new Event('change'));
            loadBacktestParameters(document.getElementById('backtest-strategy-template').value);
        });

    </script>
</body>
</html>
