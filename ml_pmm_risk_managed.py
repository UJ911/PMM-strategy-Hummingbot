import pandas as pd
import numpy as np
import pickle
import os
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.order_candidate import OrderCandidate, OrderType, TradeType
from hummingbot.core.data_type.common import PositionAction
from hummingbot.core.utils.tracking_nonce import get_tracking_nonce
import xgboost as xgb
import ccxt
from decimal import Decimal
import time

class MlPmmRiskManaged(ScriptStrategyBase):
    markets = {"binance_paper_trade": ["BTC-USDT"]}
    
    def __init__(self, connectors):
        super().__init__(connectors)
        self.connectors = connectors
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "xgboost_top5_btc_model.pkl")
        self.model = pickle.load(open(model_path, "rb"))
        self.trading_pair = "BTC-USDT"
        self.exchange = "binance_paper_trade"
        
        # OPTIMIZED CONSTANTS - Key improvements for PMM strategy
        self.inventory_risk_aversion = 0.05  
        self.k = 0.15  #
        self.gamma = 0.01 
        
        # Risk management - optimized for 1-minute refresh
        self.cooldown_period = 180  
        self.stop_loss_pct = -0.02  
        
        # Order management - optimized for 1-minute strategy
        self.last_order_refresh = 0
        self.order_refresh_time = 60  
        self.order_amount = Decimal("0.002") 
        
        # Performance tracking
        self.last_loss_time = None
        self.last_filled_price = None
        self.last_filled_side = None
        self.last_filled_amount = None
        
        # Volatility-based spread adjustment
        self.min_spread = 0.0005  
        self.max_spread = 0.005   
        
        # Moving average for price smoothing
        self.price_history = []
        self.max_price_history = 10

    @property
    def candle_config(self):
        return {
            self.exchange: {
                self.trading_pair: ["1m"]
            }
        }

    def tick_interval(self) -> float:
        return 10.0  

    def fetch_binance_candles(self, symbol="BTC/USDT", timeframe="1m", limit=30):
        """Increased limit for better feature calculation"""
        exchange = ccxt.binance({'enableRateLimit': True})
        candles = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        return candles

    def did_fill_order(self, event):
        """Track filled orders and update inventory risk"""
        self.last_filled_price = event.price
        self.last_filled_side = event.trade_type
        self.last_filled_amount = event.amount
        self.logger().info(f"Order filled: {event.trade_type.name} {event.amount} at {event.price}")

    def calculate_inventory_skew(self):
        """Calculate inventory imbalance for asymmetric quotes"""
        try:
            usdt_balance = self.connectors[self.exchange].get_balance("USDT")
            btc_balance = self.connectors[self.exchange].get_balance("BTC")
            mid_price = self.connectors[self.exchange].get_mid_price(self.trading_pair)
            
            total_value = usdt_balance + btc_balance * mid_price
            target_btc_value = total_value * 0.5  # 50% target allocation
            current_btc_value = btc_balance * mid_price
            
            # Inventory skew: positive = too much BTC, negative = too much USDT
            inventory_skew = (current_btc_value - target_btc_value) / total_value
            return np.clip(inventory_skew, -0.5, 0.5)
        except:
            return 0.0

    def calculate_dynamic_spread(self, volatility, predicted_return):
        """Calculate spread based on volatility and prediction confidence"""
        # Base spread from volatility
        vol_spread = self.k * (1 + volatility)
        
        # Prediction uncertainty spread
        pred_spread = self.gamma * abs(predicted_return)
        
        # Combined spread
        total_spread = vol_spread + pred_spread
        
        # Clamp to reasonable bounds
        return np.clip(total_spread, self.min_spread, self.max_spread)

    def on_tick(self):
        self.logger().info("Tick started...")

        # Stop loss check with improved logic
        if self.should_execute_stop_loss():
            return

        # Fetch and validate candles
        candles = self.fetch_binance_candles(symbol="BTC/USDT", timeframe="1m", limit=30)
        if not candles or len(candles) < 20:
            self.logger().info("Waiting for enough external candle data...")
            return

        # Process candles
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = self.add_features(df)
        
        # Calculate volatility for spread adjustment
        volatility = df["close"].pct_change().rolling(10).std().iloc[-1]
        
        # Get ML prediction
        latest_features = self.get_model_features(df)
        predicted_return = self.model.predict(np.array([latest_features]))[0]
        
        # Price smoothing
        current_price = df["close"].iloc[-1]
        self.price_history.append(current_price)
        if len(self.price_history) > self.max_price_history:
            self.price_history.pop(0)
        
        smoothed_price = np.mean(self.price_history)
        
        # Calculate inventory skew
        inventory_skew = self.calculate_inventory_skew()
        
        # Adjusted reservation price with inventory consideration
        reservation_price = smoothed_price + (self.inventory_risk_aversion * predicted_return) - (inventory_skew * smoothed_price * 0.001)
        
        # Dynamic spread calculation
        spread = self.calculate_dynamic_spread(volatility, predicted_return)
        
        # Asymmetric quotes based on inventory
        bid_adjustment = inventory_skew * spread * 0.3  # Widen bid if too much BTC
        ask_adjustment = -inventory_skew * spread * 0.3  # Widen ask if too much USDT
        
        bid_price = reservation_price - (spread / 2) + bid_adjustment
        ask_price = reservation_price + (spread / 2) + ask_adjustment

        self.logger().info(f"Price: {current_price:.2f}, Smooth: {smoothed_price:.2f}, Volatility: {volatility:.4f}")
        self.logger().info(f"Predicted return: {predicted_return:.6f}, Inventory skew: {inventory_skew:.4f}")
        self.logger().info(f"Reservation: {reservation_price:.2f}, Spread: {spread:.4f}")

        # Risk management checks
        if self.should_skip_trading():
            return

        # Order management
        if self.should_refresh_orders():
            self.refresh_orders(bid_price, ask_price)

        # Portfolio logging
        self.log_portfolio_status()

    def should_execute_stop_loss(self):
        """Improved stop loss logic"""
        if self.last_filled_price is None or self.last_filled_side is None:
            return False
            
        current_price = self.connectors[self.exchange].get_mid_price(self.trading_pair)
        
        if self.last_filled_side.name == "BUY":
            pnl_pct = (current_price - self.last_filled_price) / self.last_filled_price
            if pnl_pct <= self.stop_loss_pct:
                self.execute_stop_loss("SELL", current_price)
                return True
        elif self.last_filled_side.name == "SELL":
            pnl_pct = (self.last_filled_price - current_price) / self.last_filled_price
            if pnl_pct <= self.stop_loss_pct:
                self.execute_stop_loss("BUY", current_price)
                return True
        
        return False

    def execute_stop_loss(self, side, price):
        """Execute stop loss order"""
        self.logger().warning(f"Stop loss triggered! {side} {self.last_filled_amount} at {price}")
        
        if side == "SELL":
            self.sell(self.exchange, self.trading_pair, Decimal(str(self.last_filled_amount)), OrderType.MARKET)
        else:
            self.buy(self.exchange, self.trading_pair, Decimal(str(self.last_filled_amount)), OrderType.MARKET)
        
        self.last_loss_time = self.current_timestamp
        self.reset_fill_tracking()

    def should_skip_trading(self):
        """Check if trading should be skipped due to risk management"""
        if self.last_loss_time and (self.current_timestamp - self.last_loss_time) < self.cooldown_period:
            self.logger().info("In cooldown period due to recent stop-loss.")
            return True
        return False

    def should_refresh_orders(self):
        """Check if orders should be refreshed"""
        return self.current_timestamp - self.last_order_refresh >= self.order_refresh_time

    def refresh_orders(self, bid_price, ask_price):
        """Cancel existing orders and place new ones"""
        self.cancel_all_orders()
        self.logger().info("All previous orders cancelled.")

        # Place new orders with optimized amounts
        bid_price_decimal = Decimal(str(round(bid_price, 2)))
        ask_price_decimal = Decimal(str(round(ask_price, 2)))

        self.buy(self.exchange, self.trading_pair, self.order_amount, OrderType.LIMIT, price=bid_price_decimal)
        self.sell(self.exchange, self.trading_pair, self.order_amount, OrderType.LIMIT, price=ask_price_decimal)
        
        self.logger().info(f"Orders placed: Buy at {bid_price_decimal}, Sell at {ask_price_decimal}")
        self.last_order_refresh = self.current_timestamp

    def reset_fill_tracking(self):
        """Reset fill tracking variables"""
        self.last_filled_price = None
        self.last_filled_side = None
        self.last_filled_amount = None

    def log_portfolio_status(self):
        """Log current portfolio status"""
        try:
            usdt_balance = self.connectors[self.exchange].get_balance("USDT")
            btc_balance = self.connectors[self.exchange].get_balance("BTC")
            btc_price = self.connectors[self.exchange].get_mid_price("BTC-USDT")
            portfolio_value = usdt_balance + btc_balance * btc_price
            
            self.logger().info(f"Portfolio: USDT={usdt_balance:.2f}, BTC={btc_balance:.6f}, Value={portfolio_value:.2f}")
        except Exception as e:
            self.logger().error(f"Error logging portfolio: {e}")

    def ready_to_trade(self):
        """Check if strategy is ready to trade"""
        ready = self.connector_ready(self.exchange) and self.has_active_market(self.exchange, self.trading_pair)
        self.logger().info(f"Ready to trade: {ready}")
        return ready

    def add_features(self, df):
        """Enhanced feature engineering"""
        # EMAs with different periods
        df["ema_5"] = df["close"].ewm(span=5, adjust=False).mean()
        df["ema_15"] = df["close"].ewm(span=15, adjust=False).mean()
        df["ema_30"] = df["close"].ewm(span=30, adjust=False).mean()
        
        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi_14"] = 100 - (100 / (1 + rs))
        
        # Price-based features
        df["mid_price"] = (df["high"] + df["low"]) / 2
        df["spread"] = df["high"] - df["low"]
        df["volume_ma"] = df["volume"].rolling(window=10).mean()
        
        # Volatility
        df["volatility"] = df["close"].pct_change().rolling(window=10).std()
        
        return df

    def get_model_features(self, df):
        """Get features for ML model prediction"""
        latest = df.iloc[-1]
        return [
            latest["mid_price"],
            latest["ema_15"],
            latest["ema_5"],
            latest["rsi_14"],
            latest["spread"],
        ]

    def cancel_all_orders(self):
        """Fixed cancel_all_orders method"""
        try:
            for order in self.get_active_orders(connector_name=self.exchange):
                # Fixed: Pass the trading_pair parameter
                market_pair = self._market_trading_pair_tuple(self.exchange, self.trading_pair)
                if market_pair is not None:
                    self.cancel_order(market_pair, order.client_order_id)
        except Exception as e:
            self.logger().error(f"Error cancelling orders: {e}")
