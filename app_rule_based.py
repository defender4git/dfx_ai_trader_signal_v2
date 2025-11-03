"""
MT5 Rule-Based Trading Expert Advisor (FREE VERSION)
Professional technical analysis without AI API costs
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class TradingSignal:
    """Trading signal with complete trade parameters"""
    symbol: str
    signal_type: str  # 'LONG', 'SHORT', 'NEUTRAL'
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    position_size: float
    confidence: str
    reasoning: str
    timestamp: datetime
    indicators: Dict
    trend: str
    entry_strategy: str


class TechnicalAnalyzer:
    """Calculate technical indicators"""
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        
        return atr
    
    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        close = df['close']
        delta = close.diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1]
    
    @staticmethod
    def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
        """Calculate MACD, Signal line, and Histogram"""
        close = df['close']
        
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        
        return macd.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]
    
    @staticmethod
    def calculate_cci(df: pd.DataFrame, period: int = 20) -> float:
        """Calculate Commodity Channel Index"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean()
        )
        
        cci = (typical_price - sma) / (0.015 * mean_deviation)
        
        return cci.iloc[-1]
    
    @staticmethod
    def calculate_stochastic(df: pd.DataFrame, period: int = 14, smooth: int = 3) -> Tuple[float, float]:
        """Calculate Stochastic Oscillator %K and %D"""
        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()
        
        k = 100 * (df['close'] - low_min) / (high_max - low_min)
        k_smooth = k.rolling(window=smooth).mean()
        d = k_smooth.rolling(window=smooth).mean()
        
        return k_smooth.iloc[-1], d.iloc[-1]
    
    @staticmethod
    def calculate_ema(df: pd.DataFrame, period: int = 20) -> float:
        """Calculate Exponential Moving Average"""
        return df['close'].ewm(span=period, adjust=False).mean().iloc[-1]


class RuleBasedSignalGenerator:
    """Generate trading signals using rule-based logic"""
    
    def __init__(self):
        self.signal_strength = 0
        self.bullish_signals = []
        self.bearish_signals = []
        self.neutral_signals = []
    
    def analyze_market(self, indicators: Dict) -> Dict:
        """
        Professional rule-based market analysis
        Returns signal with reasoning
        """
        self.signal_strength = 0
        self.bullish_signals = []
        self.bearish_signals = []
        self.neutral_signals = []
        
        # Extract indicators
        rsi = indicators['rsi']
        macd = indicators['macd']
        macd_signal = indicators['macd_signal']
        macd_histogram = indicators['macd_histogram']
        cci = indicators['cci']
        stoch_k = indicators['stochastic_k']
        stoch_d = indicators['stochastic_d']
        atr = indicators['atr']
        ema_20 = indicators['ema_20']
        ema_50 = indicators['ema_50']
        current_price = indicators['current_price']
        
        # === TREND ANALYSIS ===
        
        # 1. MACD Analysis (Weight: 3 points)
        if macd > macd_signal and macd_histogram > 0:
            if macd > 0:
                self.signal_strength += 3
                self.bullish_signals.append("MACD bullish crossover above zero line")
            else:
                self.signal_strength += 2
                self.bullish_signals.append("MACD bullish crossover")
        elif macd < macd_signal and macd_histogram < 0:
            if macd < 0:
                self.signal_strength -= 3
                self.bearish_signals.append("MACD bearish crossover below zero line")
            else:
                self.signal_strength -= 2
                self.bearish_signals.append("MACD bearish crossover")
        
        # 2. RSI Analysis (Weight: 2 points)
        if rsi < 30:
            self.signal_strength += 2
            self.bullish_signals.append(f"RSI oversold at {rsi:.1f}")
        elif rsi > 70:
            self.signal_strength -= 2
            self.bearish_signals.append(f"RSI overbought at {rsi:.1f}")
        elif 45 <= rsi <= 55:
            self.neutral_signals.append(f"RSI neutral at {rsi:.1f}")
        elif rsi > 50:
            self.signal_strength += 1
            self.bullish_signals.append(f"RSI bullish at {rsi:.1f}")
        else:
            self.signal_strength -= 1
            self.bearish_signals.append(f"RSI bearish at {rsi:.1f}")
        
        # 3. CCI Analysis (Weight: 2 points)
        if cci > 100:
            self.signal_strength += 2
            self.bullish_signals.append(f"CCI strong bullish at {cci:.1f}")
        elif cci < -100:
            self.signal_strength -= 2
            self.bearish_signals.append(f"CCI strong bearish at {cci:.1f}")
        elif cci > 0:
            self.signal_strength += 1
            self.bullish_signals.append(f"CCI bullish at {cci:.1f}")
        elif cci < 0:
            self.signal_strength -= 1
            self.bearish_signals.append(f"CCI bearish at {cci:.1f}")
        
        # 4. Stochastic Analysis (Weight: 2 points)
        if stoch_k < 20 and stoch_k > stoch_d:
            self.signal_strength += 2
            self.bullish_signals.append("Stochastic oversold with bullish crossover")
        elif stoch_k > 80 and stoch_k < stoch_d:
            self.signal_strength -= 2
            self.bearish_signals.append("Stochastic overbought with bearish crossover")
        elif stoch_k > stoch_d:
            self.signal_strength += 1
            self.bullish_signals.append("Stochastic bullish")
        else:
            self.signal_strength -= 1
            self.bearish_signals.append("Stochastic bearish")
        
        # 5. EMA Trend Analysis (Weight: 2 points)
        if current_price > ema_20 > ema_50:
            self.signal_strength += 2
            self.bullish_signals.append("Price above both EMAs - strong uptrend")
        elif current_price < ema_20 < ema_50:
            self.signal_strength -= 2
            self.bearish_signals.append("Price below both EMAs - strong downtrend")
        elif current_price > ema_20:
            self.signal_strength += 1
            self.bullish_signals.append("Price above EMA20")
        elif current_price < ema_20:
            self.signal_strength -= 1
            self.bearish_signals.append("Price below EMA20")
        
        # === DETERMINE SIGNAL ===
        
        if self.signal_strength >= 5:
            signal = "LONG"
            confidence = "HIGH" if self.signal_strength >= 7 else "MEDIUM"
            trend = "BULLISH"
            entry_strategy = "IMMEDIATE" if self.signal_strength >= 7 else "WAIT_PULLBACK"
        elif self.signal_strength <= -5:
            signal = "SHORT"
            confidence = "HIGH" if self.signal_strength <= -7 else "MEDIUM"
            trend = "BEARISH"
            entry_strategy = "IMMEDIATE" if self.signal_strength <= -7 else "WAIT_PULLBACK"
        elif -2 <= self.signal_strength <= 2:
            signal = "NEUTRAL"
            confidence = "LOW"
            trend = "CONSOLIDATION"
            entry_strategy = "WAIT_BREAKOUT"
        else:
            signal = "NEUTRAL"
            confidence = "LOW"
            trend = "MIXED"
            entry_strategy = "WAIT_CONFIRMATION"
        
        # === VOLATILITY-BASED ADJUSTMENTS ===
        
        volatility_multiplier = 1.0
        position_adjustment = 1.0
        
        if atr > 15:
            volatility_level = "Very High"
            position_adjustment = 0.5
            sl_multiplier = 1.8
        elif atr > 10:
            volatility_level = "High"
            position_adjustment = 0.7
            sl_multiplier = 1.5
        elif atr > 5:
            volatility_level = "Medium"
            position_adjustment = 0.9
            sl_multiplier = 1.3
        else:
            volatility_level = "Low"
            position_adjustment = 1.0
            sl_multiplier = 1.2
        
        # === BUILD REASONING ===
        
        reasoning_parts = []
        reasoning_parts.append(f"Signal Strength: {self.signal_strength}/10")
        reasoning_parts.append(f"Trend: {trend}")
        reasoning_parts.append(f"Volatility: {volatility_level} (ATR: {atr:.2f})")
        
        if self.bullish_signals:
            reasoning_parts.append(f"Bullish Factors: {', '.join(self.bullish_signals[:3])}")
        if self.bearish_signals:
            reasoning_parts.append(f"Bearish Factors: {', '.join(self.bearish_signals[:3])}")
        if self.neutral_signals:
            reasoning_parts.append(f"Neutral Factors: {', '.join(self.neutral_signals)}")
        
        reasoning = " | ".join(reasoning_parts)
        
        return {
            "signal": signal,
            "confidence": confidence,
            "trend": trend,
            "entry_strategy": entry_strategy,
            "stop_loss_atr_multiplier": sl_multiplier,
            "take_profit_1_atr_multiplier": 1.5,
            "take_profit_2_atr_multiplier": 2.5,
            "take_profit_3_atr_multiplier": 3.5,
            "position_size_adjustment": position_adjustment,
            "reasoning": reasoning,
            "signal_strength": self.signal_strength,
            "bullish_count": len(self.bullish_signals),
            "bearish_count": len(self.bearish_signals)
        }


class MT5TradingEA:
    """Main Expert Advisor class - FREE VERSION"""
    
    def __init__(self, symbol: str, timeframe: int, lookback: int, 
                 account_risk_percent: float = 1.0):
        """
        Initialize MT5 Trading EA (Free Version)
        
        Args:
            symbol: Trading symbol (e.g., 'XAUUSD')
            timeframe: MT5 timeframe (e.g., mt5.TIMEFRAME_H1)
            lookback: Number of candles to analyze
            account_risk_percent: Risk per trade as % of account
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.lookback = lookback
        self.account_risk_percent = account_risk_percent
        
        self.analyzer = TechnicalAnalyzer()
        self.signal_generator = RuleBasedSignalGenerator()
        
        self.current_signal: Optional[TradingSignal] = None
        
    def connect_mt5(self) -> bool:
        """Connect to MT5 terminal"""
        if not mt5.initialize():
            print(f"MT5 initialization failed: {mt5.last_error()}")
            return False
        
        print(f"MT5 Connected: {mt5.version()}")
        account_info = mt5.account_info()
        if account_info:
            print(f"Account: {account_info.login}")
            print(f"Balance: ${account_info.balance:.2f}")
        return True
    
    def disconnect_mt5(self):
        """Disconnect from MT5"""
        mt5.shutdown()
        print("MT5 Disconnected")
    
    def get_market_data(self) -> Optional[pd.DataFrame]:
        """Fetch market data from MT5"""
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, self.lookback)
        
        if rates is None or len(rates) == 0:
            print(f"Failed to get data for {self.symbol}")
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        return df
    
    def calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate all technical indicators"""
        indicators = {
            'atr': self.analyzer.calculate_atr(df),
            'rsi': self.analyzer.calculate_rsi(df),
            'macd': 0.0,
            'macd_signal': 0.0,
            'macd_histogram': 0.0,
            'cci': self.analyzer.calculate_cci(df),
            'stochastic_k': 0.0,
            'stochastic_d': 0.0,
            'ema_20': self.analyzer.calculate_ema(df, 20),
            'ema_50': self.analyzer.calculate_ema(df, 50),
            'current_price': df['close'].iloc[-1]
        }
        
        macd, signal, histogram = self.analyzer.calculate_macd(df)
        indicators['macd'] = macd
        indicators['macd_signal'] = signal
        indicators['macd_histogram'] = histogram
        
        stoch_k, stoch_d = self.analyzer.calculate_stochastic(df)
        indicators['stochastic_k'] = stoch_k
        indicators['stochastic_d'] = stoch_d
        
        # Volatility classification
        atr = indicators['atr']
        if atr > 15:
            indicators['volatility_level'] = "Very High Volatility"
        elif atr > 10:
            indicators['volatility_level'] = "High Volatility"
        elif atr > 5:
            indicators['volatility_level'] = "Medium Volatility"
        else:
            indicators['volatility_level'] = "Low Volatility"
        
        return indicators
    
    def generate_signal(self, indicators: Dict) -> TradingSignal:
        """Generate trading signal using rule-based analysis"""
        
        # Get timeframe name
        timeframe_map = {
            mt5.TIMEFRAME_M1: "M1", mt5.TIMEFRAME_M5: "M5", 
            mt5.TIMEFRAME_M15: "M15", mt5.TIMEFRAME_M30: "M30",
            mt5.TIMEFRAME_H1: "H1", mt5.TIMEFRAME_H4: "H4",
            mt5.TIMEFRAME_D1: "D1"
        }
        tf_name = timeframe_map.get(self.timeframe, "H1")
        
        # Get rule-based analysis
        analysis = self.signal_generator.analyze_market(indicators)
        
        # Calculate trade parameters
        current_price = indicators['current_price']
        atr = indicators['atr']
        
        signal_type = analysis['signal']
        
        # Calculate position size based on risk
        account_info = mt5.account_info()
        account_balance = account_info.balance
        risk_amount = account_balance * (self.account_risk_percent / 100)
        
        # Adjust for volatility
        position_adjustment = analysis['position_size_adjustment']
        
        if signal_type == "LONG":
            stop_loss = current_price - (atr * analysis['stop_loss_atr_multiplier'])
            tp1 = current_price + (atr * analysis['take_profit_1_atr_multiplier'])
            tp2 = current_price + (atr * analysis['take_profit_2_atr_multiplier'])
            tp3 = current_price + (atr * analysis['take_profit_3_atr_multiplier'])
            
        elif signal_type == "SHORT":
            stop_loss = current_price + (atr * analysis['stop_loss_atr_multiplier'])
            tp1 = current_price - (atr * analysis['take_profit_1_atr_multiplier'])
            tp2 = current_price - (atr * analysis['take_profit_2_atr_multiplier'])
            tp3 = current_price - (atr * analysis['take_profit_3_atr_multiplier'])
            
        else:  # NEUTRAL
            stop_loss = current_price
            tp1 = tp2 = tp3 = current_price
        
        # Calculate lot size
        symbol_info = mt5.symbol_info(self.symbol)
        if not symbol_info:
            print(f"Symbol {self.symbol} not found")
            position_size = 0.01
        else:
            point = symbol_info.point
            stop_distance = abs(current_price - stop_loss)
            
            if stop_distance > 0:
                # Risk per pip
                lot_size = risk_amount / (stop_distance / point * symbol_info.trade_contract_size)
                lot_size *= position_adjustment
                
                # Round to allowed lot size
                lot_size = max(symbol_info.volume_min, 
                              min(lot_size, symbol_info.volume_max))
                lot_size = round(lot_size / symbol_info.volume_step) * symbol_info.volume_step
                position_size = lot_size
            else:
                position_size = symbol_info.volume_min
        
        # Create signal
        signal = TradingSignal(
            symbol=self.symbol,
            signal_type=signal_type,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            take_profit_3=tp3,
            position_size=position_size,
            confidence=analysis['confidence'],
            reasoning=analysis['reasoning'],
            timestamp=datetime.now(),
            indicators=indicators,
            trend=analysis['trend'],
            entry_strategy=analysis['entry_strategy']
        )
        
        return signal
    
    def print_signal(self, signal: TradingSignal):
        """Display trading signal in professional format"""
        print("\n" + "="*80)
        print(f"📊 PROFESSIONAL TRADING SIGNAL - {signal.symbol}")
        print("="*80)
        print(f"Timestamp: {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\n📈 TECHNICAL INDICATORS:")
        print(f"  ATR: {signal.indicators['atr']:.6f} ({signal.indicators['volatility_level']})")
        print(f"  RSI: {signal.indicators['rsi']:.2f}")
        print(f"  MACD: {signal.indicators['macd']:.6f} | Signal: {signal.indicators['macd_signal']:.6f} | Hist: {signal.indicators['macd_histogram']:.6f}")
        print(f"  CCI: {signal.indicators['cci']:.2f}")
        print(f"  Stochastic: K={signal.indicators['stochastic_k']:.2f} | D={signal.indicators['stochastic_d']:.2f}")
        print(f"  EMA20: {signal.indicators['ema_20']:.5f} | EMA50: {signal.indicators['ema_50']:.5f}")
        
        print(f"\n🎯 TRADING SIGNAL: {signal.signal_type}")
        print(f"  Confidence: {signal.confidence}")
        print(f"  Trend: {signal.trend}")
        print(f"  Entry Strategy: {signal.entry_strategy}")
        print(f"  Entry Price: {signal.entry_price:.5f}")
        print(f"  Position Size: {signal.position_size:.2f} lots")
        
        if signal.signal_type != "NEUTRAL":
            risk = abs(signal.entry_price - signal.stop_loss)
            print(f"\n🛡️ RISK MANAGEMENT:")
            print(f"  Stop Loss: {signal.stop_loss:.5f} (Risk: {risk:.5f} pts)")
            print(f"  TP1: {signal.take_profit_1:.5f} (R:R {abs(signal.take_profit_1 - signal.entry_price)/risk:.2f})")
            print(f"  TP2: {signal.take_profit_2:.5f} (R:R {abs(signal.take_profit_2 - signal.entry_price)/risk:.2f})")
            print(f"  TP3: {signal.take_profit_3:.5f} (R:R {abs(signal.take_profit_3 - signal.entry_price)/risk:.2f})")
        
        print(f"\n💡 ANALYSIS:")
        print(f"  {signal.reasoning}")
        print("="*80 + "\n")
    
    def execute_trade(self, signal: TradingSignal, auto_execute: bool = False) -> bool:
        """Execute trade on MT5"""
        if signal.signal_type == "NEUTRAL":
            print("⚠️  No trade signal - Market is NEUTRAL")
            return False
        
        if not auto_execute:
            response = input(f"\n🔔 Execute {signal.signal_type} trade? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                print("❌ Trade execution cancelled")
                return False
        
        # Prepare trade request
        order_type = mt5.ORDER_TYPE_BUY if signal.signal_type == "LONG" else mt5.ORDER_TYPE_SELL
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": signal.symbol,
            "volume": signal.position_size,
            "type": order_type,
            "price": signal.entry_price,
            "sl": signal.stop_loss,
            "tp": signal.take_profit_1,
            "deviation": 20,
            "magic": 234000,
            "comment": f"RuleBased_{signal.confidence}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send order
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"❌ Trade execution failed: {result.comment}")
            return False
        
        print(f"✅ Trade executed successfully!")
        print(f"   Order: {result.order}")
        print(f"   Volume: {result.volume}")
        print(f"   Price: {result.price}")
        
        return True
    
    def run(self, interval_seconds: int = 300, auto_execute: bool = False):
        """Run EA continuously"""
        if not self.connect_mt5():
            return
        
        print(f"\n🚀 FREE Rule-Based Trading EA Started")
        print(f"   Symbol: {self.symbol}")
        print(f"   Timeframe: {self.timeframe}")
        print(f"   Lookback: {self.lookback} candles")
        print(f"   Risk per trade: {self.account_risk_percent}%")
        print(f"   Auto-execute: {auto_execute}")
        print(f"   Analysis interval: {interval_seconds}s")
        print(f"   💰 100% FREE - No API costs!\n")
        
        try:
            while True:
                # Get market data
                df = self.get_market_data()
                if df is None:
                    time.sleep(interval_seconds)
                    continue
                
                # Calculate indicators
                indicators = self.calculate_indicators(df)
                
                # Generate signal
                signal = self.generate_signal(indicators)
                self.current_signal = signal
                
                # Display signal
                self.print_signal(signal)
                
                # Execute if configured
                if signal.signal_type != "NEUTRAL" and auto_execute:
                    self.execute_trade(signal, auto_execute=True)
                
                # Wait for next cycle
                print(f"⏳ Next analysis in {interval_seconds} seconds...")
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            print("\n\n⛔ EA stopped by user")
        finally:
            self.disconnect_mt5()


def main():
    """Main execution function"""
    
    # Configuration
    SYMBOL = "Volatility 10 Index"
    TIMEFRAME = mt5.TIMEFRAME_H1
    LOOKBACK = 150
    RISK_PERCENT = 1.0  # Risk 1% per trade
    
    # Create EA instance (NO API KEY NEEDED!)
    ea = MT5TradingEA(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        lookback=LOOKBACK,
        account_risk_percent=RISK_PERCENT
    )
    
    # Run EA
    # Set auto_execute=True to automatically execute trades
    # Set auto_execute=False to manually confirm each trade
    ea.run(interval_seconds=300, auto_execute=False)


if __name__ == "__main__":
    main()