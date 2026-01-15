from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, ConversationHandler, CallbackQueryHandler
from telegram.request import HTTPXRequest
import os
import logging
from dotenv import load_dotenv
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


# Load environment variables from .env file
load_dotenv()

# Set up logging (optional but recommended)
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

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
    timeframe: str
    timestamp: datetime
    indicators: Dict
    trend: str
    entry_strategy: str

# States for conversation
SELECT_SYMBOL, SELECT_TIMEFRAME = range(2)

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
            timeframe=tf_name,
            indicators=indicators,
            trend=analysis['trend'],
            entry_strategy=analysis['entry_strategy']
        )

        return signal

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    # Create quick access buttons for popular pairs
    keyboard = [
        [
            InlineKeyboardButton("📊 Generate Signal", callback_data="menu_signal")
        ],
        [
            InlineKeyboardButton("🥇 GOLD (H1)", callback_data="quick_gold_h1"),
            InlineKeyboardButton("🥇 GOLD (M15)", callback_data="quick_gold_m15")
        ],
        [
            InlineKeyboardButton("📈 US30 (H1)", callback_data="quick_us30_h1"),
            InlineKeyboardButton("📈 US30 (M15)", callback_data="quick_us30_m15")
        ],
        [
            InlineKeyboardButton("💶 EURUSD (H1)", callback_data="quick_eurusd_h1"),
            InlineKeyboardButton("💶 EURUSD (M15)", callback_data="quick_eurusd_m15")
        ],
        [
            InlineKeyboardButton("ℹ️ Help", callback_data="menu_help")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        '👋 *Welcome to Advanced Trading Signal Bot!*\n\n'
        '🚀 Quick access to popular pairs or generate custom signals:\n\n'
        '• Click buttons below for instant analysis\n'
        '• Or use /signal for custom symbol/timeframe\n'
        '• Use /help for more information',
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    help_text = """
📖 *Trading Signal Bot - Help Guide*

*Quick Access Buttons:*
Use the menu buttons for instant signal generation on popular pairs:
• 🥇 GOLD (XAUUSD variants)
• 📈 US30 (US30, US30Cash, etc.)
• 💶 EURUSD

*Commands:*
/start - Show main menu with quick access buttons
/signal - Interactive signal generation (choose any symbol)
/help - Show this help message
/cancel - Cancel current operation

*How It Works:*
1. Click a quick button OR use /signal
2. Bot analyzes market using multiple indicators
3. Receive actionable signals with entry, SL, and TPs

*Signal Confidence Levels:*
• HIGH - Strong confirmation from multiple indicators
• MEDIUM - Good setup with some confirmation
• LOW - Wait for better opportunity

⚠️ *Important:* Ensure MT5 is running and symbols are in Market Watch!

💡 *Tip:* Signals are valid for 5 minutes after generation.
    """.strip()
    
    if update.message:
        await update.message.reply_text(help_text, parse_mode='Markdown')
    elif update.callback_query:
        await update.callback_query.message.reply_text(help_text, parse_mode='Markdown')
        await update.callback_query.answer()

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Echo the user message."""
    await update.message.reply_text(update.message.text)

async def start_signal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Start the signal generation conversation."""
    
    # Send initial message first
    await update.message.reply_text("🔄 Connecting to server and fetching data...")
    
    # Run MT5 operations in executor to avoid blocking
    import asyncio
    loop = asyncio.get_event_loop()
    
    def get_mt5_symbols():
        # Connect to MT5 to get symbols
        if not mt5.initialize():
            return None
        
        # Get only symbols visible in Market Watch
        symbols = mt5.symbols_get()
        if symbols is None or len(symbols) == 0:
            mt5.shutdown()
            return []
        
        # Filter only symbols that are visible in Market Watch
        market_watch_symbols = [s.name for s in symbols if s.visible]
        
        mt5.shutdown()
        return market_watch_symbols
    
    # Run in executor to avoid blocking
    market_watch_symbols = await loop.run_in_executor(None, get_mt5_symbols)
    
    if market_watch_symbols is None:
        await update.message.reply_text("❌ Failed to connect to MT5. Make sure MT5 is running.")
        return ConversationHandler.END
    
    if len(market_watch_symbols) == 0:
        await update.message.reply_text("❌ No symbols in Market Watch. Please add symbols to Market Watch in MT5.")
        return ConversationHandler.END

    context.user_data['symbols'] = market_watch_symbols

    symbol_list = "\n".join(f"• {s}" for s in market_watch_symbols)
    await update.message.reply_text(f"✅ Available symbols (from Market Watch):\n{symbol_list}\n\n📝 Reply with the symbol name:")

    return SELECT_SYMBOL

async def select_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle symbol selection."""
    symbol = update.message.text.strip()  # Normalize symbol input
    if symbol not in [s for s in context.user_data['symbols']]: #removed .upper()
        await update.message.reply_text("Invalid symbol. Please select from the list.")
        return SELECT_SYMBOL

    context.user_data['selected_symbol'] = symbol

    timeframes = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]
    timeframe_list = "\n".join(f"• {tf}" for tf in timeframes)
    await update.message.reply_text(f"Selected symbol: {symbol}\n\nAvailable timeframes:\n{timeframe_list}\n\nReply with the timeframe:")

    return SELECT_TIMEFRAME

async def select_timeframe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle timeframe selection and generate signal."""
    timeframe_str = update.message.text.strip().upper()
    valid_timeframes = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]
    if timeframe_str not in valid_timeframes:
        await update.message.reply_text("❌ Invalid timeframe. Please select from the list.")
        return SELECT_TIMEFRAME

    symbol = context.user_data['selected_symbol']
    timeframe_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1
    }
    timeframe = timeframe_map[timeframe_str]

    # Send processing message
    await update.message.reply_text(f"🔄 Analyzing {symbol} on {timeframe_str} timeframe...")

    LOOKBACK = 150
    RISK_PERCENT = 1.0

    # Run MT5 operations in executor
    import asyncio
    loop = asyncio.get_event_loop()
    
    def generate_trading_signal():
        # Create EA instance
        ea = MT5TradingEA(
            symbol=symbol,
            timeframe=timeframe,
            lookback=LOOKBACK,
            account_risk_percent=RISK_PERCENT
        )

        if not ea.connect_mt5():
            return None, "Failed to connect to MT5."

        try:
            # Get market data
            df = ea.get_market_data()
            if df is None:
                ea.disconnect_mt5()
                return None, "Failed to get market data."

            # Calculate indicators
            indicators = ea.calculate_indicators(df)

            # Generate signal
            signal = ea.generate_signal(indicators)
            
            ea.disconnect_mt5()
            return signal, None

        except Exception as e:
            ea.disconnect_mt5()
            return None, f"Error: {str(e)}"
    
    # Run signal generation in executor
    signal, error = await loop.run_in_executor(None, generate_trading_signal)
    
    if error:
        await update.message.reply_text(f"❌ {error}")
        return ConversationHandler.END
    
    if signal is None:
        await update.message.reply_text("❌ Failed to generate signal.")
        return ConversationHandler.END

    # Check confidence
    if signal.confidence.upper() in ["HIGH", "MEDIUM"]:
        # Format message
        direction_emoji = "🟢" if signal.signal_type == "LONG" else "🔴"
        message = f"""
🚨 *TRADING SIGNAL ALERT* 🚨

{direction_emoji} *{signal.signal_type}* on {signal.symbol}
📊 *Confidence:* {signal.confidence}
📈 *Timeframe:* {signal.timeframe}

💰 *Entry Price:* {signal.entry_price:.5f}
📏 *Position Size:* {signal.position_size:.2f} lots

🛡️ *Risk Management:*
• Stop Loss: {signal.stop_loss:.5f}
• Take Profit 1: {signal.take_profit_1:.5f}
• Take Profit 2: {signal.take_profit_2:.5f} (Move SL to BE!)
• Take Profit 3: {signal.take_profit_3:.5f} (Manual closure)

💡 *Reasoning:* {signal.reasoning}

⏰ *Generated:* {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
⏱️ Valid for 5 minutes for entry.
        """.strip()

        await update.message.reply_text(message, parse_mode='Markdown')
    else:
        await update.message.reply_text(f"📊 Signal generated but confidence is {signal.confidence}. No actionable signal at this time.")

    return ConversationHandler.END

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancel the conversation."""
    await update.message.reply_text("Signal generation cancelled.")
    return ConversationHandler.END

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle button callbacks for quick signal generation."""
    query = update.callback_query
    await query.answer()
    
    if query.data == "menu_help":
        await help_command(update, context)
        return
    
    if query.data == "menu_signal":
        # Redirect to interactive signal generation
        await query.message.reply_text("Please use /signal command to start interactive signal generation.")
        return
    
    # Parse quick signal request
    if query.data.startswith("quick_"):
        parts = query.data.replace("quick_", "").split("_")
        if len(parts) != 2:
            await query.message.reply_text("❌ Invalid button data.")
            return
        
        pair_key = parts[0]
        timeframe_str = parts[1].upper()
        
        # Map pair keys to possible broker symbols
        symbol_variants = {
            "gold": ["XAUUSD", "XAUUSD.a", "XAUUSD.", "GOLD", "XAUUSDm"],
            "us30": ["US30", "US30Cash", "US30.cash", "US30USD", "US30.", "DJ30", "USTEC"],
            "eurusd": ["EURUSD", "EURUSD.a", "EURUSD.", "EURUSDm"]
        }
        
        if pair_key not in symbol_variants:
            await query.message.reply_text("❌ Unknown symbol key.")
            return
        
        # Send processing message
        await query.message.reply_text(f"🔄 Analyzing {pair_key} on {timeframe_str} timeframe...")
        
        # Get available symbols from MT5
        import asyncio
        loop = asyncio.get_event_loop()
        
        def get_matching_symbol():
            if not mt5.initialize():
                return None, "Failed to connect to MT5"
            
            symbols = mt5.symbols_get()
            if symbols is None:
                mt5.shutdown()
                return None, "Failed to get symbols"
            
            market_watch_symbols = [s.name for s in symbols if s.visible]
            
            # Find matching symbol
            for variant in symbol_variants[pair_key]:
                if variant in market_watch_symbols:
                    mt5.shutdown()
                    return variant, None
            
            mt5.shutdown()
            return None, f"No {pair_key} symbol found in Market Watch. Available symbols: {', '.join(market_watch_symbols[:5])}"
        
        symbol, error = await loop.run_in_executor(None, get_matching_symbol)
        
        if error:
            await query.message.reply_text(f"❌ {error}")
            return
        
        # Map timeframe
        timeframe_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1
        }
        
        if timeframe_str not in timeframe_map:
            await query.message.reply_text("❌ Invalid timeframe.")
            return
        
        timeframe = timeframe_map[timeframe_str]
        
        # Generate signal
        LOOKBACK = 150
        RISK_PERCENT = 1.0
        
        def generate_trading_signal():
            ea = MT5TradingEA(
                symbol=symbol,
                timeframe=timeframe,
                lookback=LOOKBACK,
                account_risk_percent=RISK_PERCENT
            )
            
            if not ea.connect_mt5():
                return None, "Failed to connect to MT5."
            
            try:
                df = ea.get_market_data()
                if df is None:
                    ea.disconnect_mt5()
                    return None, "Failed to get market data."
                
                indicators = ea.calculate_indicators(df)
                signal = ea.generate_signal(indicators)
                
                ea.disconnect_mt5()
                return signal, None
            
            except Exception as e:
                ea.disconnect_mt5()
                return None, f"Error: {str(e)}"
        
        signal, error = await loop.run_in_executor(None, generate_trading_signal)
        
        if error:
            await query.message.reply_text(f"❌ {error}")
            return
        
        if signal is None:
            await query.message.reply_text("❌ Failed to generate signal.")
            return
        
        # Check confidence and send signal
        if signal.confidence.upper() in ["HIGH", "MEDIUM"]:
            direction_emoji = "🟢" if signal.signal_type == "LONG" else "🔴"
            message = f"""
🚨 *TRADING SIGNAL ALERT* 🚨

{direction_emoji} *{signal.signal_type}* on {signal.symbol}
📊 *Confidence:* {signal.confidence}
📈 *Timeframe:* {signal.timeframe}

💰 *Entry Price:* {signal.entry_price:.5f}
📏 *Position Size:* {signal.position_size:.2f} lots

🛡️ *Risk Management:*
• Stop Loss: {signal.stop_loss:.5f}
• Take Profit 1: {signal.take_profit_1:.5f}
• Take Profit 2: {signal.take_profit_2:.5f} (Move SL to BE!)
• Take Profit 3: {signal.take_profit_3:.5f} (Manual closure)

💡 *Reasoning:* {signal.reasoning}

⏰ *Generated:* {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
⏱️ Valid for 5 minutes for entry.
            """.strip()
            
            await query.message.reply_text(message, parse_mode='Markdown')
        else:
            await query.message.reply_text(
                f"📊 Signal generated but confidence is {signal.confidence}. "
                f"No actionable signal at this time.\n\n"
                f"💡 Current analysis: {signal.reasoning}"
            )
        return ConversationHandler.END

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancel the conversation."""
    await update.message.reply_text("Signal generation cancelled.")
    return ConversationHandler.END

def main():
    """Start the bot."""
    # Create custom request with proper timeouts
    request = HTTPXRequest(
        connection_pool_size=8,
        connect_timeout=10.0,
        read_timeout=30.0,
    )
    
    # Create the Application with custom request
    application = (
        Application.builder()
        .token(os.getenv("TELEGRAM_BOT_TOKEN"))
        .request(request)
        .build()
    )
    
    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))

    # Signal conversation handler
    signal_conv_handler = ConversationHandler(
        entry_points=[CommandHandler("signal", start_signal)],
        states={
            SELECT_SYMBOL: [MessageHandler(filters.TEXT & ~filters.COMMAND, select_symbol)],
            SELECT_TIMEFRAME: [MessageHandler(filters.TEXT & ~filters.COMMAND, select_timeframe)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )
    application.add_handler(signal_conv_handler)

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    # Start the bot
    print("Bot started...")

    # Run the bot until you press Ctrl-C (this is blocking and handles the event loop)
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()