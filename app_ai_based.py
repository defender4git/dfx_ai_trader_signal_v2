"""
MT5 AI-Powered Trading Expert Advisor
Analyzes technical indicators and generates professional trading signals
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import anthropic
import openai
import os
from dotenv import load_dotenv
import twilio
from twilio.rest import Client as TwilioClient
import telegram
import requests

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    filename='mail_notifications.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
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
    timestamp: datetime
    expiry_timestamp: datetime
    trend: str
    entry_strategy: str
    indicators: Dict

    def is_expired(self) -> bool:
        """Check if the signal has expired (5 minutes validity)"""
        return datetime.now() > self.expiry_timestamp

    def time_remaining(self) -> str:
        """Get remaining time before expiry in human readable format"""
        if self.is_expired():
            return "EXPIRED"

        remaining = self.expiry_timestamp - datetime.now()
        minutes = int(remaining.total_seconds() // 60)
        seconds = int(remaining.total_seconds() % 60)

        if minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
class NotificationManager:
    """Handles notifications via Telegram and WhatsApp"""

    def __init__(self):
        # Twilio for WhatsApp
        twilio_sid = os.getenv("TWILIO_ACCOUNT_SID")
        twilio_token = os.getenv("TWILIO_AUTH_TOKEN")
        twilio_whatsapp = os.getenv("TWILIO_WHATSAPP_NUMBER")
        recipient_whatsapp = os.getenv("RECIPIENT_WHATSAPP_NUMBER")

        # Validate Twilio configuration
        if not all([twilio_sid, twilio_token, twilio_whatsapp, recipient_whatsapp]):
            logging.error("WhatsApp notification disabled: Missing Twilio configuration")
            self.twilio_client = None
            self.twilio_whatsapp_number = None
            self.recipient_whatsapp_number = None
        else:
            try:
                self.twilio_client = TwilioClient(twilio_sid, twilio_token)
                self.twilio_whatsapp_number = twilio_whatsapp
                self.recipient_whatsapp_number = recipient_whatsapp
                logging.info("WhatsApp notification initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize Twilio client: {e}")
                self.twilio_client = None
                self.twilio_whatsapp_number = None
                self.recipient_whatsapp_number = None

        # Telegram
        self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")

        # Rate limiting
        self.last_notification_time = 0
        self.min_interval_seconds = 60  # Minimum 1 minute between notifications

    def send_whatsapp_message(self, message: str, max_retries: int = 3) -> bool:
        """Send message via WhatsApp with retry logic using Content Templates"""
        if not self.twilio_client:
            logging.warning("WhatsApp client not initialized - skipping notification")
            return False

        # Rate limiting check
        current_time = time.time()
        if current_time - self.last_notification_time < self.min_interval_seconds:
            logging.info("Rate limit exceeded - skipping WhatsApp notification")
            return False

        # Get content SID from environment
        content_sid = os.getenv("TWILIO_CONTENT_SID", "")

        # Use the signal message as content variable
        content_variables = {"1": message}

        for attempt in range(max_retries):
            try:
                # Use Content Templates API
                twilio_message = self.twilio_client.messages.create(
                    from_=f"whatsapp:{self.twilio_whatsapp_number}",
                    content_sid=content_sid,
                    content_variables=content_variables,
                    to=f"whatsapp:{self.recipient_whatsapp_number}"
                )
                self.last_notification_time = current_time
                logging.info(f"WhatsApp message sent successfully: SID {twilio_message.sid}")
                print(f"WhatsApp message sent: {twilio_message.sid}")
                return True
            except Exception as e:
                wait_time = 2 ** attempt  # Exponential backoff
                logging.warning(f"WhatsApp send attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s")
                if attempt < max_retries - 1:
                    time.sleep(wait_time)

        logging.error(f"WhatsApp send failed after {max_retries} attempts")
        return False

    def send_telegram_message(self, message: str, max_retries: int = 3) -> bool:
        """Send message via Telegram with retry logic"""
        if not self.telegram_bot_token or not self.telegram_chat_id:
            logging.warning("Telegram configuration missing - skipping notification")
            return False

        for attempt in range(max_retries):
            try:
                url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
                data = {
                    "chat_id": self.telegram_chat_id,
                    "text": message,
                    "parse_mode": "Markdown"
                }
                response = requests.post(url, data=data, timeout=10)
                if response.status_code == 200:
                    logging.info("Telegram message sent successfully")
                    print("Telegram message sent successfully")
                    return True
                else:
                    logging.warning(f"Telegram send failed: HTTP {response.status_code} - {response.text}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
            except Exception as e:
                logging.warning(f"Telegram send attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

        logging.error(f"Telegram send failed after {max_retries} attempts")
        return False

    def send_signal_notification(self, signal: TradingSignal, test_mode: bool = False):
        """Send notification only for HIGH and MEDIUM confidence signals"""
        if signal.confidence.upper() not in ["HIGH", "MEDIUM"]:
            logging.info(f"Signal confidence is {signal.confidence}, skipping notification")
            print(f"Signal confidence is {signal.confidence}, skipping notification")
            return

        message = self._format_signal_message(signal)

        # In test mode, log the message without sending
        if test_mode:
            logging.info(f"TEST MODE: Would send notification - {message[:100]}...")
            print(f"TEST MODE: Notification would be sent (not actually sent)")
            return

        # Send to both channels
        whatsapp_sent = self.send_whatsapp_message(message)
        telegram_sent = self.send_telegram_message(message)

        if whatsapp_sent and telegram_sent:
            logging.info("Signal notification sent to both WhatsApp and Telegram")
            print("Signal notification sent to both WhatsApp and Telegram")
        elif whatsapp_sent:
            logging.info("Signal notification sent to WhatsApp only")
            print("Signal notification sent to WhatsApp only")
        elif telegram_sent:
            logging.info("Signal notification sent to Telegram only")
            print("Signal notification sent to Telegram only")
        else:
            logging.error("Failed to send signal notification to any channel")
            print("Failed to send signal notification to any channel")
            # Log more details for debugging
            logging.error(f"WhatsApp client initialized: {self.twilio_client is not None}")
            logging.error(f"Telegram bot token: {bool(self.telegram_bot_token)}")
            logging.error(f"Telegram chat ID: {bool(self.telegram_chat_id)}")

    def _format_signal_message(self, signal: TradingSignal) -> str:
        """Format signal for notification"""
        direction_emoji = "🟢" if signal.signal_type == "LONG" else "🔴"
        expiry_status = "⚠️ EXPIRED" if signal.is_expired() else f"⏰ Valid for: {signal.time_remaining()}"

        message = f"""
🚨 *AI TRADING SIGNAL ALERT* 🚨

{direction_emoji} *{signal.signal_type}* on {signal.symbol}
📊 *Confidence:* {signal.confidence}
{expiry_status}

💰 *Entry Price:* {signal.entry_price:.5f}
📏 *Position Size:* {signal.position_size:.2f} lots

🛡️ *Risk Management:*
• Stop Loss: {signal.stop_loss:.5f}
• Take Profit 1: {signal.take_profit_1:.5f}
• Take Profit 2: {signal.take_profit_2:.5f}
• Take Profit 3: {signal.take_profit_3:.5f}

📈 *Technical Indicators:*
• ATR: {signal.indicators['atr']:.4f} ({signal.indicators['volatility_level']})
• RSI: {signal.indicators['rsi']:.1f}
• MACD: {signal.indicators['macd']:.4f}
• CCI: {signal.indicators['cci']:.1f}

💡 *AI Reasoning:* {signal.reasoning}

⏰ *Generated:* {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
⏰ *Expires:* {signal.expiry_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
        """.strip()

        return message

    indicators: Dict


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
    def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float]:
        """Calculate MACD and Signal line"""
        close = df['close']
        
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        
        return macd.iloc[-1], signal_line.iloc[-1]
    
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
    def calculate_stochastic(df: pd.DataFrame, period: int = 14, smooth: int = 3) -> float:
        """Calculate Stochastic Oscillator"""
        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()
        
        k = 100 * (df['close'] - low_min) / (high_max - low_min)
        stoch = k.rolling(window=smooth).mean()
        
        return stoch.iloc[-1]


class AITradingAgent:
    """AI Agent for market analysis and signal generation"""

    def __init__(self, api_key: str, provider: str = "anthropic"):
        """Initialize AI client (Anthropic Claude, OpenAI GPT, DeepSeek, or Grok)"""
        self.provider = provider.lower()
        if self.provider == "anthropic":
            self.client = anthropic.Anthropic(api_key=api_key)
        elif self.provider == "openai":
            self.client = openai.OpenAI(api_key=api_key)
        elif self.provider == "deepseek":
            self.api_key = api_key
            self.base_url = "https://api.deepseek.com/v1"
        else:
            raise ValueError("Provider must be 'anthropic', 'openai', or 'deepseek'")

        # Rate limiting for API calls
        self.request_times = []
        self.max_requests_per_minute = 10  # Conservative limit to avoid rate limits
        self.min_interval_seconds = 60.0 / self.max_requests_per_minute  # Minimum time between requests
    
    def analyze_market(self, indicators: Dict, symbol: str, timeframe: str) -> Dict:
        """
        Send indicators to Claude AI for professional analysis
        Returns trading signal with complete parameters
        """

        # Rate limiting check
        current_time = time.time()
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if current_time - t < 60]

        if len(self.request_times) >= self.max_requests_per_minute:
            logging.warning(f"Rate limit reached ({self.max_requests_per_minute} requests/minute). Skipping AI analysis.")
            return self._default_analysis()

        # Check minimum interval between requests
        if self.request_times and current_time - self.request_times[-1] < self.min_interval_seconds:
            wait_time = self.min_interval_seconds - (current_time - self.request_times[-1])
            logging.info(f"Rate limiting: waiting {wait_time:.2f} seconds before next API call")
            time.sleep(wait_time)

        self.request_times.append(current_time)

        prompt = f"""Analyze {symbol} {timeframe} data: ATR={indicators['atr']:.4f}({indicators['volatility_level'][:1]}), RSI={indicators['rsi']:.1f}, MACD={indicators['macd']:.4f}/{indicators['macd_signal']:.4f}, CCI={indicators['cci']:.1f}, Stoch={indicators['stochastic']:.1f}, Bias={indicators['bias'][:3]}.

Return JSON: {{"signal":"LONG/SHORT/NEUTRAL","confidence":"HIGH/MEDIUM/LOW","trend":"BULLISH/BEARISH/NEUTRAL/CONSOLIDATION","entry_strategy":"IMMEDIATE/WAIT_PULLBACK/WAIT_BREAKOUT","stop_loss_atr_multiplier":1.3-2.0,"take_profit_1_atr_multiplier":1.5,"take_profit_2_atr_multiplier":2.0,"take_profit_3_atr_multiplier":3.0,"position_size_adjustment":0.5-1.0,"reasoning":"Brief analysis","key_observations":["obs1","obs2","obs3"]}}"""

        try:
            if self.provider == "anthropic":
                message = self.client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=512,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                response_text = message.content[0].text
                usage = getattr(message, 'usage', None)
                input_tokens = getattr(usage, 'input_tokens', 0) if usage else 0
                output_tokens = getattr(usage, 'output_tokens', 0) if usage else 0
            elif self.provider == "openai":
                try:
                    response = self.client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=512,
                        temperature=0.7
                    )
                    response_text = response.choices[0].message.content
                    usage = getattr(response, 'usage', None)
                    input_tokens = getattr(usage, 'prompt_tokens', 0) if usage else 0
                    output_tokens = getattr(usage, 'completion_tokens', 0) if usage else 0

                    # Debug: Check if response_text is empty or not JSON
                    if not response_text or not response_text.strip():
                        print(f"OpenAI returned empty response")
                        return self._default_analysis()
                except openai.RateLimitError as rate_error:
                    logging.warning(f"OpenAI rate limit exceeded: {rate_error}")
                    print(f"OpenAI rate limit exceeded - using fallback analysis")
                    return self._default_analysis()
                except Exception as api_error:
                    print(f"OpenAI API Error: {api_error}")
                    return self._default_analysis()
            elif self.provider == "deepseek":
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                data = {
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False,
                    "max_tokens": 512,
                    "temperature": 0.7
                }
                response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=data)
                response.raise_for_status()
                response_data = response.json()
                response_text = response_data["choices"][0]["message"]["content"]
                usage = response_data.get("usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)

            # Extract JSON from response
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            json_str = response_text[start:end]

            if not json_str:
                print(f"AI Analysis Error ({self.provider}): No JSON found in response")
                print(f"Raw response: {response_text}")
                return self._default_analysis()

            try:
                analysis = json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"AI Analysis Error ({self.provider}): Invalid JSON - {e}")
                print(f"JSON string: {json_str}")
                return self._default_analysis()
            # Add token usage to analysis
            analysis['token_usage'] = {
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': input_tokens + output_tokens
            }
            return analysis

        except Exception as e:
            print(f"AI Analysis Error ({self.provider}): {e}")
            if self.provider == "openai":
                print(f"OpenAI response details: {getattr(response, 'choices', 'No choices')}")
            return self._default_analysis()
    
    def _default_analysis(self) -> Dict:
        """Fallback analysis if AI fails"""
        return {
            "signal": "NEUTRAL",
            "confidence": "LOW",
            "trend": "NEUTRAL",
            "entry_strategy": "WAIT_BREAKOUT",
            "stop_loss_atr_multiplier": 1.5,
            "take_profit_1_atr_multiplier": 1.5,
            "take_profit_2_atr_multiplier": 2.0,
            "take_profit_3_atr_multiplier": 3.0,
            "position_size_adjustment": 0.5,
            "reasoning": "AI analysis unavailable - using conservative defaults",
            "key_observations": ["Awaiting clear market direction"]
        }


class MT5TradingEA:
    """Main Expert Advisor class"""
    
    def __init__(self, symbol: str, timeframe: int, lookback: int,
                 ai_api_key: str, account_risk_percent: float = 1.0, ai_provider: str = "anthropic"):
        """
        Initialize MT5 Trading EA

        Args:
            symbol: Trading symbol (e.g., 'XAUUSD')
            timeframe: MT5 timeframe (e.g., mt5.TIMEFRAME_H1)
            lookback: Number of candles to analyze
            ai_api_key: API key for AI provider (Anthropic or OpenAI)
            account_risk_percent: Risk per trade as % of account
            ai_provider: AI provider ('anthropic' or 'openai')
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.lookback = lookback
        self.account_risk_percent = account_risk_percent
        self.ai_provider = ai_provider

        self.analyzer = TechnicalAnalyzer()
        self.ai_agent = AITradingAgent(ai_api_key, ai_provider)
        self.notification_manager = NotificationManager()

        self.current_signal: Optional[TradingSignal] = None
        
    def connect_mt5(self) -> bool:
        """Connect to MT5 terminal"""
        if not mt5.initialize():
            print(f"MT5 initialization failed: {mt5.last_error()}")
            return False
        
        print(f"MT5 Connected: {mt5.version()}")
        print(f"Account: {mt5.account_info().login}")
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
            'cci': self.analyzer.calculate_cci(df),
            'stochastic': self.analyzer.calculate_stochastic(df),
            'current_price': df['close'].iloc[-1]
        }
        
        macd, signal = self.analyzer.calculate_macd(df)
        indicators['macd'] = macd
        indicators['macd_signal'] = signal
        
        # Ensure ATR is valid
        if not np.isfinite(indicators['atr']) or indicators['atr'] <= 0:
            indicators['atr'] = 1.0  # Default ATR if invalid

        # Volatility classification
        if indicators['atr'] > 15:
            indicators['volatility_level'] = "Very High Volatility"
        elif indicators['atr'] > 10:
            indicators['volatility_level'] = "High Volatility"
        elif indicators['atr'] > 5:
            indicators['volatility_level'] = "Medium Volatility"
        else:
            indicators['volatility_level'] = "Low Volatility"

        # Market bias based on last candle
        indicators['bias'] = 'bullish' if df['close'].iloc[-1] > df['open'].iloc[-1] else 'bearish'

        return indicators
    
    def generate_signal(self, indicators: Dict) -> TradingSignal:
        """Generate trading signal using AI analysis"""
        
        # Get timeframe name
        timeframe_map = {
            mt5.TIMEFRAME_M1: "M1", mt5.TIMEFRAME_M5: "M5", 
            mt5.TIMEFRAME_M15: "M15", mt5.TIMEFRAME_M30: "M30",
            mt5.TIMEFRAME_H1: "H1", mt5.TIMEFRAME_H4: "H4",
            mt5.TIMEFRAME_D1: "D1"
        }
        tf_name = timeframe_map.get(self.timeframe, "H1")
        
        # Get AI analysis
        ai_analysis = self.ai_agent.analyze_market(indicators, self.symbol, tf_name)
        
        # Calculate trade parameters
        current_price = indicators['current_price']
        atr = indicators['atr']
        
        signal_type = ai_analysis['signal']
        
        # Calculate position size based on risk
        account_info = mt5.account_info()
        account_balance = account_info.balance
        risk_amount = account_balance * (self.account_risk_percent / 100)
        
        # Adjust for volatility
        position_adjustment = ai_analysis.get('position_size_adjustment', 1.0)
        
        if signal_type == "LONG":
            stop_loss = current_price - (atr * ai_analysis['stop_loss_atr_multiplier'])
            tp1 = current_price + (atr * ai_analysis['take_profit_1_atr_multiplier'])
            tp2 = current_price + (atr * ai_analysis['take_profit_2_atr_multiplier'])
            tp3 = current_price + (atr * ai_analysis['take_profit_3_atr_multiplier'])
            
        elif signal_type == "SHORT":
            stop_loss = current_price + (atr * ai_analysis['stop_loss_atr_multiplier'])
            tp1 = current_price - (atr * ai_analysis['take_profit_1_atr_multiplier'])
            tp2 = current_price - (atr * ai_analysis['take_profit_2_atr_multiplier'])
            tp3 = current_price - (atr * ai_analysis['take_profit_3_atr_multiplier'])
            
        else:  # NEUTRAL
            stop_loss = current_price - (atr * 1.5)  # Default SL for neutral
            tp1 = current_price + (atr * 1.5)  # Default TP for neutral
            tp2 = current_price + (atr * 2.0)
            tp3 = current_price + (atr * 3.0)
        
        # Calculate lot size
        symbol_info = mt5.symbol_info(self.symbol)
        point = symbol_info.point
        stop_distance = abs(current_price - stop_loss)
        
        # Risk per pip
        lot_size = risk_amount / (stop_distance / point * symbol_info.trade_contract_size)
        lot_size *= position_adjustment
        
        # Round to allowed lot size
        lot_size = max(symbol_info.volume_min, 
                      min(lot_size, symbol_info.volume_max))
        lot_size = round(lot_size / symbol_info.volume_step) * symbol_info.volume_step
        
        # Create signal with 5-minute expiry
        expiry_time = datetime.now() + timedelta(minutes=5)
        signal = TradingSignal(
            symbol=self.symbol,
            signal_type=signal_type,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            take_profit_3=tp3,
            position_size=lot_size,
            confidence=ai_analysis['confidence'],
            reasoning=ai_analysis['reasoning'],
            timestamp=datetime.now(),
            expiry_timestamp=expiry_time,
            indicators=indicators,
            trend=ai_analysis.get('trend', 'NEUTRAL'),
            entry_strategy=ai_analysis.get('entry_strategy', 'IMMEDIATE')
        )
        
        return signal
    
    def print_signal(self, signal: TradingSignal):
        """Display trading signal in professional format"""
        print("\n" + "="*80)
        print(f"🤖 AI TRADING SIGNAL - {signal.symbol}")
        print("="*80)
        print(f"Timestamp: {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏰ Signal Expiry: {signal.expiry_timestamp.strftime('%Y-%m-%d %H:%M:%S')} ({signal.time_remaining()})")
        print(f"\n📊 MARKET ANALYSIS:")
        print(f"  ATR: {signal.indicators['atr']:.6f} ({signal.indicators['volatility_level']})")
        print(f"  RSI: {signal.indicators['rsi']:.2f}")
        print(f"  MACD: {signal.indicators['macd']:.6f} | Signal: {signal.indicators['macd_signal']:.6f}")
        print(f"  CCI: {signal.indicators['cci']:.2f}")
        print(f"  Stochastic: {signal.indicators['stochastic']:.2f}")
        print(f"  Market Bias: {signal.indicators['bias']}")

        print(f"\n🎯 TRADING SIGNAL: {signal.signal_type}")
        print(f"  Confidence: {signal.confidence}")
        print(f"  Entry Price: {signal.entry_price:.5f}")
        print(f"  Position Size: {signal.position_size:.2f} lots")

        print(f"\n🛡️ RISK MANAGEMENT:")
        stop_distance = abs(signal.entry_price - signal.stop_loss)
        print(f"  Stop Loss: {signal.stop_loss:.5f} ({stop_distance:.5f} pts)")

        if stop_distance > 0:
            rr1 = abs(signal.take_profit_1 - signal.entry_price) / stop_distance
            rr2 = abs(signal.take_profit_2 - signal.entry_price) / stop_distance
            rr3 = abs(signal.take_profit_3 - signal.entry_price) / stop_distance
            print(f"  Take Profit 1: {signal.take_profit_1:.5f} (R:R {rr1:.2f})")
            print(f"  Take Profit 2: {signal.take_profit_2:.5f} (R:R {rr2:.2f})")
            print(f"  Take Profit 3: {signal.take_profit_3:.5f} (R:R {rr3:.2f})")
        else:
            print(f"  Take Profit 1: {signal.take_profit_1:.5f} (R:R N/A)")
            print(f"  Take Profit 2: {signal.take_profit_2:.5f} (R:R N/A)")
            print(f"  Take Profit 3: {signal.take_profit_3:.5f} (R:R N/A)")

        print(f"\n💡 AI REASONING:")
        print(f"  {signal.reasoning}")
        print("="*80 + "\n")
    
    def execute_trade(self, signal: TradingSignal, auto_execute: bool = False) -> bool:
        """
        Execute trade on MT5

        Args:
            signal: TradingSignal object
            auto_execute: If True, execute without confirmation
        """
        if signal.signal_type == "NEUTRAL":
            print("⚠️  No trade signal - Market is NEUTRAL")
            return False

        # Check if signal has expired
        if signal.is_expired():
            print(f"❌ Signal has expired! Cannot execute trade. Signal expired at {signal.expiry_timestamp}")
            return False

        if not auto_execute:
            response = input(f"\n Execute {signal.signal_type} trade? (Signal expires in {signal.time_remaining()}) (yes/no): ")
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
            "tp": signal.take_profit_1,  # Set first TP by default
            "deviation": 20,
            "magic": 234000,
            "comment": f"AI_EA_{signal.confidence}",
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
    
    def run(self, interval_seconds: int = 300, auto_execute: bool = False, test_notifications: bool = False):
        """
        Run EA continuously

        Args:
            interval_seconds: Time between analysis cycles
            auto_execute: Automatically execute trades
            test_notifications: Enable test mode for notifications (no actual sending)
        """
        if not self.connect_mt5():
            return

        print(f"\n🚀 AI Trading EA Started")
        print(f"   Symbol: {self.symbol}")
        print(f"   Timeframe: {self.timeframe}")
        print(f"   Lookback: {self.lookback} candles")
        print(f"   Risk per trade: {self.account_risk_percent}%")
        print(f"   AI Provider: {self.ai_provider}")
        print(f"   Auto-execute: {auto_execute}")
        print(f"   Test notifications: {test_notifications}")
        print(f"   Analysis interval: {interval_seconds}s\n")

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

                # Send notification for HIGH confidence signals
                if signal.confidence.upper() == "HIGH":
                    self.notification_manager.send_signal_notification(signal, test_mode=test_notifications)

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
    SYMBOL = "Volatility 75 Index"
    TIMEFRAME = mt5.TIMEFRAME_H1
    LOOKBACK = 150
    RISK_PERCENT = 1.0  # Risk 1% per trade

    # AI Provider selection - default to OpenAI from .env
    AI_PROVIDER = os.getenv("AI_PROVIDER", "openai").lower()
    if AI_PROVIDER == "anthropic":
        API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    elif AI_PROVIDER == "openai":
        API_KEY = os.getenv("OPENAI_API_KEY", openai.api_key)
    else:
        print("Invalid AI provider in .env, defaulting to OpenAI")
        AI_PROVIDER = "openai"
        API_KEY = os.getenv("OPENAI_API_KEY", openai.api_key)

    # Create EA instance
    ea = MT5TradingEA(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        lookback=LOOKBACK,
        ai_api_key=API_KEY,
        account_risk_percent=RISK_PERCENT,
        ai_provider=AI_PROVIDER
    )

    # Run EA
    # Set auto_execute=True to automatically execute trades
    # Set auto_execute=False to manually confirm each trade
    # Set test_notifications=True to test notifications without sending
    ea.run(interval_seconds=300, auto_execute=True, test_notifications=False)


if __name__ == "__main__":
    main()