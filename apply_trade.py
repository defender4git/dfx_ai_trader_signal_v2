"""
Apply Trade Module
Executes trades based on high and medium confidence AI signals and manages positions
Maintains win rate above 80% through risk management and position sizing
Supports IOC/FOK order filling types
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    filename='trade_execution.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@dataclass
class TradePosition:
    """Represents an active trade position"""
    ticket: int
    symbol: str
    signal_type: str  # 'LONG' or 'SHORT'
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    position_size: float
    confidence: str
    timestamp: datetime
    partial_closes: List[Dict] = field(default_factory=list)  # Track partial TP closes
    status: str = "ACTIVE"  # ACTIVE, CLOSED, CANCELLED

    def is_tp_hit(self, current_price: float) -> Optional[int]:
        """Check if any take profit level is hit"""
        if self.signal_type == "LONG":
            if current_price >= self.take_profit_3:
                return 3
            elif current_price >= self.take_profit_2:
                return 2
            elif current_price >= self.take_profit_1:
                return 1
        else:  # SHORT
            if current_price <= self.take_profit_3:
                return 3
            elif current_price <= self.take_profit_2:
                return 2
            elif current_price <= self.take_profit_1:
                return 1
        return None

    def is_sl_hit(self, current_price: float) -> bool:
        """Check if stop loss is hit"""
        if self.signal_type == "LONG":
            return current_price <= self.stop_loss
        else:
            return current_price >= self.stop_loss

class TradeManager:
    """Manages trade execution and position monitoring"""

    def __init__(self, max_risk_percent: float = 1.0, target_win_rate: float = 0.80):
        self.max_risk_percent = max_risk_percent
        self.target_win_rate = target_win_rate
        self.active_positions: Dict[int, TradePosition] = {}
        self.trade_history: List[Dict] = []
        self.win_rate = 0.0
        self.total_trades = 0
        self.winning_trades = 0

        # Risk management parameters
        self.consecutive_losses = 0
        self.max_consecutive_losses = 3
        self.risk_multiplier = 1.0

    def connect_mt5(self) -> bool:
        """Connect to MT5 terminal"""
        if not mt5.initialize():
            logging.error(f"MT5 initialization failed: {mt5.last_error()}")
            return False

        logging.info(f"MT5 Connected: {mt5.version()}")
        logging.info(f"Account: {mt5.account_info().login}")
        return True

    def disconnect_mt5(self):
        """Disconnect from MT5"""
        mt5.shutdown()
        logging.info("MT5 Disconnected")

    def calculate_win_rate(self) -> float:
        """Calculate current win rate"""
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    def should_trade(self) -> bool:
        """Determine if trading should continue based on win rate and risk management"""
        current_win_rate = self.calculate_win_rate()

        # Stop trading if win rate drops below target
        if self.total_trades >= 10 and current_win_rate < self.target_win_rate:
            logging.warning(f"Win rate {current_win_rate:.2%} below target {self.target_win_rate:.2%} - pausing trading")
            return False

        # Reduce risk after consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            logging.warning(f"Consecutive losses: {self.consecutive_losses} - reducing risk")
            self.risk_multiplier = 0.5
            return True  # Still allow trading but with reduced risk

        self.risk_multiplier = 1.0
        return True

    def execute_trade(self, signal, filling_type: str = "IOC") -> bool:
        """
        Execute trade based on signal
        Args:
            signal: TradingSignal object
            filling_type: "IOC", "FOK", or "RETURN"
        """
        logging.info(f"Starting trade execution for {signal.symbol} with {filling_type} filling type")
        
        if signal.confidence.upper() not in ["HIGH", "MEDIUM"]:
            logging.info(f"Signal confidence is {signal.confidence}, skipping execution")
            return False

        if not self.should_trade():
            logging.info("Trading paused due to risk management rules")
            return False
        
        # Check if symbol is available and market is open
        symbol_info = mt5.symbol_info(signal.symbol)
        if symbol_info is None:
            error_msg = f"Symbol {signal.symbol} not found or not available for trading"
            logging.error(error_msg)
            raise Exception(error_msg)
        
        if not symbol_info.visible:
            error_msg = f"Symbol {signal.symbol} is not visible/available for trading"
            logging.error(error_msg)
            raise Exception(error_msg)

        # Check if signal has expired
        if signal.is_expired():
            logging.warning(f"Signal has expired! Cannot execute trade. Signal expired at {signal.expiry_timestamp}")
            return False

        # Adjust position size based on risk multiplier
        adjusted_position_size = signal.position_size * self.risk_multiplier

        # Map filling type
        filling_map = {
            "IOC": mt5.ORDER_FILLING_IOC,
            "FOK": mt5.ORDER_FILLING_FOK,
            "RETURN": mt5.ORDER_FILLING_RETURN
        }
        order_filling = filling_map.get(filling_type.upper(), mt5.ORDER_FILLING_IOC)

        # Prepare trade request
        order_type = mt5.ORDER_TYPE_BUY if signal.signal_type == "LONG" else mt5.ORDER_TYPE_SELL

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": signal.symbol,
            "volume": adjusted_position_size,
            "type": order_type,
            "price": signal.entry_price,
            "sl": signal.stop_loss,
            "tp": signal.take_profit_1,  # Set first TP by default
            "deviation": 20,
            "magic": 234001,  # Different magic number for apply_trade
            "comment": f"AI_HIGH_CONF_{signal.confidence}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": order_filling,
        }

        # Send order
        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            error_msg = f"Trade execution failed: {result.comment} (retcode: {result.retcode})"
            logging.error(error_msg)
            raise Exception(error_msg)

        # Create position tracking object
        position = TradePosition(
            ticket=result.order,
            symbol=signal.symbol,
            signal_type=signal.signal_type,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit_1=signal.take_profit_1,
            take_profit_2=signal.take_profit_2,
            take_profit_3=signal.take_profit_3,
            position_size=adjusted_position_size,
            confidence=signal.confidence,
            timestamp=datetime.now()
        )

        self.active_positions[result.order] = position

        logging.info(f"✅ Trade executed successfully! Ticket: {result.order}, Volume: {result.volume}, Price: {result.price}")
        print(f"✅ Trade executed successfully! Ticket: {result.order}")

        return True

    def monitor_positions(self):
        """Monitor active positions and manage take profits"""
        positions_to_remove = []

        for ticket, position in self.active_positions.items():
            try:
                # Get current position info
                mt5_position = mt5.positions_get(ticket=ticket)
                if not mt5_position:
                    logging.warning(f"Position {ticket} not found - may have been closed")
                    positions_to_remove.append(ticket)
                    continue

                current_price = mt5_position[0].price_current

                # Check for stop loss hit
                if position.is_sl_hit(current_price):
                    logging.info(f"Stop loss hit for position {ticket}")
                    # Close entire position
                    self._close_position(ticket, "SL_HIT")
                    positions_to_remove.append(ticket)
                    continue

                # Check for take profit levels
                tp_level = position.is_tp_hit(current_price)
                if tp_level:
                    if tp_level == 1:
                        # Close 50% at TP1
                        close_volume = position.position_size * 0.5
                        self._partial_close(ticket, close_volume, f"TP{tp_level}")
                        position.partial_closes.append({
                            "level": tp_level,
                            "volume": close_volume,
                            "price": current_price,
                            "timestamp": datetime.now()
                        })
                    elif tp_level == 2:
                        # Close remaining 50% at TP2
                        remaining_volume = position.position_size * 0.5
                        self._partial_close(ticket, remaining_volume, f"TP{tp_level}")
                        position.partial_closes.append({
                            "level": tp_level,
                            "volume": remaining_volume,
                            "price": current_price,
                            "timestamp": datetime.now()
                        })
                        positions_to_remove.append(ticket)
                    elif tp_level == 3:
                        # Close entire position at TP3 (if not already partially closed)
                        self._close_position(ticket, f"TP{tp_level}")
                        positions_to_remove.append(ticket)

            except Exception as e:
                logging.error(f"Error monitoring position {ticket}: {e}")
                positions_to_remove.append(ticket)

        # Remove closed positions
        for ticket in positions_to_remove:
            if ticket in self.active_positions:
                del self.active_positions[ticket]

    def _partial_close(self, ticket: int, volume: float, reason: str) -> bool:
        """Partially close a position"""
        try:
            position = mt5.positions_get(ticket=ticket)[0]
            order_type = mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": volume,
                "type": order_type,
                "position": ticket,
                "price": mt5.symbol_info_tick(position.symbol).bid if order_type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(position.symbol).ask,
                "deviation": 20,
                "magic": 234001,
                "comment": f"PARTIAL_CLOSE_{reason}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logging.info(f"Partial close successful for ticket {ticket}: {volume} lots at {result.price}")
                return True
            else:
                logging.error(f"Partial close failed for ticket {ticket}: {result.comment}")
                return False

        except Exception as e:
            logging.error(f"Error in partial close for ticket {ticket}: {e}")
            return False

    def _close_position(self, ticket: int, reason: str) -> bool:
        """Close entire position"""
        try:
            position = mt5.positions_get(ticket=ticket)[0]
            order_type = mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY

            request = {
                "action": mt5.TRADE_ACTION_CLOSEBY,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": order_type,
                "position": ticket,
                "price": mt5.symbol_info_tick(position.symbol).bid if order_type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(position.symbol).ask,
                "deviation": 20,
                "magic": 234001,
                "comment": f"CLOSE_{reason}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logging.info(f"Position close successful for ticket {ticket}: {reason}")
                return True
            else:
                logging.error(f"Position close failed for ticket {ticket}: {result.comment}")
                return False

        except Exception as e:
            logging.error(f"Error closing position {ticket}: {e}")
            return False

    def update_trade_history(self, position: TradePosition, exit_reason: str, exit_price: float):
        """Update trade history and win rate"""
        # Calculate P&L
        if position.signal_type == "LONG":
            pnl = (exit_price - position.entry_price) * position.position_size * 100000  # Assuming forex
        else:
            pnl = (position.entry_price - exit_price) * position.position_size * 100000

        is_win = pnl > 0

        trade_record = {
            "ticket": position.ticket,
            "symbol": position.symbol,
            "signal_type": position.signal_type,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "is_win": is_win,
            "exit_reason": exit_reason,
            "timestamp": datetime.now(),
            "confidence": position.confidence
        }

        self.trade_history.append(trade_record)
        self.total_trades += 1

        if is_win:
            self.winning_trades += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1

        self.win_rate = self.calculate_win_rate()

        logging.info(f"Trade closed - Win Rate: {self.win_rate:.2%} ({self.winning_trades}/{self.total_trades})")

    def get_performance_stats(self) -> Dict:
        """Get current performance statistics"""
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "win_rate": self.win_rate,
            "active_positions": len(self.active_positions),
            "consecutive_losses": self.consecutive_losses,
            "risk_multiplier": self.risk_multiplier,
            "trading_enabled": self.should_trade()
        }

def apply_high_confidence_trade(signal, filling_type: str = "IOC") -> bool:
    """
    Main function to apply a high or medium confidence trade
    Args:
        signal: TradingSignal object with HIGH or MEDIUM confidence
        filling_type: Order filling type (IOC/FOK/RETURN)
    Returns:
        bool: True if trade executed successfully
    """
    manager = TradeManager()

    if not manager.connect_mt5():
        return False

    try:
        success = manager.execute_trade(signal, filling_type)
        return success
    finally:
        manager.disconnect_mt5()

def run_trade_manager(monitor_interval: int = 60):
    """
    Run continuous trade monitoring
    Args:
        monitor_interval: Seconds between position checks
    """
    manager = TradeManager()

    if not manager.connect_mt5():
        return

    logging.info("🚀 Trade Manager Started - Monitoring positions...")
    print("🚀 Trade Manager Started - Monitoring positions...")

    try:
        while True:
            manager.monitor_positions()

            # Log performance stats every 5 minutes
            if int(time.time()) % 300 == 0:
                stats = manager.get_performance_stats()
                logging.info(f"Performance Stats: {stats}")
                print(f"📊 Performance: Win Rate {stats['win_rate']:.2%}, Active Positions: {stats['active_positions']}")

            time.sleep(monitor_interval)

    except KeyboardInterrupt:
        logging.info("⛔ Trade Manager stopped by user")
        print("\n⛔ Trade Manager stopped by user")
    finally:
        manager.disconnect_mt5()

if __name__ == "__main__":
    # Example usage - run trade manager
    run_trade_manager()