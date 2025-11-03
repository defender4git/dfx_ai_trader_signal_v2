"""
AI Trading Expert Advisor Web Interface
Provides a professional web UI for the AI-based trading system
"""

from flask import Flask, render_template, request, jsonify
import sys
import os
import json
from datetime import datetime

import openai

loadenv = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(loadenv):
    from dotenv import load_dotenv
    load_dotenv(loadenv)

# Add the parent directory to the path to import the trading modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app_ai_based import MT5TradingEA, TradingSignal

app = Flask(__name__,
            template_folder='templates',
            static_folder='static')

# Global variables to store the EA instance and current signal
ea_instance = None
current_signal = None

@app.route('/')
def index():
    """Main page"""
    return render_template('ai_trader.html')

@app.route('/run_ai_analysis', methods=['POST'])
def run_ai_analysis():
    """Run AI analysis and return results"""
    global ea_instance, current_signal

    try:
        data = request.get_json()

        # Extract parameters
        symbol = data.get('symbol', 'XAUUSD')
        timeframe = data.get('timeframe', 16385)  # H1 default
        ai_provider = data.get('aiProvider', 'anthropic')
        risk_percent = data.get('riskPercent', 1.0)

        # Get API key based on provider
        if ai_provider == 'anthropic':
            api_key = os.getenv('ANTHROPIC_API_KEY', 'your-anthropic-api-key-here')
        elif ai_provider == 'openai':
            api_key = os.getenv('OPENAI_API_KEY', 'your-openai-api-key-here')
        elif ai_provider == 'deepseek':
            api_key = os.getenv('DEEPSEEKER_API_KEY', 'your-deepseek-api-key-here')
        elif ai_provider == 'grok':
            api_key = os.getenv('GROK_API_KEY', 'your-grok-api-key-here')
        elif ai_provider == 'kilo_code':
            api_key = os.getenv('KILO_CODE_API_KEY', 'your-kilo-code-api-key-here')
        else:
            return jsonify({'error': 'Invalid AI provider'}), 400

        # Create or update EA instance
        if ea_instance is None or ea_instance.symbol != symbol or ea_instance.timeframe != timeframe:
            ea_instance = MT5TradingEA(
                symbol=symbol,
                timeframe=timeframe,
                lookback=150,
                ai_api_key=api_key,
                account_risk_percent=risk_percent,
                ai_provider=ai_provider
            )

            # Connect to MT5 if not already connected
            if not ea_instance.connect_mt5():
                return jsonify({'error': 'Failed to connect to MT5'}), 500

        # Get market data
        df = ea_instance.get_market_data()
        if df is None:
            return jsonify({'error': 'Failed to get market data'}), 500

        # Calculate indicators
        indicators = ea_instance.calculate_indicators(df)

        # Generate signal
        signal = ea_instance.generate_signal(indicators)
        current_signal = signal

        # Prepare response
        response = {
            'signal': {
                'symbol': signal.symbol,
                'signal_type': signal.signal_type,
                'entry_price': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'take_profit_1': signal.take_profit_1,
                'take_profit_2': signal.take_profit_2,
                'take_profit_3': signal.take_profit_3,
                'position_size': signal.position_size,
                'confidence': signal.confidence,
                'reasoning': signal.reasoning,
                'timestamp': signal.timestamp.isoformat()
            },
            'analysis': {
                'atr': indicators['atr'],
                'rsi': indicators['rsi'],
                'macd': indicators['macd'],
                'macd_signal': indicators['macd_signal'],
                'cci': indicators['cci'],
                'stochastic': indicators['stochastic'],
                'volatility_level': indicators['volatility_level'],
                'bias': indicators['bias'],
                'current_price': indicators['current_price']
            },
            'token_usage': signal.indicators.get('token_usage', {
                'input_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0
            })
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_current_signal')
def get_current_signal():
    """Get the current trading signal"""
    global current_signal

    if current_signal is None:
        return jsonify({'error': 'No signal available'}), 404

    response = {
        'symbol': current_signal.symbol,
        'signal_type': current_signal.signal_type,
        'entry_price': current_signal.entry_price,
        'stop_loss': current_signal.stop_loss,
        'take_profit_1': current_signal.take_profit_1,
        'take_profit_2': current_signal.take_profit_2,
        'take_profit_3': current_signal.take_profit_3,
        'position_size': current_signal.position_size,
        'confidence': current_signal.confidence,
        'reasoning': current_signal.reasoning,
        'timestamp': current_signal.timestamp.isoformat(),
        'indicators': current_signal.indicators
    }

    return jsonify(response)

@app.route('/execute_trade', methods=['POST'])
def execute_trade():
    """Execute the current trading signal"""
    global ea_instance, current_signal

    if ea_instance is None or current_signal is None:
        return jsonify({'error': 'No active EA or signal'}), 400

    try:
        data = request.get_json()
        auto_execute = data.get('auto_execute', False)

        success = ea_instance.execute_trade(current_signal, auto_execute)

        if success:
            return jsonify({'message': 'Trade executed successfully'})
        else:
            return jsonify({'error': 'Trade execution failed'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_mt5_status')
def get_mt5_status():
    """Get MT5 connection status"""
    try:
        import MetaTrader5 as mt5
        if mt5.initialize():
            account_info = mt5.account_info()
            mt5.shutdown()
            return jsonify({
                'connected': True,
                'account': account_info.login if account_info else None
            })
        else:
            return jsonify({'connected': False})
    except Exception as e:
        return jsonify({'connected': False, 'error': str(e)})

@app.route('/get_symbols')
def get_symbols():
    """Get available trading symbols from MT5"""
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            return jsonify({'error': 'Failed to connect to MT5'}), 500

        # Get all available symbols
        symbols = mt5.symbols_get()

        if symbols is None:
            mt5.shutdown()
            return jsonify({'error': 'No symbols available'}), 500

        # Filter for forex and major symbols
        filtered_symbols = []
        for symbol in symbols:
            symbol_name = symbol.name
            # Include major forex pairs and common symbols
            if any(pair in symbol_name.upper() for pair in ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']) or \
               'GOLD' in symbol_name.upper() or 'XAU' in symbol_name.upper() or \
               'VOLATILITY' in symbol_name.upper() or 'CRYPTO' in symbol_name.upper():
                filtered_symbols.append({
                    'name': symbol_name,
                    'description': getattr(symbol, 'description', symbol_name),
                    'path': getattr(symbol, 'path', ''),
                    'currency_base': getattr(symbol, 'currency_base', ''),
                    'currency_profit': getattr(symbol, 'currency_profit', ''),
                    'point': symbol.point,
                    'volume_min': symbol.volume_min,
                    'volume_max': symbol.volume_max
                })

        mt5.shutdown()

        # Sort by name for better UX
        filtered_symbols.sort(key=lambda x: x['name'])

        return jsonify({'symbols': filtered_symbols})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting AI Trading Expert Advisor Web Interface...")
    print("📊 Open your browser to http://localhost:9000")
    app.run(debug=True, host='0.0.0.0', port=9000)