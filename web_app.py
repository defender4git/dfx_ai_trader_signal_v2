"""
AI Trading Signal Generator Web Interface
Provides a professional web UI for the AI-based trading system
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import sys
import os
import json
import logging
from datetime import datetime

import openai

loadenv = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(loadenv):
    from dotenv import load_dotenv
    load_dotenv(loadenv)

# Add the parent directory to the path to import the trading modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app_ai_based import MT5TradingEA, TradingSignal

# Import Flask extensions for user management
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail, Message
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__,
            template_folder='templates',
            static_folder='static')

# Flask configuration for user management
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'lIMITED123!XsfegthhhttbMF34R9FSSWW')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Flask-Mail configuration
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS', 'True').lower() == 'true'
app.config['MAIL_USE_SSL'] = os.getenv('MAIL_USE_SSL', 'False').lower() == 'true'
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER', app.config['MAIL_USERNAME'])
app.config['MAIL_CHARSET'] = 'utf-8'

# Configure logging
logging.basicConfig(
    filename='mail_notifications.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
mail = Mail(app)

# Global variables to store the EA instance and current signal
ea_instance = None
current_signal = None

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(150), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_admin = db.Column(db.Boolean, default=False)
    email_notifications = db.Column(db.Boolean, default=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# Activity Log model
class ActivityLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    activity_type = db.Column(db.String(100), nullable=False)  # e.g., 'email_notification'
    description = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(50), nullable=False)  # 'success', 'failure'
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    details = db.Column(db.Text)  # Additional details like error messages

    user = db.relationship('User', backref=db.backref('activity_logs', lazy=True))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def send_signal_alert(user_email, signal_data):
    """Send email alert for new trading signal"""
    # Get user from database
    user = User.query.filter_by(email=user_email).first()
    if not user:
        logging.error(f"User not found for email {user_email}")
        return

    if not app.config['MAIL_USERNAME'] or not app.config['MAIL_PASSWORD']:
        logging.warning("Email configuration not found - skipping alert")
        # Log to activity log
        activity_log = ActivityLog(
            user_id=user.id,
            activity_type='email_notification',
            description=f'Failed to send signal alert for {signal_data["symbol"]} - Email configuration missing',
            status='failure',
            details='MAIL_USERNAME or MAIL_PASSWORD not configured'
        )
        db.session.add(activity_log)
        db.session.commit()
        print("⚠️  Email configuration not found - skipping alert")
        return

    try:
        subject = f"🚨 AI Trading Signal Alert - {signal_data['symbol']}"

        body = f"""
AI Trading Signal Generator Alert

📊 Signal Details:
• Symbol: {signal_data['symbol']}
• Signal Type: {signal_data['signal_type']}
• Confidence: {signal_data['confidence']}
• Entry Price: {signal_data['entry_price']:.5f}
• Stop Loss: {signal_data['stop_loss']:.5f}
• Take Profit 1: {signal_data['take_profit_1']:.5f}
• Position Size: {signal_data['position_size']:.2f} lots

💡 AI Reasoning:
{signal_data['reasoning']}

📈 Technical Indicators:
• ATR: {signal_data.get('atr', 'N/A')}
• RSI: {signal_data.get('rsi', 'N/A')}
• MACD: {signal_data.get('macd', 'N/A')}
• CCI: {signal_data.get('cci', 'N/A')}

⏰ Generated: {signal_data['timestamp']}

---
AI Trading Signal Generator
Real-time market analysis and automated signals
        """

        msg = Message(
            subject=subject,
            recipients=[user_email],
            body=body,
            charset='utf-8'
        )

        mail.send(msg)

        # Log success to activity log
        activity_log = ActivityLog(
            user_id=user.id,
            activity_type='email_notification',
            description=f'Signal alert sent for {signal_data["symbol"]} ({signal_data["signal_type"]})',
            status='success',
            details=f'Signal: {signal_data["signal_type"]}, Confidence: {signal_data["confidence"]}'
        )
        db.session.add(activity_log)
        db.session.commit()

        logging.info(f"Signal alert sent successfully to {user_email} for symbol {signal_data['symbol']}")
        print(f"✅ Signal alert sent to {user_email}")

    except Exception as e:
        # Log failure to activity log
        activity_log = ActivityLog(
            user_id=user.id,
            activity_type='email_notification',
            description=f'Failed to send signal alert for {signal_data["symbol"]}',
            status='failure',
            details=str(e)
        )
        db.session.add(activity_log)
        db.session.commit()

        logging.error(f"Failed to send email alert to {user_email}: {str(e)}")
        print(f"❌ Failed to send email alert: {e}")

@app.route('/')
def index():
    """Main page - redirect to login if not authenticated"""
    if current_user.is_authenticated:
        return render_template('ai_trader.html')
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login page"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('index'))

        return render_template('login.html', error='Invalid username or password')

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration page"""
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        if User.query.filter_by(username=username).first():
            return render_template('register.html', error='Username already exists')

        if User.query.filter_by(email=email).first():
            return render_template('register.html', error='Email already exists')

        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    """User logout"""
    logout_user()
    return redirect(url_for('login'))

@app.route('/profile')
@login_required
def profile():
    """User profile page"""
    # Get recent activity logs for the user
    activity_logs = ActivityLog.query.filter_by(user_id=current_user.id)\
        .order_by(ActivityLog.timestamp.desc())\
        .limit(20)\
        .all()
    return render_template('profile.html', activity_logs=activity_logs)

@app.route('/run_ai_analysis', methods=['POST'])
@login_required
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

        # Prepare detailed logs for frontend
        logs = [
            f"🔗 Connected to MT5 for symbol {symbol}",
            f"📊 Retrieved market data: {len(df)} candles from MT5",
            f"📈 Calculated technical indicators: ATR={indicators['atr']:.4f}, RSI={indicators['rsi']:.1f}, MACD={indicators['macd']:.4f}",
            f"🎯 Generated trading signal: {signal.signal_type} with {signal.confidence} confidence",
            f"💡 AI Analysis: {signal.reasoning[:100]}...",
            f"✅ Analysis completed successfully for {symbol}"
        ]

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
            }),
            'logs': logs
        }

        # Send email alerts to all users with notifications enabled
        try:
            users_with_notifications = User.query.filter_by(email_notifications=True).all()
            for user in users_with_notifications:
                signal_alert_data = {
                    'symbol': signal.symbol,
                    'signal_type': signal.signal_type,
                    'confidence': signal.confidence,
                    'entry_price': signal.entry_price,
                    'stop_loss': signal.stop_loss,
                    'take_profit_1': signal.take_profit_1,
                    'take_profit_2': signal.take_profit_2,
                    'take_profit_3': signal.take_profit_3,
                    'position_size': signal.position_size,
                    'reasoning': signal.reasoning,
                    'timestamp': signal.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC'),
                    # 'atr': indicators.get('atr', 0),
                    # 'rsi': indicators.get('rsi', 0),
                    # 'macd': indicators.get('macd', 0),
                    # 'cci': indicators.get('cci', 0)
                }
                send_signal_alert(user.email, signal_alert_data)
        except Exception as e:
            print(f"⚠️  Error sending email alerts: {e}")

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_current_signal')
@login_required
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
@login_required
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
@login_required
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
@login_required
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

@app.route('/get_user_signals')
@login_required
def get_user_signals():
    """Get trading signals filtered by user parameters"""
    try:
        # Get query parameters
        symbol = request.args.get('symbol')
        signal_type = request.args.get('signal_type')
        confidence = request.args.get('confidence')
        date_from = request.args.get('date_from')
        date_to = request.args.get('date_to')

        # For now, return the current signal if it matches filters
        # In a real implementation, you'd query a database
        global current_signal

        if current_signal is None:
            return jsonify({'signals': []})

        # Apply filters
        matches = True

        if symbol and current_signal.symbol != symbol:
            matches = False
        if signal_type and current_signal.signal_type != signal_type:
            matches = False
        if confidence and current_signal.confidence != confidence:
            matches = False

        if matches:
            signals = [{
                'id': 1,
                'symbol': current_signal.symbol,
                'signal_type': current_signal.signal_type,
                'entry_price': current_signal.entry_price,
                'stop_loss': current_signal.stop_loss,
                'take_profit_1': current_signal.take_profit_1,
                'position_size': current_signal.position_size,
                'confidence': current_signal.confidence,
                'reasoning': current_signal.reasoning,
                'timestamp': current_signal.timestamp.isoformat(),
                'user_id': current_user.id
            }]
        else:
            signals = []

        return jsonify({'signals': signals})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create database tables and handle migrations
    with app.app_context():
        # Create all tables
        db.create_all()

        # Add email_notifications column if it doesn't exist (migration)
        try:
            # Check if column exists by trying to query it
            User.query.filter_by(email_notifications=True).first()
        except Exception as e:
            if 'no such column' in str(e):
                print("Adding email_notifications column to User table...")
                # For SQLAlchemy 2.0+, use connection.execute()
                with db.engine.connect() as conn:
                    conn.execute(db.text('ALTER TABLE user ADD COLUMN email_notifications BOOLEAN DEFAULT 1'))
                    conn.commit()
                print("Database migration completed")

        # Create ActivityLog table if it doesn't exist
        try:
            ActivityLog.query.first()
        except Exception as e:
            if 'no such table' in str(e):
                print("Creating ActivityLog table...")
                db.create_all()
                print("ActivityLog table created")

    print("Starting AI Trading Signal Generator Web Interface...")
    print("Open your browser to http://localhost:9000")
    app.run(debug=True, host='0.0.0.0', port=9000)