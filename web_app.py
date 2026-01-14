"""
AI Trading Signal Generator Web Interface
Provides a professional web UI for the AI-based trading system
"""

import signal
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
import sys
import os
import json
import logging
from datetime import datetime, timedelta

import openai

loadenv = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(loadenv):
    from dotenv import load_dotenv
    load_dotenv(loadenv)

# Add the parent directory to the path to import the trading modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app_ai_based import MT5TradingEA, TradingSignal
from apply_trade import apply_high_confidence_trade, TradeManager

# Import Flask extensions for user management
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail, Message
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__,
            template_folder='templates',
            static_folder='static')

# Flask configuration for user management
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', '')
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
#email library file
email_filename = 'static/vip_email.txt'

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
mail = Mail(app)

# Global variables to store the EA instance, current signal, and trade manager
ea_instance = None
current_signal = None
trade_manager = None

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(150), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_admin = db.Column(db.Boolean, default=False)
    email_notifications = db.Column(db.Boolean, default=True)
    plan = db.Column(db.String(50), default='basic')  # basic, pro, vip

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

# Mailing List model
class MailingList(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    list_type = db.Column(db.String(50), nullable=False)  # 'vip'
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def send_signal_alert(user_email, signal_data):
    """Send email alert for new trading signal"""
    # Get user from database
    user = User.query.filter_by(email=user_email).first()
    if not user:
        logging.warning(f"No user found for email {user_email}, sending as VIP")

    # Check daily signal limit for user
    if user and user.email_notifications:
        plan_limits = {'basic': 2, 'pro': 5, 'vip': 8}
        daily_limit = plan_limits.get(user.plan, 2)
        today = datetime.utcnow().date()
        today_start = datetime(today.year, today.month, today.day)
        today_end = today_start + timedelta(days=1)

        # Count successful email notifications sent today
        today_count = ActivityLog.query.filter_by(
            user_id=user.id,
            activity_type='email_notification',
            status='success'
        ).filter(ActivityLog.timestamp >= today_start, ActivityLog.timestamp < today_end).count()

        if today_count >= daily_limit:
            logging.info(f"Daily signal limit ({daily_limit}) reached for user {user.username} - skipping alert")
            # Log the skip
            activity_log = ActivityLog(
                user_id=user.id,
                activity_type='email_notification',
                description=f'Daily signal limit ({daily_limit}) reached for {signal_data["symbol"]} - alert skipped',
                status='failure',
                details=f'Limit: {daily_limit}, Sent today: {today_count}'
            )
            db.session.add(activity_log)
            db.session.commit()
            print(f"⚠️  Daily limit reached for {user.username} - skipping alert")
            return

    if not app.config['MAIL_USERNAME'] or not app.config['MAIL_PASSWORD']:
        logging.warning("Email configuration not found - skipping alert")
        # Log to activity log only if user exists
        if user:
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

        # Check if signal is expired
        expiry_time = datetime.strptime(signal_data['timestamp'], '%Y-%m-%d %H:%M:%S UTC') + timedelta(minutes=5)
        is_expired = datetime.now() > expiry_time
        expiry_status = "⚠️ SIGNAL EXPIRED" if is_expired else "⏰ Valid for: 5 minutes"

        body = f"""
AI Trading Signal Generator Alert

📊 Signal Details:
• Symbol: {signal_data['symbol']}
• Signal Type: {signal_data['signal_type']}
• Confidence: {signal_data['confidence']}
• Entry Price: {signal_data['entry_price']:.5f}
• Stop Loss: {signal_data['stop_loss']:.5f}
• Take Profit 1: {signal_data['take_profit_1']:.5f}
• Take Profit 2: {signal_data['take_profit_2']:.5f}
• Take Profit 3: {signal_data['take_profit_3']:.5f}
• Position Size: {signal_data['position_size']:.2f} lots
• {expiry_status}

💡 AI Reasoning:
{signal_data['reasoning']}

⏰ Generated: {signal_data['timestamp']}
⏰ Expires: {expiry_time.strftime('%Y-%m-%d %H:%M:%S UTC')}

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

        if user:
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
        # Retrieve saved settings from session
        selected_symbol = session.get('selected_symbol','')
        selected_timeframe = session.get('selected_timeframe', 16385)  # H1 default
        selected_ai_provider = session.get('selected_ai_provider', 'anthropic')
        selected_risk_percent = session.get('selected_risk_percent', 1.0)

        return render_template('ai_trader.html',
                             selected_symbol=selected_symbol,
                             selected_timeframe=selected_timeframe,
                             selected_ai_provider=selected_ai_provider,
                             selected_risk_percent=selected_risk_percent)
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

@app.route('/mlm', methods=['GET', 'POST'])
@login_required
def mlm():
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('index'))

    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'add_vip':
            email = request.form.get('email')
            if email and '@' in email:
                if not MailingList.query.filter_by(email=email).first():
                    new_entry = MailingList(email=email, list_type='vip')
                    db.session.add(new_entry)
                    db.session.commit()
                    flash('VIP email added successfully', 'success')
                else:
                    flash('Email already exists', 'error')
            else:
                flash('Invalid email address', 'error')
        elif action == 'delete_vip':
            email_id = request.form.get('email_id')
            entry = MailingList.query.get(email_id)
            if entry:
                db.session.delete(entry)
                db.session.commit()
                flash('VIP email deleted', 'success')
        elif action == 'toggle_vip':
            email_id = request.form.get('email_id')
            entry = MailingList.query.get(email_id)
            if entry:
                entry.is_active = not entry.is_active
                db.session.commit()
                flash('VIP email status updated', 'success')
        elif action == 'change_plan':
            user_id = request.form.get('user_id')
            new_plan = request.form.get('new_plan')
            user = User.query.get(user_id)
            if user and new_plan in ['basic', 'pro', 'vip']:
                user.plan = new_plan
                db.session.commit()
                flash(f'User plan updated to {new_plan.title()}', 'success')
        elif action == 'toggle_user':
            user_id = request.form.get('user_id')
            user = User.query.get(user_id)
            if user:
                user.email_notifications = not user.email_notifications
                db.session.commit()
                flash('User email notifications updated', 'success')
        return redirect(url_for('mlm'))

    # GET request
    vip_emails = MailingList.query.filter_by(list_type='vip').all()
    users = User.query.all()
    return render_template('mlm.html', vip_emails=vip_emails, users=users)

@app.route('/run_ai_analysis', methods=['POST'])
@login_required
def run_ai_analysis():
    """Run AI analysis and return results"""
    global ea_instance, current_signal, trade_manager

    try:
        data = request.get_json()

        # Extract parameters
        symbol = data.get('symbol', '')
        timeframe = data.get('timeframe', 16385)  # H1 default
        ai_provider = data.get('aiProvider', 'anthropic')
        risk_percent = data.get('riskPercent', 1.0)

        # Save settings to session for state preservation
        session['selected_symbol'] = symbol
        session['selected_timeframe'] = timeframe
        session['selected_ai_provider'] = ai_provider
        session['selected_risk_percent'] = risk_percent

        # Get API key based on provider
        if ai_provider == 'anthropic':
            api_key = os.getenv('ANTHROPIC_API_KEY', '')
        elif ai_provider == 'openai':
            api_key = os.getenv('OPENAI_API_KEY', '')
        elif ai_provider == 'deepseek':
            api_key = os.getenv('DEEPSEEKER_API_KEY', '')
        elif ai_provider == 'grok':
            api_key = os.getenv('GROK_API_KEY', '')
        
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
            f"⏰ Signal validity: {signal.time_remaining()} (expires at {signal.expiry_timestamp.strftime('%H:%M:%S UTC')})",
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
                'timestamp': signal.timestamp.isoformat(),
                'expiry_timestamp': signal.expiry_timestamp.isoformat(),
                'is_expired': signal.is_expired(),
                'time_remaining': signal.time_remaining()
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

        # Send email alerts to all users with notifications enabled (separate from WhatsApp/Telegram)
        # Send for HIGH and MEDIUM confidence signals
        try:
            if signal.confidence.upper() in ["HIGH", "MEDIUM"]:
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
            logging.error(f"Error sending email alerts: {e}")
            print(f"⚠️  Error sending email alerts: {e}")

        # Send email alerts to VIP emails in database
        try:
            if signal.confidence.upper() in ["HIGH", "MEDIUM"]:
                vip_entries = MailingList.query.filter_by(list_type='vip', is_active=True).all()
                for entry in vip_entries:
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
                    }
                    send_signal_alert(entry.email, signal_alert_data)
        except Exception as e:
            logging.error(f"Error sending VIP email alerts: {e}")
            print(f"⚠️  Error sending VIP email alerts: {e}")

        # Send WhatsApp/Telegram notifications for HIGH and MEDIUM confidence signals (separate from email)
        try:
            if signal.confidence.upper() in ["HIGH", "MEDIUM"]:
                # Create notification manager instance for web app
                from app_ai_based import NotificationManager
                notification_manager = NotificationManager()
                notification_manager.send_signal_notification(signal)
        except Exception as e:
            logging.error(f"Error sending WhatsApp/Telegram notifications: {e}")
            print(f"⚠️  Error sending WhatsApp/Telegram notifications: {e}")

        # Log expiry information
        logging.info(f"Signal generated for {signal.symbol} - Expires at {signal.expiry_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"📅 Signal expiry: {signal.expiry_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')} ({signal.time_remaining()})")

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_current_signal')
@login_required
def get_current_signal():
    """Get the current trading signals (with and without patterns) and trade manager status"""
    global current_signal, trade_manager

    if current_signal is None:
        return jsonify({'error': 'No signal available'}), 404

    # For backward compatibility, return the signal with patterns as the main signal
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
        'indicators': current_signal.indicators,
        'is_expired': current_signal.is_expired(),
        'time_remaining': current_signal.time_remaining()
    }

    # Include trade manager performance stats if available
    if trade_manager:
        response['trade_manager_stats'] = trade_manager.get_performance_stats()

    return jsonify(response)

@app.route('/execute_trade', methods=['POST'])
@login_required
def execute_trade():
    """Execute the current trading signal using the advanced trade manager with fallback filling types"""
    global current_signal, trade_manager

    if current_signal is None:
        return jsonify({'error': 'No signal available'}), 400

    try:
        # Initialize trade manager if not already done
        if trade_manager is None:
            trade_manager = TradeManager()

        # Try different filling types in order: IOC -> FOK -> RETURN
        filling_types = ['IOC', 'FOK', 'RETURN']
        last_error = None
        ioc_error = None
        fok_error = None
        return_error = None

        for filling_type in filling_types:
            try:
                logging.info(f"Attempting trade execution with {filling_type} filling type")
                success = trade_manager.execute_trade(current_signal, filling_type)

                if success:
                    # Get updated performance stats
                    stats = trade_manager.get_performance_stats()
                    return jsonify({
                        'message': f'Trade executed successfully with {filling_type} filling type and advanced risk management',
                        'filling_type_used': filling_type,
                        'performance_stats': stats
                    })

            except Exception as e:
                last_error = str(e)
                logging.warning(f"Trade execution failed with {filling_type} filling type: {e}")
                # Store the specific error for this filling type
                if filling_type == 'IOC':
                    ioc_error = last_error
                elif filling_type == 'FOK':
                    fok_error = last_error
                elif filling_type == 'RETURN':
                    return_error = last_error
                continue

        # If all filling types failed
        error_details = []
        if 'ioc_error' in locals():
            error_details.append(f"IOC: {ioc_error}")
        if 'fok_error' in locals():
            error_details.append(f"FOK: {fok_error}")
        if 'return_error' in locals():
            error_details.append(f"RETURN: {return_error}")
        
        error_msg = f"Trade execution failed with all filling types. Specific errors:\n" + "\n".join(error_details)
        logging.error(error_msg)
        return jsonify({'error': error_msg}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_mt5_status')
@login_required
def get_mt5_status():
    """Get MT5 connection status and trade manager stats"""
    try:
        import MetaTrader5 as mt5
        mt5_connected = mt5.initialize()

        response = {'connected': bool(mt5_connected)}

        if mt5_connected:
            account_info = mt5.account_info()
            response['account'] = account_info.login if account_info else None
            mt5.shutdown()

        # Add trade manager performance stats if available
        global trade_manager
        if trade_manager:
            response['trade_manager_stats'] = trade_manager.get_performance_stats()

        return jsonify(response)
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
        all_symbols = mt5.symbols_get()
        symbols = [symbol.name for symbol in all_symbols if symbol.visible]

        print(f"Total symbols retrieved: {len(symbols)}")

        if symbols is None:
            mt5.shutdown()
            return jsonify({'error': 'No symbols available'}), 500

        # Filter for forex and major symbols
        filtered_symbols = []
        for symbol_name in symbols:
            # Include major forex pairs and common symbols
            if any(pair in symbol_name.upper() for pair in ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD','US']) or \
               'GOLD' in symbol_name.upper() or 'XAU' in symbol_name.upper() or \
               'VOLATILITY' in symbol_name.upper() or 'CRYPTO' in symbol_name.upper():
                # Get full symbol info for additional details
                symbol_info = mt5.symbol_info(symbol_name)
                if symbol_info:
                    filtered_symbols.append({
                        'name': symbol_name,
                        'description': getattr(symbol_info, 'description', symbol_name),
                        'path': getattr(symbol_info, 'path', ''),
                        'currency_base': getattr(symbol_info, 'currency_base', ''),
                        'currency_profit': getattr(symbol_info, 'currency_profit', ''),
                        'point': symbol_info.point,
                        'volume_min': symbol_info.volume_min,
                        'volume_max': symbol_info.volume_max
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
    """Get trading signals and trade manager performance stats"""
    try:
        # Get query parameters
        symbol = request.args.get('symbol')
        signal_type = request.args.get('signal_type')
        confidence = request.args.get('confidence')
        date_from = request.args.get('date_from')
        date_to = request.args.get('date_to')

        # For now, return the current signal if it matches filters
        # In a real implementation, you'd query a database
        global current_signal, trade_manager

        signals = []

        if current_signal is not None:
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

        # Include trade manager performance stats
        response = {'signals': signals}
        if trade_manager:
            response['performance_stats'] = trade_manager.get_performance_stats()

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create database tables and handle migrations
    with app.app_context():
        # Create all tables
        db.create_all()

        # Add email_notifications column if it doesn't exist (migration)
        try:
            with db.engine.connect() as conn:
                result = conn.execute(db.text("PRAGMA table_info(user)")).fetchall()
                columns = [row[1] for row in result]
                if 'email_notifications' not in columns:
                    print("Adding email_notifications column to User table...")
                    conn.execute(db.text('ALTER TABLE user ADD COLUMN email_notifications BOOLEAN DEFAULT 1'))
                    conn.commit()
                    print("Database migration completed")
        except Exception as e:
            print(f"Error during email_notifications migration: {e}")

        # Add plan column if it doesn't exist (migration)
        try:
            with db.engine.connect() as conn:
                result = conn.execute(db.text("PRAGMA table_info(user)")).fetchall()
                columns = [row[1] for row in result]
                if 'plan' not in columns:
                    print("Adding plan column to User table...")
                    conn.execute(db.text("ALTER TABLE user ADD COLUMN plan VARCHAR(50) DEFAULT 'basic'"))
                    conn.commit()
                    print("Plan column migration completed")
        except Exception as e:
            print(f"Error during plan migration: {e}")

        # Create ActivityLog table if it doesn't exist
        try:
            ActivityLog.query.first()
        except Exception as e:
            if 'no such table' in str(e):
                print("Creating ActivityLog table...")
                db.create_all()
                print("ActivityLog table created")

        # Migrate VIP emails from file to database
        try:
            with open('static/vip_email.txt', 'r') as f:
                vip_emails = [line.strip() for line in f if line.strip()]
            for email in vip_emails:
                if not MailingList.query.filter_by(email=email, list_type='vip').first():
                    new_entry = MailingList(email=email, list_type='vip')
                    db.session.add(new_entry)
            db.session.commit()
            print("VIP emails migrated to database")
        except Exception as e:
            print(f"Error migrating VIP emails: {e}")

    print("Starting AI Trading Signal Generator Web Interface with Advanced Trade Management...")
    print("Features:")
    print("- AI-powered signal generation (HIGH/MEDIUM confidence)")
    print("- Advanced trade execution with risk management")
    print("- Win rate monitoring (target: 80%)")
    print("- Partial take profit closures")
    print("- Multi-channel notifications (Email/WhatsApp/Telegram)")
    print("- Session state preservation")
    print("")
    print("Open your browser to http://localhost:9000")
    app.run(debug=True, host='0.0.0.0', port=9000)