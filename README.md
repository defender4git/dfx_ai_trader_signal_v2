<<<<<<< HEAD
# 🤖 AI Trading Signal Generator
=======
# AI Trading Signal Generator
>>>>>>> fde187184aa8c6c5ea01ae626ab26ece046c86c8

<div align="center">

![AI Trading Signal Generator Advisor](https://img.shields.io/badge/AI--Trading-Expert--Advisor-blue?style=for-the-badge&logo=brain&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.3+-000000?style=flat-square&logo=flask&logoColor=white)
![MetaTrader 5](https://img.shields.io/badge/MetaTrader--5-FFD700?style=flat-square&logo=metatrader&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

<<<<<<< HEAD
**Professional AI-Powered Trading System with Modern Web Interface**
=======
### Core Functionality
- **Multi-Provider AI Analysis**: Support for 5 different AI providers (Anthropic Claude, OpenAI GPT, DeepSeek, Grok)
- **Real-time Market Analysis**: Live technical indicator calculations (ATR, RSI, MACD, CCI, Stochastic)
- **Professional Trading Signals**: Complete trade parameters with risk management
- **Dynamic Symbol Loading**: Automatically loads available symbols from connected MT5 broker
- **Token Usage Tracking**: Real-time monitoring of API usage and TPM limits
- **Web-based Interface**: Modern, responsive UI built with Tailwind CSS
>>>>>>> fde187184aa8c6c5ea01ae626ab26ece046c86c8

*Real-time market analysis, automated trading signals, and comprehensive risk management*

[🚀 Quick Start](#-installation) • [📖 Documentation](#-features) • [🔧 API Reference](#-api-reference) • [📞 Support](#-contact)

</div>

---

## 📋 Table of Contents

- [✨ Overview](#-overview)
- [🚀 Key Features](#-key-features)
- [🛠️ System Requirements](#️-system-requirements)
- [📦 Installation](#-installation)
- [⚙️ Configuration](#️-configuration)
- [🎯 Usage](#-usage)
- [🔧 API Reference](#-api-reference)
- [📊 Token Management](#-token-management)
- [🛡️ Security & Best Practices](#️-security--best-practices)
- [🐛 Troubleshooting](#-troubleshooting)
- [📈 Performance Optimization](#-performance-optimization)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [⚠️ Disclaimer](#️-disclaimer)
- [📞 Contact](#-contact)

---

## ✨ Overview

**AI Trading Signal Generator** is a cutting-edge, professional-grade trading system that leverages multiple AI providers to deliver sophisticated market analysis and automated trading signals. Built with modern web technologies and comprehensive risk management, it provides traders with institutional-quality tools in an accessible, user-friendly interface.

The system integrates seamlessly with MetaTrader 5, offering real-time technical analysis, multi-timeframe support, and intelligent position sizing with advanced risk controls.

---

## 🚀 Key Features

### 🤖 AI-Powered Analysis
- **Multi-Provider Support**: Anthropic Claude, OpenAI GPT, DeepSeek, Grok, and Kilo Code
- **Intelligent Signal Generation**: Context-aware trading signals with confidence scoring
- **Adaptive Analysis**: Dynamic market condition assessment and strategy adjustment

### 📊 Technical Analysis Suite
- **Complete Indicator Set**: ATR, RSI, MACD, CCI, Stochastic Oscillator
- **Volatility Analysis**: Real-time volatility classification and adjustment
- **Market Bias Detection**: Bullish/bearish trend identification
- **Multi-Timeframe Support**: M1 to D1 analysis capabilities

### 🖥️ Modern Web Interface
- **Responsive Design**: Optimized for desktop, tablet, and mobile devices
- **Real-Time Updates**: Live market data and signal monitoring
- **User Management**: Secure authentication and personalized dashboards
- **Token Tracking**: Comprehensive API usage monitoring

### 🛡️ Risk Management
- **Dynamic Position Sizing**: Account balance and volatility-based calculations
- **Configurable Risk Parameters**: 0.1% to 5.0% per trade
- **Advanced Stop Loss**: ATR-based protective stops
- **Multiple Take Profit Levels**: R:R optimization with 1:1, 1:2, 1:3 targets

### 🔗 Integration Capabilities
- **MT5 Direct Connection**: Native MetaTrader 5 integration
- **Dynamic Symbol Loading**: Automatic broker symbol detection
- **WebSocket Updates**: Real-time data streaming
- **RESTful API**: Complete programmatic access

---

## 🛠️ System Requirements

### Minimum Requirements
- **Operating System**: Windows 10/11 (64-bit)
- **Python Version**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 500MB free space
- **Network**: Stable internet connection

### Required Software
- **MetaTrader 5 Terminal**: Latest version installed and configured
- **Python Environment**: Virtual environment recommended
- **Web Browser**: Modern browser with JavaScript enabled

### API Access
- **AI Provider Account**: At least one supported AI service
- **Trading Account**: MetaTrader 5 account with market data access
- **Broker Permissions**: Automated trading enabled

---

## 📦 Installation

### 1. Environment Setup

```bash
# Clone or download the project
git clone https://github.com/your-repo/ai-trading-ea.git
cd ai-trading-ea

# Create virtual environment (recommended)
python -m venv trading_env
trading_env\Scripts\activate  # Windows
```

### 2. Dependencies Installation

```bash
# Install required packages
pip install -r requirements.txt

# Verify installation
python -c "import MetaTrader5, flask, anthropic; print('✅ All dependencies installed')"
```

### 3. MetaTrader 5 Configuration

1. **Install MT5 Terminal**: Download from official MetaQuotes website
2. **Enable Automation**:
   - Launch MetaTrader 5
   - Click "Algo Trading" button in toolbar
   - Enable automated trading
3. **Configure Web Access**:
   - Tools → Options → Expert Advisors
   - Check "Allow automated trading"
   - Add `localhost` to WebRequest URLs

---

## ⚙️ Configuration

### Environment Variables Setup

Create a `.env` file in the project root:

```env
# ===========================================
# AI TRADING EXPERT ADVISOR CONFIGURATION
# ===========================================

# AI PROVIDER API KEYS
# Configure at least one provider
ANTHROPIC_API_KEY=sk-ant-api3-your-anthropic-key-here
OPENAI_API_KEY=sk-proj-your-openai-key-here
DEEPSEEKER_API_KEY=sk-your-deepseek-key-here
GROK_API_KEY=xai-your-grok-key-here
KILO_CODE_API_KEY=your-kilo-code-key-here

# APPLICATION SETTINGS
SECRET_KEY=your-secure-random-secret-key-here
AI_PROVIDER=anthropic
DEBUG=False

# TRADING PARAMETERS
DEFAULT_RISK_PERCENT=1.0
DEFAULT_TIMEFRAME=16385
DEFAULT_SYMBOL=XAUUSD
```

### AI Provider Selection Guide

| Provider | Best For | Token Efficiency | Cost |
|----------|----------|------------------|------|
| **Anthropic Claude** | Balanced analysis, reliability | High | Medium |
| **OpenAI GPT** | Speed, consistency | Medium | Medium |
| **DeepSeek** | Cost optimization | High | Low |
| **Grok** | Innovative insights | Medium | Low |
| **Kilo Code** | Trading specialization | High | Variable |

---

## 🎯 Usage

### Web Interface (Recommended)

```bash
# Start the web application
python web_app.py
```

Navigate to `http://localhost:9000` and log in to access the trading dashboard.

#### Dashboard Features:
- **📊 Real-time Analysis**: Live market data and indicator monitoring
- **🎯 Signal Generation**: AI-powered trading signals with full parameters
- **📱 Responsive Design**: Optimized for all device types
- **👤 User Management**: Secure authentication and profile management
- **📈 Performance Tracking**: Historical signal analysis and filtering

### Command Line Operation

```bash
# Run console-based analysis
python app_ai_based.py
```

### Configuration Panel Options

1. **Symbol Selection**: Dynamic loading from MT5 broker
2. **Timeframe**: M1, M5, M15, M30, H1, H4, D1
3. **AI Provider**: Select based on preference and token limits
4. **Risk Management**: 0.1% to 5.0% per trade
5. **Analysis Control**: Start/stop automated analysis

---

## 🔧 API Reference

### Authentication Endpoints

#### `POST /login`
User authentication
```json
{
  "username": "trader_username",
  "password": "secure_password"
}
```

#### `POST /register`
User registration
```json
{
  "username": "trader_username",
  "email": "trader@example.com",
  "password": "secure_password"
}
```

### Trading Endpoints

#### `POST /run_ai_analysis`
Execute market analysis
```json
{
  "symbol": "XAUUSD",
  "timeframe": 16385,
  "aiProvider": "anthropic",
  "riskPercent": 1.0
}
```

**Response:**
```json
{
  "signal": {
    "symbol": "XAUUSD",
    "signal_type": "LONG",
    "entry_price": 1950.25000,
    "stop_loss": 1945.50000,
    "take_profit_1": 1955.00000,
    "position_size": 0.05,
    "confidence": "HIGH",
    "reasoning": "Strong bullish momentum with RSI divergence"
  },
  "analysis": {
    "atr": 4.75,
    "rsi": 65.2,
    "macd": 2.45,
    "volatility_level": "Medium Volatility"
  },
  "token_usage": {
    "input_tokens": 125,
    "output_tokens": 89,
    "total_tokens": 214
  }
}
```

#### `GET /get_symbols`
Retrieve available trading symbols

#### `GET /get_user_signals`
Get filtered trading signals for authenticated user
```json
{
  "symbol": "XAUUSD",
  "signal_type": "LONG",
  "confidence": "HIGH",
  "date_from": "2024-01-01",
  "date_to": "2024-12-31"
}
```

#### `POST /execute_trade`
Execute trading signal
```json
{
  "auto_execute": false
}
```

---

## 📊 Token Management

### Usage Optimization
- **Condensed Prompts**: 50% reduction in input tokens through optimized prompts
- **Structured Output**: Consistent JSON responses for efficient parsing
- **Real-time Monitoring**: Live token usage tracking and TPM limit alerts
- **Provider Switching**: Automatic failover based on limits and performance

### TPM Limits by Provider

| Provider | TPM Limit | Recommended Usage |
|----------|-----------|-------------------|
| Anthropic Claude | 50,000 | Balanced analysis |
| OpenAI GPT-4 | 10,000 | Fast responses |
| DeepSeek | 100,000 | High-frequency |
| Grok | 50,000 | Innovative analysis |
| Kilo Code | Variable | Specialized trading |

### Cost Optimization Strategies
1. **Provider Selection**: Choose based on analysis requirements
2. **Analysis Frequency**: Adjust based on timeframe and market conditions
3. **Batch Processing**: Group analysis requests when possible
4. **Caching**: Implement signal caching for repeated analysis

---

<<<<<<< HEAD
## 🛡️ Security & Best Practices

### 🔐 API Security
- **Environment Variables**: Never commit API keys to version control
- **Key Rotation**: Regular key updates and monitoring
- **Access Control**: IP whitelisting and rate limiting
- **Encryption**: Secure storage of sensitive data

### 📈 Trading Safety
- **Demo Testing**: Always test on demo account first
- **Gradual Scaling**: Start with small position sizes
- **Risk Limits**: Never exceed predetermined risk thresholds
- **Monitoring**: Continuous account balance and P&L monitoring

### 🔧 System Security
- **Regular Updates**: Keep dependencies and MT5 updated
- **Backup Strategy**: Regular configuration and data backups
- **Access Logging**: Monitor system access and API usage
- **Error Handling**: Comprehensive error logging and alerting

---

## 🐛 Troubleshooting

### Common Issues & Solutions

#### MT5 Connection Problems
```bash
# Check MT5 status
python -c "import MetaTrader5 as mt5; print(mt5.initialize())"
```
- **Solutions**:
  - Ensure MT5 is running
  - Enable automated trading
  - Add localhost to WebRequest URLs
  - Check firewall settings

#### AI API Errors
- **Rate Limits**: Switch providers or reduce frequency
- **Invalid Keys**: Verify API keys in `.env` file
- **Network Issues**: Check internet connectivity
- **Token Limits**: Monitor usage and upgrade plans if needed

#### Symbol Loading Issues
- **Refresh Interface**: Reload the web interface
- **Broker Verification**: Confirm symbol availability with broker
- **MT5 Synchronization**: Restart MT5 terminal

### Debug Mode
Enable debug logging:
```bash
export DEBUG=True
python web_app.py
```

### Log Analysis
- Check console output for error messages
- Review web interface analysis logs
- Monitor token usage in dashboard footer
- Examine MT5 terminal logs

---

## 📈 Performance Optimization

### System Performance
- **Memory Management**: Efficient data structures and cleanup
- **CPU Optimization**: Asynchronous processing for API calls
- **Network Efficiency**: Connection pooling and retry logic
- **Database Optimization**: Indexed queries and connection pooling

### Trading Performance
- **Signal Quality**: AI provider selection based on market conditions
- **Execution Speed**: Optimized order placement and confirmation
- **Risk Assessment**: Real-time position sizing calculations
- **Market Adaptation**: Dynamic parameter adjustment

### Monitoring & Metrics
- **Response Times**: API call latency tracking
- **Success Rates**: Signal accuracy and execution success
- **Resource Usage**: Memory, CPU, and network monitoring
- **Error Rates**: Comprehensive error tracking and alerting

---

## 🤝 Contributing

We welcome contributions from the trading and development community!

### Development Process
1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** changes (`git commit -m 'Add amazing feature'`)
4. **Push** to branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Contribution Guidelines
- **Code Quality**: Follow PEP 8 standards
- **Testing**: Include unit tests for new features
- **Documentation**: Update README and docstrings
- **Security**: Never commit sensitive data or API keys
- **Compatibility**: Test across different MT5 versions and Python environments

### Feature Requests
- Use GitHub Issues for feature requests
- Provide detailed use cases and benefits
- Include mockups or examples when possible

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

**Educational and Personal Use**: This software is provided for educational purposes and personal trading use. Commercial use requires explicit permission from the author.

---

## ⚠️ Disclaimer

**IMPORTANT RISK WARNING**

This software is provided "as-is" for educational and informational purposes only. Trading foreign exchange and other financial instruments involves substantial risk of loss and is not suitable for all investors.

**Key Points:**
- Past performance does not guarantee future results
- Always test thoroughly on a demo account before live trading
- Never risk more than you can afford to lose
- Consult with financial advisors before using automated trading systems
- The author assumes no responsibility for trading losses or damages

**Regulatory Compliance**: Ensure compliance with your local regulations regarding automated trading systems and algorithmic trading.

---

## 📞 Contact

### Developer Information

**Kilo Code**  
*AI-Powered Software Development Agent*

### Support Channels

- **📧 Email**: support@kilocode.com
- **🐛 Issues**: [GitHub Issues](https://github.com/kilocode/ai-trading-ea/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/kilocode/ai-trading-ea/discussions)
- **📖 Documentation**: [Wiki](https://github.com/kilocode/ai-trading-ea/wiki)

### Professional Services

- **Custom Development**: AI-powered trading system customization
- **Integration Services**: Third-party platform integration
- **Consulting**: Trading strategy optimization and risk management
- **Training**: Automated trading system development workshops

### Response Time
- **Bug Reports**: Within 24 hours
- **Feature Requests**: Within 48 hours
- **General Inquiries**: Within 72 hours

---

<div align="center">

**⚡ Powered by AI • Built for Traders • Designed for Performance**

---

*Happy Trading with AI Trading Signal Generator!* 🚀📈

</div>
=======
**Happy Trading!** 📈💰
>>>>>>> fde187184aa8c6c5ea01ae626ab26ece046c86c8
