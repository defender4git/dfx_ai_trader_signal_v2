# AI Trading Expert Advisor

A professional AI-powered trading system that analyzes technical indicators and generates trading signals using multiple AI providers. Features a modern web interface for real-time market analysis and automated trading execution.

## 🚀 Features

### Core Functionality
- **Multi-Provider AI Analysis**: Support for 5 different AI providers (Anthropic Claude, OpenAI GPT, DeepSeek, Grok, Kilo Code)
- **Real-time Market Analysis**: Live technical indicator calculations (ATR, RSI, MACD, CCI, Stochastic)
- **Professional Trading Signals**: Complete trade parameters with risk management
- **Dynamic Symbol Loading**: Automatically loads available symbols from connected MT5 broker
- **Token Usage Tracking**: Real-time monitoring of API usage and TPM limits
- **Web-based Interface**: Modern, responsive UI built with Tailwind CSS

### Technical Indicators
- Average True Range (ATR) with volatility classification
- Relative Strength Index (RSI)
- MACD with signal line
- Commodity Channel Index (CCI)
- Stochastic Oscillator
- Market bias analysis (bullish/bearish)

### Risk Management
- Configurable risk per trade (0.1% - 5.0%)
- Position size calculation based on account balance
- Stop loss and take profit levels
- Volatility-adjusted position sizing

## 📋 Prerequisites

### System Requirements
- Windows 10/11
- Python 3.8+
- MetaTrader 5 terminal installed and running
- Active internet connection for AI API calls

### Required Accounts & API Keys
1. **MetaTrader 5**: Trading account with market data access
2. **AI Provider API Keys** (choose at least one):
   - Anthropic Claude API key
   - OpenAI API key
   - DeepSeek API key
   - xAI Grok API key
   - Kilo Code API key

## 🛠️ Installation

### 1. Clone or Download
```bash
# Navigate to your project directory
cd AItrader/
```

### 2. Install Dependencies
```bash
# Install required Python packages
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the AItrader directory:

```env
# AI Provider API Keys (configure at least one)
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
OPENAI_API_KEY=sk-proj-your-openai-key-here
DEEPSEEKER_API_KEY=sk-your-deepseek-key-here
GROK_API_KEY=xai-your-grok-key-here
KILO_CODE_API_KEY=your-kilo-code-key-here

# Default AI Provider (anthropic, openai, deepseek, grok, kilo_code)
AI_PROVIDER=anthropic
```

### 4. Configure MetaTrader 5
- Ensure MetaTrader 5 is installed and running
- Enable automated trading in MT5 (Algo Trading button)
- Allow WebRequest from localhost in MT5 options

## 🚀 Usage

### Web Interface (Recommended)
```bash
# Run the web application
python web_app.py
```

Open your browser and navigate to `http://localhost:9000`

#### Web Interface Features:
1. **Configuration Panel**:
   - Select trading symbol (dynamically loaded from MT5)
   - Choose timeframe (M1, M5, M15, M30, H1, H4, D1)
   - Select AI provider
   - Set risk percentage per trade

2. **Real-time Analysis**:
   - Start/Stop analysis with automatic intervals
   - View current trading signals
   - Monitor technical indicators
   - Track token usage and TPM limits

3. **System Status**:
   - MT5 connection status
   - AI service availability
   - Analysis state

### Command Line Interface
```bash
# Run the console application
python app_ai_based.py
```

## 📊 Configuration Options

### AI Providers
- **Anthropic Claude**: High-quality analysis with balanced token usage
- **OpenAI GPT**: Fast responses with good accuracy
- **DeepSeek**: Cost-effective alternative with good performance
- **Grok**: Innovative analysis with unique insights
- **Kilo Code**: Specialized trading-focused AI

### Timeframes
- M1 (1 minute) - Scalping
- M5 (5 minutes) - Short-term
- M15 (15 minutes) - Short-term
- M30 (30 minutes) - Medium-term
- H1 (1 hour) - Medium-term
- H4 (4 hours) - Long-term
- D1 (Daily) - Long-term

### Risk Management
- Risk per trade: 0.1% to 5.0% of account balance
- Automatic position sizing based on stop loss distance
- Volatility-adjusted risk calculations

## 🔧 API Reference

### Endpoints

#### `GET /`
Main web interface

#### `POST /run_ai_analysis`
Run market analysis with specified parameters
```json
{
  "symbol": "XAUUSD",
  "timeframe": 16385,
  "aiProvider": "anthropic",
  "riskPercent": 1.0
}
```

#### `GET /get_symbols`
Retrieve available trading symbols from MT5

#### `GET /get_mt5_status`
Check MT5 connection status

#### `POST /execute_trade`
Execute a trading signal
```json
{
  "auto_execute": false
}
```

## 📈 Token Usage Optimization

The system is optimized for efficient token usage:
- **Condensed prompts**: ~50% reduction in input tokens
- **Structured responses**: Consistent JSON output format
- **Real-time tracking**: Monitor usage vs. TPM limits
- **Provider selection**: Choose based on cost/performance needs

### TPM Limits (approximate)
- Anthropic Claude: 50,000 tokens/minute
- OpenAI GPT-4: 10,000 tokens/minute
- DeepSeek: 100,000 tokens/minute
- Grok: 50,000 tokens/minute
- Kilo Code: Varies by plan

## 🛡️ Security & Best Practices

### API Key Security
- Store API keys in `.env` file (never commit to version control)
- Use environment variables for sensitive data
- Rotate keys regularly

### Trading Safety
- Start with small position sizes
- Test on demo account first
- Monitor account balance and risk exposure
- Enable proper stop losses

### System Monitoring
- Check MT5 connection status regularly
- Monitor token usage to avoid rate limits
- Keep logs for analysis and debugging

## 🐛 Troubleshooting

### Common Issues

#### MT5 Connection Failed
- Ensure MetaTrader 5 is running
- Check if automated trading is enabled
- Verify WebRequest permissions for localhost

#### AI API Errors
- Verify API keys are correct in `.env`
- Check internet connection
- Monitor token usage limits

#### Symbol Not Available
- Refresh symbol list in web interface
- Check if symbol is enabled in MT5
- Verify broker offers the selected symbol

#### Token Limit Exceeded
- Switch to different AI provider
- Reduce analysis frequency
- Upgrade API plan if needed

### Logs and Debugging
- Check console output for error messages
- Review analysis logs in web interface
- Monitor token usage in footer

## 📝 License

This project is for educational and personal use. Please ensure compliance with your broker's terms of service and local regulations regarding automated trading.

## ⚠️ Disclaimer

This software is provided as-is for educational purposes. Trading involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results. Always test thoroughly on a demo account before using with real money.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Test thoroughly
4. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review logs for error messages
3. Ensure all prerequisites are met
4. Test with demo account first

---

**Happy Trading!** 📈💰