"""
Telegram Bot for AI Trading Signal Notifications
Handles Telegram messaging for trading signals
"""

import os
import logging
import requests
from dotenv import load_dotenv
import time
import telegram

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    filename='mail_notifications.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class TelegramBot:
    """Handles Telegram notifications for trading signals"""

    def __init__(self):
        # Telegram configuration
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")

        # Rate limiting
        self.last_notification_time = 0
        self.min_interval_seconds = 60  # Minimum 1 minute between notifications

    def send_message(self, message: str, max_retries: int = 3) -> bool:
        """Send message via Telegram with retry logic"""
        if not self.bot_token or not self.chat_id:
            logging.warning("Telegram configuration missing - skipping notification")
            return False

        for attempt in range(max_retries):
            try:
                url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
                data = {
                    "chat_id": self.chat_id,
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
                        import time
                        time.sleep(2 ** attempt)
            except Exception as e:
                logging.warning(f"Telegram send attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)

        logging.error(f"Telegram send failed after {max_retries} attempts")
        return False
telegram_bot = TelegramBot()