# BIGBOLZ Trading Bot

## Overview

BIGBOLZ is a cryptocurrency trading signal bot (v11) that monitors multiple trading pairs on the MEXC exchange and sends automated alerts via Telegram. The bot implements a technical analysis strategy using candlestick pattern detection to identify bullish and bearish signals across multiple timeframes. It features advanced filtering mechanisms including cooldown periods, spam detection, signal strength validation, and risk assessment to ensure high-quality trading signals.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Application Architecture
**Decision**: Hybrid polling + webhook architecture using Flask for web server and asyncio for bot logic  
**Rationale**: The bot runs continuously in a polling loop to check market conditions while maintaining a Flask web server (likely for health checks or webhooks). This allows the application to run on platforms like Replit that expect HTTP endpoints while performing scheduled market analysis.

**Alternatives considered**: Pure webhook-based or pure polling  
**Pros**: Compatible with cloud hosting platforms, allows both scheduled tasks and HTTP endpoints  
**Cons**: Slightly higher resource usage due to running multiple threads

### Data Processing & Analysis
**Decision**: Real-time OHLCV (candlestick) data analysis with configurable lookback periods  
**Rationale**: Uses 4-hour timeframe candlestick data with up to 500 historical bars for technical analysis. Implements custom pattern detection algorithms rather than relying on third-party indicator libraries.

**Key parameters**:
- Timeframe: 4 hours
- Historical lookback: 500 bars maximum
- Pattern detection: 11-period length with configurable bull/bear thresholds
- Body-based vs wick-based candle analysis (configurable)

**Pros**: Full control over analysis logic, optimized for specific strategy  
**Cons**: Requires manual maintenance of technical indicators

### Signal Generation & Filtering
**Decision**: Multi-layered filtering system to reduce false signals  
**Rationale**: Implements several filtering mechanisms to improve signal quality:

1. **Cooldown System**: 15-bar cooldown between signals per symbol to prevent over-trading
2. **Invalidation Window**: 5-bar window to invalidate conflicting signals
3. **Anti-Spam Detection**: Tracks signal frequency (3 signals within 15 bars triggers 20-bar cooldown)
4. **Risk Assessment**: Flags "risky" signals based on recent signal density (2+ signals in 20 bars)
5. **Strong Signal Detection**: Identifies high-confidence signals using 15% threshold percentage

**Pros**: Reduces noise and false positives significantly  
**Cons**: May miss some valid trading opportunities due to aggressive filtering

### State Management
**Decision**: JSON file-based persistence for bot state  
**Rationale**: Stores signal history, cooldowns, and timestamps in `bigbolz_v11_state.json` to maintain context across restarts.

**State includes**:
- Last signal timestamps per symbol
- Signal types (bull/bear)
- Cooldown tracking
- Historical signal counts for spam detection

**Pros**: Simple, no database dependency, easy to debug  
**Cons**: Not suitable for high-frequency updates or distributed systems

### Rate Limiting & API Management
**Decision**: Batched requests with delays to respect exchange rate limits  
**Rationale**: MEXC exchange configured with 1200ms rate limit. Implements additional 0.1s delay between requests and processes symbols in batches of 10.

**Pros**: Prevents API bans, maintains stable connection  
**Cons**: Slower data retrieval for large symbol lists

### Multi-Symbol Monitoring
**Decision**: Sequential symbol processing with batch optimization  
**Rationale**: Monitors 14+ cryptocurrency pairs including major coins (BTC, ETH, SOL) and altcoins. Processes symbols sequentially with batching to balance speed and rate limits.

**Pros**: Diversified signal coverage, adaptable symbol list  
**Cons**: Processing time scales linearly with symbol count

### Notification System
**Decision**: Telegram Bot API for alert delivery  
**Rationale**: Uses environment variables for bot token and chat ID to send formatted trading signals with emojis and signal metadata.

**Alert includes**:
- Signal type (bull/bear)
- Symbol
- Risk level flags
- Strength indicators
- Timestamp

**Pros**: Real-time mobile notifications, widely used platform  
**Cons**: Dependent on Telegram API availability

### Configuration Management
**Decision**: Environment variables for sensitive data, constants for strategy parameters  
**Rationale**: Bot token and chat ID stored as environment variables. All strategy parameters (timeframes, thresholds, filters) defined as module-level constants for easy tuning.

**Pros**: Secure credential management, easy parameter adjustment  
**Cons**: Requires restart to change strategy parameters

## External Dependencies

### Cryptocurrency Exchange
- **MEXC Exchange** (via ccxt library)
  - Purpose: Market data retrieval and price information
  - Data: OHLCV candlestick data for technical analysis
  - Rate limit: 1200ms between requests

### Messaging Platform
- **Telegram Bot API**
  - Purpose: Real-time signal notifications
  - Authentication: Bot token (environment variable)
  - Target: Chat ID (environment variable)

### Python Libraries
- **ccxt**: Exchange connectivity and market data
- **numpy**: Numerical computations for technical analysis
- **requests**: HTTP client for Telegram API
- **Flask**: Web server framework (health checks/webhooks)
- **asyncio**: Asynchronous programming for concurrent operations

### File System
- **State persistence**: JSON file (`bigbolz_v11_state.json`) for maintaining bot state across restarts

### Environment Variables Required
- `BOT_TOKEN`: Telegram bot authentication token
- `CHAT_ID`: Telegram chat/channel ID for notifications

## Deployment Status

**Current Status**: ✅ DEPLOYED AND RUNNING

### Deployment Details (October 14, 2025)
- **Platform**: Replit (24/7 hosting)
- **Workflow**: Configured to run `python main.py` automatically
- **Web Server**: Flask running on port 5000 for keep-alive
- **Bot Status**: Successfully monitoring 300+ trading pairs on MEXC
- **Environment**: All secrets (BOT_TOKEN, CHAT_ID) configured securely
- **State Management**: JSON file (`bigbolz_v11_state.json`) for persistence

### How It Works
1. Bot runs continuously in the background
2. Monitors cryptocurrency pairs on MEXC exchange every 2 seconds
3. Analyzes 4-hour candlestick patterns using TradingView indicator logic
4. Sends instant Telegram alerts when signals are detected
5. Maintains state across restarts for continuity

### Monitoring
- Flask web interface available at port 5000 showing bot status
- Logs track historical data building and signal generation
- State file preserves signal history and cooldown tracking