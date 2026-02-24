"""
Telegram Bot Integration for Polymarket ARB Bot
- Sends notifications on new bets, position resolves, scan results
- Supports commands: /status, /positions, /history, /start, /stop, /scan
- Runs as async background task alongside the web server
"""

import asyncio
import aiohttp
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("telegram-bot")

# Load env
PROJECT_ROOT = Path(__file__).parent.parent
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")


class TelegramBot:
    """Telegram bot for Polymarket ARB Bot notifications and commands"""

    def __init__(self, token: str = "", chat_id: str = ""):
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self.base_url = f"https://api.telegram.org/bot{self.token}" if self.token else ""
        self.enabled = bool(self.token and self.chat_id)
        self._polling_task = None
        self._last_update_id = 0
        self._sim_engine = None
        self._auto_trader = None
        self._live_cache = None

        if self.enabled:
            log.info(f"Telegram bot enabled (chat_id: {self.chat_id})")
        else:
            log.info("Telegram bot disabled (no token/chat_id configured)")

    def set_engines(self, sim_engine, auto_trader, live_cache=None):
        """Connect bot to trading engines"""
        self._sim_engine = sim_engine
        self._auto_trader = auto_trader
        self._live_cache = live_cache

    # ─── SEND MESSAGES ────────────────────────────────────────────

    async def send_message(self, text: str, parse_mode: str = "HTML"):
        """Send a message to the configured chat"""
        if not self.enabled:
            return False
        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                resp = await session.post(f"{self.base_url}/sendMessage", json={
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": parse_mode,
                    "disable_web_page_preview": True,
                })
                data = await resp.json()
                if not data.get("ok"):
                    log.warning(f"Telegram send failed: {data.get('description', 'unknown error')}")
                    return False
                return True
        except Exception as e:
            log.error(f"Telegram send error: {e}")
            return False

    # ─── NOTIFICATION METHODS ────────────────────────────────────

    async def notify_new_bet(self, bet_info: dict):
        """Notify when a new bet is placed"""
        side = bet_info.get('side', '?')
        amount = bet_info.get('amount', 0)
        price = bet_info.get('entry_price', 0)
        edge = bet_info.get('edge_at_entry', 0)
        market = bet_info.get('market', 'Unknown')[:60]
        mtype = bet_info.get('type', '').upper()

        text = (
            f"🎲 <b>NEW BET</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"📌 {market}\n"
            f"📊 Side: <b>{side}</b> | Type: {mtype}\n"
            f"💰 Amount: <b>${amount:.2f}</b> @ {price:.3f}\n"
            f"📈 Edge: {edge*100:+.1f}%"
        )
        await self.send_message(text)

    async def notify_resolve(self, result: dict, new_balance: float = 0):
        """Notify when a position is resolved/sold"""
        pnl = result.get('pnl', 0)
        status = result.get('status', 'unknown').upper()
        market = result.get('market', 'Unknown')[:60]
        outcome = result.get('outcome', '?')

        emoji = "✅" if pnl >= 0 else "❌"
        text = (
            f"{emoji} <b>POSITION CLOSED</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"📌 {market}\n"
            f"📊 Result: <b>{status}</b> → {outcome}\n"
            f"💰 PnL: <b>{'+'if pnl>=0 else ''}${pnl:.2f}</b>\n"
            f"💼 Balance: ${new_balance:.2f}"
        )
        await self.send_message(text)

    async def notify_scan_complete(self, scan_result: dict):
        """Notify on scan cycle completion (only if bets were placed or resolved)"""
        bets = scan_result.get('bets_placed', 0)
        resolved = scan_result.get('resolved', 0)
        total_opps = scan_result.get('total_opportunities', 0)
        cycle = scan_result.get('cycle', 0)
        portfolio = scan_result.get('portfolio', {})

        # Only send notification if something happened
        if bets == 0 and resolved == 0:
            return

        text = (
            f"🔍 <b>SCAN #{cycle} COMPLETE</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"📊 Opportunities: {total_opps}\n"
            f"🎲 Bets placed: <b>{bets}</b>\n"
            f"✅ Resolved: <b>{resolved}</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"💼 Balance: ${portfolio.get('balance', 0):.2f}\n"
            f"📈 Total PnL: {'+'if portfolio.get('total_pnl',0)>=0 else ''}${portfolio.get('total_pnl', 0):.2f}\n"
            f"📊 Win Rate: {portfolio.get('win_rate', 0):.1f}%\n"
            f"📌 Open: {portfolio.get('open_positions', 0)} positions"
        )
        await self.send_message(text)

    async def send_portfolio_status(self):
        """Send current portfolio status"""
        if not self._sim_engine:
            await self.send_message("⚠️ Engine not connected")
            return

        p = self._sim_engine.get_portfolio_summary()
        positions = self._sim_engine.positions

        # Format open positions
        pos_text = ""
        if positions:
            pos_lines = []
            for pos in positions[:5]:
                pos_lines.append(
                    f"  • {pos.get('side','')} ${pos.get('amount',0):.2f} @ "
                    f"{pos.get('entry_price',0):.3f} | {pos.get('market','')[:35]}"
                )
            pos_text = "\n".join(pos_lines)
        else:
            pos_text = "  No open positions"

        text = (
            f"📊 <b>PORTFOLIO STATUS</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"💰 Balance: <b>${p.get('balance', 0):.2f}</b>\n"
            f"💼 Portfolio Value: ${p.get('portfolio_value', 0):.2f}\n"
            f"📈 Total PnL: <b>{'+'if p.get('total_pnl',0)>=0 else ''}${p.get('total_pnl', 0):.2f}</b> "
            f"({p.get('pnl_pct', 0):+.1f}%)\n"
            f"📊 Win Rate: {p.get('win_rate', 0):.1f}%\n"
            f"🔢 Trades: {p.get('closed_trades', 0)} closed | {p.get('open_positions', 0)} open\n"
            f"━━━━━━━━━━━━━━━\n"
            f"<b>Open Positions:</b>\n{pos_text}"
        )
        await self.send_message(text)

    async def send_history(self, limit: int = 10):
        """Send recent trade history"""
        if not self._sim_engine:
            await self.send_message("⚠️ Engine not connected")
            return

        trades = self._sim_engine.closed_trades[-limit:]
        if not trades:
            await self.send_message("📋 No closed trades yet")
            return

        lines = []
        for t in reversed(trades):
            pnl = t.get('pnl', 0)
            emoji = "✅" if pnl > 0 else "❌" if pnl < 0 else "➖"
            lines.append(
                f"{emoji} {t.get('side','')} ${t.get('amount',0):.2f} → "
                f"{'+'if pnl>=0 else ''}${pnl:.2f} | {t.get('market','')[:30]}"
            )

        text = (
            f"📋 <b>RECENT TRADES</b> (last {len(trades)})\n"
            f"━━━━━━━━━━━━━━━\n" +
            "\n".join(lines)
        )
        await self.send_message(text)

    # ─── COMMAND HANDLER ─────────────────────────────────────────

    async def handle_command(self, text: str, chat_id: str):
        """Handle incoming Telegram commands"""
        # Security: only respond to configured chat_id
        if str(chat_id) != str(self.chat_id):
            log.warning(f"Telegram: unauthorized chat_id {chat_id}")
            return

        cmd = text.strip().lower().split()[0] if text.strip() else ""

        if cmd == "/status" or cmd == "/s":
            await self.send_portfolio_status()

        elif cmd == "/positions" or cmd == "/pos" or cmd == "/p":
            await self.send_portfolio_status()

        elif cmd == "/history" or cmd == "/h":
            await self.send_history()

        elif cmd == "/start_auto" or cmd == "/start":
            if self._auto_trader:
                if self._auto_trader.running:
                    await self.send_message("⚡ AutoTrader is already running")
                else:
                    self._auto_trader.start()
                    await self.send_message("▶️ AutoTrader <b>STARTED</b>")
            else:
                await self.send_message("⚠️ AutoTrader not connected")

        elif cmd == "/stop_auto" or cmd == "/stop":
            if self._auto_trader:
                if not self._auto_trader.running:
                    await self.send_message("⏹ AutoTrader is already stopped")
                else:
                    self._auto_trader.stop()
                    await self.send_message("⏹ AutoTrader <b>STOPPED</b>")
            else:
                await self.send_message("⚠️ AutoTrader not connected")

        elif cmd == "/scan":
            if self._auto_trader:
                await self.send_message("🔍 Running scan...")
                try:
                    result = await self._auto_trader.run_scan_cycle()
                    await self.notify_scan_complete(result)
                    if result.get('bets_placed', 0) == 0 and result.get('resolved', 0) == 0:
                        await self.send_message(
                            f"🔍 Scan complete: {result.get('total_opportunities', 0)} "
                            f"opportunities found, no action taken"
                        )
                except Exception as e:
                    await self.send_message(f"❌ Scan error: {e}")
            else:
                await self.send_message("⚠️ AutoTrader not connected")

        elif cmd == "/help":
            await self.send_message(
                "🤖 <b>Polymarket ARB Bot</b>\n"
                "━━━━━━━━━━━━━━━\n"
                "/status - Portfolio status\n"
                "/positions - Open positions\n"
                "/history - Recent trades\n"
                "/start - Start AutoTrader\n"
                "/stop - Stop AutoTrader\n"
                "/scan - Run manual scan\n"
                "/help - Show this message"
            )

        else:
            await self.send_message(
                "❓ Unknown command. Use /help to see available commands."
            )

    # ─── POLLING LOOP ────────────────────────────────────────────

    async def _poll_updates(self):
        """Long-poll Telegram for new messages/commands"""
        log.info("Telegram polling started")
        await self.send_message("🟢 <b>Polymarket ARB Bot</b> is online!\nType /help for commands.")

        while True:
            try:
                timeout = aiohttp.ClientTimeout(total=35)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    params = {
                        "offset": self._last_update_id + 1,
                        "timeout": 30,
                        "allowed_updates": ["message"],
                    }
                    async with session.get(
                        f"{self.base_url}/getUpdates", params=params
                    ) as resp:
                        data = await resp.json()
                        if not data.get("ok"):
                            log.warning(f"Telegram poll error: {data}")
                            await asyncio.sleep(5)
                            continue

                        for update in data.get("result", []):
                            self._last_update_id = update["update_id"]
                            message = update.get("message", {})
                            text = message.get("text", "")
                            chat_id = str(message.get("chat", {}).get("id", ""))

                            if text.startswith("/"):
                                await self.handle_command(text, chat_id)

            except asyncio.CancelledError:
                log.info("Telegram polling stopped")
                break
            except Exception as e:
                log.error(f"Telegram poll error: {e}")
                await asyncio.sleep(10)

    def start_polling(self):
        """Start the polling task"""
        if not self.enabled:
            log.info("Telegram bot not enabled, skipping polling")
            return None
        if self._polling_task and not self._polling_task.done():
            return self._polling_task
        self._polling_task = asyncio.ensure_future(self._poll_updates())
        return self._polling_task

    def stop_polling(self):
        """Stop the polling task"""
        if self._polling_task:
            self._polling_task.cancel()
            self._polling_task = None


# ─── SINGLETON INSTANCE ──────────────────────────────────────────

_bot_instance = None

def get_telegram_bot() -> TelegramBot:
    """Get or create the singleton Telegram bot instance"""
    global _bot_instance
    if _bot_instance is None:
        _bot_instance = TelegramBot()
    return _bot_instance
