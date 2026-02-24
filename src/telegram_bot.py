"""
Telegram Bot — Professional Forex EA-Style Notifications
═════════════════════════════════════════════════════════
• Trade opened/closed alerts with full metrics
• Daily performance reports (auto at midnight + /report command)
• Startup summary with account snapshot
• Commands: /status /history /report /start /stop /scan /help
"""

import asyncio
import aiohttp
import logging
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

log = logging.getLogger("telegram-bot")

# Load env
PROJECT_ROOT = Path(__file__).parent.parent
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")


class TelegramBot:
    """Professional Telegram notifications for Polymarket ARB Bot"""

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

        # Daily report tracking
        self._daily_trades_open = 0
        self._daily_trades_closed = 0
        self._daily_pnl_start = 0.0
        self._daily_best_trade = None   # {pnl, market}
        self._daily_worst_trade = None  # {pnl, market}
        self._last_report_date = None

        if self.enabled:
            log.info(f"Telegram bot enabled (chat_id: {self.chat_id})")
        else:
            log.info("Telegram bot disabled (no token/chat_id configured)")

    def set_engines(self, sim_engine, auto_trader, live_cache=None):
        """Connect bot to trading engines"""
        self._sim_engine = sim_engine
        self._auto_trader = auto_trader
        self._live_cache = live_cache

    # ═══════════════════════════════════════════════════════════
    # SEND MESSAGE
    # ═══════════════════════════════════════════════════════════

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
                    log.warning(f"Telegram send failed: {data.get('description', 'unknown')}")
                    return False
                return True
        except Exception as e:
            log.error(f"Telegram send error: {e}")
            return False

    # ═══════════════════════════════════════════════════════════
    # FOREX EA-STYLE NOTIFICATIONS
    # ═══════════════════════════════════════════════════════════

    async def notify_trade_opened(self, trade: dict):
        """
        🟢 TRADE OPENED — Forex EA style
        Called when a new bet is placed (manual or auto)
        """
        side = trade.get('side', '?')
        amount = trade.get('amount', 0)
        price = trade.get('entry_price', 0)
        edge = trade.get('edge_at_entry', 0)
        market = trade.get('market', 'Unknown')
        mtype = (trade.get('type', 'general') or 'general').upper()
        trade_id = trade.get('id', '?')

        # Get current account state
        bal = 0.0
        open_count = 0
        if self._sim_engine:
            p = self._sim_engine.get_portfolio_summary()
            bal = p.get('balance', 0)
            open_count = p.get('open_positions', 0)

        potential_win = amount * (1 / price - 1) if price > 0 else 0
        potential_loss = amount

        text = (
            f"{'🟢' if side == 'YES' else '🔴'} <b>TRADE OPENED</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 <b>{market[:65]}</b>\n"
            f"\n"
            f"▸ Side: <b>{side}</b>  |  Type: <code>{mtype}</code>\n"
            f"▸ Size: <b>${amount:.2f}</b>  @  <code>{price:.4f}</code>\n"
            f"▸ Edge: <b>{edge*100:+.1f}%</b>\n"
            f"▸ Potential: +${potential_win:.2f} / -${potential_loss:.2f}\n"
            f"\n"
            f"💼 Balance: <b>${bal:.2f}</b>  |  Open: {open_count}\n"
            f"🔖 ID: <code>{trade_id}</code>"
        )
        await self.send_message(text)

        # Track daily stats
        self._daily_trades_open += 1

    async def notify_trade_closed(self, trade: dict):
        """
        ✅/❌ TRADE CLOSED — Forex EA style
        Called when a position is resolved or sold
        """
        pnl = trade.get('pnl', 0)
        amount = trade.get('amount', 0)
        market = trade.get('market', 'Unknown')
        side = trade.get('side', '?')
        outcome = trade.get('outcome', '?')
        entry = trade.get('entry_price', 0)
        trade_id = trade.get('id', '?')
        status = trade.get('status', 'unknown')

        # ROI calculation
        roi = (pnl / amount * 100) if amount > 0 else 0

        # Duration
        duration_str = "—"
        try:
            ts_open = trade.get('ts', '')
            ts_close = trade.get('resolved_at', '')
            if ts_open and ts_close:
                t1 = datetime.fromisoformat(str(ts_open).replace('Z', '+00:00'))
                t2 = datetime.fromisoformat(str(ts_close).replace('Z', '+00:00'))
                delta = t2 - t1
                hours = delta.total_seconds() / 3600
                if hours < 1:
                    duration_str = f"{int(delta.total_seconds() / 60)}m"
                elif hours < 24:
                    duration_str = f"{hours:.1f}h"
                else:
                    duration_str = f"{delta.days}d {int(hours % 24)}h"
        except Exception:
            pass

        # Get current account state
        bal = 0.0
        total_pnl = 0.0
        win_rate = 0.0
        win_count = 0
        loss_count = 0
        if self._sim_engine:
            p = self._sim_engine.get_portfolio_summary()
            bal = p.get('balance', 0)
            total_pnl = p.get('total_pnl', 0)
            win_rate = p.get('win_rate', 0)
            win_count = p.get('win_count', 0)
            loss_count = max(0, p.get('closed_trades', 0) - win_count)

        is_win = pnl >= 0
        emoji = "✅" if is_win else "❌"
        pnl_sign = "+" if pnl >= 0 else ""

        text = (
            f"{emoji} <b>TRADE CLOSED</b>  {'WIN' if is_win else 'LOSS'}\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 <b>{market[:65]}</b>\n"
            f"\n"
            f"▸ {side} → {outcome}  |  Duration: {duration_str}\n"
            f"▸ Entry: <code>{entry:.4f}</code>  |  Size: ${amount:.2f}\n"
            f"▸ P/L: <b>{pnl_sign}${abs(pnl):.2f}</b>  ({pnl_sign}{roi:.1f}%)\n"
            f"\n"
            f"💼 Balance: <b>${bal:.2f}</b>\n"
            f"📈 Total P/L: {'+' if total_pnl >= 0 else ''}${total_pnl:.2f}\n"
            f"📊 Record: {win_count}W / {loss_count}L  ({win_rate:.1f}%)\n"
            f"🔖 ID: <code>{trade_id}</code>"
        )
        await self.send_message(text)

        # Track daily stats
        self._daily_trades_closed += 1
        if self._daily_best_trade is None or pnl > self._daily_best_trade['pnl']:
            self._daily_best_trade = {'pnl': pnl, 'market': market[:40]}
        if self._daily_worst_trade is None or pnl < self._daily_worst_trade['pnl']:
            self._daily_worst_trade = {'pnl': pnl, 'market': market[:40]}

    async def send_daily_report(self):
        """
        📋 DAILY PERFORMANCE REPORT — Forex EA style
        Sent at midnight or on demand via /report
        """
        if not self._sim_engine:
            return

        p = self._sim_engine.get_portfolio_summary()
        bal = p.get('balance', 0)
        total_pnl = p.get('total_pnl', 0)
        win_rate = p.get('win_rate', 0)
        win_count = p.get('win_count', 0)
        total_trades = p.get('closed_trades', 0)
        loss_count = max(0, total_trades - win_count)
        open_pos = p.get('open_positions', 0)
        initial = p.get('initial_balance', 10000)
        invested = p.get('total_invested', 0)

        # Daily P/L
        daily_pnl = bal - self._daily_pnl_start if self._daily_pnl_start > 0 else total_pnl

        # Best / Worst
        best = self._daily_best_trade
        worst = self._daily_worst_trade
        best_str = f"+${best['pnl']:.2f} ({best['market']})" if best and best['pnl'] != 0 else "—"
        worst_str = f"-${abs(worst['pnl']):.2f} ({worst['market']})" if worst and worst['pnl'] != 0 else "—"

        # AutoTrader stats
        auto_scans = 0
        auto_bets = 0
        auto_running = False
        if self._auto_trader:
            auto_scans = self._auto_trader.scan_count
            auto_bets = self._auto_trader.auto_bets_placed
            auto_running = self._auto_trader.running

        now = datetime.now()
        dpnl_sign = "+" if daily_pnl >= 0 else ""

        text = (
            f"📋 <b>DAILY REPORT</b>  —  {now.strftime('%b %d, %Y')}\n"
            f"═══════════════════════════\n"
            f"\n"
            f"💰 <b>ACCOUNT</b>\n"
            f"▸ Balance: <b>${bal:.2f}</b>\n"
            f"▸ Invested: ${invested:.2f}\n"
            f"▸ Initial: ${initial:.2f}\n"
            f"\n"
            f"📈 <b>PERFORMANCE</b>\n"
            f"▸ Day P/L: <b>{dpnl_sign}${abs(daily_pnl):.2f}</b>\n"
            f"▸ Total P/L: {'+' if total_pnl >= 0 else ''}${total_pnl:.2f}\n"
            f"▸ ROI: {'+' if total_pnl >= 0 else ''}{(total_pnl / initial * 100) if initial > 0 else 0:.2f}%\n"
            f"\n"
            f"📊 <b>TRADES</b>\n"
            f"▸ Opened today: {self._daily_trades_open}\n"
            f"▸ Closed today: {self._daily_trades_closed}\n"
            f"▸ Open positions: {open_pos}\n"
            f"▸ Record: {win_count}W / {loss_count}L ({win_rate:.1f}%)\n"
            f"\n"
            f"🏆 Best: {best_str}\n"
            f"💀 Worst: {worst_str}\n"
            f"\n"
            f"🤖 <b>AUTOTRADER</b>\n"
            f"▸ Status: {'🟢 RUNNING' if auto_running else '🔴 STOPPED'}\n"
            f"▸ Scans: {auto_scans}  |  Auto bets: {auto_bets}\n"
            f"\n"
            f"═══════════════════════════\n"
            f"<i>POLYARB TRADING TERMINAL v1.3</i>"
        )
        await self.send_message(text)

        # Reset daily counters
        self._daily_trades_open = 0
        self._daily_trades_closed = 0
        self._daily_pnl_start = bal
        self._daily_best_trade = None
        self._daily_worst_trade = None
        self._last_report_date = now.date()

    async def send_startup_summary(self):
        """
        🚀 STARTUP SUMMARY — sent when bot comes online
        """
        if not self._sim_engine:
            await self.send_message("🟢 <b>Polymarket ARB Bot</b> is online!\nType /help for commands.")
            return

        p = self._sim_engine.get_portfolio_summary()
        bal = p.get('balance', 0)
        total_pnl = p.get('total_pnl', 0)
        win_rate = p.get('win_rate', 0)
        open_pos = p.get('open_positions', 0)
        total_trades = p.get('closed_trades', 0)

        self._daily_pnl_start = bal  # Track for daily report

        text = (
            f"🚀 <b>BOT ONLINE</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"💼 Balance: <b>${bal:.2f}</b>\n"
            f"📈 Total P/L: {'+' if total_pnl >= 0 else ''}${total_pnl:.2f}\n"
            f"📊 Win Rate: {win_rate:.1f}%  |  Trades: {total_trades}\n"
            f"📌 Open positions: {open_pos}\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"<i>POLYARB TRADING TERMINAL v1.3</i>\n"
            f"Type /help for commands"
        )
        await self.send_message(text)

    async def notify_scan_complete(self, scan_result: dict):
        """Notify on scan cycle (only if bets placed or resolved)"""
        bets = scan_result.get('bets_placed', 0)
        resolved = scan_result.get('resolved', 0)
        total_opps = scan_result.get('total_opportunities', 0)
        cycle = scan_result.get('cycle', 0)

        # Only send if something happened
        if bets == 0 and resolved == 0:
            return

        portfolio = scan_result.get('portfolio', {})
        text = (
            f"🔍 <b>SCAN #{cycle}</b>\n"
            f"▸ Opportunities: {total_opps}\n"
            f"▸ Bets placed: <b>{bets}</b>  |  Resolved: <b>{resolved}</b>\n"
            f"▸ Balance: ${portfolio.get('balance', 0):.2f}"
        )
        await self.send_message(text)

    # ═══════════════════════════════════════════════════════════
    # PORTFOLIO STATUS (for /status command)
    # ═══════════════════════════════════════════════════════════

    async def send_portfolio_status(self):
        """Send current portfolio status"""
        if not self._sim_engine:
            await self.send_message("⚠️ Engine not connected")
            return

        p = self._sim_engine.get_portfolio_summary()
        positions = self._sim_engine.positions
        bal = p.get('balance', 0)
        total_pnl = p.get('total_pnl', 0)
        win_rate = p.get('win_rate', 0)
        win_count = p.get('win_count', 0)
        total_trades = p.get('closed_trades', 0)
        loss_count = max(0, total_trades - win_count)
        invested = p.get('total_invested', 0)

        pos_text = ""
        if positions:
            lines = []
            for pos in positions[:8]:
                side = pos.get('side', '')
                amt = pos.get('amount', 0)
                entry = pos.get('entry_price', 0)
                mkt = pos.get('market', '')[:35]
                lines.append(f"  {'🟢' if side=='YES' else '🔴'} {side} ${amt:.2f} @ {entry:.3f} | {mkt}")
            pos_text = "\n".join(lines)
        else:
            pos_text = "  No open positions"

        auto_status = "🟢 RUNNING" if (self._auto_trader and self._auto_trader.running) else "🔴 STOPPED"

        text = (
            f"📊 <b>PORTFOLIO STATUS</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 Balance: <b>${bal:.2f}</b>\n"
            f"💼 Invested: ${invested:.2f}\n"
            f"📈 Total P/L: <b>{'+' if total_pnl >= 0 else ''}${total_pnl:.2f}</b>\n"
            f"📊 Record: {win_count}W / {loss_count}L ({win_rate:.1f}%)\n"
            f"🤖 AutoTrader: {auto_status}\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>Open Positions ({len(positions)}):</b>\n{pos_text}"
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

        total_pnl = sum(t.get('pnl', 0) for t in trades)
        wins = sum(1 for t in trades if t.get('pnl', 0) > 0)

        lines = []
        for t in reversed(trades):
            pnl = t.get('pnl', 0)
            emoji = "✅" if pnl > 0 else "❌" if pnl < 0 else "➖"
            lines.append(
                f"{emoji} {t.get('side', '')} ${t.get('amount', 0):.2f} → "
                f"{'+'if pnl >= 0 else ''}${pnl:.2f} | {t.get('market', '')[:30]}"
            )

        text = (
            f"📋 <b>RECENT TRADES</b> (last {len(trades)})\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n" +
            "\n".join(lines) +
            f"\n━━━━━━━━━━━━━━━━━━━━━\n"
            f"Sum: {'+'if total_pnl >= 0 else ''}${total_pnl:.2f}  |  {wins}W / {len(trades) - wins}L"
        )
        await self.send_message(text)

    # ═══════════════════════════════════════════════════════════
    # DAILY REPORT CHECK (called from ws_broadcast_task)
    # ═══════════════════════════════════════════════════════════

    async def check_daily_report(self):
        """Check if midnight crossed → send daily report"""
        now = datetime.now()
        today = now.date()
        if self._last_report_date is None:
            self._last_report_date = today
            if self._sim_engine:
                self._daily_pnl_start = self._sim_engine.get_portfolio_summary().get('balance', 0)
            return False
        if today > self._last_report_date:
            await self.send_daily_report()
            return True
        return False

    # ═══════════════════════════════════════════════════════════
    # COMMAND HANDLER
    # ═══════════════════════════════════════════════════════════

    async def handle_command(self, text: str, chat_id: str):
        """Handle incoming Telegram commands"""
        if str(chat_id) != str(self.chat_id):
            log.warning(f"Telegram: unauthorized chat_id {chat_id}")
            return

        cmd = text.strip().lower().split()[0] if text.strip() else ""

        if cmd in ("/status", "/s"):
            await self.send_portfolio_status()

        elif cmd in ("/positions", "/pos", "/p"):
            await self.send_portfolio_status()

        elif cmd in ("/history", "/h"):
            await self.send_history()

        elif cmd in ("/report", "/r", "/daily"):
            await self.send_daily_report()

        elif cmd in ("/start_auto", "/start"):
            if self._auto_trader:
                if self._auto_trader.running:
                    await self.send_message("⚡ AutoTrader is already running")
                else:
                    self._auto_trader.start()
                    await self.send_message("▶️ AutoTrader <b>STARTED</b>")
            else:
                await self.send_message("⚠️ AutoTrader not connected")

        elif cmd in ("/stop_auto", "/stop"):
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
                            f"🔍 Scan: {result.get('total_opportunities', 0)} "
                            f"opportunities, no action taken"
                        )
                except Exception as e:
                    await self.send_message(f"❌ Scan error: {e}")
            else:
                await self.send_message("⚠️ AutoTrader not connected")

        elif cmd == "/help":
            await self.send_message(
                "🤖 <b>POLYARB TRADING TERMINAL</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━\n"
                "/status  — Portfolio & positions\n"
                "/history — Recent closed trades\n"
                "/report  — Daily performance report\n"
                "/start   — Start AutoTrader\n"
                "/stop    — Stop AutoTrader\n"
                "/scan    — Run manual scan\n"
                "/help    — Show this message\n"
                "━━━━━━━━━━━━━━━━━━━━━\n"
                "<i>v1.3 — Forex EA Notifications</i>"
            )

        else:
            await self.send_message("❓ Unknown command. /help for commands.")

    # ═══════════════════════════════════════════════════════════
    # POLLING LOOP
    # ═══════════════════════════════════════════════════════════

    async def _poll_updates(self):
        """Long-poll Telegram for new messages/commands"""
        log.info("Telegram polling started")
        await self.send_startup_summary()

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


# ═══════════════════════════════════════════════════════════
# SINGLETON INSTANCE
# ═══════════════════════════════════════════════════════════

_bot_instance = None

def get_telegram_bot() -> TelegramBot:
    """Get or create the singleton Telegram bot instance"""
    global _bot_instance
    if _bot_instance is None:
        _bot_instance = TelegramBot()
    return _bot_instance
