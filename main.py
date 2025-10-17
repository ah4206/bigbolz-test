#!/usr/bin/env python3
# BIGBOLZ BOT v11
import asyncio
import json
import logging
import math
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from threading import Thread

import ccxt
import numpy as np
import requests
from flask import Flask

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('bigbolz_v11')

MEXC = ccxt.mexc({'rateLimit': 1200})
BOT_TOKEN = '8460657561:AAHVmExG6-LthruXAzTMXAxtSKbQeVuIQSM'
CHAT_ID = '-1003167969696'
TIMEFRAME = '1m'
MAX_BARS_BACK = 500
POLL_INTERVAL = 2
STATE_FILE = 'bigbolz_v11_state.json'
REQUEST_DELAY = 0.1
BATCH_SIZE = 10
CONFIRMED_MODE = False

LENGTH = 11
SHOW_BULL = 3
SHOW_BEAR = 3
USE_BODY = True
COOLDOWN_BARS = 15
INVALIDATION_WINDOW = 5
ENABLE_RISKY = True
RISKY_THRESHOLD = 2
RISKY_PERIOD = 20
ENABLE_ANTI_SPAM = True
SPAM_THRESHOLD = 3
SPAM_DETECTION_WINDOW = 15
ANTI_SPAM_COOLDOWN = 20
ENABLE_STRONG = True
STRONG_THRESHOLD_PCT = 15.0

SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'ASTUSDT', 'XRPUSDT',
    'BNBUSDT'
]

app = Flask(__name__)


@app.route('/')
def home():
    return f'<h1>BIGBOLZ v11</h1><p>Tracking {len(SYMBOLS)} symbols on {TIMEFRAME}</p>'


def start_flask():
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)


class Swing:

    def __init__(self, y=math.nan, x=math.nan, crossed=False):
        self.y, self.x, self.crossed = y, x, crossed

    def to_dict(self):
        return {
            'y': None if math.isnan(self.y) else self.y,
            'x': None if math.isnan(self.x) else self.x,
            'crossed': self.crossed
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            d.get('y') if d.get('y') is not None else math.nan,
            d.get('x') if d.get('x') is not None else math.nan,
            d.get('crossed', False))


class OB:
    NAN_FIELDS = [
        'top', 'btm', 'loc', 'break_loc', 'break_bar', 'last_signal_bar',
        'ref_extreme', 'cooling_end_bar', 'third_signal_bar',
        'cycle_start_bar', 'cooldown_end_bar', 'last_sig_bar_for_strong',
        'last_sig_anchor_extreme', 'last_updated_bar'
    ]
    BOOL_FIELDS = [
        'breaker', 'invalidated', 'invalidation_marker_placed',
        'last_sig_is_long_for_strong', 'current_strong'
    ]

    def __init__(self, top=math.nan, btm=math.nan, loc=math.nan):
        self.top, self.btm, self.loc = top, btm, loc
        self.breaker = False
        self.break_loc = self.break_bar = self.last_signal_bar = self.ref_extreme = self.cooling_end_bar = math.nan
        self.invalidated = self.invalidation_marker_placed = False
        self.signal_count_risky = 0
        self.third_signal_bar = self.cycle_start_bar = math.nan
        self.cycle_signal_count = 0
        self.cooldown_end_bar = self.last_sig_bar_for_strong = self.last_sig_anchor_extreme = self.last_updated_bar = math.nan
        self.last_sig_is_long_for_strong = self.current_strong = False

    def full_reset(self):
        self.last_signal_bar = self.ref_extreme = self.cooling_end_bar = self.third_signal_bar = self.cycle_start_bar = self.cooldown_end_bar = self.last_sig_bar_for_strong = self.last_sig_anchor_extreme = self.last_updated_bar = math.nan
        self.invalidated = self.invalidation_marker_placed = self.last_sig_is_long_for_strong = self.current_strong = False
        self.signal_count_risky = self.cycle_signal_count = 0

    def to_dict(self):
        d = self.__dict__.copy()
        for k in self.NAN_FIELDS:
            if math.isnan(d.get(k, math.nan)): d[k] = None
        for k in self.BOOL_FIELDS:
            d[k] = bool(d.get(k, False))
        return d

    @classmethod
    def from_dict(cls, d):
        ob = cls(
            d.get('top') if d.get('top') is not None else math.nan,
            d.get('btm') if d.get('btm') is not None else math.nan,
            d.get('loc') if d.get('loc') is not None else math.nan)
        for k, v in d.items():
            if v is None and k in cls.NAN_FIELDS: setattr(ob, k, math.nan)
            elif k in cls.BOOL_FIELDS: setattr(ob, k, bool(v))
            elif isinstance(v, (int, float)): setattr(ob, k, float(v))
            elif v is None:
                setattr(ob, k, math.nan if k in cls.NAN_FIELDS else 0)
            else:
                setattr(ob, k, v)
        return ob


class BigBolzBot:

    def __init__(self):
        self.state = defaultdict(dict)
        self.last_alert_bar = defaultdict(lambda: math.nan)
        self.last_confirmed_ts = {}
        self.strong_map = defaultdict(dict)
        self.last_seen_bar = defaultdict(lambda: -1)
        self.last_computed_bar = defaultdict(lambda: -1)
        self.live_bar = defaultdict(lambda: {
            'high': math.nan,
            'low': math.nan,
            'close': math.nan,
            'start_ts': 0
        })
        self.load_state()

    def load_state(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, 'r') as f:
                    data = json.load(f)
                for sym, st_data in data.get('storage', {}).items():
                    self.state[sym]['bull_ob'] = [
                        OB.from_dict(x) for x in st_data.get('bull_ob', [])
                    ]
                    self.state[sym]['bear_ob'] = [
                        OB.from_dict(x) for x in st_data.get('bear_ob', [])
                    ]
                    self.state[sym]['top'] = Swing.from_dict(
                        st_data.get('top', {}))
                    self.state[sym]['btm'] = Swing.from_dict(
                        st_data.get('btm', {}))
                    self.state[sym]['os'] = st_data.get('os', 0)
                    self.state[sym]['prev_os'] = st_data.get('prev_os', 0)
                    self.state[sym]['historical_built'] = st_data.get(
                        'historical_built', False)
                    self.state[sym]['bull_ob'] = self.state[sym][
                        'bull_ob'][:SHOW_BULL]
                    self.state[sym]['bear_ob'] = self.state[sym][
                        'bear_ob'][:SHOW_BEAR]
                self.last_confirmed_ts = data.get('last_ts', {})
                self.last_seen_bar = defaultdict(
                    lambda: -1, {
                        k: int(v)
                        for k, v in data.get('last_seen_bar', {}).items()
                        if v is not None
                    })
                self.last_computed_bar = defaultdict(
                    lambda: -1, {
                        k: int(v)
                        for k, v in data.get('last_computed_bar', {}).items()
                        if v is not None
                    })
                live_data = data.get('live_bar', {})
                for sym, ld in live_data.items():
                    if isinstance(ld, dict):
                        self.live_bar[sym] = {
                            'high':
                            float(ld['high'])
                            if ld.get('high') is not None else math.nan,
                            'low':
                            float(ld['low'])
                            if ld.get('low') is not None else math.nan,
                            'close':
                            float(ld['close'])
                            if ld.get('close') is not None else math.nan,
                            'start_ts':
                            int(ld.get('start_ts', 0))
                            if ld.get('start_ts') is not None else 0
                        }
                logger.info('State loaded')
            except Exception as e:
                logger.error(f'State load error: {e}')

    def save_state(self):
        data = {
            'last_ts': self.last_confirmed_ts,
            'last_seen_bar': dict(self.last_seen_bar),
            'last_computed_bar': dict(self.last_computed_bar),
            'live_bar': dict(self.live_bar),
            'storage': {}
        }
        for sym, st in self.state.items():
            data['storage'][sym] = {
                'bull_ob': [ob.to_dict() for ob in st.get('bull_ob', [])],
                'bear_ob': [ob.to_dict() for ob in st.get('bear_ob', [])],
                'top': st.get('top', Swing()).to_dict(),
                'btm': st.get('btm', Swing()).to_dict(),
                'os': st.get('os', 0),
                'prev_os': st.get('prev_os', 0),
                'historical_built': st.get('historical_built', False)
            }
        try:
            with open(STATE_FILE, 'w') as f:
                json.dump(data, f, default=str)
        except Exception as e:
            logger.error(f'State save error: {e}')

    def fetch_klines(self, symbol):
        try:
            ohlcv = MEXC.fetch_ohlcv(symbol,
                                     TIMEFRAME,
                                     limit=MAX_BARS_BACK + LENGTH + 10)
            return [(c[0], float(c[1]), float(c[2]), float(c[3]), float(c[4]))
                    for c in ohlcv]
        except Exception as e:
            logger.error(f'Klines fetch error {symbol}: {e}')
            return []

    def fetch_ticker(self, symbol):
        try:
            t = MEXC.fetch_ticker(symbol)
            return t.get('last') if t.get('last') is not None and isinstance(
                t.get('last'), (int, float)) else None
        except Exception as e:
            logger.error(f'Ticker fetch error {symbol}: {e}')
            return None

    def is_forming_bar(self, klines):
        return not (not klines or CONFIRMED_MODE) and int(
            time.time() * 1000) - klines[-1][0] < 60000

    def has_new_confirmed_bar(self, symbol, klines):
        if len(klines) < 2:
            return False
        confirmed_ts = klines[-2][0]
        if confirmed_ts > self.last_confirmed_ts.get(symbol, 0):
            self.last_confirmed_ts[symbol] = confirmed_ts
            self.live_bar[symbol] = {
                'high': math.nan,
                'low': math.nan,
                'close': math.nan,
                'start_ts': klines[-1][0]
            }
            return True
        return False

    def ta_highest(self, arr, length):
        return float(np.max(
            arr[-length:])) if length > 0 and len(arr) >= length else math.nan

    def ta_lowest(self, arr, length):
        return float(np.min(
            arr[-length:])) if length > 0 and len(arr) >= length else math.nan

    def touches_ob(self, ob, high, low, close, open_, bar_idx):
        if math.isnan(ob.top) or math.isnan(ob.btm) or ob.invalidated:
            return False
        if not (not ob.breaker or
                (not math.isnan(ob.break_bar) and bar_idx > ob.break_bar)):
            return False
        price_overlap = (high >= ob.btm
                         and low <= ob.top) or (close >= ob.btm
                                                and close <= ob.top)
        if USE_BODY:
            price_overlap = price_overlap or (max(open_, close) >= ob.btm
                                              and min(open_, close) <= ob.top)
        return price_overlap

    def calc_is_strong(self, ob, eval_is_long, eval_sig_bar, highs, lows,
                       prior_anchor):
        if not ENABLE_STRONG or math.isnan(
                ob.last_sig_bar_for_strong
        ) or ob.last_sig_is_long_for_strong != eval_is_long or ob.last_sig_bar_for_strong >= eval_sig_bar:
            return False
        run_length = int(eval_sig_bar - ob.last_sig_bar_for_strong - 1)
        if run_length <= 0 or math.isnan(prior_anchor) or prior_anchor == 0:
            return False
        effective_length = min(run_length, MAX_BARS_BACK)
        if effective_length <= 0:
            return False
        start_idx, end_idx = int(ob.last_sig_bar_for_strong +
                                 1), int(eval_sig_bar)
        if start_idx >= end_idx:
            return False
        inter_highs, inter_lows = highs[start_idx:end_idx], lows[
            start_idx:end_idx]
        if len(inter_highs) < effective_length:
            return False
        if eval_is_long:
            inter_extreme = self.ta_highest(inter_highs, effective_length)
            return not math.isnan(inter_extreme) and (
                inter_extreme -
                prior_anchor) / prior_anchor * 100.0 >= STRONG_THRESHOLD_PCT
        else:
            inter_extreme = self.ta_lowest(inter_lows, effective_length)
            return not math.isnan(inter_extreme) and (
                prior_anchor -
                inter_extreme) / prior_anchor * 100.0 >= STRONG_THRESHOLD_PCT

    def reset_strong_anchor(self, ob, new_is_long, new_sig_bar, new_low,
                            new_high):
        ob.last_sig_bar_for_strong, ob.last_sig_is_long_for_strong, ob.last_sig_anchor_extreme, ob.current_strong = new_sig_bar, new_is_long, new_low if new_is_long else new_high, False

    def build_historical(self, symbol, klines):
        st = self.state[symbol]
        if len(klines) < LENGTH + 1:
            return
        ts_list, o_list, h_list, l_list, c_list = zip(*klines[:-1])
        o, h, l, c, ts_list = list(o_list), list(h_list), list(l_list), list(
            c_list), list(ts_list)
        st['os'], st['prev_os'], st['top'], st['btm'], st['bull_ob'], st[
            'bear_ob'] = math.nan, math.nan, Swing(), Swing(), [], []
        n = len(klines) - 1
        logger.info(f'{symbol} - Starting swing detection: LENGTH={LENGTH}, n={n}, total bars={len(h)}')
        for i in range(LENGTH, n):
            check_idx = i - LENGTH
            if check_idx < 0 or check_idx >= len(h):
                st['prev_os'] = st['os']
                continue
            # For swing detection, compare the current bar against the LENGTH bars AFTER it
            # upper/lower represent the extreme values in the next LENGTH bars
            upper = self.ta_highest(h[check_idx + 1:min(check_idx + 1 + LENGTH, len(h))], LENGTH)
            lower = self.ta_lowest(l[check_idx + 1:min(check_idx + 1 + LENGTH, len(l))], LENGTH)
            
            # Log first few iterations for debugging
            if i <= LENGTH + 3:
                logger.info(f'{symbol} - i={i}, check_idx={check_idx}, h[{check_idx}]={h[check_idx]:.6f}, upper={upper:.6f}, h>{upper}={h[check_idx] > upper}')
            
            st['prev_os'] = st['os']
            if not math.isnan(upper) and h[check_idx] > upper:
                st['os'] = 0
                if math.isnan(st['prev_os']) or st['os'] != st['prev_os']:
                    st['top'] = Swing(h[check_idx], check_idx, False)
                    logger.info(f'{symbol} - Swing HIGH detected at bar {check_idx}, price {h[check_idx]:.6f}')
            elif not math.isnan(lower) and l[check_idx] < lower:
                st['os'] = 1
                if math.isnan(st['prev_os']) or st['os'] != st['prev_os']:
                    st['btm'] = Swing(l[check_idx], check_idx, False)
                    logger.info(f'{symbol} - Swing LOW detected at bar {check_idx}, price {l[check_idx]:.6f}')
            else:
                if not math.isnan(st['prev_os']):
                    st['os'] = st['prev_os']
            self.process_confirmed_obs(st, o, h, l, c, ts_list, i)
        st['bull_ob'], st['bear_ob'], st['historical_built'] = st[
            'bull_ob'][:SHOW_BULL], st['bear_ob'][:SHOW_BEAR], True
        logger.info(f'Historical built for {symbol} - Created {len(st["bull_ob"])} Bull OBs and {len(st["bear_ob"])} Bear OBs')

    def process_confirmed_obs(self, st, o, h, l, c, ts, bar_idx):
        top, btm = st['top'], st['btm']

        if not math.isnan(top.y) and c[bar_idx] > top.y and not top.crossed:
            logger.info(f'Price {c[bar_idx]:.6f} crossed ABOVE swing high {top.y:.6f} at bar {bar_idx}')
            top.crossed = True
            maxima = max(o[bar_idx - 1], c[bar_idx -
                                           1]) if USE_BODY else h[bar_idx - 1]
            minima = min(o[bar_idx - 1], c[bar_idx -
                                           1]) if USE_BODY else l[bar_idx - 1]
            loc = ts[bar_idx - 1]

            for j in range(int(top.x) + 1, bar_idx + 1):
                temp_min = min(o[j], c[j]) if USE_BODY else l[j]
                temp_max = max(o[j], c[j]) if USE_BODY else h[j]

                minima = min(temp_min, minima)
                if minima == temp_min:
                    maxima = temp_max
                    loc = ts[j]

            st['bull_ob'].insert(0, OB(maxima, minima, loc))
            logger.info(f'Created Bull OB: Top={maxima:.6f}, Btm={minima:.6f} at bar {bar_idx}')

        if not math.isnan(btm.y) and c[bar_idx] < btm.y and not btm.crossed:
            logger.info(f'Price {c[bar_idx]:.6f} crossed BELOW swing low {btm.y:.6f} at bar {bar_idx}')
            btm.crossed = True
            maxima = max(o[bar_idx - 1], c[bar_idx -
                                           1]) if USE_BODY else h[bar_idx - 1]
            minima = min(o[bar_idx - 1], c[bar_idx -
                                           1]) if USE_BODY else l[bar_idx - 1]
            loc = ts[bar_idx - 1]

            for j in range(int(btm.x) + 1, bar_idx + 1):
                temp_max = max(o[j], c[j]) if USE_BODY else h[j]
                temp_min = min(o[j], c[j]) if USE_BODY else l[j]

                maxima = max(temp_max, maxima)
                if maxima == temp_max:
                    minima = temp_min
                    loc = ts[j]

            st['bear_ob'].insert(0, OB(maxima, minima, loc))
            logger.info(f'Created Bear OB: Top={maxima:.6f}, Btm={minima:.6f} at bar {bar_idx}')

        for i in range(len(st['bull_ob']) - 1, -1, -1):
            ob = st['bull_ob'][i]
            if not ob.breaker:
                if (min(o[bar_idx], c[bar_idx])
                        if USE_BODY else l[bar_idx]) < ob.btm:
                    new_breaker = OB(ob.top, ob.btm, ob.loc)
                    new_breaker.breaker, new_breaker.break_loc, new_breaker.break_bar = True, ts[
                        bar_idx], bar_idx
                    new_breaker.full_reset()
                    st['bull_ob'].insert(0, new_breaker)
                    del st['bull_ob'][i + 1]
            else:
                if c[bar_idx] > ob.top:
                    del st['bull_ob'][i]

        for i in range(len(st['bear_ob']) - 1, -1, -1):
            ob = st['bear_ob'][i]
            if not ob.breaker:
                if (max(o[bar_idx], c[bar_idx])
                        if USE_BODY else h[bar_idx]) > ob.top:
                    new_breaker = OB(ob.top, ob.btm, ob.loc)
                    new_breaker.breaker, new_breaker.break_loc, new_breaker.break_bar = True, ts[
                        bar_idx], bar_idx
                    new_breaker.full_reset()
                    st['bear_ob'].insert(0, new_breaker)
                    del st['bear_ob'][i + 1]
            else:
                if c[bar_idx] < ob.btm:
                    del st['bear_ob'][i]

        for obs_list, is_bull_group in [(st['bull_ob'], True),
                                        (st['bear_ob'], False)]:
            for ob in obs_list:
                if not math.isnan(ob.last_signal_bar) and not ob.invalidated:
                    is_long = not ob.breaker if is_bull_group else ob.breaker
                    window_start, window_end = int(ob.last_signal_bar), int(
                        ob.last_signal_bar) + INVALIDATION_WINDOW
                    if window_start <= bar_idx <= window_end:
                        if (c[bar_idx] < ob.btm
                                if is_long else c[bar_idx] > ob.top):
                            ob.invalidated = True
                            ob.full_reset()

    async def send_alert(self, symbol, sig_type, price):
        emojis = {
            'Risky Strong Long': '🚀💪',
            'Risky Strong Short': '🔻💪',
            'Strong Long': '🚀',
            'Strong Short': '🔻',
            'Risky Long': '🚀⚠️',
            'Risky Short': '🔻⚠️',
            'Long': '🚀',
            'Short': '🔻'
        }
        msg = f"{emojis.get(sig_type, '')} {sig_type} {symbol.replace('USDT', '')} @ {price:.6f} {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')} {TIMEFRAME}"
        if not BOT_TOKEN or not CHAT_ID:
            return
        try:
            r = requests.post(
                f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage',
                json={
                    'chat_id': CHAT_ID,
                    'text': msg
                },
                timeout=10)
            if r.status_code == 200:
                logger.info(f'Alert sent: {sig_type} {symbol}')
        except Exception as e:
            logger.error(f'Telegram error: {e}')

    async def process_symbol(self, symbol):
        klines = self.fetch_klines(symbol)
        if len(klines) < LENGTH + 2:
            return
        st = self.state[symbol]
        if not st:
            st['bull_ob'], st['bear_ob'], st['top'], st['btm'] = [], [], Swing(
            ), Swing()
            st['os'], st['prev_os'] = 0, 0

        if not st.get('historical_built', False):
            self.build_historical(symbol, klines)
        ts_list, o_list, h_list, l_list, c_list = zip(*klines)
        o, h, l, c = list(o_list), list(h_list), list(l_list), list(c_list)
        bar_idx = len(klines) - 1
        new_conf = self.has_new_confirmed_bar(symbol, klines)
        forming = self.is_forming_bar(klines)
        live = self.live_bar[symbol]
        if math.isnan(live['high']) or live['start_ts'] != klines[-1][0]:
            live['high'], live['low'], live['close'], live['start_ts'] = h[
                -1], l[-1], c[-1], klines[-1][0]
        high, low, close = live['high'], live['low'], live['close']
        if forming and not CONFIRMED_MODE:
            t_last = self.fetch_ticker(symbol)
            if t_last is not None:
                close, high, low = t_last, max(
                    high, t_last) if not math.isnan(high) else t_last, min(
                        low, t_last) if not math.isnan(low) else t_last
                live['close'], live['high'], live['low'] = close, high, low
        if new_conf or bar_idx > self.last_seen_bar[symbol]:
            self.last_seen_bar[symbol] = bar_idx
        elif not forming:
            return
        if new_conf:
            self.process_confirmed_obs(st, o, h, l, c, list(ts_list),
                                       bar_idx - 1)
            st['bull_ob'], st['bear_ob'] = st['bull_ob'][:SHOW_BULL], st[
                'bear_ob'][:SHOW_BEAR]
        if bar_idx < self.last_computed_bar[symbol]:
            return
        if bar_idx > self.last_computed_bar[symbol]:
            self.last_computed_bar[symbol] = bar_idx
        for obs_list, is_bull_group in [(st['bull_ob'], True),
                                        (st['bear_ob'], False)]:
            for ob in obs_list[:SHOW_BULL if is_bull_group else SHOW_BEAR]:
                is_long = not ob.breaker if is_bull_group else ob.breaker
                ob.current_strong = self.calc_is_strong(
                    ob, is_long, bar_idx, h, l, ob.last_sig_anchor_extreme)

        # Debug logging
        total_bull = len(st.get('bull_ob', []))
        total_bear = len(st.get('bear_ob', []))
        if total_bull > 0 or total_bear > 0:
            logger.info(f'{symbol} - Price: {close:.6f}, Bull OBs: {total_bull}, Bear OBs: {total_bear}, Bar: {bar_idx}')
            for i, ob in enumerate(st.get('bull_ob', [])):
                logger.info(f'{symbol} - Bull OB #{i+1}: Top={ob.top:.6f}, Btm={ob.btm:.6f}, Breaker={ob.breaker}, Invalidated={ob.invalidated}')
            for i, ob in enumerate(st.get('bear_ob', [])):
                logger.info(f'{symbol} - Bear OB #{i+1}: Top={ob.top:.6f}, Btm={ob.btm:.6f}, Breaker={ob.breaker}, Invalidated={ob.invalidated}')

        for obs_list, is_bull_group in [(st['bull_ob'], True),
                                        (st['bear_ob'], False)]:
            for ob in obs_list:
                if not self.touches_ob(ob, high, low, close, o[-1], bar_idx):
                    continue
                is_long = not ob.breaker if is_bull_group else ob.breaker
                logger.info(
                    f'{symbol} - OB touch detected: {"Long" if is_long else "Short"} at {close}'
                )
                in_cooling = not math.isnan(
                    ob.cooling_end_bar) and bar_idx <= ob.cooling_end_bar
                ref_breached = not math.isnan(ob.ref_extreme) and (
                    low < ob.ref_extreme if is_long else high > ob.ref_extreme)
                if in_cooling and not ref_breached:
                    continue
                can_signal = True
                if ENABLE_ANTI_SPAM:
                    if not math.isnan(ob.cooldown_end_bar
                                      ) and bar_idx <= ob.cooldown_end_bar:
                        can_signal = False
                    else:
                        if math.isnan(
                                ob.cycle_start_bar
                        ) or bar_idx - ob.cycle_start_bar > SPAM_DETECTION_WINDOW:
                            ob.cycle_start_bar, ob.cycle_signal_count = bar_idx, 1
                        else:
                            ob.cycle_signal_count += 1
                        if ob.cycle_signal_count >= SPAM_THRESHOLD and bar_idx - ob.cycle_start_bar <= SPAM_DETECTION_WINDOW:
                            ob.cooldown_end_bar, can_signal = bar_idx + ANTI_SPAM_COOLDOWN, False
                if not can_signal:
                    continue
                is_risky = False
                if ENABLE_RISKY:
                    if not math.isnan(
                            ob.third_signal_bar
                    ) and bar_idx - ob.third_signal_bar > RISKY_PERIOD:
                        ob.signal_count_risky, ob.third_signal_bar = 0, math.nan
                    ob.signal_count_risky += 1
                    if ob.signal_count_risky == RISKY_THRESHOLD:
                        ob.third_signal_bar = bar_idx
                    is_risky = ob.signal_count_risky > RISKY_THRESHOLD and not math.isnan(
                        ob.third_signal_bar
                    ) and bar_idx - ob.third_signal_bar <= RISKY_PERIOD
                ob.last_signal_bar, ob.ref_extreme, ob.cooling_end_bar, ob.invalidated, ob.invalidation_marker_placed = bar_idx, low if is_long else high, bar_idx + COOLDOWN_BARS, False, False
                is_strong = ob.current_strong
                sig_label = ('Risky Strong ' if is_risky and is_strong else
                             'Strong ' if is_strong else 'Risky ' if is_risky
                             else '') + ('Long' if is_long else 'Short')
                await self.send_alert(symbol, sig_label, close)
                self.reset_strong_anchor(ob, is_long, bar_idx, low, high)

    async def run(self):
        logger.info('BIGBOLZ v11 starting')
        Thread(target=start_flask, daemon=True).start()
        logger.info('Flask thread started, beginning main monitoring loop')
        loop_count = 0
        while True:
            try:
                loop_count += 1
                if loop_count % 10 == 0:
                    logger.info(
                        f'Monitoring loop #{loop_count} - checking {len(SYMBOLS)} symbols'
                    )
                for i in range(0, len(SYMBOLS), BATCH_SIZE):
                    results = await asyncio.gather(*[
                        self.process_symbol(s)
                        for s in SYMBOLS[i:i + BATCH_SIZE]
                    ],
                                                   return_exceptions=True)
                    for idx, res in enumerate(results):
                        if isinstance(res, Exception):
                            logger.error(
                                f'Error processing {SYMBOLS[i+idx]}: {res}')
                    await asyncio.sleep(REQUEST_DELAY)
                self.save_state()
            except Exception as e:
                logger.error(f'Main loop error: {e}', exc_info=True)
            await asyncio.sleep(POLL_INTERVAL)


if __name__ == '__main__':
    asyncio.run(BigBolzBot().run())
