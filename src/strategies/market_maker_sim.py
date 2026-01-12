"""
Market Maker Simulation (Multi-Asset, Synthetic Order Book, OOP Design)

This script simulates a simplified limit order book (LOB) environment
and a market-making agent providing liquidity across multiple 
correlated assets, using a fully object-oriented design.

Core features:
  • Abstract base classes for processes and agents
  • Encapsulation via properties (getters/setters)
  • Classmethods & staticmethods for object creation/utlities
  • Magic methods for clean introspection
  • Multi-asset midprice dynamics (OU or GBM)
  • Poisson order arrivals and price impact
  • Dynamic quoting (Avellaneda–Stoikov model)
  • Inventory and risk control
  • PnL tracking and performance metrics
  • Visualization utilities for prices, inventory, and PnL

Created by Thomas Vrije for a quant-research portfolio project.
"""
import abc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Literal
import time
import math


# ======================================================================
# 0. Global Configuration Dataclass
# ======================================================================
@dataclass
class SimulationConfig:
    """
    Global configuration object controlling every component of the market simulation.
    Each field has sensible defaults but can be overridden at runtime to run 
    experiments under different assumptions or regimes.

    Attributes
    ----------
    symbols : List[str]
        List of asset tickers to simulate.
    steps : int
        Number of discrete time steps in the simulation.
    dt : float
        Time resolution per step (e.g., 1/390 for 1-minute data, 1/23400 for 1-second).
    model : str
        Price process model, either 'OU' (mean-reverting) or 'GBM' (diffusive).
    seed : int
        Random seed for reproducibility of stochastic components.
    tick_size : float
        Minimum allowed price increment (used for rounding and matching).
    sec_fee_bps_sell = 0.0
    mm_params : Dict
        Parameters passed to the MarketMaker agent, e.g.:
            {
                "spread": 0.1,
                "size": 10.0,
                "risk_aversion": 0.05,
                "vol_sensitivity": 5.0,
                "initial_cash": 1000000.0,
                "inventory_limit": 100.0,
                "inventory_penalty": 1.0,
                "min_unwind_frac": 0.10,
                "levels": 3,
                "level_step_ticks": 1,
                "depth_decay": 0.6,
                "mm_queue_share": 0.6,
                "queue_jitter": 0.02,
                "risk_horizon_secs": 600,
                "taker_unwind_enabled": True,
                "taker_unwind_trigger": 0.6,
                "taker_unwind_target": 0.4,
                "taker_unwind_max_clips": 10,
                "taker_unwind_stale_mult": 2.0
                "adverse_mm_ticks": 0.5,
                "cancel_slip_ticks": 0.5,
                "overnight_penalty_bps": 0.0,
                "inventory_skew_gamma": 0.002,
                "quote_update_cost_ps": 0.00005,
                "ioc_cooldown_steps": 20,
            }
    process_params : Dict
        Parameters for the PriceProcess controlling price dynamics, e.g.:
            {
                "diag_interval": 5000,
                "init_price": 100.0,
                # GBM-only
                "mu": 1e-4,
                "vol": 0.01,
                # OU-only
                "theta": 100.0,
                "kappa": 0.05,
                "sigma": 0.02,
                # Cross-asset correlation
                "cov_scale": 1.0,
                "correlation": 0.7,
                # The rest
                "impact_eta": 0.0,        # ticks per (trade_size / mm_size)
                "impact_decay": 0.0,        # 0-1 optional decay towards process anchor
                "mm_latency_prob": 0.15,
                "stale_impact_boost": 1.0,
                "mm_refresh_interval": 50,
                "mm_refresh_ticks": 1,
                "markout_steps": 0,
                "maker_markout_ticks": 0.8,
                "maker_adverse_eta": 0.3,
                "bg_liquidity": True,
                "bg_inside_ticks": 1,
                "bg_levels": 3,
                "bg_size": 400.0,
                "bg_depth_decay": 0.7,
                "maker_drag_decay": 0.85,
                "maker_drag_stepcap": 1.5,
                "impact_max_ticks_per_step": 3.0,
                "max_anchor_move_ticks": 3.0,
                "annual_mu": 0.085,
                "regime_on": True,
                "regimes": {
                    "side": {"mu_ann": 0, "vol_mult": 1.0, "p_stay": 0.98},
                    "bull": {"mu_ann": 0.18, "vol_mult": 0.85, "p_stay": 0.985},
                    "bear": {"mu_ann": -0.25, "vol_mult": 1.60, "p_stay": 0.975}
                }

                "innovations": "student_t",      # "gaussian" or "student_t"
                "df": 5,                         # t dof (fat tails)

                # GARCH(1,1) for daily variance
                "garch_enabled": True,
                "garch_omega": 1e-6,
                "garch_alpha": 0.05,
                "garch_beta": 0.92,

                # Merton jumps in log-returns
                "jump_lambda": 0.04,
                "jump_mu": 0.0,
                "jump_sigma": 0.03,

                # Intraday U-shape
                "intraday_u_shape": True,
                "u_open_close": 1.7,
                "u_midday": 1.0,
                "event_times_steps": [],          # e.g., [60, 120, 22800] if you want spikes
                "event_sigma_mult": 3.0,
                "event_half_life_secs": 180.0,
                "basis_enabled": True,
                "basis_half_life_secs": 90.0,
                "basis_sigma_ticks": 0.4,

                # Flow impact
                "micro_trade_impact_eta": 0.0,     # keep old per-trade impact off
                "flow_impact_enabled": True,
                "flow_impact_gamma": 0.25,         # ticks per (order_size^alpha) flow
                "flow_impact_alpha": 0.6,          # concave impact
                "flow_impact_beta": 120.0,         # per-day decay (roughly 2.25 min half-life @ 1/23400 dt)
                "flow_cross_rho": 0.3,             # cross-asset mixing
                "flow_include_mm": False,          # impact from external aggressors only
                "flow_vol_norm": True,             # divide by current daily vol
                "flow_shock_cancel_ticks": 0.5,
            }
    external_order_params : Dict
        Parameters governing random external order flow (Poisson arrivals, size, mix), e.g.:
            {
                "lambda": 70000,
                "price_sigma": 0.0005,
                "offset_tick_sigma": 2.0,
                "marketable_frac": 0.4,
                "retail_mu": 5,
                "inst_mu": 50,
                "retail_frac": 0.7,
                "buy_prob": 0.5,
                "use_price_bias": True,
                "bias_mode": "ath_contra",
                "buyprob_alpha": 2.0,
                "marketable_alpha": 1.0,

                # Hawkes Process Parameters
                "hawkes_enabled": True,
                "hawkes_alpha_self": 0.20,        # jump in per-step intensity per same-side event
                "hawkes_alpha_cross": 0.10,       # jump in per-step intensity per opposide-side event
                "hawkes_beta": 50.0,              # decay rate per DAY (roughly 5-min half life with dt=1/23400)
                "hawkes_cap_mult": 5.0,           # cap = cap_mult * baseline (per symbol)

                # Price-linked intensity tilt and aggression/size scalers
                "trend_eta": 4.0,                 # tilts buy/sell intensities with signed return
                "vol_eta": 2.0,                   # boosts both intensities with |return|/σ
                "mkt_trend_slope": 6.0,           # raises p_mkt with trend-aligned move
                "mkt_vol_slope": 4.0,             # raises p_mkt with |return|/σ
                "size_vol_slope": 6.0             # scales sizes with |return|/σ
            }
    fee_rate : float
        Per-trade fee applied to takers (fraction of notional).
    rebate_rate : float
        Rebate paid to makers (fraction of notional).
    routing_fee_rate : float
        Per-trade fee applied to all fills
    verbose : bool
        If True, prints periodic simulation progress and performance summaries.

    # ETF creation/redemption
    "etf_ap": {
        "enabled": True,
        "prem_threshold": 5e-4,
        "ap_kappa": 10.0,
        "ap_cost_bps": 2e-4,
        "sigma_nav_mult": 0.6,
        "max_ap_flow_per_day": 0.05
        }

    # Per-share fees (USD per share, set to 0 if not needed)
    per_share_fees: Dict[str, float] = field(default_factory=lambda: {
        "taker": 0.0028,
        "maker": 0.0014,
        "routing": 0.0003,
        "clearing": 0.0002,
        "finra_taf_ps": 0.000119,     #sells only
        "finra_taf_cap": 5.95,         # max per execution
    })
    """
    symbols: List[str] = field(default_factory=lambda: ["SPY"])
    steps: int = 1000
    dt: float = 1/390 # 1-min default
    model: str = "OU"
    seed: int = 42
    tick_size: float = 0.01
    sec_fee_bps_sell: float = 0.0

    mm_params: Dict = field(default_factory=lambda: {
        "spread": 0.1, 
        "size": 10.0,
        "risk_aversion": 0.05,
        "vol_sensitivity": 5.0,
        "initial_cash": 1000000.0,
        "inventory_limit": 100.0,
        "inventory_penalty": 1.0,
        "min_unwind_frac": 0.10,
        "levels": 3,
        "level_step_ticks": 1,
        "depth_decay": 0.6,
        "mm_queue_share": 0.6,
        "queue_jitter": 0.02,
        "risk_horizon_secs": 600,
        "taker_unwind_enabled": True,
        "taker_unwind_trigger": 0.6,
        "taker_unwind_target": 0.4,
        "taker_unwind_max_clips": 10,
        "taker_unwind_stale_mult": 2.0,
        "adverse_mm_ticks": 0.5,              # shift against MM (in ticks) after maker fill
        "cancel_slip_ticks": 0.5,             # extra adverse ticks if stale when hit
        "overnight_penalty_bps": 0.0,         # charge on EOD |inventory| notional
        "inventory_skew_gamma": 0.002,
        "quote_update_cost_ps": 0.00005,
        "ioc_cooldown_steps": 20,
    })

    process_params: Dict = field(default_factory= lambda: {
        "diag_interval": 5000,
        "init_price": 100.0,
        # GBM-only
        "mu": 1e-4, 
        "vol": 0.01,
        # OU-only
        "theta": 100.0,
        "kappa": 0.05,
        "sigma": 0.02,
        # Cross-asset correlation
        "correlation": 0.7,
        "cov_scale": 1.0,
        # Price impact
        "impact_eta": 0.0,
        "impact_decay": 0.0,
        "mm_latency_prob": 0.15,
        "stale_impact_boost": 1.0,
        "mm_refresh_interval": 50,
        "mm_refresh_ticks": 1,
        "markout_steps": 0,
        "maker_markout_ticks": 0.8,
        "maker_adverse_eta": 0.3,
        "bg_liquidity": True,
        "bg_inside_ticks": 1,
        "bg_levels": 3,
        "bg_size": 400.0,
        "bg_depth_decay": 0.7,
        "maker_drag_decay": 0.85,
        "maker_drag_stepcap": 1.5,
        "impact_max_ticks_per_step": 3.0,
        "max_anchor_move_ticks": 3.0,
        "annual_mu": 0.085,
        "regime_on": True,
        "regimes": {
            "side": {"mu_ann": 0, "vol_mult": 1.0, "p_stay": 0.98},
            "bull": {"mu_ann": 0.18, "vol_mult": 0.85, "p_stay": 0.985},
            "bear": {"mu_ann": -0.25, "vol_mult": 1.60, "p_stay": 0.975}
        },
        # Innovations
        "innovations": "student_t",
        "df": 5,
        # GARCH(1,1)
        "garch_enabled": True,
        "garch_omega": 1e-6,
        "garch_alpha": 0.05,
        "garch_beta": 0.92,
        # Merton jumps in log-returns
        "jump_lambda": 0.04,
        "jump_mu": 0.0,
        "jump_sigma": 0.03,
        # Intraday U-shape
        "intraday_u_shape": True,
        "u_open_close": 1.7,
        "u_midday": 1.0,
        "event_times_steps": [],          # e.g., [60, 120, 22800] if you want spikes
        "event_sigma_mult": 3.0,
        "event_half_life_secs": 180.0,
        "basis_enabled": True,
        "basis_half_life_secs": 90.0,
        "basis_sigma_ticks": 0.4,
        # Flow impact
        "micro_trade_impact_eta": 0.0,
        "flow_impact_enabled": True,
        "flow_impact_gamma": 0.25,
        "flow_impact_alpha": 0.6,
        "flow_impact_beta": 120.0,
        "flow_cross_rho": 0.3,
        "flow_include_mm": False,
        "flow_vol_norm": True,
        "flow_shock_cancel_ticks": 0.5,
    })

    external_order_params: Dict = field(default_factory= lambda: {
        "lambda": 70000,
        "price_sigma": 0.0005,
        "offset_tick_sigma": 2.0,
        "marketable_frac": 0.4,
        "retail_mu": 5, 
        "inst_mu": 50, 
        "retail_frac": 0.7,
        "buy_prob": 0.5,

        # --- REALISTIC FLOW BIASES (optional) ---
        # If True, modulate buy-vs-sell and aggression using a logistic transform of signals
        "use_price_bias": True,
        # "ath_contra": fewer buys as price pushes above prior-highs (profit-taking / supply),
        # "trend": more buys on positive short-term returns (herding/momentum),
        # "none": disable biasing.
        "bias_mode": "ath_contra",
        # Coefficients for buy probability and marketable fraction logits
        "buyprob_alpha": 2.0,
        "marketable_alpha": 1.0,

        # Hawkes Processes to model trade arrivals
        "hawkes_enabled": True,
        "hawkes_alpha_self": 0.20,
        "hawkes_alpha_cross": 0.10,
        "hawkes_beta": 50.0,
        "hawkes_cap_mult": 5.0,
        "trend_eta": 4.0,
        "vol_eta": 2.0,
        "mkt_trend_slope": 6.0,
        "mkt_vol_slope": 4.0,
        "size_vol_slope": 6.0,
    })

    fee_rate: float = 3e-5
    rebate_rate: float = 2.5e-5
    routing_fee_rate: float = 7e-6
    verbose: bool = True

    # ETF creation/redemption
    etf_ap: Dict = field(default_factory= lambda: {
        "enabled": True,
        "prem_threshold": 5e-4,
        "ap_kappa": 10.0,
         "ap_cost_bps": 2e-4,
        "sigma_nav_mult": 0.6,
        "max_ap_flow_per_day": 0.05
    })

    # Per-share fees (USD per share, set to 0 if not needed)
    per_share_fees: Dict[str, float] = field(default_factory=lambda: {
        "taker": 0.0030,
        "maker": 0.0015,
        "routing": 0.0002,
        "clearing":0.0001,
        "finra_taf_ps": 0.000119,
        "finra_taf_cap": 5.95,
    })

# ======================================================================
# 1. Abstract Base Classes
# ======================================================================

class BaseProcess(abc.ABC):
    """
    Abstract base class for all stochastic price processes.

    Subclasses must implement:
        step() → advances one simulation timestep
                    and returns a dictionary mapping
                    symbol → updated price.
    
    Subclasses can optionally accept a SimulationConfig for parameterized setup.
    """
    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config


    @abc.abstractmethod
    def step(self) -> Dict[str, float]:
        """Advance the process one step and return updated prices."""
        raise NotImplementedError("Subclasses must implement step().")
    

    @classmethod
    def configure(cls, config: SimulationConfig):
        """Instantiate subclass with configuration object."""
        return cls(config=config)


class BaseAgent(abc.ABC):
    """
    Abstract base class for all trading agents.

    Subclasses must implement:
        act() → define agent behavior at each simulation step.

    Subclasses can optionally accept a SimulationConfig for parameterized setup.
    """
    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config


    @abc.abstractmethod
    def act(self) -> None:
        """Define agent behavior per simulation step."""
        raise NotImplementedError("Subclasses must implement act().")
    
    
    @classmethod
    def configure(cls, config: SimulationConfig):
        return cls(config=config)


# ======================================================================
# 2. Core Entities
# ======================================================================

@dataclass
class Order:
    """
    Represents a single limit or market order.

    Attributes
    ----------
    symbol : str
        Asset identifier.
    price : float
        The quoted order price (rounded to tick size if config provided).
    size : float
        Quantity to buy or sell.
    side : {'buy', 'sell'}
        Order direction.
    trader : str, optional
        Owner identifier ('external' or 'mm').
    timestamp : float
        Creation time (UNIX seconds).
    """
    symbol: str
    price: float
    size: float
    side: Literal["buy", "sell"]
    trader: str = "external"  
    timestamp: float = 0.0
    _tick_size: float = 0.01 # internal default, overriden via config

    def __post_init__(self):
        """Basic validation after dataclass initialization."""
        if self.side not in ("buy", "sell"):
            raise ValueError(f"Invalid side '{self.side}'. Must be 'buy' or 'sell'.")
        if self.price <= 0 or self.size <= 0:
            raise ValueError("Order price and size must be positive numbers.")
    

    @staticmethod
    def _decimals_from_tick(tick: float) -> int:
        d, x = 0, float(tick)
        while d < 8 and abs(x - round(x)) > 1e-9:
            x *= 10.0; d += 1
        return d


    @staticmethod
    def snap(price: float, tick: float, side: str) -> float:
        k = price / tick
        k = math.floor(k + 1e-12) if side == "buy" else math.ceil(k - 1e-12)
        decimals = Order._decimals_from_tick(tick)
        return round(k * tick, decimals)


    @classmethod
    def from_config(cls, symbol: str, price: float, size: float,
                    side: str, trader: str, config: SimulationConfig,
                    timestamp: Optional[float] = None):
        """Factory to create an order aligned with simulation config."""
        return cls(symbol=symbol,
                   price=cls.snap(price, config.tick_size, side),
                   size=size, side=side, trader=trader,
                   _tick_size=config.tick_size,
                   timestamp=(time.time() if timestamp is None else float(timestamp)))
    

    @property
    def value(self) -> float:
        """Nominal notional value of the order."""
        return self.price * self.size

    def __repr__(self) -> str:
        """Human-readable summary of order details for debugging."""
        return (
            f"Order({self.side}@{self.price:.2f}, size={self.size:.2f}, "
            f"trader={self.trader})"
        )



@dataclass
class Trade:
    """
    Represents an executed trade.

    Attributes
    ----------
    symbol : str
        Asset identifier.
    price : float
        Execution price.
    size: float
        Execution size.
    buyer : str
        Trader ID of the buyer.
    seller : str
        Trader ID of the seller.
    timestamp : float
        Execution timestamp.
    liquidity_flag : {'maker', 'taker'}
        Optional tag for fee computation.
    fee_rate : float
        Trading fee applied to takers.
    rebate_rate : float
        Rebate received by makers.
    maker : str 
        Maker ID tag for fee computation.
    """
    symbol: str
    price: float
    size: float
    buyer: str
    seller: str
    timestamp: float = field(default_factory=time.time)
    liquidity_flag: str = "maker" # or "taker"
    fee_rate: float = 0.0001
    rebate_rate: float = 0.00005
    maker: str = "" # ("mm" or "external")
    bid_px: Optional[float] = None
    ask_px: Optional[float] = None
    aggressor: Optional[str] = None
    aggressor_side: Optional[str] = None
    resting_side: Optional[str] = None
    stale: bool = False
    step: Optional[int] = None

    @property
    def notional(self) -> float:
        """Total traded value."""
        return self.price * self.size


    @property
    def fee(self) -> float:
        """Transaction fee (taker pays, maker may receive rebate)."""
        if self.liquidity_flag == "taker":
            return -self.notional * self.fee_rate
        elif self.liquidity_flag == "maker":
            return self.notional * self.rebate_rate
        return 0.0
    

    @classmethod
    def from_config(cls, symbol: str, price: float, size: float,
                    buyer: str, seller: str, liquidity_flag: str,
                    config: SimulationConfig):
        """Factory constructor with config-based fees and tick size."""
        trade = cls(
            symbol=symbol,
            price=price,
            size=size,
            buyer=buyer,
            seller=seller,
            liquidity_flag=liquidity_flag,
            fee_rate=config.fee_rate,
            rebate_rate=config.rebate_rate,
        )
        return trade
    

    def __repr__(self) -> str:
        """Human-readable summary of trade details."""
        return (
            f"Trade(price={self.price:.2f}, size={self.size:.2f}, "
            f"buyer={self.buyer}, seller={self.seller}, flag={self.liquidity_flag})"
        )
# ======================================================================
# 3. Order Book
# ======================================================================

class OrderBook:
    """
    Simplified limit order book supporting multiple assets.

    Attributes
    ----------
    _books : Dict[str, Dict[str, List[Order]]]
        Nested dict: {symbol: {'buy' : [...], 'sell' : [...]}}
    _mid_prices : Dict[str, float]
        Current synthetic midprice per asset.
    config : SimulationConfig
        Provides tick size, depth limit, and matching preferences.

        
    Methods
    -------
    place_order(order: Order, symbol: str)
        Insert order into the correct book side, sorted.
    
    match_orders(symbol: str)
        Match best bid/ask; return executed trades.

    update_midprice(symbol: str, new_price: float)
        Refresh midprice after simulated price dynamics or trades.
    """
    def __init__(self, config: SimulationConfig):
        self.config = config
        self._symbols = config.symbols
        self._tick_size = config.tick_size
        self._max_depth = config.process_params.get("max_depth", 100)
        self._mm_queue_share = float(config.mm_params.get("mm_queue_share", 0.35))
        
        self._books: Dict[str, Dict[str, List[Order]]] = {
            s: {"buy": [], "sell": []} for s in self._symbols
            }
        init_px = config.process_params.get("init_price", 100.0)
        self._mid_prices: Dict[str, float] = {s: init_px for s in self._symbols}

        self.stale_mm: Dict[str, bool] = {s: False for s in self._symbols}


    # --- Properties ---
    @property
    def symbols(self) -> List[str]:
        """List of asset symbols tracked in this book."""
        return self._symbols
    
    @property
    def mid_prices(self) -> Dict[str, float]:
        """Dictionary of current midprices per symbol."""
        return self._mid_prices
    
    @mid_prices.setter
    def mid_prices(self, value: Dict[str, float]):
        """Update one or more midprices."""
        if isinstance(value, dict):
            self._mid_prices.update(value)
        else:
            try:
                d = dict(zip(self._symbols, list(value)))
                self._mid_prices.update(d)
            except Exception as e:
                raise TypeError(f"mid_prices expects dict or array-like of len {len(self._symbols)}") from e


    # --- Core Methods ---
    def place_order(self, symbol: str, order: Order):
        """
        Add an order to the right side and maintain sorted order book per side.
        
        Respects depth and tick-size rules.

        Buys are sorted descending (highest bid first).
        Sells are sorted ascending (lowest ask first).
        """
        if symbol not in self._books:
            raise ValueError(f"Unknown symbol '{symbol}'.")
        
        # snap to tick grid
        order.price = Order.snap(order.price, self._tick_size, order.side)

        book_side = self._books[symbol][order.side]
        book_side.append(order)

        if order.side == "buy":
            book_side.sort(key=lambda o: (-o.price, o.timestamp))
        else:
            book_side.sort(key=lambda o: (o.price, o.timestamp))

        if len(book_side) > self._max_depth:
            book_side.pop(-1)


    def match_orders(self, symbol: str) -> List[Trade]:
        """
        Match top of book orders and return executed trades.

        Returns
        -------
        trades : List[Trade]
            List of executed trades during this call
        """
        book = self._books[symbol]
        buys, sells = book["buy"], book["sell"]
        trades: List[Trade] = []

        # Continue matching as long as best bid >= best ask
        while buys and sells and buys[0].price >= sells[0].price:
            best_bid, best_ask = buys[0], sells[0]

            maker_is_bid = best_bid.timestamp <= best_ask.timestamp
            if best_bid.timestamp == best_ask.timestamp:
                maker_is_bid = (np.random.rand() >= 0.5)
            maker_trader = best_bid.trader if maker_is_bid else best_ask.trader
            aggressor_trader = best_ask.trader if maker_is_bid else best_bid.trader
            aggressor_side = "sell" if maker_is_bid else "buy"

            trade_price = best_bid.price if maker_is_bid else best_ask.price
            trade_size = min(best_bid.size, best_ask.size)
        
            trade = Trade.from_config(
                symbol=symbol,
                price=trade_price,
                size=trade_size,
                buyer=best_bid.trader,
                seller=best_ask.trader,
                liquidity_flag="maker" if maker_trader == "mm" else "taker",
                config=self.config,
            )
            trade.maker = maker_trader
            trade.bid_px = best_bid.price
            trade.ask_px = best_ask.price

            is_stale = (self.stale_mm.get(symbol, False) and maker_trader == "mm")
            trade.stale = is_stale

            trade.aggressor = aggressor_trader
            trade.aggressor_side = aggressor_side
            trade.resting_side = "buy" if maker_is_bid else "sell"

            trades.append(trade)

            # Update sizes or remove filled orders
            best_bid.size -= trade_size
            best_ask.size -= trade_size
            if best_bid.size <= 0:
                buys.pop(0)
            if best_ask.size <= 0:
                sells.pop(0)

        return trades


    def execute_market(self, symbol: str, side: str, size: float, trader: str):
        opp = "sell" if side == "buy" else "buy"
        fills = []
        while size > 1e-9 and self._books[symbol][opp]:
            best_px = self._books[symbol][opp][0].price
            level = [o for o in self._books[symbol][opp] if abs(o.price - best_px) < 1e-12]
            others = [o for o in level if o.trader != "mm"]
            mms    = [o for o in level if o.trader == "mm"]

            sum_oth = sum(o.size for o in others)
            sum_mm  = sum(o.size for o in mms)
            if sum_oth + sum_mm <= 1e-12:
                # clean empty
                for o in level: self._books[symbol][opp].remove(o)
                continue

        # target split
            tgt_mm  = min(size * self._mm_queue_share, sum_mm)
            tgt_oth = min(size - tgt_mm,              sum_oth)

            def _pro_rata(tgt, bucket):
                left = tgt
                for o in bucket:
                    if left <= 1e-9: break
                    take = min(left, o.size)
                    px   = o.price
                    bid_px, ask_px = self.top_of_book(symbol)
                    buyer = trader if side == "buy" else o.trader
                    seller= o.trader if side == "buy" else trader
                    t = Trade.from_config(symbol, px, take, buyer, seller, "taker", self.config)
                    t.maker = o.trader; t.aggressor = trader; t.aggressor_side = side
                    t.resting_side = opp; t.bid_px = bid_px; t.ask_px = ask_px
                    fills.append(t)
                    o.size -= take; left -= take

            _pro_rata(tgt_oth, others)
            _pro_rata(tgt_mm,  mms)
            size -= (tgt_oth + tgt_mm)

            # remove depleted orders
            for o in level:
                if o.size <= 1e-9: self._books[symbol][opp].remove(o)
            # if still size left at same price, loop continues; if not, next price level will be taken automatically
        return fills


    def update_midprice(self, symbol: str, new_price: float):
        """Update or drift the midprice for a symbol."""
        if symbol not in self._mid_prices:
            raise ValueError(f"Unknown symbol '{symbol}'.")
        self._mid_prices[symbol] = float(new_price)


    def top_of_book(self, symbol: str, exclude_trader: Optional[str] = None) -> Tuple[float, float]:
        buys = [o for o in self._books[symbol]["buy"] if exclude_trader is None or o.trader != exclude_trader]
        sells = [o for o in self._books[symbol]["sell"] if exclude_trader is None or o.trader != exclude_trader]
        mid = self._mid_prices[symbol]
        best_bid = max((o.price for o in buys), default = max(self._tick_size, mid - self._tick_size))
        best_ask = min((o.price for o in sells), default = (mid + self._tick_size))
        return best_bid, best_ask
    

    def cancel_trader(self, symbol: str, trader: str):
        b = self._books[symbol]
        b["buy"] = [o for o in b["buy"] if o.trader != trader]
        b["sell"] = [o for o in b["sell"] if o.trader != trader]


    # --- Magic Methods ---
    def __repr__(self):
        return f"<Orderbook symbols={self._symbols}, tick={self._tick_size}>"
    
    def __len__(self):
        """Total number of resting orders across all symbols."""
        return sum(len(v['buy']) + len(v['sell']) for v in self._books.values())


# ======================================================================
# 4. Price Process
# ======================================================================

class PriceProcess(BaseProcess):
    """
    Simulates midprice evolution for each asset.

    Supports correlated OU (mean-reverting) or GBM (diffusive) processes.

    Parameterized by SimulationConfig.

    Attributes
    ----------
    config : SimulationConfig
        Simulation-wide configuration.
    _symbols : List[str]
        Tracked asset identifiers.
    _dt : float
        Simulation timestep (fraction of a trading day).
    _model : str
        Type of stochastic process ('OU' or 'GBM').
    _prices: np.ndarray
        Current midprice vector.
    _cov : np.ndarray
        Covariance matrix defining inter-asset correlations.

    Methods
    -------
    step()
        Advance one time step and return updated midprices.
    """

    def __init__(self, config: SimulationConfig):
        super().__init__(config)
        self._symbols = config.symbols
        self._model = config.model.upper()
        self._dt = config.dt
        self._params = config.process_params

        n = len(self._symbols)
        self._prices = np.ones(n) * self._params.get("init_price", 100.0)

        # base parameters
        self._mu = np.full(n, self._params.get("mu", 1e-4))
        self._vol = np.full(n, self._params.get("vol", 0.02))
        self._theta = np.full(n, self._params.get("theta", 100.0))
        self._kappa = np.full(n, self._params.get("kappa", 0.05))
        self._sigma = np.full(n, self._params.get("sigma", 0.02)) 

        # covariance and correlation scaling
        self._cov_scale = self._params.get("cov_scale", 1)
        corr = self._params.get("correlation", 0.7)
        self._cov = self._build_cov_matrix(n, self._params.get("correlation", 0.7))

        # innovations + GARCH`
        self._steps_per_day = max(1, int(round(1.0 / self._dt)))
        self._acc_z = np.zeros(n)
        self._t_in_day = 0
        self._innov = str(self._params.get("innovations", "gaussian")).lower()
        self._df = int(self._params.get("df", 5))
        self._garch_on = bool(self._params.get("garch_enabled", False))
        self._go = float(self._params.get("garch_omega", 1e-6))
        self._ga = float(self._params.get("garch_alpha", 0.05))
        self._gb = float(self._params.get("garch_beta", 0.92))
        # daily variance state for GBM branch
        init_var = float(self._params.get("vol", 0.02)) ** 2
        self._sigma2 = np.full(n, init_var)

        # jumps (Merton)
        self._jl = float(self._params.get("jump_lambda", 0.0))
        self._jm = float(self._params.get("jump_mu", 0.0))
        self._js = float(self._params.get("jump_sigma", 0.0))

        # Intraday U-shape + event spike scaffolding
        intr = self._params
        self._intraday_on = bool(intr.get("intraday_u_shape", True))
        self._u_high = float(intr.get("u_open_close", 1.7))
        self._u_low = float(intr.get("u_midday", 1.0))
        self._last_intraday = 1.0

        self._event_times_steps = list(map(int, intr.get("event_times_steps", [])))
        self._event_sigma_mult = float(intr.get("event_sigma_mult", 3.0))
        hl_secs = float(intr.get("event_half_life_secs", 180.0))
        half_steps = max(1, int(round(hl_secs * (self._steps_per_day / 23400))))
        self._event_decay = math.exp(-math.log(2) / half_steps)
        self._event_level = 0.0

        # Annual drift + regimes
        self._mu_base_daily = math.log1p(float(self._params.get("annual_mu", 0.085))) / 252.0
        self._max_step_abs_r = None
        self._regime_on = bool(self._params.get("regime_on", True))
        self._regimes = dict(self._params.get("regimes", {}))
        self._reg_keys = list(self._regimes.keys())
        self._reg_idx = 0

        # ETF AP state
        if getattr(self.config, "etf_ap", None) and self.config.etf_ap.get("enabled", False):
            self._nav = self._prices.astype(float).copy()
            if not hasattr(self, "_steps_per_day") or self._steps_per_day is None:
                self._steps_per_day = max(1, int(round(1.0 / self._dt)))
            self._ap_last_flow = np.zeros_like(self._prices, dtype=float)
            self._ap_cost_accum = 0.0

    @staticmethod
    def _build_cov_matrix(n: int, corr: float) -> np.ndarray:
        """Constructs an N x N matrix with constant correlation."""
        C = np.full((n,n), corr, dtype=float)
        np.fill_diagonal(C, 1.0)
        return C


    @staticmethod
    def _random_noise(n: int, corr_mat: np.ndarray, innov: str = "gaussian", df: int = 5) -> np.ndarray:
        """Correlated unit-variance shocks (Gaussian or Student-t)."""
        try:
            L = np.linalg.cholesky(corr_mat)
        except np.linalg.LinAlgError:
            eps = 1e-10
            L = np.linalg.cholesky(corr_mat + np.eye(corr_mat.shape[0]) * eps)
        if innov == "student_t":
            z = np.random.standard_t(df, size=n)
            z = z * math.sqrt((df - 2) / df) # unit variance
        else:
            z = np.random.normal(size=n)
        return L @ z
    

    @property
    def prices(self) -> Dict[str, float]:
        return {s: float(p) for s, p in zip(self._symbols, self._prices)}


    def _intraday_mult(self) -> float:
        if not self._intraday_on:
            return 1.0
        t = (self._t_in_day % self._steps_per_day) / float(self._steps_per_day)
        return self._u_low + (self._u_high - self._u_low) * 0.5 * (1.0 + math.cos(2.0 * math.pi * t))


    # Core price evolution
    def step(self) -> Dict[str, float]:
        """
        Simulate one step of correlated price movement.

        The volatility is scaled appropriately for dt granularity.

        Returns
        -------
        Dict[str, float]
            Updated midprices for all symbols.
        """
        n = len(self._symbols)
        z = self._random_noise(n, self._cov, self._innov, self._df) * math.sqrt(self._cov_scale)

        if self._regime_on:
            R = self._regimes[self._reg_keys[self._reg_idx]]
            if np.random.rand() > float(R["p_stay"]):
                self._reg_idx = np.random.choice([i for i in range(len(self._reg_keys)) if i != self._reg_idx])
            R = self._regimes[self._reg_keys[self._reg_idx]]

            mu_daily = math.log1p(float(R.get("mu_ann", 0.0))) / 252.0
            vol_mult = float(R["vol_mult"])
        else:
            mu_daily = self._mu_base_daily
            vol_mult = 1.0


        intraday = self._intraday_mult()
        self._last_intraday = float(intraday)

        if (self._t_in_day in set(self._event_times_steps)) and self._event_times_steps:
            self._event_level = max(self._event_level, self._event_sigma_mult)
        else:
            self._event_level *= self._event_decay
        spike = 1.0 + self._event_level


        if self._model == "OU":
            # Ornstein-Uhlenbeck: mean-reverting
            sigma = self._sigma * vol_mult * intraday * spike
            drift = self._kappa * (self._theta - self._prices) * self._dt
            diffusion = sigma * math.sqrt(self._dt) * z
            self._prices += drift + diffusion

        elif self._model ==  "GBM":
            # Geometric Brownian Motion: log-normal diffusion
            # jumps in log-space (compound Poisson)
            if self._jl > 0.0:
                k = np.random.poisson(self._jl * self._dt * intraday, size=n)
                J = np.zeros(n)
                nz = k > 0
                # sum of k normals -> Normal(k*mu, sqrt(k)*sigma)
                if np.any(nz):
                    J[nz] = np.random.normal(loc=self._jm * k[nz], scale=self._js * np.sqrt(k[nz]))
            else:
                J = 0.0

            if self._garch_on:
                if not hasattr(self, "_steps_per_day") or self._steps_per_day is None:
                    self._steps_per_day = max(1, int(round(1.0 / self._dt)))
                if (not hasattr(self, "_acc_z")) or (self._acc_z.shape[0] != n):
                    self._acc_z = np.zeros(n)
                    self._t_in_day = 0

                sigma_daily = np.sqrt(self._sigma2) * vol_mult * intraday * spike
                drift = (mu_daily - 0.5 * (sigma_daily ** 2)) * self._dt
                diffusion = sigma_daily * math.sqrt(self._dt) * z
                r = drift + diffusion + J

                if getattr(self, "_max_step_abs_r", None) is not None:
                    r = np.clip(r, -self._max_step_abs_r, self._max_step_abs_r)

                self._prices *= np.exp(r)

                self._acc_z += z
                self._t_in_day += 1
                if self._t_in_day >= self._steps_per_day:
                    z_day = self._acc_z / math.sqrt(self._steps_per_day)

                    if self._ga + self._gb >= 0.999:
                        self._gb = 0.999 - self._ga

                    self._sigma2 = self._go + (self._ga * (z_day ** 2) + self._gb) * self._sigma2
                    self._sigma2 = np.clip(self._sigma2, 1e-8, (0.40 ** 2))
                    self._acc_z[:] = 0.0
                    self._t_in_day = 0
            
            else:
                sigma_daily = self._vol * vol_mult * intraday * spike
                drift = (mu_daily - 0.5 * (sigma_daily ** 2)) * self._dt
                diffusion = sigma_daily * math.sqrt(self._dt) * z
                r = drift + diffusion + J
                if getattr(self, "_max_step_abs_r", None) is not None:
                    r = np.clip(r, -self._max_step_abs_r, self._max_step_abs_r)
                self._prices *= np.exp(r)

        else:
            raise ValueError(f"Unknown model type '{self._model}'.")
        
        # ETF NAV & AP arbitrage
        if getattr(self.config, "etf_ap", None) and self.config.etf_ap.get("enabled", False):
            sigma_nav_mult = self.config.etf_ap.get("sigma_nav_mult", 0.6)
            base_sigma_daily = (np.sqrt(self._sigma2) if self._garch_on else self._vol) * vol_mult
            sigma_daily_nav = base_sigma_daily * sigma_nav_mult

            r_nav = mu_daily * self._dt + sigma_daily_nav * math.sqrt(self._dt) * z
            self._nav *= np.exp(r_nav)

            prem = (self._prices / self._nav) - 1.0
            thr = self.config.etf_ap.get("prem_threshold", 5e-4)
            excess = np.sign(prem) * np.maximum(np.abs(prem) - thr, 0.0)

            ap_kappa = self.config.etf_ap.get("ap_kappa", 10.0)
            r_ap = ap_kappa * self._dt * (np.log(self._nav) - np.log(self._prices))
            self._prices *= np.exp(r_ap)

            max_rate = self.config.etf_ap.get("max_ap_flow_per_day", 0.05) / self._steps_per_day
            flow_frac = np.clip(50.0 * excess, -max_rate, max_rate)
            self._ap_last_flow = flow_frac

            ap_cost_bps = self.config.etf_ap.get("ap_cost_bps", 2e-4)
            notional_flow = np.abs(flow_frac) * self._prices
            self._ap_cost_accum += ap_cost_bps * float(np.sum(notional_flow))

        # Prevent negative prices
        floor_px = max(1e-4, float(getattr(self.config, "tick_size", 0.01)))
        self._prices = np.maximum(self._prices, floor_px)

        return self.prices
    

    def pop_ap_cost(self) -> float:
        c = getattr(self, "_ap_cost_accum", 0.0)
        self._ap_cost_accum = 0.0
        return float(c)


    def __repr__(self):
        return f"<PriceProcess model={self._model} symbols={self._symbols}>"


# ======================================================================
# 5. Market Maker Agent
# ======================================================================

class MarketMaker(BaseAgent):
    """
    Configurable liquidity-providing agent quoting bid/ask prices 
    around mid and adapting spreads dynamically based on volatility
    and inventory (Avellaneda-Stoikov style).

    Attributes
    ----------
    config : SimulationConfig
        Global simulation parameters.
    symbols : List[str]
        Assets the market maker trades.
    _inventory : Dict[str, float]
        Current position per asset.
    _cash : float
        Account cash balance.
    _base_spread : float
        Initial symmetric spread
    _order_size : float
        Size per quote
    _gamma : float
        Risk-aversion coefficient
    _inventory_limit : float
        Maximum absolute inventory per asset

        
    Methods
    -------
    quote(symbol, , volatility)
        Generate bid/ask quotes for a symbol given mid and volatility estimate.

    update_inventory(trades)
        Adjust inventory and cash based on executed trades.

    adapt_spread(volatility, inventory)
        Dynamically adjust spread based on volatility and inventory risk.
    
    act()
        Placeholder required by BaseAgent.
    """
    def __init__(self, config: SimulationConfig):
        super().__init__(config)
        mm_cfg = config.mm_params

        self.symbols = config.symbols
        self._tick_size = config.tick_size

        # core quoting parameters
        self._base_spread = mm_cfg.get("spread", 0.1)
        self._order_size = mm_cfg.get("size", 10.0)
        self._gamma = mm_cfg.get("risk_aversion", 0.1)
        self._inventory_limit = mm_cfg.get("inventory_limit", 100.0)
        self._vol_sensitivity = mm_cfg.get("vol_sensitivity", 5.0)
        self._inventory_penalty = mm_cfg.get("inventory_penalty", 1.0)
        self._min_unwind = self._order_size * mm_cfg.get("min_unwind_frac", 0.10)
        self._levels = mm_cfg.get("levels", 3)
        self._level_step_ticks = mm_cfg.get("level_step_ticks", 1)
        self._depth_decay = mm_cfg.get("depth_decay", 0.6)
        self._risk_horizon_T = mm_cfg.get("risk_horizon_secs", 600) / 23400   # 10 min default
        self._skew_gamma = float(mm_cfg.get("inventory_skew_gamma",
                                            mm_cfg.get("risk_aversion", 0.1)))

        # state
        self._inventory = {s: 0.0 for s in self.symbols}
        self._cash = float(mm_cfg.get("initial_cash", 1000000.0))
        self._initial_cash = self._cash


    # --- Properties ---
    @property
    def cash(self) -> float:
        """Current cash balance."""
        return self._cash
    

    @cash.setter
    def cash(self, value: float):
        self._cash = value


    @property
    def inventory(self) -> Dict[str, float]:
        """Current inventory dictionary per symbol."""
        return self._inventory


    # --- Methods ---
    def quote(self, symbol: str, mid: float, volatility: float = 0.02) -> Tuple[Optional[Order], Optional[Order]]:
        """
        Return a bid and ask Order around the current midprice.
        
        Features:
            - Smoothly reduces order size as inventory approaches limit.
            - Widens spread based on volatility and inventory pressure.
            - Stops quoting entirely if at the hard boundary.
        """
        inv = self._inventory[symbol]
        lim = self._inventory_limit
        x = inv / lim
        inv_abs = abs(x)

        adjusted_spread = self.adapt_spread(volatility, inv)
        half = adjusted_spread / 2
        tau = self._risk_horizon_T
        sigma_ticks = max(1e-12, (volatility * mid) / self._tick_size)
        skew_ticks = self._skew_gamma * (inv / lim) * (sigma_ticks ** 2) * tau
        skew = skew_ticks * self._tick_size

        scaled = self._order_size * max(0.0, 1.0 - inv_abs**2)

        buy_size = scaled
        sell_size = scaled
        if inv >= lim:
            buy_size = 0.0
            sell_size = max(sell_size, self._min_unwind)
        elif inv <= -lim:
            sell_size = 0.0
            buy_size = max(buy_size, self._min_unwind)

        bid, ask = None, None
        if buy_size > 1e-9 and inv + buy_size <= lim:
            raw_bid = mid - half - skew
            bid = Order.from_config(symbol, raw_bid, buy_size, "buy", "mm", self.config)
        if sell_size > 1e-9 and inv - sell_size >= -lim:
            raw_ask = mid + half - skew
            ask = Order.from_config(symbol, raw_ask, sell_size, "sell", "mm", self.config)
        
        return bid, ask


    def _apply_fees(self, *, is_maker: bool, notional: float, size: float, side: Optional[str] = None, routed: bool=True) -> float:
        """
        Net cash impact from fees/rebates for THIS fill.
        Mix % of notional with per-share fees (config.per_share_fees).
        positive value increases cash, negative decreases cash.
        """
        taker = (self.config.fee_rate * notional) if (not is_maker) else 0.0
        rebate = (self.config.rebate_rate * notional) if is_maker else 0.0
        route = (getattr(self.config, "routing_fee_rate", 0.0) * notional) if routed else 0.0
        
        ps = getattr(self.config, "per_share_fees", {}) or {}
        taker_ps = float(ps.get("taker", 0.0))
        maker_ps = float(ps.get("maker", 0.0))
        routing_ps = float(ps.get("routing", 0.0))
        clearing_ps = float(ps.get("clearing", 0.0))
        finra_ps = float(ps.get("finra_taf_ps", 0.0))
        finra_cap = float(ps.get("finra_taf_cap", 0.0))

        per_share_core = -(maker_ps if is_maker else taker_ps) - routing_ps - clearing_ps

        taf = 0.0
        if (side == "sell") and (size > 0):
            taf = -min(finra_ps * float(size), finra_cap)

        sec_bps = float(getattr(self.config, "sec_fee_bps_sell", 0.0))
        sec_fee = -(sec_bps / 1e4) * notional if side == "sell" else 0.0

        return rebate - taker - route + per_share_core * max(0.0, float(size)) + taf + sec_fee


    def update_inventory(self, trades: List[Trade]):
        """
        Update inventory and cash from executed trades.
        Positive size = buy (long inventory), negative = sell (reduce inventory).

        Fee model:
        - The MM always pays a routing/clearing cost on its fills: fee_rate * notional
        - The MM receives a maker rebate only when it was the resting (maker) side
          on that trade: rebate_rate * notional
        - Net cash fee impact = rebate_if_maker - taker_fee - routing_cost
        """
        taker_rate = self.config.fee_rate
        maker_rebate = self.config.rebate_rate
        routing_rate = getattr(self.config, "routing_fee_rate", 7e-6)

        for t in trades:
            if not ("mm" in (t.buyer, t.seller)):
                continue

            notional = t.price * t.size
            is_mm_maker = (t.maker == "mm")
            mm_side = "buy" if t.buyer == "mm" else "sell"
           
            cash_fee = self._apply_fees(is_maker=is_mm_maker, notional=notional, size=t.size, side = mm_side, routed=True)

            if t.buyer == "mm":
                # MM bought - increase inventory, reduce cash
                self._inventory[t.symbol] = self._inventory.get(t.symbol, 0.0) + t.size
                self._cash -= notional
                self._cash += cash_fee
            else:
                # MM sold - decrease inventory, increase cash
                self._inventory[t.symbol] = self._inventory.get(t.symbol, 0) - t.size
                self._cash += notional
                self._cash += cash_fee


    def adapt_spread(self, volatility: float, inventory: float) -> float:
        """
        Risk-based spread adjustment function.

        The spread widens if:
            - volatility increases
            - inventory deviates from 0
        """
        inv_abs = abs(inventory / self._inventory_limit)
        base = max(2 * self._tick_size, self._base_spread)
        raw = base * (
            1.0
            + 0.1 * max(0.0, self._gamma)
            + self._vol_sensitivity * float(volatility)
            + 2.0 * (inv_abs ** 1.5)
        )
        return float(np.clip(raw, 2 * self._tick_size, 12 * self._tick_size))


    def quote_ladder(self, symbol: str, mid: float, volatility: float=0.02):
        inv = self._inventory[symbol]; lim = self._inventory_limit
        x = inv / lim; inv_abs = abs(x)
        spread = self.adapt_spread(volatility, inv)
        half = spread / 2
        tau = self._risk_horizon_T
        sigma_ticks = max(1e-12, (volatility * mid) / self._tick_size)
        skew_ticks = self._skew_gamma * (inv / lim) * (sigma_ticks ** 2) * tau
        skew = skew_ticks * self._tick_size

        base_size = self._order_size * max(0.0, 1.0 - inv_abs**2)
        if inv >= 0.999 * lim:
            base_size = max(base_size, self._min_unwind)
        elif inv <= -0.999 * lim:
            base_size = max(base_size, self._min_unwind)

        cap_buy = max(0.0, lim - inv)
        cap_sell = max(0.0, lim + inv)

        orders = []
        step_px = self._level_step_ticks * self._tick_size
        cum_buy = 0.0; cum_sell = 0.0

        for k in range(self._levels):
            size_k = base_size * (self._depth_decay ** k)

            if size_k < 1e-9: break

            rem_buy = max(0.0, cap_buy - cum_buy)
            s_buy = min(size_k, rem_buy)
            if s_buy > 1e-9 and (inv + s_buy) <= lim:
                raw_bid = mid - half - skew - k * step_px
                orders.append(Order.from_config(symbol, raw_bid, s_buy, "buy", "mm",self.config))
                cum_buy += s_buy

            rem_sell = max(0.0, cap_sell - cum_sell)
            s_sell = min(size_k, rem_sell)
            if s_sell > 1e-9 and (inv - s_sell) >= -lim:
                raw_ask = mid + half - skew + k * step_px
                orders.append(Order.from_config(symbol, raw_ask, s_sell, "sell", "mm", self.config))
                cum_sell += s_sell

            if cum_buy >= cap_buy and cum_sell >= cap_sell:
                break         
        
        return orders
    

    def within_limits(self, symbol: str) -> bool:
        """Check whether current inventory is within allowed limits."""
        return abs(self._inventory[symbol]) < self._inventory_limit


    def act(self):
        """Satisfy BaseAgent abstract method."""
        pass

    
    def __repr__(self):
        inv_str = {k: round(v, 2) for k, v in self._inventory.items()}
        return f"<MarketMaker spread={self._base_spread:.4f}, cash={self._cash:.2f}, inv={inv_str}>"
    

    def __len__(self):
        """Number of symbols handled by the market maker."""
        return len(self.symbols)
    

# ======================================================================
# 6. Simulation Engine
# ======================================================================  

class MarketSimulation:
    """
    Orchestrates the entire market simulation.

    Components:
        - PriceProcess : models underlying price dynamics.
        - OrderBook : maintains order flow and trade execution.
        - MarketMaker : provides continuous liquidity.
        - ExternalFlow: synthetic external orders (stochastic intensity).

    Controlled entirely by SimulationConfig.

    Methods
    -------
    from_config(config)
        Factory method to construct a simulation from config dict.

    run()
        Run simulation for N steps; collect stats.
    
    generate_external_orders()
        Randomly create external buy/sell orders.
    
    compute_pnl()
        Calculate cumulative PnL and performance metrics.

    plot_results(df)
        Static utility for plotting results.
    """
    def __init__(self, config: SimulationConfig):
        np.random.seed(config.seed)
        self.config = config
        self.symbols = config.symbols
        self.steps = config.steps

        self.order_book = OrderBook(config)
        self.price_process = PriceProcess(config)
        self.market_maker = MarketMaker(config)

        self.trades: List[Trade] = []
        self._ext_market_trades: List[Trade] = []
        self.pnl_history: List[Dict[str, float]] = []
        self._sym_idx: Dict[str, int] = {s: i for i,s in enumerate(self.symbols)}

        flow_cfg = config.external_order_params
        self.flow_intensity = float(flow_cfg.get("lambda", 70000)) # λ arrivals per day
        self.buy_prob_base = float(flow_cfg.get("buy_prob", 0.5))
        self.retail_frac = float(flow_cfg.get("retail_frac", 0.7))
        self.retail_mu = float(flow_cfg.get("retail_mu", 5))
        self.inst_mu = float(flow_cfg.get("inst_mu", 50))
        self.offset_tick_sigma = float(flow_cfg.get("offset_tick_sigma", 0.0))
        self.marketable_frac_base = float(flow_cfg.get("marketable_frac", 0.4))
        self.flow_vol = float(flow_cfg.get("price_sigma", 0.0))
        self.tick = self.config.tick_size

        self.use_price_bias = bool(flow_cfg.get("use_price_bias", True))
        self.bias_mode = str(flow_cfg.get("bias_mode", "ath_contra")).lower()
        self.buyprob_alpha = float(flow_cfg.get("buyprob_alpha", 2.0))
        self.marketable_alpha = float(flow_cfg.get("marketable_alpha", 1.0))

        self.hawkes_on = bool(flow_cfg.get("hawkes_enabled", True))
        self.h_alpha_self = float(flow_cfg.get("hawkes_alpha_self", 0.20))
        self.h_alpha_cross = float(flow_cfg.get("hawkes_alpha_cross", 0.10))
        self.h_beta = float(flow_cfg.get("hawkes_beta", 50.0))  # per day
        self.h_cap_mult = float(flow_cfg.get("hawkes_cap_mult", 5.0))
        self.trend_eta = float(flow_cfg.get("trend_eta", 4.0))
        self.vol_eta = float(flow_cfg.get("vol_eta", 2.0))
        self.mkt_trend_slope = float(flow_cfg.get("mkt_trend_slope", 6.0))
        self.mkt_vol_slope = float(flow_cfg.get("mkt_vol_slope", 4.0))
        self.size_vol_slope = float(flow_cfg.get("size_vol_slope", 6.0))

        self._hawkes_decay = math.exp(-self.h_beta * self.config.dt)
        self._xi_buy = {s: 0.0 for s in self.symbols}
        self._xi_sell = {s: 0.0 for s in self.symbols}
        self._sym_share = 1.0 / max(1, len(self.symbols))

        init_mids = {s: self.order_book.mid_prices[s] for s in self.symbols}
        self._prev_mid = dict(init_mids)
        self._rolling_max = dict(init_mids)

        self._cum_abs_inv = 0.0

        pp = config.process_params
        self.impact_eta = float(pp.get("impact_eta", 0.0))
        self.impact_decay = float(pp.get("impact_decay", 0.0))
        self.mm_latency_prob = float(pp.get("mm_latency_prob", 0.15))
        self.stale_impact_boost = float(pp.get("stale_impact_boost", 1.0))
        self._last_mm_mid = {s: self.order_book.mid_prices[s] for s in self.symbols}
        self.mm_refresh_interval = int(pp.get("mm_refresh_interval", 50))
        self.mm_refresh_ticks = float(pp.get("mm_refresh_ticks", 1))
        self.max_anchor_move_ticks = float(pp.get("max_anchor_move_ticks", 3.0))

        self.mm_queue_share = float(self.config.mm_params.get("mm_queue_share", 0.6))
        self.queue_jitter = float(self.config.mm_params.get("queue_jitter", 0.02))

        self._maker_drag = {s: 0.0 for s in self.symbols}
        self._maker_drag_decay = float(pp.get("maker_drag_decay", 0.85))
        self._maker_drag_stepcap = float(pp.get("maker_drag_stepcap", 1.5))

        self._bg_on = bool(pp.get("bg_liquidity", True))
        self._bg_inside_ticks = int(pp.get("bg_inside_ticks", 1))
        self._bg_levels = int(pp.get("bg_levels", 3))
        self._bg_size0 = float(pp.get("bg_size", 400.0))
        self._bg_decay = float(pp.get("bg_depth_decay", 0.7))

        self.micro_eta = float(pp.get("micro_trade_impact_eta", pp.get("impact_eta", 0.0)))
        self.flow_impact_on = bool(pp.get("flow_impact_enabled", True))
        self.flow_gamma = float(pp.get("flow_impact_gamma", 0.25))
        self.flow_alpha = float(pp.get("flow_impact_alpha", 0.6))
        self.flow_beta = float(pp.get("flow_impact_beta", 120.0))
        self.flow_cross_rho = float(pp.get("flow_cross_rho", 0.3))
        self.flow_include_mm = bool(pp.get("flow_include_mm", False))
        self.flow_vol_norm = bool(pp.get("flow_vol_norm", True))
        self._flow_decay = math.exp(-self.flow_beta * self.config.dt)
        self._z_flow = {s: 0.0 for s in self.symbols}
        self._flow_shock_cancel = float(pp.get("flow_shock_cancel_ticks", 0.5))

        self._mid_hist = {s: [] for s in self.symbols}

        self._basis_on = bool(pp.get("basis_enabled", True))
        hl_secs = float(pp.get("basis_half_life_secs", 90.0))
        steps_per_sec = max(1e-9, (1.0 / self.config.dt) / 23400.0)
        hl_steps = max(1, int(round(hl_secs * steps_per_sec)))
        self._basis_phi = math.exp(-math.log(2) / hl_steps)
        self._basis_sigma_ticks = float(pp.get("basis_sigma_ticks", 0.4))
        self._basis = {s: 0.0 for s in self.symbols}

        self._ioc_cd_len = int(self.config.mm_params.get("ioc_cooldown_steps", 20))
        self._ioc_cd = {s: 0 for s in self.symbols}

        # taker-unwind params
        mm = config.mm_params
        self._tw_enabled = bool(mm.get("taker_unwind_enabled", True))
        self._tw_trigger = float(mm.get("taker_unwind_trigger", 0.6))
        self._tw_target = float(mm.get("taker_unwind_target", 0.4))
        self._tw_max_clips = int(mm.get("taker_unwind_max_clips", 10))
        self._tw_stale_mult = float(mm.get("taker_unwind_stale_mult", 2.0))

        # new params
        self._prev_equity_cons = float(self.market_maker.cash)
        self._cb_long_px = {s: 0.0 for s in self.symbols}
        self._cb_long_qty = {s: 0.0 for s in self.symbols}
        self._cb_short_px = {s: 0.0 for s in self.symbols}
        self._cb_short_qty = {s: 0.0 for s in self.symbols}
        self._pnl_fee_prev = 0.0
        self._cum_realized = 0.0

        self.maker_adverse_eta = float(self.config.process_params.get("maker_adverse_eta", 0.0))

        # --- Diagnostics ---
        self._diag_every = int(self.config.process_params.get("diag_interval", 5000))
        self._diag_history = []
        self._diag_ex_counts = {s: {"buy":0, "sell":0, "mkt_buy":0, "mkt_sell":0, "vol_buy":0.0, "vol_sell":0.0} for s in self.symbols}
        self._diag_last_lambda = {s:(0.0,0.0) for s in self.symbols}
        self._diag_last_spread = {s:0.0 for s in self.symbols}
        self._diag_last_flow_delta = {s:0.0 for s in self.symbols}
        self._diag_window = {"maker": 0.0, "taker": 0.0}
        self._pnl_fee = 0.0
        self._pnl_spread = 0.0
        self._diag_ex_counts_win = {s: {"buy":0, "sell":0, "mkt_buy":0, "mkt_sell":0, "vol_buy":0, "vol_sell":0} for s in self.symbols}
        self._win_maker_edge_sum = 0.0
        self._win_maker_edge_n = 0
        self._cum_abs_inv_notional = 0.0

        cov = getattr(self.price_process, "_cov", None)
        if isinstance(cov, np.ndarray) and cov.shape[0] == len(self.symbols):
            d = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
            self._corr = cov / (d[:, None] * d[None, :])
        else:
            self._corr = np.eye(len(self.symbols))

        self.verbose = config.verbose


    @classmethod
    def from_config(cls, config: Dict):
        """Factory constructor from raw dict -> SimulationConfig."""
        return cls(SimulationConfig(**config))
    

    @staticmethod
    def _clip01(p: float, eps: float = 1e-6) -> float:
        return min(1 - eps, max(eps, p))


    @staticmethod
    def _logit(p: float, eps: float = 1e-6) -> float:
        p = MarketSimulation._clip01(p, eps)
        return math.log(p / (1 - p))


    @staticmethod
    def _inv_logit(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))


    def _quote_background_orders(self):
        if not self._bg_on: return
        step_px = self.config.tick_size
        for s in self.symbols:
            self.order_book.cancel_trader(s, "bg")
            mid = self.order_book.mid_prices[s]
            for k in range(self._bg_levels):
                off = (self._bg_inside_ticks + k) * step_px
                sz = self._bg_size0 * (self._bg_decay ** k)
                self.order_book.place_order(s, Order.from_config(s, mid - off, sz, "buy", "bg", self.config))
                self.order_book.place_order(s, Order.from_config(s, mid + off, sz, "sell", "bg", self.config))


    def _inventory_value_conservative(self) -> float:
        v = 0.0
        for s in self.symbols:
            bid_ex, ask_ex = self.order_book.top_of_book(s, exclude_trader="mm")
            spr = max(self.tick, ask_ex - bid_ex)
            hair = max(self.tick, 0.5 * spr + 0.5 * self.tick)
            q = self.market_maker.inventory[s]
            v += q * ((bid_ex - hair) if q >= 0 else (ask_ex + hair))
        return float(v)
    

    def _apply_trade_cost_basis(self, t: Trade) -> float:
        """Average-cost realized PnL with separate long/short buckets."""
        if "mm" not in (t.buyer, t.seller):
            return 0.0
        s, px, qty = t.symbol, float(t.price), float(t.size)
        realized = 0.0
        mm_side = "buy" if t.buyer == "mm" else "sell"

        if mm_side == "sell":
            close = min(qty, self._cb_long_qty[s])
            if close > 0:
                realized += (px - self._cb_long_px[s]) * close
                self._cb_long_qty[s] -= close
                qty -= close
            if qty > 0:
                tot = self._cb_short_qty[s] + qty
                self._cb_short_px[s] = (self._cb_short_px[s] * self._cb_short_qty[s] + px * qty) / tot
                self._cb_short_qty[s] = tot
        else:
            cover = min(qty, self._cb_short_qty[s])
            if cover > 0:
                realized += (self._cb_short_px[s] - px) * cover
                self._cb_short_qty[s] -= cover
                qty -= cover
            if qty > 0:
                tot = self._cb_long_qty[s] + qty
                self._cb_long_px[s] = (self._cb_long_px[s] * self._cb_long_qty[s] + px * qty) / tot
                self._cb_long_qty[s] = tot

        return realized
    

    def _reset_step_diag(self):
        self._diag_mm_maker_notional = 0.0
        self._diag_mm_taker_notional = 0.0
        self._step_micro_delta = {s: 0.0 for s in self.symbols}
        self._step_flow_delta = {s: 0.0 for s in self.symbols}

    def _print_diag(self, step, cash, inv_val):
        if step % self._diag_every:
            return

        mids = " ".join(f"{s}:{self.order_book.mid_prices[s]:.2f}" for s in self.symbols)
        invs = " ".join(f"{s}:{self.market_maker.inventory[s]:.0f}" for s in self.symbols)
        spds = " ".join(f"{s}:{self._diag_last_spread[s]/self.tick:.1f}t" for s in self.symbols)
        fimp = " ".join(f"{s}:{self._diag_last_flow_delta[s]/self.tick:.2f}t" for s in self.symbols)
        lamb = " ".join(f"{s}:{lb:.2f}/{ls:.2f}" for s,(lb,ls) in self._diag_last_lambda.items())
        
        maker_edge_t = (self._win_maker_edge_sum / self._win_maker_edge_n) if self._win_maker_edge_n else 0.0
        exln = " ".join(
            f"{s}:B{c['buy']}({c['mkt_buy']})/S{c['sell']}({c['mkt_sell']})"
            for s,c in self._diag_ex_counts_win.items()
        )
        
        print(f"step {step:>6d} | mids {mids} | inv {invs} | spr {spds} | flowΔ {fimp} | λ {lamb}")
        print(f"        | fills {exln} | maker/taker ${int(self._diag_window['maker'])}/{int(self._diag_window['taker'])} | maker_edge:{maker_edge_t:.2f} ticks")

        # reset window
        self._diag_window = {"maker": 0.0, "taker": 0.0}
        self._diag_ex_counts_win = {s: {"buy":0,"sell":0,"mkt_buy":0,"mkt_sell":0,"vol_buy":0.0,"vol_sell":0.0} for s in self.symbols}
        self._win_maker_edge_sum = 0.0
        self._win_maker_edge_n = 0


    def _update_rolling_max(self) -> None:
        for s in self.symbols:
            self._rolling_max[s] = max(self._rolling_max[s], self.order_book.mid_prices[s])
    

    def _set_prev_mid(self) -> None:
        for s in self.symbols:
            self._prev_mid[s] = self.order_book.mid_prices[s]


    def _mark_price(self, s: str) -> float:
        return self.order_book.mid_prices[s]
    

    def _step_ret(self, s: str) -> float:
        prev = max(self.tick, self._prev_mid[s])
        cur = max(self.tick, self.order_book.mid_prices[s])
        return math.log(cur / prev)
    

    def _cur_daily_vol(self, s: str) -> float:
        idx = self._sym_idx[s]
        if getattr(self.price_process, "_garch_on", False):
            return float(np.sqrt(self.price_process._sigma2[idx]))
        return float(self.price_process._vol[idx])
    

    def _dynamic_p_mkt(self, s: str, side: str, r: float, vol: float) -> float:
        base = self.marketable_frac_base
        align = max(r, 0.0) if side == "buy" else max(-r, 0.0)
        x = self.mkt_trend_slope * align + self.mkt_vol_slope * (abs(r) / (vol + 1e-12))
        return self._inv_logit(self._logit(base) +  self.marketable_alpha * x)
    

    def _size_scale(self, r: float, vol: float) -> float:
        return max(0.2, 1.0 + self.size_vol_slope * (abs(r) / (vol + 1e-12)))


    def _mm_ioc_hedge(self) -> List[Trade]:
        fills = []
        clip = float(self.config.mm_params["size"])
        min_frac = float(self.config.mm_params.get("min_unwind_frac", 0.10))

        for s in self.symbols:
            if self._ioc_cd.get(s, 0) > 0:
                continue

            inv, lim = self.market_maker.inventory[s], self.market_maker._inventory_limit
            if lim <= 0:
                continue

            bid, ask = self.order_book.top_of_book(s, exclude_trader="mm")
            mid = self.order_book.mid_prices[s]

            vol = self._cur_daily_vol(s); tau = self.market_maker._risk_horizon_T
            sigma_ticks = max(1e-12, (vol * mid) / self.tick)
            skew_ticks = self.market_maker._skew_gamma * (inv / lim) * (sigma_ticks ** 2) * tau
            r = mid - skew_ticks * self.tick

            need_inv = max(0.0, abs(inv) - self._tw_target * lim)
            inv_trigger = (abs(inv) / lim) >= self._tw_trigger

            edge_buy = (r >= (ask - 0.5 * self.tick))
            edge_sell = (r <= (bid + 0.5 * self.tick))
            need_edge = (edge_buy and inv <= 0) or (edge_sell and inv >= 0)
            stale = bool(self.order_book.stale_mm.get(s, False))

            fire = inv_trigger or need_edge or (stale and (edge_buy or edge_sell))
            if not fire:
                continue

            if inv_trigger:
                side = "sell" if inv > 0 else "buy"
                qty = need_inv
            else:
                if edge_buy and not edge_sell:
                    side = "buy"
                elif edge_sell and not edge_buy:
                    side = "sell"
                else:
                    side = "sell" if inv > 0 else "buy"
                qty = min_frac * clip
                if stale:
                    qty *= float(self._tw_stale_mult)

            qty = min(qty, self._tw_max_clips * clip)
            if qty <= 1e-9:
                continue

            self.order_book.cancel_trader(s, "mm")
            fills.extend(self.order_book.execute_market(s, side, qty, "mm"))
            self._ioc_cd[s] = self._ioc_cd_len

        return fills


    def _maybe_mm_unwind(self) -> List[Trade]:
        if not self._tw_enabled:
            return []
        fills: List[Trade] = []
        clip = float(self.config.mm_params.get("size", 10.0))
        for s in self.symbols:
            inv = self.market_maker.inventory[s]
            lim = float(self.market_maker._inventory_limit)
            if lim <= 0:
                continue
            ratio = abs(inv) / lim
            if ratio < self._tw_trigger:
                continue
            tgt_units = max(0.0, abs(inv) - self._tw_target * lim)
            if tgt_units <= 1e-9:
                continue
            mult = self._tw_stale_mult if self.order_book.stale_mm.get(s, False) else 1.0
            qty = min(tgt_units, clip * self._tw_max_clips) * mult
            side = "sell" if inv > 0 else "buy"
            self.order_book.cancel_trader(s, "mm")
            step_fills = self.order_book.execute_market(s, side, qty, "mm")
            fills.extend(step_fills)
        return fills


    def generate_external_orders(self, step: int) -> List[Tuple[str, Order]]:
        orders: List[Tuple[str, Order]] = []

        for s in self.symbols:
            mid = self.order_book.mid_prices[s]
            best_bid, best_ask = self.order_book.top_of_book(s)
            r = self._step_ret(s)
            vol = max(1e-12, self._cur_daily_vol(s))

            base_total_sym = self.flow_intensity * self._sym_share * self.config.dt

            p_buy = self.buy_prob_base
            if self.use_price_bias:
                tilt = 0.0
                if self.bias_mode == "trend":
                    tilt += self.buyprob_alpha * np.tanh(5.0 * r / (vol + 1e-12))
                elif self.bias_mode == "ath_contra":
                    dist_ath = (mid / max(self._rolling_max[s], self.tick)) - 1.0
                    tilt -= self.buyprob_alpha * np.tanh(50.0 * dist_ath)
                p_buy = self._inv_logit(self._logit(self.buy_prob_base) + tilt)
            
            base_buy_sym = base_total_sym * p_buy
            base_sell_sym = base_total_sym * (1.0 - p_buy)

            self._xi_buy[s] *= self._hawkes_decay
            self._xi_sell[s] *= self._hawkes_decay

            lam_buy = base_buy_sym + self.h_alpha_self * self._xi_buy[s] + self.h_alpha_cross * self._xi_sell[s]
            lam_sell = base_sell_sym + self.h_alpha_self * self._xi_sell[s] + self.h_alpha_cross * self._xi_buy[s]

            trend_mult = math.exp(self.trend_eta * r)
            vol_mult = 1.0 + self.vol_eta * (abs(r) / vol)
            lam_buy *= trend_mult * vol_mult
            lam_sell *= (1.0 / trend_mult) * vol_mult

            cap = self.h_cap_mult * base_total_sym
            lam_buy = min(max(lam_buy, 0.0), cap)
            lam_sell = min(max(lam_sell, 0.0), cap)

            self._diag_last_lambda[s] = (lam_buy, lam_sell)
            cnt = self._diag_ex_counts_win[s]

            n_buy = np.random.poisson(lam_buy)
            n_sell = np.random.poisson(lam_sell)

            self._xi_buy[s] += n_buy
            self._xi_sell[s] += n_sell


            def _mk(side: str):
                p_mkt = self._dynamic_p_mkt(s, side, r, vol)
                is_mkt = (np.random.rand() < p_mkt)

                is_retail = (np.random.rand() < self.retail_frac)
                mu = self.retail_mu if is_retail else self.inst_mu
                size = np.random.exponential(mu * self._size_scale(r, vol))

                if side == "buy":
                    cnt["buy"] += 1; cnt["vol_buy"] += size
                    if is_mkt: cnt["mkt_buy"] += 1
                else:
                    cnt["sell"] += 1; cnt["vol_sell"] += size
                    if is_mkt: cnt["mkt_sell"] += 1

                if is_mkt:
                    fills = self.order_book.execute_market(s, side, size, "external")
                    if fills:
                        self._ext_market_trades.extend(fills)
                    return

                if self.offset_tick_sigma > 0:
                    n_ticks = max(1, int(round(abs(np.random.normal(0, self.offset_tick_sigma)))))
                    px_off = n_ticks * self.tick
                    if side == "buy":
                        price = min(best_ask - self.tick, max(self.tick, best_bid - px_off))
                    else:
                        price = max(best_bid + self.tick, best_ask + px_off)
                else:
                    sigma = self.flow_vol if self.flow_vol > 0 else 5e-4
                    off = abs(np.random.normal(0, sigma))
                    if side == "buy":
                        price = min(best_ask - self.tick, max(self.tick, mid * (1 - off)))
                    else:
                        price = max(best_bid + self.tick, mid * (1 + off))
                    price = max(self.tick, price)

                ts = step + np.random.uniform(-0.5, 0.5)
                orders.append((s, Order.from_config(s, price, size, side, "external", self.config, timestamp=ts)))

            for _ in range(n_buy):
                _mk("buy")
            for _ in range(n_sell):
                _mk("sell")

        return orders


    def _apply_trade_impact(self, step_trades: List[Trade]) -> None:
        if self.micro_eta <= 0 or not step_trades:
            return
        max_ticks_per_step = float(self.config.process_params.get("impact_max_ticks_per_step", 3.0))

        for t in step_trades:
            side = getattr(t, "aggressor_side", None)
            if side not in ("buy", "sell"):
                continue
            # mid moves up when buyer aggresses, down when seller aggresses
            direction = +1.0 if side == "buy" else -1.0
        
            base_size = max(1e-9, getattr(self.market_maker, "_order_size", 1.0))
            scale = t.size / base_size

            # stale fills amplify adverse selection
            boost = (1.0 + self.stale_impact_boost) if getattr(t, "stale", False) else 1.0
            if getattr(t, "maker", "") == "mm":
                boost *= (1.0 + self.maker_adverse_eta)

            raw = direction * self.micro_eta * scale * boost
            delta = np.sign(raw) * min(abs(raw), max_ticks_per_step - abs(self._step_micro_delta[t.symbol]))
            if abs(delta) <= 0.0:
                continue
            self._step_micro_delta[t.symbol] += delta
            mid = self.order_book.mid_prices[t.symbol]
            self.order_book.update_midprice(t.symbol, max(self.tick, mid + float(delta) * self.tick))

            adverse_ticks = float(self.config.mm_params.get("adverse_mm_ticks", 0.5))
            cancel_ticks = float(self.config.mm_params.get("cancel_slip_ticks", 0.25))
            if getattr(t, "maker", "") == "mm":
                dir_sign = +1.0 if getattr(t, "resting_side", None) == "sell" else -1.0
                extra = dir_sign * (adverse_ticks + (cancel_ticks if getattr(t, "stale", False) else 0.0))

                room = max(0.0, max_ticks_per_step - abs(self._step_micro_delta[t.symbol]))
                inc = float(np.sign(extra) * min(abs(extra), room))
                if inc:
                    self._step_micro_delta[t.symbol] += inc
                    mid = self.order_book.mid_prices[t.symbol]
                    self.order_book.update_midprice(t.symbol, max(self.tick, mid + inc * self.tick))



    def _apply_flow_impact(self, step_trades: List[Trade]) -> None:
        if not self.flow_impact_on:
            return
        
        # decay flow memory
        for s in self.symbols:
            self._z_flow[s] *= self._flow_decay

        # accumulate signed external marketable flow (normalized by Mm size)
        q = np.zeros(len(self.symbols))
        base = max(1e-9, getattr(self.market_maker, "_order_size", 1.0))
        for t in step_trades:
            if (not self.flow_include_mm) and getattr(t, "aggressor", None) != "external":
                continue
            i = self._sym_idx[t.symbol]
            sgn = +1.0 if getattr(t, "aggressor_side", None) == "buy" else -1.0
            q[i] += sgn * (t.size / base)
        
        # update state
        for s in self.symbols:
            i = self._sym_idx[s]
            self._z_flow[s] += q[i]

        # concave transform + cross-mix
        z = np.array([self._z_flow[s] for s in self.symbols], dtype=float)
        phi = np.sign(z) * (np.abs(z) ** self.flow_alpha)
        mix = (1.0 - self.flow_cross_rho) * phi + self.flow_cross_rho * (self._corr @ phi)
        mix *= (1.0 - self._flow_decay)

        # optional vol normalization
        if self.flow_vol_norm:
            vols = np.array([max(1e-6, self._cur_daily_vol(s)) for s in self.symbols], dtype = float)
            scale = vols / float(np.median(vols))
            mix = mix / np.clip(scale, 0.5, 2.0)

        # apply in ticks
        mix = np.clip(mix, -5.0, 5.0)
        max_ticks_per_step = 1.0
        for s in self.symbols:
            i = self._sym_idx[s]
            mid = self.order_book.mid_prices[s]
            raw_ticks = self.flow_gamma * mix[i]
            room = max(0.0, max_ticks_per_step - abs(self._step_flow_delta[s]))
            inc = float(np.sign(raw_ticks) * min(abs(raw_ticks), room))
            if inc == 0.0:
                self._diag_last_flow_delta[s] = 0.0
                continue 
            self._step_flow_delta[s] += inc
            self._diag_last_flow_delta[s] = inc
            self.order_book.update_midprice(s, max(self.tick, mid + inc * self.tick))
            if abs(inc) >= self._flow_shock_cancel:
                self.order_book.cancel_trader(s, "mm")
                self.order_book.stale_mm[s] = True


    def _apply_maker_markout_penalty(self, step_trades: List[Trade]) -> None:
        ticks_per_clip = float(self.config.process_params.get("maker_markout_ticks", 1.2))
        base = max(1e-9, getattr(self.market_maker, "_order_size", 1.0))
        for t in step_trades:
            if getattr(t, "maker", "") != "mm":
                continue
            s = t.symbol
            sign = +1.0 if t.buyer == "mm" else -1.0
            self._maker_drag[s] += (-sign) * min(3.0, (t.size / base) * ticks_per_clip)


    def _apply_maker_drag(self):
        for s in self.symbols:
            d = self._maker_drag[s]
            if abs(d) > 1e-6:
                step = np.sign(d) * min(abs(d), self._maker_drag_stepcap)
                mid = self.order_book.mid_prices[s]
                self.order_book.update_midprice(s, max(self.tick, mid + step * self.tick))
            self._maker_drag[s] *= self._maker_drag_decay


    def _decay_impact(self) -> None:
        if self.impact_decay <= 0:
            return
        intr = float(getattr(self.price_process, "_last_intraday", 1.0))
        decay = self.impact_decay * (1.0 / max(1.0, intr))
        if decay <= 0:
            return
        for s in self.symbols:
            mid = self.order_book.mid_prices[s]
            anchor = self.price_process.prices[s]
            gap = anchor - mid
            adj = np.sign(gap) * min(abs(gap), self.max_anchor_move_ticks * self.tick)
            self.order_book.update_midprice(s, max(self.tick, mid + decay * adj))  


    def run(self):
        """
        Main simulation loop:
            - Advance price process
            - Market maker quotes bid/ask
            - Generate external orders
            - Match trades 
            - Update inventory, cash, PnL
        """
        for step in range(self.steps):
            self._reset_step_diag()

            for s in self.symbols:
                if self._ioc_cd[s] > 0:
                    self._ioc_cd[s] -= 1
            posted_mm = {s: False for s in self.symbols}

            # Step 1: evolve midprices
            new_prices = self.price_process.step()
            self.order_book.mid_prices = new_prices

            if self._basis_on:
                for s in self.symbols:
                    eps_ticks = np.random.normal(0.0, self._basis_sigma_ticks)
                    self._basis[s] = self._basis_phi * self._basis[s] + eps_ticks * self.tick
                    mid = self.order_book.mid_prices[s]
                    self.order_book.update_midprice(s, max(self.tick, mid + self._basis[s]))

            ap_cost = self.price_process.pop_ap_cost()
            if ap_cost:
                self.market_maker.cash -= ap_cost
                self._pnl_fee -= ap_cost

            # Helpers
            def _quote_mm():
                for sym in self.symbols:
                    mid = self.order_book.mid_prices[sym]
                    need_refresh = ((step % self.mm_refresh_interval) == 0) or \
                                    (abs(mid - self._last_mm_mid[sym]) >= self.mm_refresh_ticks * self.tick)
                    if not need_refresh:
                        continue

                    self.order_book.cancel_trader(sym, "mm")

                    idx = self._sym_idx[sym]
                    if self.price_process._model == "OU":
                        daily_vol = float(self.price_process._sigma[idx])
                    else:
                        daily_vol = self._cur_daily_vol(sym)
                    vol_est = daily_vol

                    spread_now = self.market_maker.adapt_spread(vol_est, self.market_maker.inventory[sym])
                    self._diag_last_spread[sym] = spread_now

                    tick = self.config.tick_size
                    fee_ps = self.config.per_share_fees
                    px = mid
                    maker_cost_usd = max(0.0, float(fee_ps.get("routing", 0))) + max(0.0, float(fee_ps.get("clearing", 0.0)) + max(0.0, float(fee_ps.get("maker", 0.0))))
                    maker_cost_ticks = maker_cost_usd / tick
                    bps_ticks = px * (self.config.routing_fee_rate - self.config.rebate_rate) / tick
                    min_edge_ticks = maker_cost_ticks + max(0.0, bps_ticks) + 0.15
                    if (spread_now / tick) * 0.5 < min_edge_ticks:
                        continue

                    mm_orders = self.market_maker.quote_ladder(sym, mid, vol_est)

                    for o in mm_orders:
                        best_bid, best_ask = self.order_book.top_of_book(sym)
                        if o.side == "buy" and best_ask is not None and o.price >= best_ask:
                            new_px = max(self.tick, best_ask - self.tick)
                            o = Order.from_config(sym, new_px, o.size, "buy", "mm", self.config)
                        elif o.side == "sell" and o.price <= best_bid:
                            new_px = best_bid + self.tick
                            o = Order.from_config(sym, new_px, o.size, "sell", "mm", self.config)
                        
                        r = np.random.rand()
                        jitter = self.queue_jitter * np.random.rand()
                        o.timestamp = step + (-jitter if r < self.mm_queue_share else jitter)
                        self.order_book.place_order(sym, o)

                    posted_mm[sym] = bool(mm_orders)
                    q_cost_ps = float(self.config.mm_params.get("quote_update_cost_ps", 0.0))
                    if q_cost_ps and mm_orders:
                        posted_shares = sum(o.size for o in mm_orders)
                        cost = q_cost_ps * posted_shares
                        self.market_maker.cash -= cost
                        self._pnl_fee -= cost

                    self._last_mm_mid[sym] = mid

            
            self._quote_background_orders()

            def _push_external():
                for sym, order in self.generate_external_orders(step):
                    self.order_book.place_order(sym, order)

            
            has_quotes = any(self.order_book._books[s]["buy"] or self.order_book._books[s]["sell"] for s in self.symbols)
            latency = has_quotes and (np.random.rand() < self.mm_latency_prob)

            # Step 2/3: Sequence with latency
            if latency:
                _push_external(); _quote_mm()
            else:
                _quote_mm(); _push_external()

            for sym in self.symbols:
                self.order_book.stale_mm[sym] = bool(latency and posted_mm[sym])

            step_trades: List[Trade] = []
            if getattr(self, "_mm_ioc_hedge", None) is not None:
                unwind_fills = self._mm_ioc_hedge()
                for t in unwind_fills:
                    t.step = step
                step_trades.extend(unwind_fills)

            # Step 4: match and record trades
            for sym in self.symbols:
                trades = self.order_book.match_orders(sym)
                for t in trades:
                    t.symbol = sym
                    t.step = step
                step_trades.extend(trades)

            if self._ext_market_trades:
                for t in self._ext_market_trades:
                    t.step = step
                step_trades.extend(self._ext_market_trades)
                self._ext_market_trades = []

            if step_trades:
                self.trades.extend(step_trades)
                self.market_maker.update_inventory(step_trades)
                self._apply_trade_impact(step_trades)
                self._apply_maker_markout_penalty(step_trades)
            self._apply_flow_impact(step_trades)
            self._apply_maker_drag()

            if step_trades:
                for t in step_trades:
                    if "mm" not in (t.buyer, t.seller):
                        continue
                    
                    bid0, ask0 = self.order_book.top_of_book(t.symbol)
                    mid0 = (bid0 + ask0) / 2.0
                    
                    if t.maker == "mm":
                        self._diag_mm_maker_notional += t.notional
                        edge0 = (t.price - mid0) if t.seller == "mm" else (mid0 - t.price)
                        self._pnl_spread += edge0 * t.size
                        self._win_maker_edge_sum += (edge0 / self.tick)
                        self._win_maker_edge_n += 1
                    else:
                        self._diag_mm_taker_notional += t.notional

                    fee_effect = self.market_maker._apply_fees(
                        is_maker=(t.maker=="mm"),
                        notional=t.notional,
                        size=t.size,
                        side=("buy" if t.buyer == "mm" else "sell"),
                        routed=True
                    )
                    self._pnl_fee += fee_effect

            steps_per_day = max(1, int(round(1.0 / self.config.dt)))
            if (step + 1) % steps_per_day == 0:
                pen = float(self.config.mm_params.get("overnight_penalty_bps", 0.0)) * 1e-4
                eod_notional = sum(abs(self.market_maker.inventory[s]) * self._mark_price(s) for s in self.symbols)
                self.market_maker.cash -= pen * eod_notional
                self._pnl_fee -= pen * eod_notional

            self._decay_impact()
            for sym in self.symbols:
                self.order_book.stale_mm[sym] = False

            # Step 5: record PnL snapshot
            cash_now = self.market_maker.cash
            
            realized_step = 0.0
            for _t in step_trades:
                realized_step += self._apply_trade_cost_basis(_t)
            self._cum_realized += realized_step

            fees_step = self._pnl_fee - self._pnl_fee_prev
            self._pnl_fee_prev = self._pnl_fee

            inv_val_mid = sum(self.market_maker.inventory[s] * self._mark_price(s) for s in self.symbols)
            inv_val_cons = self._inventory_value_conservative()
            equity_now = cash_now + inv_val_cons
            self._cum_abs_inv_notional += sum(
                abs(self.market_maker.inventory[s]) * self._mark_price(s) for s in self.symbols
            )

            step_change = equity_now - self._prev_equity_cons
            reval_step = step_change - realized_step - fees_step
            self._prev_equity_cons = equity_now

            abs_inv = sum(abs(self.market_maker.inventory[s]) for s in self.symbols)
            self._cum_abs_inv += abs_inv

            self.pnl_history.append({
                "step": step, 
                "cash": cash_now, 
                "inventory_value_mid": inv_val_mid, 
                "inventory_value_cons": inv_val_cons,
                "inventory_value": inv_val_mid,
                "equity": equity_now,
                "step_realized": realized_step,
                "step_reval": reval_step,
                "step_fees": fees_step,
                "net_worth": equity_now,
            })

            self._update_rolling_max() # update ATH with current mids
            for s in self.symbols:
                self._mid_hist[s].append(self.order_book.mid_prices[s])
            self._set_prev_mid() # set prev for next step

            self._diag_window["maker"] += self._diag_mm_maker_notional
            self._diag_window["taker"] += self._diag_mm_taker_notional

            if self.verbose and step % 5000 == 0:
                print(f"Step {step:>6d}/{self.steps} - net worth: {cash_now + inv_val_cons:,.2f}")
            self._print_diag(step, cash_now, inv_val_cons)

        if self.verbose:
            print(f"Simulation complete ({self.steps} steps, {len(self.trades)} trades).")


    # --- Metrics ---
    def compute_pnl(self) -> pd.DataFrame:
        """
        Compute mark-to-market equity and key performance metrics.

        Returns
        -------
        pd.DataFrame
            Contains columns with PnL components and summary stats.
        """
        df = pd.DataFrame(self.pnl_history)
        if df.empty:
            raise ValueError("PnL history is empty. Did you run the simulation?")

        # --- Core components ---
        df["equity_cons"] = df["equity"].astype(float)
        df["equity_mid"] = df["cash"].astype(float) + df["inventory_value_mid"].astype(float)
        df["step_pnl"] = df["equity_cons"].diff().fillna(0.0)
        df["realized_cum"] = df["step_realized"].cumsum()
        df["reval_cum"] = df["step_reval"].cumsum()
        df["fees_cum"] = df["step_fees"].cumsum()
        df["total_pnl"] = df["equity_cons"] - float(df["equity_cons"].iloc[0])
        
        # Daily aggregation
        steps_per_day = max(1, int(round(1.0 / float(self.config.dt))))
        grp = np.arange(len(df)) // steps_per_day
        daily_equity = df["equity_cons"].groupby(grp).last().astype(float)
        daily_ret = daily_equity.pct_change().dropna().values

        rf_annual = float(getattr(self.config, "risk_free_annual", 0.0) or 0.0)
        rf_daily = (1.0 + rf_annual) ** (1.0 / 252.0) - 1.0

        # Excess returns + robust stats
        ex = daily_ret - rf_daily
        eps = 1e-12
        if ex.size >= 1:
            mu_d = float(np.nanmean(ex))
            sig_d = float(np.nanstd(ex, ddof=1))
            sharpe = mu_d / max(sig_d, eps) * math.sqrt(252.0)

            dn = ex[ex < 0.0]
            if dn.size == 0:
                sortino = np.nan
            else:
                dn_sigma = float(np.nanstd(dn, ddof=1)) 
                sortino = mu_d / max(dn_sigma, eps) * math.sqrt(252.0)
        else:
            mu_d = sig_d = sharpe = sortino = np.nan

        # --- Risk metrics ---
        max_dd_abs, max_dd_pct = self._compute_drawdown(df["equity_cons"])
        sim_years = (len(df) / steps_per_day) / 252.0
        if sim_years > 0 and float(df["equity_cons"].iloc[0]) > 0:
            cagr = (float(df["equity_cons"].iloc[-1]) / float(df["equity_cons"].iloc[0])) ** (1.0 / sim_years) - 1.0 
        else:
            cagr = np.nan

        hit_ratio = self._compute_hit_ratio()
        turnover = self._estimate_turnover()

        # --- Summary dictionary ---
        summary = {
            "Final Equity": float(df["equity_cons"].iloc[-1]),
            "Step PnL (mean)": float(df["step_pnl"].mean()),
            "Total PnL": float(df["total_pnl"].iloc[-1]),
            "Realized PnL (cum)": float(df["realized_cum"].iloc[-1]),
            "Reval PnL (cum)": float(df["reval_cum"].iloc[-1]),
            "Fees/Rebates (cum)": float(df["fees_cum"].iloc[-1]),
            "Daily Return (mean)": float(mu_d) if np.isfinite(mu_d) else np.nan,
            "Daily Volatility": float(sig_d) if np.isfinite(sig_d) else np.nan,
            "Sharpe (ann.)": float(sharpe) if np.isfinite(sharpe) else np.nan,
            "Sortino (ann.)": float(sortino) if np.isfinite(sortino) else np.nan,
            "Max Drawdown (abs)": float(max_dd_abs),
            "Max Drawdown (%)": float(max_dd_pct * 100),
            "CAGR (annualized)": float(cagr),
            "Inventory Turnover": float(turnover),
            "Hit Ratio": float(hit_ratio),
            "Total Trades": len(self.trades),
            "Steps": self.steps,
        }

        if self.verbose:
            print("\n📊 Performance Summary")
            for k, v in summary.items():
                print(f"{k:25s}: {v:,.4f}")

        df.attrs["summary"] = summary
        return df


    # --- Helpers ---
    @staticmethod
    def _compute_drawdown(pnl_series: pd.Series) -> float:
        """Compute maximum drawdown in absolute value."""
        peak = pnl_series.cummax()
        dd_abs = (pnl_series - peak)
        dd_pct = (pnl_series / peak - 1.0).fillna(0.0)
        return float(-dd_abs.min()), float(-dd_pct.min())
    

    def _compute_cagr(self, pnl_series: pd.Series) -> float:
        """Approximate CAGR given total return over simulation length."""
        if pnl_series.iloc[0] <= 0:
            return np.nan

        total_return = pnl_series.iloc[-1] / pnl_series.iloc[0] - 1 
        simulated_years = len(pnl_series) * self.config.dt / 252
        if simulated_years <= 0 or total_return <= -1:
            return np.nan

        return (1 + total_return) ** (1 / simulated_years) - 1 
    

    def _estimate_turnover(self) -> float:
        """Rough estimate of inventory turnover (avg absolute trade size / avg inventory)."""
        mm_trades = [t for t in self.trades if "mm" in (t.buyer, t.seller)]
        if not mm_trades:
            return 0.0
        vol_notional = sum(t.notional for t in mm_trades)
        avg_inv_notional = self._cum_abs_inv_notional / max(1, self.steps)
        return vol_notional / max(1e-9, avg_inv_notional)

    
    def _compute_hit_ratio(self) -> float:
        """Fraction of profitable trades (simplified)."""
        mm_trades = [t for t in self.trades if "mm" in (t.buyer, t.seller) and t.step is not None]
        if not mm_trades:
            return np.nan
        steps_per_day = max(1, int(round(1.0 / self.config.dt)))
        cfg_k = int(self.config.process_params.get("markout_steps", 0))
        if cfg_k > 0:
            k = cfg_k
        else:
            steps_per_min = max(1, int(round(steps_per_day / 390)))
            k = steps_per_min
 
        wins = 0; denom = 0
        for t in mm_trades:
            hist = self._mid_hist.get(t.symbol)
            if not hist:
                continue
            idx = t.step + k
            if idx >= len(hist):
                continue
            future_mid = float(hist[idx])
            sign = +1.0 if t.buyer == "mm" else -1.0
            edge = sign * (future_mid - t.price)
            if edge > 0:
                wins += 1
            denom += 1
        return (wins / denom) if denom else np.nan
    

    # --- Plotting ---
    @staticmethod
    def plot_results(df: pd.DataFrame):
        """
        Plot midprice trajectories, inventory evolution, and PnL curves.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame returned by compute_pnl().
        """
        plt.figure(figsize=(10,5))
        plt.plot(df["step"], df["equity"], label="Equity (conservative)", lw=2)
        if "inventory_value_mid" in df:
            plt.plot(df["step"], df["inventory_value_mid"], label="Inventory (mid)", alpha=0.7)
        if "inventory_value_cons" in df:
            plt.plot(df["step"], df["inventory_value_cons"], label="Inventory (conservative)", alpha=0.7)
        plt.plot(df["step"], df["cash"], label="Cash", alpha=0.7)
        plt.title("Market Maker PnL Components")
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def __repr__(self):
        return f"<MarketSimulation steps={self.steps}, symbols={self.symbols}>"
    
    def __iter__(self):
        """Iterate over all traded symbols."""
        yield from self.symbols



# ======================================================================
# 7. Demo Entry Point
# ====================================================================== 

if __name__ == "__main__":
    # --- Configuration ---
    config = {
        "symbols": ["SPY", "QQQ", "DIA"], 
        "steps": 23400*42,
        "seed": 42,
        "dt": 1 / 23400,
        "model": "GBM",
        "tick_size": 0.01,
        
        "mm_params": {
            "spread": 0.01,
            "size": 200.0,
            "risk_aversion": 0.05,
            "vol_sensitivity": 0.50,
            "initial_cash": 5_000_000.0,
            "inventory_limit": 20000.0,
            "inventory_penalty": 2.0,
            "min_unwind_frac": 0.25,
            "levels": 5,
            "level_step_ticks": 1,
            "depth_decay": 0.5,
            "mm_queue_share": 0.5,
            "queue_jitter": 0.01,
            "risk_horizon_secs": 300,
            "taker_unwind_enabled": True,
            "taker_unwind_trigger": 0.6,
            "taker_unwind_target": 0.30,
            "taker_unwind_max_clips": 2,
            "taker_unwind_stale_mult": 2.0,
            "adverse_mm_ticks": 0.1,
            "cancel_slip_ticks": 0.05,
            "overnight_penalty_bps": 1.5,
            "inventory_skew_gamma": 0.002,
            "quote_update_cost_ps": 0.00002,
            "ioc_cooldown_steps": 30,
        },

        "process_params": {
            "diag_interval": 5000,
            "init_price": 100.0,
            "mu": 1e-4,
            "vol": 0.015,
            "theta": 100.0,
            "kappa": 0.05,
            "sigma": 0.02,
            "correlation": 0.88,
            "cov_scale": 1.0,
            "impact_eta": 0.0,
            "impact_decay": 0.02,
            "mm_latency_prob": 0.2,
            "stale_impact_boost": 2.0,
            "mm_refresh_interval": 10,
            "mm_refresh_ticks": 0.5,
            "innovations": "student_t",
            "df": 3,
            "garch_enabled": True,
            "garch_omega": 1e-6,
            "garch_alpha": 0.05,
            "garch_beta": 0.92,
            "jump_lambda": 0.04,
            "jump_mu": 0.0,
            "jump_sigma": 0.04,
            "intraday_u_shape": True,
            "u_open_close": 1.7,
            "u_midday": 1.0,
            "event_times_steps": [60, 120, 22800],          # e.g., [60, 120, 22800] if you want spikes
            "event_sigma_mult": 3.0,
            "event_half_life_secs": 180.0,
            "basis_enabled": True,
            "basis_half_life_secs": 90.0,
            "basis_sigma_ticks": 0.4,
            "micro_trade_impact_eta": 0.015,   
            "flow_impact_enabled": True,
            "flow_impact_gamma": 0.02,      
            "flow_impact_alpha": 0.6,       
            "flow_impact_beta": 60.0,       
            "flow_cross_rho": 0.3,           
            "flow_include_mm": True,        
            "flow_vol_norm": True,
            "flow_shock_cancel_ticks": 0.5,
            "markout_steps": 0,
            "maker_markout_ticks": 0.4,
            "maker_adverse_eta": 0.5,
            "bg_liquidity": True,
            "bg_inside_ticks": 2,
            "bg_levels": 4,
            "bg_size": 300.0,
            "bg_depth_decay": 0.6,
            "maker_drag_decay": 0.88,
            "maker_drag_stepcap": 1.0,
            "impact_max_ticks_per_step": 3.0,
            "max_anchor_move_ticks": 5.0, 
            "annual_mu": 0.085,
            "regime_on": True,
            "regimes": {
                "side": {"mu_ann": 0, "vol_mult": 1.0, "p_stay": 0.98},
                "bull": {"mu_ann": 0.18, "vol_mult": 0.85, "p_stay": 0.985},
                "bear": {"mu_ann": -0.25, "vol_mult": 1.60, "p_stay": 0.975}
            }          
        },

        "external_order_params": {
            "lambda": 20000,
            "price_sigma": 0.0005,
            "offset_tick_sigma": 1,
            "marketable_frac": 0.35,
            "retail_mu": 5,
            "inst_mu": 40,
            "retail_frac": 0.7,
            "buy_prob": 0.55,
            "use_price_bias": True,
            "bias_mode": "trend",
            "buyprob_alpha": 2.0,
            "marketable_alpha": 1.0,
            "hawkes_enabled": True,
            "hawkes_alpha_self": 0.13,
            "hawkes_alpha_cross": 0.05,
            "hawkes_beta": 50.0,
            "hawkes_cap_mult": 2.0,
            "trend_eta": 0.9,
            "vol_eta": 0.9,
            "mkt_trend_slope": 4.0,
            "mkt_vol_slope": 3.0,
            "size_vol_slope": 1.0,
        },

        "fee_rate": 0.0,           # taker = 0.2 bps
        "rebate_rate": 0.0,        # maker rebate = 0.05 bps
        "routing_fee_rate": 0.0, # = 0.002 bps (near-zero)
        "sec_fee_bps_sell": 0.0,
        "verbose": True,

        # ETF creation/redemption
        "etf_ap": {
            "enabled": False,
            "prem_threshold": 5e-4,        # 5 bps premium/discount trigger
            "ap_kappa": 5.0,              # tether strength (annualized)
            "ap_cost_bps": 2e-4,          # 2 bps = 0.02% cost on AP notional flow
            "sigma_nav_mult": 0.6,         # NAV smoother than trade price
            "max_ap_flow_per_day": 0.05
        },

        # Per-share fees (USD per share, set to 0 if not needed)
        "per_share_fees": {
            "taker": 0.0030,
            "maker": 0.0010,
            "routing": 0.00005,
            "clearing":0.00020,
            "finra_taf_ps": 0.000119,
            "finra_taf_cap": 5.95,
        },
    }
    
    sim_config = SimulationConfig(**config)


    # --- Display run summary ---
    print("\n🚀 Starting Market Maker Simulation...")
    print(
        f"Assets: {sim_config.symbols} | Steps: {sim_config.steps} | Seed: {sim_config.seed}\n"
        f"Model: {sim_config.model} | dt={sim_config.dt:.6f}"
    )

    # --- Run Simulation ---
    sim = MarketSimulation(sim_config)
    sim.run()

    # --- Compute Performance ---
    df_pnl = sim.compute_pnl()

    # --- Plot or save results---
    sim.plot_results(df_pnl)

    