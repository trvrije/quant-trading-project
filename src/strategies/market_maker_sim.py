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
import bisect
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
    mm_params : Dict
        Parameters passed to the MarketMaker agent, e.g.:
            {
                "spread": 0.1,
                "size": 10.0,
                "risk_aversion": 0.1,
                "vol_sensitivity": 5.0,
                "initial_cash": 1000000.0,
                "inventory_limit": 100.0,
                "inventory_penalty": 1.0,
                "min_unwind_frac": 0.10
            }
    process_params : Dict
        Parameters for the PriceProcess controlling price dynamics, e.g.:
            {
                "init_price": 100.0,
                # GBM-only
                "mu": 1e-4,
                "vol": 0.01,
                # OU-only
                "theta": 100.0,
                "kappa": 0.05,
                "sigma": 0.02,
                # Cross-asset correlation
                "correlation": 0.8,
                "cov_scale": 1e-4
            }
    external_order_params : Dict
        Parameters governing random external order flow (Poisson arrivals, size, mix), e.g.:
            {
                "lambda": 5,
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
                "marketable_alpha": 1.0
            }
    fee_rate : float
        Per-trade fee applied to takers (fraction of notional).
    rebate_rate : float
        Rebate paid to makers (fraction of notional).
    verbose : bool
        If True, prints periodic simulation progress and performance summaries.
    """
    symbols: List[str] = field(default_factory=lambda: ["SPY"])
    steps: int = 1000
    dt: float = 1/390 # 1-min default
    model: str = "OU"
    seed: int = 42
    tick_size: float = 0.01

    mm_params: Dict = field(default_factory=lambda: {
        "spread": 0.1, 
        "size": 10.0,
        "risk_aversion": 0.1,
        "vol_sensitivity": 5.0,
        "initial_cash": 1000000.0,
        "inventory_limit": 100.0,
        "inventory_penalty": 1.0,
        "min_unwind_frac": 0.10,
    })

    process_params: Dict = field(default_factory= lambda: {
        "init_price": 100.0,
        # GBM-only
        "mu": 1e-4, 
        "vol": 0.01,
        # OU-only
        "theta": 100.0,
        "kappa": 0.05,
        "sigma": 0.02,
        # Cross-asset correlation
        "cov_scale": 1e-4,
        "correlation": 0.8,
    })

    external_order_params: Dict = field(default_factory= lambda: {
        "lambda": 5,
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
    })

    fee_rate: float = 0.0001
    rebate_rate: float = 0.00005
    verbose: bool = True


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
    timestamp: float = field(default_factory=time.time)
    _tick_size: float = 0.01 # internal default, overriden via config

    def __post_init__(self):
        """Basic validation and price snapping to tick size after dataclass initialization."""
        if self.side not in ("buy", "sell"):
            raise ValueError(f"Invalid side '{self.side}'. Must be 'buy' or 'sell'.")
        if self.price <= 0 or self.size <= 0:
            raise ValueError("Order price and size must be positive numbers.")
        self.price = self._apply_tick_size(self.price, self._tick_size)
    

    @classmethod
    def from_config(cls, symbol: str, price: float, size: float,
                    side: str, trader: str, config: SimulationConfig):
        """Factory to create an order aligned with simulation config."""
        order = cls(
            symbol=symbol,
            price=price,
            size=size,
            side=side,
            trader=trader,
            _tick_size=config.tick_size,
        )
        return order
    

    @staticmethod
    def _apply_tick_size(price: float, tick_size: float = 0.01) -> float:
        """Snap price to nearest multiple of tick size."""
        return round(price / tick_size) * tick_size
    

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
        self._match_mode = config.process_params.get("match_mode", "FIFO") # or 'pro-rata'
        
        self._books: Dict[str, Dict[str, List[Order]]] = {
            s: {"buy": [], "sell": []} for s in self._symbols
            }
        init_px = config.process_params.get("init_price", 100.0)
        self._mid_prices: Dict[str, float] = {s: init_px for s in self._symbols}


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
        self._mid_prices.update(value)


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
        order.price = Order._apply_tick_size(order.price, self._tick_size)

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
            maker_trader = best_bid.trader if maker_is_bid else best_ask.trader
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
            trades.append(trade)

            # Update sizes or remove filled orders
            best_bid.size -= trade_size
            best_ask.size -= trade_size
            if best_bid.size <= 0:
                buys.pop(0)
            if best_ask.size <= 0:
                sells.pop(0)

        return trades


    def update_midprice(self, symbol: str, new_price: float):
        """Update or drift the midprice for a symbol."""
        if symbol not in self._mid_prices:
            raise ValueError(f"Unknown symbol '{symbol}'.")
        self._mid_prices[symbol] = Order._apply_tick_size(new_price, self._tick_size)

    def top_of_book(self, symbol: str) -> Tuple[float, float]:
        buys = self._books[symbol]["buy"]
        sells = self._books[symbol]["sell"]
        mid = self._mid_prices[symbol]
        best_bid = buys[0].price if buys else max(self._tick_size, mid - self._tick_size)
        best_ask = sells[0].price if sells else (mid + self._tick_size)
        return best_bid, best_ask
    
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

        # model parameters (vectorized)
        # GBM drift/vol
        self._mu = np.full(n, self._params.get("mu", 1e-4))
        self._vol = np.full(n, self._params.get("vol", 0.01))
        self._theta = np.full(n, self._params.get("theta", 100.0))
        self._kappa = np.full(n, self._params.get("kappa", 0.05))
        self._sigma = np.full(n, self._params.get("sigma", 0.02)) 

        # covariance and correlation scaling
        cov_scale = self._params.get("cov_scale", 1e-4)
        corr = self._params.get("correlation", 0.8)
        self._cov = self._build_cov_matrix(n, cov_scale, corr)


    @staticmethod
    def _build_cov_matrix(n: int, var_scale: float, corr: float) -> np.ndarray:
        """Constructs an N x N matrix with constant correlation."""
        cov = np.full((n,n), corr * var_scale)
        np.fill_diagonal(cov, var_scale)
        return cov
    

    @property
    def prices(self) -> Dict[str, float]:
        """Current midprice dictionary."""
        return dict(zip(self._symbols, self._prices))


    @staticmethod
    def _random_noise(n: int, cov: np.ndarray) -> np.ndarray:
        """Generate correlated Gaussian noise via Cholesky factorization."""
        L = np.linalg.cholesky(cov)
        z = np.random.normal(size=n)
        return L @ z
    

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
        eps = self._random_noise(n, self._cov)

        if self._model == "OU":
            # Ornstein-Uhlenbeck: mean-reverting
            drift = self._kappa * (self._theta - self._prices) * self._dt
            diffusion = self._sigma * math.sqrt(self._dt) * eps
            self._prices += drift + diffusion

        elif self._model ==  "GBM":
            # Geometric Brownian Motion: log-normal diffusion
            drift = (self._mu - 0.5 * self._vol**2) * self._dt
            diffusion = self._vol * math.sqrt(self._dt) * eps
            self._prices *= np.exp(drift + diffusion)

        else:
            raise ValueError(f"Unknown model type '{self._model}'.")
        

        # Prevent negative prices
        self._prices = np.maximum(self._prices, 0.01)
        self._prices = np.round(self._prices / self.config.tick_size) * self.config.tick_size

        return self.prices

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
        skew = self._gamma * x * half

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
            bid_px = Order._apply_tick_size(mid - half - skew, self._tick_size)
            bid = Order.from_config(symbol, bid_px, buy_size, "buy", "mm", self.config)
        if sell_size > 1e-9 and inv - sell_size >= -lim:
            ask_px = Order._apply_tick_size(mid + half - skew, self._tick_size)
            ask = Order.from_config(symbol, ask_px, sell_size, "sell", "mm", self.config)
        
        return bid, ask


    def update_inventory(self, trades: List[Trade]):
        """
        Update inventory and cash from executed trades.
        Positive size = buy (long inventory), negative = sell (reduce inventory).
        """
        for t in trades:
            mm_is_buyer = (t.buyer == "mm")
            mm_is_seller = (t.seller == "mm")
            if not (mm_is_buyer or mm_is_seller):
                continue

            notional = t.price * t.size

            fee = (self.config.rebate_rate * notional) if t.maker == "mm" else (-self.config.fee_rate * notional)
           
            if mm_is_buyer:
                # MM bought - increase inventory, reduce cash
                self._inventory[t.symbol] = self._inventory.get(t.symbol, 0.0) + t.size
                self._cash -= notional
                self._cash += fee
            else:
                # MM sold - decrease inventory, increase cash
                self._inventory[t.symbol] = self._inventory.get(t.symbol, 0) - t.size
                self._cash += notional
                self._cash += fee


    def adapt_spread(self, volatility: float, inventory: float) -> float:
        """
        Risk-based spread adjustment function.

        The spread widens if:
            - volatility increases
            - inventory deviates from 0
        """
        inv_pressure = abs(inventory / self._inventory_limit)
        inv_component = self._inventory_penalty * (inv_pressure ** 2)
        vol_component = self._vol_sensitivity * volatility
        gamma_component = max(0.0, self._gamma) * 0.5
        return self._base_spread * (1 + inv_component + vol_component + gamma_component)


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
        self.pnl_history: List[Dict[str, float]] = []

        self._sym_idx: Dict[str, int] = {s: i for i,s in enumerate(self.symbols)}

        flow_cfg = config.external_order_params
        self.flow_intensity = float(flow_cfg.get("lambda", 5)) # λ arrivals per step
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

        init_mids = {s: self.order_book.mid_prices[s] for s in self.symbols}
        self._prev_mid = dict(init_mids)
        self._rolling_max = dict(init_mids)

        self._cum_abs_inv = 0.0

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


    def _biased_buy_prob(self, symbol: str, mid: float) -> Tuple[float, float]:
        """
        Returns (p_buy, p-marketable) after applying a logistic transform
        to a realistic microstructure signal.

        ath_contra: signal = max(0, mid/rolling_max - 1). Higher above prior-high -> fewer buys & more profit-taking.
        trend:      signal = short-term return (mid/prev_mid - 1). Positive -> more buys, more marketables.
        none:       no biasing.
        """
        p_buy = self.buy_prob_base
        p_mkt = self.marketable_frac_base
        if not self.use_price_bias or self.bias_mode == "none":
            return p_buy, p_mkt
        
        if self.bias_mode == "ath_contra":
            denom = max(self.tick, self._rolling_max[symbol])
            signal = max(0.0, mid / denom - 1.0)
            # fewer buys as we push new highs; slightly mre marketable sells
            p_buy = self._inv_logit(self._logit(p_buy) - self.buyprob_alpha * signal)
            p_mkt = self._inv_logit(self._logit(p_mkt) + self.marketable_alpha * signal)
        
        elif self.bias_mode == "trend":
            prev = max(self.tick, self._prev_mid[symbol])
            signal = (mid / prev) - 1.0
            p_buy = self._inv_logit(self._logit(p_buy) + self.buyprob_alpha * signal)
            p_mkt = self._inv_logit(self._logit(p_mkt) + self.marketable_alpha * signal)
        
        return p_buy, p_mkt


    def _update_rolling_max(self) -> None:
        for s in self.symbols:
            self._rolling_max[s] = max(self._rolling_max[s], self.order_book.mid_prices[s])
    

    def _set_prev_mid(self) -> None:
        for s in self.symbols:
            self._prev_mid[s] = self.order_book.mid_prices[s]


    def generate_external_orders(self) -> List[Tuple[str, Order]]:
        """
        Generate stochastic batch of synthetic external market orders.

        Uses Poisson arrivals and Gaussian price offsets.
        
        Returns
        -------
        List[Tuple[str, Order]]
            A list of (symbol, Order) tuples to insert into the order book.
        """
        orders = []
        n_arrivals = np.random.poisson(self.flow_intensity)

        for _ in range(n_arrivals):
            symbol = np.random.choice(self.symbols)
            mid = self.order_book.mid_prices[symbol]
            best_bid, best_ask = self.order_book.top_of_book(symbol)

            # biasing
            p_buy, p_mkt = self._biased_buy_prob(symbol, mid)

            # random side and size
            side = "buy" if np.random.rand() < p_buy else "sell"
            size = np.random.exponential(self.retail_mu) if np.random.rand() < self.retail_frac \
                    else np.random.exponential(self.inst_mu)

            if self.offset_tick_sigma > 0:
                n_ticks = max(1, int(round(abs(np.random.normal(0, self.offset_tick_sigma)))))
                px_off = n_ticks * self.tick

                if np.random.rand() < p_mkt:
                    price = (best_ask + px_off) if side == "buy" else (best_bid - px_off)
                else:
                    price = (best_bid - px_off) if side == "buy" else (best_ask + px_off)
            
            else:
                sigma = self.flow_vol if self.flow_vol > 0 else 0.0005
                off = abs(np.random.normal(0, sigma))
                if np.random.rand() < p_mkt:
                    price = max(best_ask, mid * (1 + off)) if side == "buy" else min(best_bid, mid * (1 - off))
                else:
                    tentative = mid * (1 - off) if side == "buy" else mid * (1 + off)
                    price = min(tentative, best_bid) if side == "buy" else max(tentative, best_ask)
         
            price = max(self.tick, price)

            orders.append((symbol, Order.from_config(symbol, price, size, side, "external", self.config)))
        return orders


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
            # Step 1: evolve midprices
            new_prices = self.price_process.step()
            self.order_book.mid_prices = new_prices

            # Step 2: market maker quotes per symbol
            for sym in self.symbols:
                mid = self.order_book.mid_prices[sym]
                idx = self._sym_idx[sym]
                if self.price_process._model == "OU":
                    vol_est = float(self.price_process._sigma[idx]) * math.sqrt(self.config.dt)
                else:
                    vol_est = float(self.price_process._vol[idx]) * math.sqrt(self.config.dt)

                bid, ask = self.market_maker.quote(sym, mid, vol_est)
                if bid:
                    self.order_book.place_order(sym, bid)
                if ask:
                    self.order_book.place_order(sym, ask)

            # Step 3: add random external orders
            for sym, order in self.generate_external_orders():
                self.order_book.place_order(sym, order)

            # Step 4: match and record trades
            step_trades: List[Trade] = []
            for sym in self.symbols:
                trades = self.order_book.match_orders(sym)
                for t in trades:
                    t.symbol = sym
                step_trades.extend(trades)

            if step_trades:
                self.trades.extend(step_trades)
                self.market_maker.update_inventory(step_trades)

            # Step 5: record PnL snapshot
            cash = self.market_maker.cash
            inv_val = sum(
                self.market_maker.inventory[sym] * self.order_book.mid_prices[sym]
                for sym in self.symbols
            )
            abs_inv = sum(abs(self.market_maker.inventory[s]) for s in self.symbols)
            self._cum_abs_inv += abs_inv
            self.pnl_history.append(
                {"step": step, "cash": cash, "inventory_value": inv_val, "net_worth": cash + inv_val}
            )

            self._update_rolling_max() # update ATH with current mids
            self._set_prev_mid() # set prev for next step

            if self.verbose and step % 500 == 0:
                print(f"Step {step:>6d}/{self.steps} - net worth: {cash + inv_val:,.2f}")

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
        df["equity"] = df["cash"] + df["inventory_value"]
        df["step_pnl"] = df["equity"].diff().fillna(0)
        initial_cash = getattr(self.market_maker, "_initial_cash", 1000000)
        df["returns"] = df["step_pnl"] / max(1e-9, initial_cash)

        # --- Basic statistics ---
        avg_r = df["returns"].mean()
        vol = df["returns"].std(ddof=1)
        downside_std = df.loc[df["returns"] < 0, "returns"].std(ddof=1)
        ann_factor = self._annualization_factor()

        # --- Annualized ratios ---
        sharpe = (avg_r / vol) * np.sqrt(ann_factor) if vol > 0 else np.nan
        sortino = (avg_r / downside_std) * np.sqrt(ann_factor) if (np.isfinite(downside_std) and downside_std > 0) else np.nan

        # --- Risk metrics ---
        max_dd = self._compute_drawdown(df["equity"])
        cagr = self._compute_cagr(df["equity"])
        hit_ratio = self._compute_hit_ratio()
        turnover = self._estimate_turnover()

        # --- Summary dictionary ---
        summary = {
            "Final Equity": float(df["equity"].iloc[-1]),
            "Step PnL (mean)": float(df["step_pnl"].mean()),
            "Average Return": float(avg_r),
            "Volatility": float(vol),
            "Sharpe (ann.)": float(sharpe) if np.isfinite(sharpe) else np.nan,
            "Sortino (ann.)": float(sortino) if np.isfinite(sortino) else np.nan,
            "Max Drawdown": float(max_dd),
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
        df["total_pnl"] = df["equity"]
        return df


    # --- Helpers ---
    def _annualization_factor(self) -> float:
        """
        Compute annualization factor = 252 * steps_per_day.
        Handles arbitrary time resolution automatically.
        """
        steps_per_day = max(1, int(round(1.0 / self.config.dt)))
        return 252 * steps_per_day


    @staticmethod
    def _compute_drawdown(pnl_series: pd.Series) -> float:
        """Compute maximum drawdown in absolute value."""
        cummax = pnl_series.cummax()
        return (pnl_series - cummax).min()
    

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
        if not self.trades:
            return 0.0
        vol = sum(t.size for t in self.trades)
        avg_inv_time = self._cum_abs_inv / max(1, self.steps)
        return vol / (avg_inv_time + 1e-9)

    
    def _compute_hit_ratio(self) -> float:
        """Fraction of profitable trades (simplified)."""
        mm_trades = [t for t in self.trades if ("mm" in (t.buyer, t.seller)) and (t.bid_px is not None) and (t.ask_px is not None)]
        if not mm_trades:
            return np.nan
        wins = 0
        for t in mm_trades:
            mid = 0.5 * (t.bid_px + t.ask_px)
            if t.seller == "mm" and t.price > mid:
                wins += 1
            elif t.buyer == "mm" and t.price < mid:
                wins += 1
      
        return wins / len(mm_trades) 
    

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
        plt.plot(df["step"], df["equity"], label="Total Equity", lw=2)
        plt.plot(df["step"], df["cash"], label="Cash", alpha=0.7)
        plt.plot(df["step"], df["inventory_value"], label="Inventory Value", alpha=0.7)
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
        "steps": 23400*5,
        "seed": 42,
        "dt": 1 / 23400,
        "model": "OU",
        "tick_size": 0.01,

        "mm_params": {
            "spread": 0.1,
            "size": 10.0,
            "risk_aversion": 0.1,
            "vol_sensitivity": 5.0,
            "initial_cash": 1000000.0,
            "inventory_limit": 100.0,
            "inventory_penalty": 1.0,
            "min_unwind_frac": 0.10,
        },

        "process_params": {
            "init_price": 100.0,
            "mu": 1e-4,
            "vol": 0.01,
            "theta": 100.0,
            "kappa": 0.05,
            "sigma": 0.02,
            "correlation": 0.8,
            "cov_scale": 1e-4,
        },

        "external_order_params": {
            "lambda": 5,
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
        },

        "fee_rate": 0.0001,
        "rebate_rate": 0.00005,
        "verbose": True,
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

    