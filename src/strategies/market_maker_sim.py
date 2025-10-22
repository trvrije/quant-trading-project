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
                "vol_sensitivity": 5.0
            }
    process_params : Dict
        Parameters for the PriceProcess controlling price dynamics, e.g.:
            {
                "init_price": 100.0,
                "mu": 1e-4,
                "vol": 0.01,
                "cov_scale": 1e-4
            }
    external_order_params : Dict
        Parameters governing random external order flow (Poisson arrivals, size, mix), e.g.:
            {
                "lambda": 5,
                "retail_mu": 5,
                "inst_mu": 50,
                "mix": [0.7, 0.3]
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
        "vol_sensitivity": 5.0
    })

    process_params: Dict = field(default_factory= lambda: {
        "init_price": 100.0,
        "mu": 1e-4, 
        "vol": 0.01, 
        "cov_scale": 1e-4
    })

    external_order_params: Dict = field(default_factory= lambda: {
        "lambda": 5, 
        "retail_mu": 5, 
        "inst_mu": 50, 
        "mix": [0.7, 0.3]
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
        self._mid_prices: Dict[str, float] = {s: 100 for s in self._symbols}


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

        # depth limit check
        if len(book_side) >= self._max_depth:
            book_side.pop(-1) # drop worst order to maintain depth

        # Use bisect to keep list sorted efficiently
        prices = [o.price for o in book_side]
        if order.side == "buy":
            # insert maintaining descending order
            idx = len(prices) - bisect.bisect_left(list(reversed(prices)), order.price)
        else:
            # insert maintaining ascending order
            idx = bisect.bisect_left(prices, order.price)
        
        book_side.insert(idx, order)


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
            trade_price = Order._apply_tick_size(
            (best_bid.price + best_ask.price) / 2, self._tick_size
            )
            trade_size = min(best_bid.size, best_ask.size)

            # Determine who is taker/maker
            liquidity_flag = "maker" if best_bid.trader == "mm" or best_ask.trader == "mm" else "taker"

            trade = Trade.from_config(
                symbol=symbol,
                price=trade_price,
                size=trade_size,
                buyer=best_bid.trader,
                seller=best_ask.trader,
                liquidity_flag=liquidity_flag,
                config=self.config,
            )
            trades.append(trade)

            # Update sizes or remove filled orders
            best_bid.size -= trade_size
            best_ask.size -= trade_size
            if best_bid.size <= 0:
                buys.pop(0)
            if best_ask.size <= 0:
                sells.pop(0)

            # Update midprice to last trade price
            self._mid_prices[symbol] = trade_price

        return trades


    def update_midprice(self, symbol: str, new_price: float):
        """Update or drift the midprice for a symbol."""
        if symbol not in self._mid_prices:
            raise ValueError(f"Unknown symbol '{symbol}'.")
        self._mid_prices[symbol] = Order._apply_tick_size(new_price, self._tick_size)


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

        # state
        self._inventory = {s: 0.0 for s in self.symbols}
        self._cash = 0.0


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


    @property
    def net_worth(self) -> float:
        """Mark-to-market total wealth (cash + inventory value placeholder)."""
        return self._cash + sum(self._inventory.values())


    # --- Methods ---
    def quote(self, symbol: str, mid: float, volatility: float = 0.02) -> Tuple[Order, Order]:
        """
        Return a bid and ask Order around the current midprice.
        
        The spread can be dynamically adapted depending on inventory risk and volatility.
        """
        # Adapt spread based on current position (inventory pressure)
        inv = self._inventory[symbol]
        adjusted_spread = self.adapt_spread(volatility, inv)
        half = adjusted_spread / 2

        bid_price = Order._apply_tick_size(mid - half, self._tick_size)
        ask_price = Order._apply_tick_size(mid + half, self._tick_size)

        bid = Order.from_config(symbol, bid_price, self._order_size, "buy", "mm", self.config)
        ask = Order.from_config(symbol, ask_price, self._order_size, "sell", "mm", self.config)
        
        return bid, ask


    def update_inventory(self, trades: List[Trade]):
        """
        Update inventory and cash from executed trades.
        Positive size = buy (long inventory), negative = sell (reduce inventory).
        """
        for t in trades:
            if t.buyer == "mm":
                # MM bought - increase inventory, reduce cash
                self._inventory[t.symbol] = self._inventory.get(t.symbol, 0.0) + t.size
                self._cash -= t.price * t.size + t.fee
            elif t.seller == "mm":
                # MM sold - decrease inventory, increase cash
                self._inventory[t.symbol] = self._inventory.get(t.symbol, 0) - t.size
                self._cash += t.price * t.size + t.fee


    def adapt_spread(self, volatility: float, inventory: float) -> float:
        """
        Risk-based spread adjustment function.

        The spread widens if:
            - volatility increases
            - inventory deviates from 0
        """
        # risk-aversion coefficients
        inv_component = self._inventory_penalty * self._gamma * abs(inventory / self._inventory_limit)
        vol_component = self._vol_sensitivity * volatility
        return self._base_spread * (1 + inv_component + vol_component)


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
        self.time_index: List[int] = []

        flow_cfg = config.external_order_params
        self.flow_intensity = flow_cfg.get("lambda", 5.0) # λ arrivals per step
        self.flow_imbalance = flow_cfg.get("mix", [0.5, 0.5])[0] # buy probability
        self.flow_vol = flow_cfg.get("price_sigma", 0.05)
        self.flow_size = (flow_cfg.get("retail_mu", 5), flow_cfg.get("inst_mu", 50))

        self.verbose = config.verbose


    @classmethod
    def from_config(cls, config: Dict):
        """Factory constructor from raw dict -> SimulationConfig."""
        return cls(SimulationConfig(**config))
    
    
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

            # random side and size
            side = "buy" if np.random.rand() < self.flow_imbalance else "sell"
            size = np.random.uniform(*self.flow_size)

            # random price offset (to simulate small aggression)
            offset = np.random.normal(0, self.flow_vol)
            price = mid * (1 + offset) if side == "buy" else mid * (1 - offset)

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
                vol = self.price_process._params.get("vol", 0.01)
                bid, ask = self.market_maker.quote(sym, mid, vol)
                self.order_book.place_order(sym, bid)
                self.order_book.place_order(sym, ask)

            # Step 3: add random external orders
            for sym, order in self.generate_external_orders():
                self.order_book.place_order(sym, order)

            # Step 4: match and record trades
            step_trades = []
            for sym in self.symbols:
                trades = self.order_book.match_orders(sym)
                for t in trades:
                    # store which symbol the trade belongs to
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
            self.pnl_history.append(
                {"step": step, "cash": cash, "inventory_value": inv_val, "net_worth": cash + inv_val}
            )
            self.time_index.append(step)

            if self.verbose and step % 500 == 0:
                print(f"Step {step:>6d}/{self.steps} - net worth: {cash + inv_val:,.2f}")

        if self.verbose:
            print(f"Simulation complete ({self.steps} steps, {len(self.trades)} trades).")


    # --- Metrics ---
    def compute_pnl(self) -> pd.DataFrame:
        """
        Compute mark-to-market PnL time series and performance metrics.

        Returns
        -------
        pd.DataFrame
            Contains columns with PnL components and summary stats.
        """
        df = pd.DataFrame(self.pnl_history)
        df["total_pnl"] = df["cash"] + df["inventory_value"]
        df["returns"] = df["total_pnl"].pct_change().fillna(0)

        avg_r, vol = df["returns"].mean(), df["returns"].std()
        annual_factor = self._annualization_factor()
        sharpe = (avg_r / vol) * np.sqrt(annual_factor) if vol > 0 else np.nan 
        downside_std = df.loc[df["returns"] < 0, "returns"].std()
        sortino = (avg_r / downside_std) * np.sqrt(annual_factor) if downside_std > 0 else np.nan
        max_dd = self._compute_drawdown(df["total_pnl"])
        cagr = self._compute_cagr(df["total_pnl"])
        hit_ratio = self._compute_hit_ratio()
        turnover = self._estimate_turnover()

        summary = {
            "Final PnL": df["total_pnl"].iloc[-1],
            "Average Return": avg_r,
            "Volatility": vol,
            "Sharpe (ann.)": sharpe,
            "Sortino (ann.)": sortino,
            "Max Drawdown": max_dd,
            "CAGR (annualized)": cagr,
            "Inventory Turnover": turnover,
            "Hit ratio": hit_ratio,
            "Total Trades": len(self.trades),
            "Steps": self.steps,
        }
        if self.verbose:
            print("\n Performance Summary")
            for k, v in summary.items():
                print(f"{k:25s}: {v:,.4f}")

        df.attrs["summary"] = summary
        return df


    # --- Helpers ---
    def _annualization_factor(self) -> float:
        """Adjust annualization depending on simulation granularity."""
        if self.config.dt <= 1 / 23400: # tick-level (~6.5h*3600)
            return 252 * 23400
        elif self.config.dt <= 1 / 390: # minute-level
            return 252 * 390
        else:
            return 252


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
        avg_inv = np.mean([abs(v) for v in self.market_maker.inventory.values()]) + 1e-9
        return vol / avg_inv

    
    def _compute_hit_ratio(self) -> float:
        """Fraction of profitable trades (simplified)."""
        if not self.trades:
            return np.nan
        mm_trades = [t for t in self.trades if "mm" in (t.buyer, t.seller)]
        fair_value = getattr(self.price_process, "_theta", self.price_process._prices)
        wins = sum(1 for t in mm_trades if (t.seller == "mm" and t.price > fair_value[0]) or 
                   (t.buyer == "mm" and t.price < fair_value[0]))
        return wins / len(mm_trades) if mm_trades else np.nan
    

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
        plt.plot(df["step"], df["total_pnl"], label="Total PnL", lw=2)
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
        "steps": 200000,
        "seed": 42,
        "dt": 1 / 23400,
        "model": "OU",
        "tick_size": 0.01,

        "mm_params": {
            "spread": 0.1,
            "size": 10.0,
            "risk_aversion": 0.1,
            "vol_sensitivity": 5.0
        },

        "process_params": {
            "init_price": 100.0,
            "mu": 1e-4,
            "vol": 0.01,
            "cov_scale": 1e-4
        },

        "external_order_params": {
            "lambda": 5,
            "retail_mu": 5,
            "inst_mu": 50,
            "mix": [0.7, 0.3]
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

    