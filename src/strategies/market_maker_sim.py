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
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


# ======================================================================
# 1. Abstract Base Classes
# ======================================================================

class BaseProcess(abc.ABC):
    """Abstract base class for all stochastic price processes."""
    @abc.abstractmethod
    def step(self) -> Dict[str, float]:
        """Advance the process one step and return updated prices."""
        pass


class BaseAgent(abc.ABC):
    """Abstract base class for all trading agents."""
    @abc.abstractmethod
    def act(self):
        """Define agent behavior per simulation step."""
        pass


# ======================================================================
# 2. Core Entities
# ======================================================================

@dataclass
class Order:
    """Represents a single limit or market order."""
    price: float
    size: float
    side: str                  # 'buy' or 'sell'
    trader: str = "external"   # or "mm" for market maker

    def __repr__(self):
        """Human-readable summary of order details."""
        return f"Order({self.side}@{self.price:.2f}, size={self.size:.2f}, {self.trader})"


@dataclass
class Trade:
    """Represents an executed trade."""
    price: float
    size: float
    buyer: str
    seller: str
    timestamp: int

    def __repr__(self):
        """Human-readable summary of trade details."""
        return f"Trade(price={self.price:.2f}, size={self.size:.2f})"


# ======================================================================
# 3. Order Book
# ======================================================================

class OrderBook:
    """
    Simplified limit order book supporting multiple assets.

    Attributes
    ----------
    books: Dict[str, Dict[str, List[Order]]]
        Nested dict: {symbol: {'buy' : [...], 'sell' : [...]}}

    mid_prices: Dict[str, float]
        Current synthetic midprice per asset.

        
    Methods
    -------
    place_order(order: Order, symbol: str)
        Insert order into the correct book side.
    
    match_orders(symbol: str)
        Match best bid/ask; return executed trades.

    update_midprice(symbol: str, new_price: float)
        Refresh midprice after simulated price dynamics or trades.
    """
    def __init__(self, symbols: List[str], init_price: float = 100.0):
        self._symbols = symbols
        self._books = {s: {"buy": [], "sell": []} for s in symbols}
        self._mid_prices = {s: init_price for s in symbols}

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
        """Add an order to the right side and maintain sorted order book per side."""
        pass # TODO

    def match_orders(self, symbol: str) -> List[Trade]:
        """Match top of book orders and return executed trades."""
        pass # TODO

    def update_midprice(self, symbol: str, new_price: float):
        """Update or drift the midprice for a symbol."""
        self._mid_prices[symbol] = new_price

    # --- Magic Methods ---
    def __repr__(self):
        return f"<Orderbook symbols={self._symbols}>"
    
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

    Attributes
    ----------
    symbols : List[str]
        Tracked asset identifiers.
    dt : float
        Simulation timestep (fraction of a trading day).
    model : str
        Type of stochastic process ('OU' or 'GBM').
    prices: np.ndarray
        Current midprice vector.
    cov : np.ndarray
        Covariance matrix defining inter-asset correlations.

    Methods
    -------
    step()
        Advance one time step and return updated midprices.
    """

    def __init__(self, symbols: List[str], dt: float = 1/390, model: str = "OU"):
        self._symbols = symbols
        self._model = model
        self._dt = dt
        self._prices = np.ones(len(symbols)) * 100.0
        self._cov = np.eye(len(symbols)) * 0.0001 # correlation matrix placeholder

    @property
    def prices(self) -> Dict[str, float]:
        """Current midprice dictionary."""
        return dict(zip(self._symbols, self._prices))

    @staticmethod
    def _random_noise(n: int, cov: np.ndarray) -> np.ndarray:
        """Generate correlated Gaussian noise of dimension n."""

    def step(self) -> Dict[str, float]:
        """Simulate one step of correlated price movement"""
        pass # TODO

    def __repr__(self):
        return f"<PriceProcess model={self._model} symbols={self._symbols}>"


# ======================================================================
# 5. Market Maker Agent
# ======================================================================

class MarketMaker(BaseAgent):
    """
    Liquidity provider quoting bid/ask prices around mid.

    Attributes
    ----------
    symbols : List[str]
        Assets the market maker trades.
    inventory : Dict[str, float]
        Current position per asset.
    cash : float
        Account cash balance.
    spread : float
        Quoted bid-ask spread in price units.
    size : float
        Order size per quote.

    Methods
    -------
    quote(symbol, mid)
        Generate bid/ask quotes for a symbol.

    update_inventory(trades)
        Adjust inventory and cash based on executed trades.

    adapt_spread(volatility, inventory)
        Dynamically adjust spread based on volatility and inventory risk.
    
    act()
        Placeholder required by BaseAgent.
    """

    def __init__(self, symbols: List[str], spread: float = 0.1, size: float = 10.0):
        self.symbols = symbols
        self.spread = spread
        self.size = size
        self._inventory = {s: 0.0 for s in symbols}
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
    def net_worth(self) -> float:
        """Mark-to-market total wealth (cash + inventory value placeholder)."""
        return self._cash + sum(self._inventory.values())

    # --- Methods ---
    def quote(self, symbol: str, mid: float) -> Tuple[Order, Order]:
        """Return a bid and ask Order around the current midprice."""
        pass # TODO

    def update_inventory(self, trades: List[Trade]):
        """Update inventory and cash from executed trades."""
        pass # TODO

    def adapt_spread(self, volatility: float, inventory: float):
        """Risk-based spread adjustment (Avellaneda-Stoikov style placeholder)."""
        pass # TODO

    def act(self):
        """Satisfy BaseAgent abstract method."""
        pass

    def __repr__(self):
        return f"<MarketMaker spread={self.spread} size={self.size}>"
    
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
        - External Orders: randomly generated to simulate real flow.

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

    def __init__(self, symbols: List[str], steps: int = 1000, seed: int = 42):
        np.random.seed(seed)
        self.symbols = symbols
        self.steps = steps
        self.order_book = OrderBook(symbols)
        self.price_process = PriceProcess(symbols)
        self.market_maker = MarketMaker(symbols)
        self.trades: List[Trade] = []
        self.pnl_history: List[float] = []

    # --- Classmethod ---
    @classmethod
    def from_config(cls, config: Dict):
        """Factory constructor from configuration dict."""
        return cls(**config)
    
    # --- Core loop ---
    def generate_external_orders(self) -> List[Tuple[str, Order]]:
        """
        Generate synthetic external market orders.

        Returns
        -------
        List[Tuple[str, Order]]
            A list of (symbol, Order) tuples to insert into the order book.
        """
        pass # TODO

    def run(self):
        """
        Main simulation loop:
            - Advance price process
            - Generate external and MM orders
            - Match trades and record results
        """
        pass # TODO

    # --- Metrics ---
    def compute_pnl(self) -> pd.DataFrame:
        """
        Compute mark-to-market PnL and related stats per symbol.

        Returns
        -------
        pd.DataFrame
            Contains columns for time, inventory, cash, and total PnL.
        """
        pass # TODO

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
        pass # TODO

    def __repr__(self):
        return f"<MarketSimulation n_steps={self.steps} n_symbols={len(self.symbols)}>"
    
    def __iter__(self):
        """Iterate over all traded symbols."""
        for s in self.symbols:
            yield s


# ======================================================================
# 7. Demo Entry Point
# ====================================================================== 

if __name__ == "__main__":
    config = {"symbols": ["SPY", "QQQ", "DIA"], "steps": 1000}
    sim = MarketSimulation.from_config(config)
    sim.run()
    sim.plot_results()