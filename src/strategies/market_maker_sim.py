"""
Market Maker Simulation (Multi-Asset, Synthetic Order Book)

This script simulates a simplified limit order book (LOB) environment
and a market making agent providing liquidity across multiple correlated assets.

Core features:
  • Multi-asset midprice dynamics (OU or GBM)
  • Poisson order arrivals and price impact
  • Dynamic quoting (Avellaneda–Stoikov model)
  • Inventory and risk control
  • PnL tracking and performance metrics
  • Visualization utilities for prices, inventory, and PnL

Created by Thomas Vrije for a quant-research portfolio project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


# ======================================================================
# 1. Data Structures
# ======================================================================

@dataclass
class Order:
    """Represents a single limit or market order."""
    price: float
    size: float
    side: str
    trader: str = "external" # or "mm" for market maker


@dataclass
class Trade:
    """Represents an executed trade."""
    price: float
    size: float
    buyer: str
    seller: str
    timestamp: int



# ======================================================================
# 2. Order Book
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
        Cross best bid/ask; return executed trades.

    update_midprice(symbol: str, new_price: float)
        Refresh midprice after simulated price dynamics or trades.
    """
    def __init__(self, symbols: List[str], init_price: float = 100.0):
        self.symbols = symbols
        self.books = {s: {"buy": [], "sell": []} for s in symbols}
        self.mid_prices = {s: init_price for s in symbols}

    def place_order(self, symbol: str, order: Order):
        """Add an order and maintain sorted order book per side."""
        pass # TODO

    def match_orders(self, symbol: str) -> List[Trade]:
        """Match top of book orders and update midprice if trades occur."""
        pass # TODO

    def update_midprice(self, symbol: str, new_price: float):
        """Set or drift the midprice for a symbol."""
        pass # TODO


# ======================================================================
# 3. Market Dynamics
# ======================================================================

class PriceProcess:
    """
    Simulates midprice evolution for each asset.

    Supports correlated OU (mean-reverting) or GBM (diffusive) processes.

    Methods
    -------
    step()
        Advance one time step and return updated midprices.
    """

    def __init__(self, symbols: List[str], dt: float = 1/390, model: str = "OU"):
        self.symbols = symbols
        self.model = model
        self.dt = dt
        self.prices = np.ones(len(symbols)) * 100.0
        self.cov = np.eye(len(symbols)) * 0.0001 # correlation matrix placeholder

    def step(self) -> Dict[str, float]:
        """Simulate one step of correlated price movement"""
        pass # TODO


# ======================================================================
# 4. Market Maker Agent
# ======================================================================

class MarketMaker:
    """
    Liquidity provider quoting bid/ask prices around mid.

    Attributes
    ----------
    inventory: Dict[str, float]
    cash: float
    spread: float
    size: float

    Methods
    -------
    quote(symbol, mid_price)
        Generate bid/ask quotes for a symbol.

    update_inventory(trades)
        Adjust inventory and cash based on executed trades.

    adapt_spread(volatility, inventory)
        Optional: dynamic spread control (Avellaneda-Stoikov).
    """

    def __init__(self, symbols: List[str], spread: float = 0.1, size: float = 10.0):
        self.symbols = symbols
        self.spread = spread
        self.size = size
        self.inventory = {s: 0.0 for s in symbols}
        self.cash = 0.0

    def quote(self, symbol: str, mid: float) -> Tuple[Order, Order]:
        """Return a bid and ask Order around the current midprice."""
        pass # TODO

    def update_inventory(self, trades: List[Trade]):
        """Update inventory and cash from executed trades."""
        pass # TODO

    def adapt_spread(self, volatility: float, inventory: float):
        """Optional: risk-based spread adjustment."""
        pass # TODO


# ======================================================================
# 5. Simulation Engine
# ======================================================================  

class MarketSimulation:
    """
    Orchestrates the entire market simulation.

    Components:
        - PriceProcess(midprice evolution)
        - OrderBook (matching)
        - MarketMaker (liquidity provision)
        - External order flow

    Methods
    -------
    run(steps)
        Run simulation for N steps; collect stats.
    
    generate_external_orders()
        Randomly create external buy/sell orders.
    
    compute_pnl()
        Calculate cumulative PnL and performance metrics.
    """

    def __init__(self, symbols: List[str], steps: int = 1000, seed: int = 42):
        np.random.seed(seed)
        self.symbols = symbols
        self.steps = steps
        self.order_book = OrderBook(symbols)
        self.price_process = PriceProcess(symbols)
        self.mm = MarketMaker(symbols)
        self.trades_log: List[Trade] = []
        self.pnl_history: List[float] = []

    def generate_external_orders(self) -> List[Tuple[str, Order]]:
        """Simulate random external orders for all symbols."""
        pass # TODO

    def run(self):
        """Run full market simulation loop."""
        pass # TODO

    def compute_pnl(self):
        """Compute mark-to-market PnL and related stats."""
        pass # TODO

    def plot_results(self):
        """Visualize price_paths, inventory, and PnL."""
        pass # TODO


# ======================================================================
# 6. Demo Entry Point
# ====================================================================== 

if __name__ == "__main__":
    symbols = ["SPY", "QQQ", "DIA"]
    sim = MarketSimulation(symbols=symbols, steps=1000)
    sim.run()
    sim.plot_results()