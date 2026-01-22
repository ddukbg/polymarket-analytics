import pandas as pd
import time
import warnings
from typing import Optional
from datetime import datetime
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FuncFormatter
from polymarket_api import PolymarketAPI

warnings.filterwarnings('ignore')


class PortfolioAnalyzer:
    """
    Analyzes a Polymarket trader's portfolio with optional date filtering.
    
    Args:
        api: PolymarketAPI instance
        wallet_address: Wallet address to analyze
        start_date: Optional start date (datetime or string 'YYYY-MM-DD')
        end_date: Optional end date (datetime or string 'YYYY-MM-DD')
    """
    
    def __init__(self, api, wallet_address: str, 
                 start_date: Optional[str] = None, 
                 end_date: Optional[str] = None):
        self.api = api
        self.wallet = wallet_address
        
        # Convert dates to datetime and Unix timestamps
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None
        
        # Convert to Unix timestamps for API calls
        self.start_timestamp = (
            int(self.start_date.timestamp()) if self.start_date else None
        )
        self.end_timestamp = (
            int(self.end_date.timestamp()) if self.end_date else None
        )
        
        # Data containers
        self.df = None
        self.all_trades_df = None
        self.trade_dfs = {}
        self.title_mapping = {}
        self.summary_df = None
        self.winning_df = None
        self.losing_df = None
        
    def load_positions(self):
        """Load and process all positions for the wallet."""
        print(f"Loading positions for wallet: {self.wallet}")
        if self.start_date or self.end_date:
            print(f"Date filter: {self.start_date or 'Start'} to {self.end_date or 'End'}")
        
        # Get closed positions with API-level date filtering
        closed_positions = self.api.get_all_closed_positions(
            address=self.wallet,
            sort_by='TIMESTAMP',
            start_timestamp=self.start_timestamp,
            end_timestamp=self.end_timestamp
        )
        
        # Get open positions (these don't have timestamp filtering in API)
        open_positions = self.api.get_all_user_positions(self.wallet)
        
        # Process closed positions
        df_closed = pd.DataFrame()
        if closed_positions:
            df_closed = pd.DataFrame(closed_positions)
            df_closed['value'] = df_closed['avgPrice'] * df_closed['totalBought']
            df_closed['closed'] = True
            df_closed['timestamp'] = pd.to_datetime(df_closed['timestamp'], unit='s')
            
            # Convert endDate if it exists
            if 'endDate' in df_closed.columns:
                df_closed['endDate'] = df_closed['endDate'].replace('', None)
                df_closed['endDate'] = pd.to_datetime(df_closed['endDate'], errors='coerce', utc=True).dt.tz_localize(None)
            
            # Sort
            df_closed = df_closed.sort_values(by='timestamp', ascending=False)
       
        # Process open positions
        df_open = pd.DataFrame()
        if open_positions:
            df_open = pd.DataFrame(open_positions)
            df_open['value'] = df_open['avgPrice'] * df_open['totalBought']
            df_open = df_open.drop(columns=['realizedPnl'])
            df_open = df_open.rename(columns={'cashPnl': 'realizedPnl'})
            df_open['closed'] = False
            
            # Handle timestamp: use endDate if available, otherwise use a far future date
            df_open['endDate'] = df_open['endDate'].replace('', None)
            
            # Convert endDate to datetime and strip timezone
            df_open['endDate'] = pd.to_datetime(df_open['endDate'], errors='coerce', utc=True).dt.tz_localize(None)
            
            # Create timestamp from endDate
            df_open['timestamp'] = df_open['endDate'].copy()
            
            # For missing endDates (NaT), use a recognizable far future date
            far_future = pd.Timestamp('2099-12-31')
            df_open.loc[df_open['timestamp'].isna(), 'timestamp'] = far_future
            df_open.loc[df_open['endDate'].isna(), 'endDate'] = far_future
            
            # Sort
            df_open = df_open.sort_values(by='timestamp', ascending=False)

        # Combine positions
        self.df = pd.concat([df_closed, df_open], ignore_index=True)\
            .sort_values(by='timestamp', ascending=False)\
            .reset_index(drop=True)
        
        # Apply additional client-side date filtering for open positions
        if (self.start_date or self.end_date) and len(self.df) > 0:
            original_count = len(self.df)
            if self.start_date:
                self.df = self.df[self.df['timestamp'] >= self.start_date]
            if self.end_date:
                self.df = self.df[self.df['timestamp'] <= self.end_date]
            
            if len(self.df) < original_count:
                print(f"Client-side filtering: {original_count} -> {len(self.df)} positions")
        
        if len(self.df) == 0:
            print("\nNo positions found in date range")
            return self.df
            
        print(f"\nTotal positions: {len(self.df)}")
        print(f"  Closed: {len(self.df[self.df['closed'] == True])}")
        print(f"  Open: {len(self.df[self.df['closed'] == False])}")
        print(f"Date range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")
        
        return self.df
    
    def fetch_event_ids(self):
        """Map event slugs to event IDs."""
        if self.df is None or len(self.df) == 0:
            print("No positions to fetch event IDs for")
            return
            
        unique_slugs = self.df['eventSlug'].unique()
        slug_to_id = {}
        
        print(f"\nFetching event IDs for {len(unique_slugs)} unique slugs...")
        
        for i, slug in enumerate(unique_slugs, 1):
            try:
                event = self.api.get_event_by_slug(slug)
                slug_to_id[slug] = event['id']
                
                if i % 10 == 0:
                    print(f"Progress: {i}/{len(unique_slugs)}")
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"Error fetching {slug}: {e}")
                slug_to_id[slug] = None
        
        print(f"✓ Completed: {len(slug_to_id)} slugs processed")
        
        self.df['eventId'] = self.df['eventSlug'].map(slug_to_id)
        
    def load_trades(self):
        """Load all trades for the positions."""
        if self.df is None or len(self.df) == 0:
            print("No positions to load trades for")
            self.all_trades_df = pd.DataFrame()
            return self.all_trades_df
            
        unique_event_ids = self.df['eventId'].dropna().unique()
        print(f"\nFetching trades for {len(unique_event_ids)} unique events...")
        
        all_trades = []
        
        for i, event_id in enumerate(unique_event_ids, 1):
            try:
                event_trades = self.api.get_all_user_trades(
                    address=self.wallet,   # Changed: use 'address' parameter
                    event_id=event_id      # Changed: use named parameter
                )
                all_trades.extend(event_trades)
                
                if i % 10 == 0:
                    print(f"Progress: {i}/{len(unique_event_ids)} - Total: {len(all_trades)}")
                    
                time.sleep(0.1)
                    
            except Exception as e:
                print(f"Error fetching event {event_id}: {e}")
        
        print(f"✓ Completed: {len(all_trades)} trades fetched")
        
        if not all_trades:
            self.all_trades_df = pd.DataFrame()
            return self.all_trades_df
        
        # Assemble all trades into one dataframe
        self.all_trades_df = pd.DataFrame(all_trades)
        self.all_trades_df['timestamp'] = pd.to_datetime(
            self.all_trades_df['timestamp'], unit='s'
        )
        self.all_trades_df['value'] = (
            self.all_trades_df['size'] * self.all_trades_df['price']
        )
        
        # Map closed status from df
        asset_to_closed = self.df.set_index('asset')['closed'].to_dict()
        self.all_trades_df['closed'] = (
            self.all_trades_df['asset'].astype(str).map(asset_to_closed)
        )
        
        # Merge curPrice from df
        asset_to_curprice = self.df.set_index('asset')['curPrice'].to_dict()
        self.all_trades_df['curPrice'] = (
            self.all_trades_df['asset'].astype(str).map(asset_to_curprice)
        )
        
        # Current value of the position
        self.all_trades_df['current_value'] = (
            self.all_trades_df['curPrice'] * self.all_trades_df['size']
        )

        # Calculate trade P&L: current value - initial value
        self.all_trades_df['trade_pnl'] = (
            (self.all_trades_df['curPrice'] * self.all_trades_df['size']) - 
            self.all_trades_df['value']
        )

        # Calculate percent P&L
        self.all_trades_df['percent_pnl'] = (
            100 * self.all_trades_df['trade_pnl'] / self.all_trades_df['value']
        )
            
        # Apply date filtering to trades
        if self.start_date or self.end_date:
            original_count = len(self.all_trades_df)
            if self.start_date:
                self.all_trades_df = self.all_trades_df[
                    self.all_trades_df['timestamp'] >= self.start_date
                ]
            if self.end_date:
                self.all_trades_df = self.all_trades_df[
                    self.all_trades_df['timestamp'] <= self.end_date
                ]
            
            if len(self.all_trades_df) < original_count:
                print(f"Trade date filtering: {original_count} -> {len(self.all_trades_df)} trades")
        
        # Sort by timestamp descending (newest first)
        self.all_trades_df = self.all_trades_df.sort_values('timestamp', ascending=False).reset_index(drop=True).drop(columns=['profileImage','profileImageOptimized'], errors='ignore')
        
        return self.all_trades_df
        
    # def process_trades(self):
    #     """Create position-specific trade dataframes and calculate P&L."""
    #     if self.df is None or len(self.df) == 0:
    #         print("No positions to process trades for")
    #         return
            
    #     if self.all_trades_df is None or len(self.all_trades_df) == 0:
    #         print("No trades to process")
    #         return
            
    #     self.trade_dfs = {}
    #     self.title_mapping = {}
        
    #     for i, (idx, row) in enumerate(self.df.iterrows(), start=1):
    #         df_name = f'trade{i}'
    #         asset_id = str(row['asset'])
            
    #         position_trades = self.all_trades_df[
    #             self.all_trades_df['asset'].astype(str) == asset_id
    #         ].copy()
            
    #         if len(position_trades) == 0:
    #             continue
            
    #         self.trade_dfs[df_name] = position_trades
    #         self.title_mapping[df_name] = {
    #             'title': row['title'],
    #             'eventSlug': row['eventSlug'],
    #             'asset': asset_id,
    #             'num_trades': len(position_trades),
    #             'realizedPnl': row['realizedPnl']
    #         }
        
    #     if len(self.df) < 50:
    #         print("\n Trades Breakdown:")
    #         # Print all positions
    #         for df_name, info in self.title_mapping.items():
    #             print(f"{df_name}: {info['title']} - {info['num_trades']} trades, P&L: ${info['realizedPnl']:.2f}")
        
    #     print(f"\nCreated {len(self.trade_dfs)} position-specific trade dataframes")

    def process_trades(self):
        """Create position-specific trade dataframes and calculate P&L."""
        if self.df is None or len(self.df) == 0:
            print("No positions to process trades for")
            return
            
        if self.all_trades_df is None or len(self.all_trades_df) == 0:
            print("No trades to process")
            return
        
        # CRITICAL FIX: Deduplicate by asset, aggregate positions
        # For same asset: keep most recent, sum realizedPnl
        print(f"Original positions: {len(self.df)}")
        print(f"Unique assets: {self.df['asset'].nunique()}")
        
        # Aggregate duplicate assets
        df_aggregated = self.df.groupby('asset').agg({
            'title': 'first',
            'eventSlug': 'first',
            'asset': 'first',
            'realizedPnl': 'sum',  # Sum P&L across re-entries
            'closed': lambda x: all(x),  # True only if ALL positions closed
            'timestamp': 'max'  # Use most recent timestamp
        }).reset_index(drop=True)
        
        duplicates_removed = len(self.df) - len(df_aggregated)
        if duplicates_removed > 0:
            print(f"Aggregated {duplicates_removed} duplicate assets (re-entries)")
        
        self.trade_dfs = {}
        self.title_mapping = {}
        
        # Use aggregated dataframe
        for i, (idx, row) in enumerate(df_aggregated.iterrows(), start=1):
            df_name = f'trade{i}'
            asset_id = str(row['asset'])
            
            position_trades = self.all_trades_df[
                self.all_trades_df['asset'].astype(str) == asset_id
            ].copy()
            
            if len(position_trades) == 0:
                continue
            
            self.trade_dfs[df_name] = position_trades
            self.title_mapping[df_name] = {
                'title': row['title'],
                'eventSlug': row['eventSlug'],
                'asset': asset_id,
                'num_trades': len(position_trades),
                'realizedPnl': row['realizedPnl']
            }
        
        if len(df_aggregated) < 50:
            print("\n Trades Breakdown:")
            for df_name, info in self.title_mapping.items():
                print(f"{df_name}: {info['title']} - {info['num_trades']} trades, P&L: ${info['realizedPnl']:.2f}")
        
        print(f"\nCreated {len(self.trade_dfs)} position-specific trade dataframes")
    
    def create_summary(self):
        """Generate position summary statistics."""
        if self.df is None or len(self.df) == 0:
            print("No positions to summarize")
            self.summary_df = pd.DataFrame()
            self.winning_df = pd.DataFrame()
            self.losing_df = pd.DataFrame()
            return self.summary_df
            
        position_summary = []
        
        for key in self.trade_dfs.keys():
            if self.trade_dfs[key] is not None and len(self.trade_dfs[key]) > 0:
                asset_id = str(self.trade_dfs[key]['asset'].iloc[0])
                position_data = self.df[
                    self.df['asset'].astype(str) == asset_id
                ].iloc[0]
                
                api_pnl = position_data['realizedPnl']
                is_closed = position_data['closed']
                totalBought = position_data['totalBought']
                
                total_invested = self.trade_dfs[key][
                    self.trade_dfs[key]['side'] == 'BUY'
                ]['value'].sum()
                
                total_sells = self.trade_dfs[key][
                    self.trade_dfs[key]['side'] == 'SELL'
                ]['value'].sum()
                
                net_invested = total_invested - total_sells
                roi_total_capital = (
                    (api_pnl / total_invested * 100) if total_invested > 0 else 0
                )
                
                position_summary.append({
                    'position': key,
                    'title': self.trade_dfs[key]['title'].iloc[0],
                    'eventSlug': self.trade_dfs[key]['eventSlug'].iloc[0],
                    'first_timestamp': self.trade_dfs[key]['timestamp'].iloc[-1],
                    'totalBought': totalBought,
                    'realizedPnl': api_pnl,
                    'total_invested': total_invested,
                    'total_sells': total_sells,
                    'net_invested': net_invested,
                    'roi_pct': roi_total_capital,
                    'num_trades': len(self.trade_dfs[key]),
                    'has_sells': (self.trade_dfs[key]['side'] == 'SELL').any(),
                    'closed': is_closed,
                    'asset': asset_id
                })
        
        self.summary_df = pd.DataFrame(position_summary)
        
        if len(self.summary_df) > 0:
            self.winning_df = self.summary_df[
                self.summary_df['realizedPnl'] > 0
            ].sort_values('first_timestamp', ascending=True)
            
            self.losing_df = self.summary_df[
                self.summary_df['realizedPnl'] < 0
            ].sort_values('first_timestamp', ascending=True)
            
            print(f"\n{len(self.winning_df)} Winning Positions")
            print(f"{len(self.losing_df)} Losing Positions")
        else:
            self.winning_df = pd.DataFrame()
            self.losing_df = pd.DataFrame()
        
        return self.summary_df
    
    def run_full_analysis(self):
        """Run the complete analysis pipeline."""
        print("=" * 70)
        print("STARTING PORTFOLIO ANALYSIS")
        print("=" * 70)
        
        self.load_positions()
        
        if self.df is None or len(self.df) == 0:
            print("\nNo data to analyze. Exiting.")
            return self
            
        self.fetch_event_ids()
        self.load_trades()
        self.process_trades()
        self.create_summary()
        self.portfolio_summary()
        
        # Save to CSV
        self.save_to_csv()
        
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)
        
        return self
    
    def portfolio_summary(self):
        """Display overall portfolio performance statistics."""
        if self.df is None or len(self.df) == 0 or self.summary_df is None or len(self.summary_df) == 0:
            print("\nNo data available for portfolio summary")
            return
            
        realized_pnl = self.df[self.df['closed'] == True]['realizedPnl'].sum()
        unrealized_pnl = self.df[self.df['closed'] == False]['realizedPnl'].sum()
        total_pnl = self.df['realizedPnl'].sum()
        total_invested = self.summary_df['total_invested'].sum() if len(self.summary_df) > 0 else 0
        overall_roi = (total_pnl / total_invested * 100) if total_invested > 0 else 0
        
        closed_positions = self.summary_df[self.summary_df['closed'] == True] if len(self.summary_df) > 0 else pd.DataFrame()
        winning = closed_positions[closed_positions['realizedPnl'] > 0] if len(closed_positions) > 0 else pd.DataFrame()
        losing = closed_positions[closed_positions['realizedPnl'] < 0] if len(closed_positions) > 0 else pd.DataFrame()
        
        win_rate = (
            (len(winning) / len(closed_positions) * 100) 
            if len(closed_positions) > 0 else 0
        )
        avg_win = winning['realizedPnl'].mean() if len(winning) > 0 else 0
        avg_loss = losing['realizedPnl'].mean() if len(losing) > 0 else 0
        
        profit_factor = (
            abs((avg_win * len(winning)) / (avg_loss * len(losing))) 
            if len(losing) > 0 and avg_loss != 0 else 0
        )
        
        open_positions = self.summary_df[self.summary_df['closed'] == False] if len(self.summary_df) > 0 else pd.DataFrame()
        open_winning = open_positions[open_positions['realizedPnl'] > 0] if len(open_positions) > 0 else pd.DataFrame()
        open_losing = open_positions[open_positions['realizedPnl'] < 0] if len(open_positions) > 0 else pd.DataFrame()
        
        print("\n" + "=" * 70)
        print("PORTFOLIO SUMMARY")
        print("=" * 70)
        
        print(f"\nWallet: {self.wallet}")
        
        if self.start_date or self.end_date:
            print(f"\nDate Range Filter:")
            if self.start_date:
                print(f"  Start: {self.start_date.strftime('%Y-%m-%d')}")
            if self.end_date:
                print(f"  End: {self.end_date.strftime('%Y-%m-%d')}")
        
        print(f"\nOverall Performance:")
        print(f"  Total P&L (Mark-to-Market): ${total_pnl:,.2f}")
        print(f"    Realized P&L:   ${realized_pnl:,.2f}")
        print(f"    Unrealized P&L: ${unrealized_pnl:,.2f}")
        print(f"  Total Invested: ${total_invested:,.2f}")
        print(f"  Overall ROI: {overall_roi:.2f}%")
        
        print(f"\nClosed Positions ({len(closed_positions)} total):")
        print(f"  Winning: {len(winning)} ({win_rate:.1f}%)")
        print(f"  Losing: {len(losing)}")
        if len(winning) > 0:
            print(f"  Avg Win: ${avg_win:,.2f}")
        if len(losing) > 0:
            print(f"  Avg Loss: ${avg_loss:,.2f}")
        if profit_factor > 0:
            print(f"  Profit Factor: {profit_factor:.2f}")
        
        print(f"\nOpen Positions ({len(open_positions)} total):")
        print(f"  Winning: {len(open_winning)}")
        print(f"  Losing: {len(open_losing)}")
        if len(open_positions) > 0:
            print(f"  Capital at Risk: ${open_positions['total_invested'].sum():,.2f}")
            print(f"  Unrealized P&L: ${unrealized_pnl:,.2f}")
        
        print("=" * 70 + "\n")
    
    def position_summary(self, position_key):
        """
        Display detailed summary of a specific position.
        
        Args:
            position_key: e.g., 'trade1', 'trade91'
        
        Returns:
            DataFrame: All trades for this position
        """
        if position_key not in self.trade_dfs:
            print(f"Position '{position_key}' not found")
            available = list(self.trade_dfs.keys())[:5]
            print(f"Available positions: {available}{'...' if len(self.trade_dfs) > 5 else ''}")
            return None
        
            # Add this check
        if self.df is None or len(self.df) == 0 or self.summary_df is None or len(self.summary_df) == 0:
            print("\nNo data available for portfolio summary")
            return
        
        trades = self.trade_dfs[position_key]
        asset_id = str(trades['asset'].iloc[0])
        position_data = self.df[self.df['asset'].astype(str) == asset_id].iloc[0]
        summary_data = self.summary_df[self.summary_df['asset'] == asset_id].iloc[0]
        
        api_pnl = position_data['realizedPnl']
        is_closed = position_data['closed']
        calc_pnl = trades['trade_pnl'].sum() if 'trade_pnl' in trades.columns else None
        
        title = trades['title'].iloc[0]
        num_trades = len(trades)
        has_sells = (trades['side'] == 'SELL').any()
        
        print(f"\n{'=' * 70}")
        print(f"POSITION: {position_key}")
        print(f"{'=' * 70}")
        print(f"Title: {title}")
        print(f"Status: {'CLOSED' if is_closed else 'OPEN'}")
        print(f"Event: {position_data['eventSlug']}")
        print(f"Timestamp: {position_data['timestamp']}")
        
        print(f"\nPerformance:")
        print(f"  Realized P&L: ${api_pnl:,.2f}")
        if calc_pnl is not None:
            print(f"  Calculated P&L: ${calc_pnl:,.2f}")
        print(f"  Total Invested: ${summary_data['total_invested']:,.2f}")
        print(f"  ROI: {summary_data['roi_pct']:.2f}%")
        
        print(f"\nTrading Activity:")
        print(f"  Number of trades: {num_trades}")
        print(f"  Has sells: {'Yes' if has_sells else 'No'}")
        
        if has_sells:
            total_buys = trades[trades['side'] == 'BUY']['value'].sum()
            total_sells = trades[trades['side'] == 'SELL']['value'].sum()
            print(f"  Total bought: ${total_buys:,.2f}")
            print(f"  Total sold: ${total_sells:,.2f}")
            print(f"  Net invested: ${summary_data['net_invested']:,.2f}")
        else:
            total_shares = trades['size'].sum()
            total_cost = trades['value'].sum()
            avg_price = total_cost / total_shares if total_shares > 0 else 0
            
            print(f"\nPosition Details (Held to {'Maturity' if is_closed else 'Current'}):")
            print(f"  Total shares: {total_shares:,.2f}")
            print(f"  Average price: ${avg_price:.4f}")
            print(f"  Total cost: ${total_cost:,.2f}")
            
            if not is_closed:
                current_price = position_data['avgPrice']
                print(f"  Current price: ${current_price:.4f}")
        
        print(f"{'=' * 70}\n")
        
        return trades
    
    def plot_pnl_by_trades(self):
        """Plot cumulative P&L using individual trades with mark-to-market."""
        if self.all_trades_df is None or len(self.all_trades_df) == 0:
            print("\nNo trades to plot")
            return
        
        # Create a mapping of asset -> curPrice from analyzer.df
        price_map = self.df.set_index('asset')['curPrice'].to_dict() # type: ignore
        
        # Calculate P&L for each trade
        trades_df = self.all_trades_df.copy()
        
        # Get curPrice for each trade's asset
        trades_df['curPrice'] = trades_df['asset'].map(price_map)
        
        # Calculate P&L: (curPrice - fillPrice) * size
        # Note: For sells, size is negative, so this handles buys and sells correctly
        trades_df['trade_pnl'] = (trades_df['curPrice'] - trades_df['price']) * trades_df['size']
        
        # Sort by timestamp and calculate cumulative
        trades_df = trades_df.sort_values('timestamp')
        trades_df['cumulative_pnl'] = trades_df['trade_pnl'].cumsum()
        
        # Plot
        plt.figure(figsize=(12, 5))
        plt.plot(trades_df['timestamp'], trades_df['cumulative_pnl'], 
                linewidth=2.5, color="#855BF9")
        
        plt.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        plt.fill_between(trades_df['timestamp'], trades_df['cumulative_pnl'], 0,
                        alpha=0.2, color="#855BF9")
        
        plt.title('Cumulative P&L Over Time (Trade-by-Trade)', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('P&L ($)')
        plt.grid(True, alpha=0.3)
        plt.gca().yaxis.set_major_formatter(
            FuncFormatter(lambda x, p: f'${x:,.0f}')
        )
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def get_position_trades(self, position_key: str):
        """
        Get trades dataframe for a specific position.
        
        Args:
            position_key: Position identifier (e.g., 'trade1', 'trade5')
        
        Returns:
            DataFrame of trades for that position
        """
        if position_key not in self.trade_dfs:
            print(f"Position '{position_key}' not found")
            return None
        return self.trade_dfs[position_key]
    
    def list_available_data(self):
        """Print available dataframes and their contents."""
        print("\n" + "=" * 70)
        print("AVAILABLE DATA FOR ANALYSIS")
        print("=" * 70)
        
        print(f"\n1. Main Positions DataFrame:")
        print(f"   Access via: analyzer.df")
        if self.df is not None:
            print(f"   Shape: {self.df.shape}")
            print(f"   Columns: {list(self.df.columns)}")
        else:
            print(f"   Status: Not loaded")
        
        print(f"\n2. All Trades DataFrame:")
        print(f"   Access via: analyzer.all_trades_df")
        if self.all_trades_df is not None:
            print(f"   Shape: {self.all_trades_df.shape}")
            print(f"   Columns: {list(self.all_trades_df.columns)}")
        else:
            print(f"   Status: Not loaded")
        
        print(f"\n3. Summary DataFrame:")
        print(f"   Access via: analyzer.summary_df")
        if self.summary_df is not None and len(self.summary_df) > 0:
            print(f"   Shape: {self.summary_df.shape}")
            print(f"   Columns: {list(self.summary_df.columns)}")
        else:
            print(f"   Status: Not loaded")
        
        print(f"\n4. Winning Positions:")
        print(f"   Access via: analyzer.winning_df")
        if self.winning_df is not None and len(self.winning_df) > 0:
            print(f"   Count: {len(self.winning_df)} positions")
            print(f"   Columns: {list(self.winning_df.columns)}")
        else:
            print(f"   Status: No winning positions")
        
        print(f"\n5. Losing Positions:")
        print(f"   Access via: analyzer.losing_df")
        if self.losing_df is not None and len(self.losing_df) > 0:
            print(f"   Count: {len(self.losing_df)} positions")
            print(f"   Columns: {list(self.losing_df.columns)}")
        else:
            print(f"   Status: No losing positions")
        
        print(f"\n6. Individual Position Trade DataFrames:")
        print(f"   Access via: analyzer.trade_dfs['tradeX']")
        print(f"   or use: analyzer.get_position_trades('tradeX')")
        if self.trade_dfs:
            print(f"   Available positions: {len(self.trade_dfs)}")
            print(f"   Examples: {list(self.trade_dfs.keys())[:5]}")
        else:
            print(f"   Status: Not loaded")
        
        print(f"\n7. Title Mapping Dictionary:")
        print(f"   Access via: analyzer.title_mapping")
        if self.title_mapping:
            print(f"   Available: {len(self.title_mapping)} entries")
        else:
            print(f"   Status: Not loaded")

    def save_to_csv(self):
        """Save dataframes to CSV files and update index."""
        if self.all_trades_df is None or len(self.all_trades_df) == 0:
            print("\nNo data to save")
            return
        
        # Get info from all_trades_df
        username = self.all_trades_df['name'].iloc[0] if 'name' in self.all_trades_df.columns else self.all_trades_df['proxyWallet'].iloc[0]
        wallet = self.wallet  # The wallet address
        
        # Get today's date in MMDDYYYY format
        today = datetime.now().strftime('%m%d%Y')
        
        # Define save directory
        save_dir = 'trader analysis'
        os.makedirs(save_dir, exist_ok=True)
        
        # Save each dataframe
        saved_files = []
        
        if self.df is not None and len(self.df) > 0:
            filepath = os.path.join(save_dir, f'{username}_{today}_positions.csv')
            self.df.to_csv(filepath, index=False)
            saved_files.append(f'{username}_{today}_positions.csv')
        
        if self.all_trades_df is not None and len(self.all_trades_df) > 0:
            filepath = os.path.join(save_dir, f'{username}_{today}_trades.csv')
            self.all_trades_df.to_csv(filepath, index=False)
            saved_files.append(f'{username}_{today}_trades.csv')
        
        if self.summary_df is not None and len(self.summary_df) > 0:
            filepath = os.path.join(save_dir, f'{username}_{today}_summary.csv')
            self.summary_df.to_csv(filepath, index=False)
            saved_files.append(f'{username}_{today}_summary.csv')
        
        # Update index file
        index_path = os.path.join(save_dir, 'trader_files_index.csv')
        
        # Load existing index or create new
        if os.path.exists(index_path):
            index_df = pd.read_csv(index_path)
        else:
            index_df = pd.DataFrame(columns=['wallet_address', 'username', 'date', 'files'])
        
        # Add new entry
        new_entry = pd.DataFrame([{
            'wallet_address': wallet,
            'username': username,
            'date': today,
            'files': ', '.join(saved_files)
        }])
        index_df = pd.concat([index_df, new_entry], ignore_index=True)
        
        # Save updated index
        index_df.to_csv(index_path, index=False)
        
        print(f"\n✓ Saved {len(saved_files)} files to: {(save_dir)}")
        for filename in saved_files:
            print(f"  - {filename}")
        print(f"✓ Updated index: trader_files_index.csv")
        

# Example usage:
if __name__ == "__main__":
    from polymarket_api import PolymarketAPI
    
    api = PolymarketAPI()
    
    # Basic usage - no date filtering
    analyzer = PortfolioAnalyzer(
        api=api,
        wallet_address='0x3657862e57070b82a289b5887ec943a7c2166b14'
    )
    
    # Or with date filtering
    # analyzer = PolymarketPortfolioAnalyzer(
    #     api=api,
    #     wallet_address='0x3657862e57070b82a289b5887ec943a7c2166b14',
    #     start_date='2024-01-01',
    #     end_date='2024-12-31'
    # )
    
    # Run full analysis
    analyzer.run_full_analysis()
    
    # Display portfolio summary
    analyzer.portfolio_summary()
    
    # Plot P&L
    analyzer.plot_pnl_by_trades()
    
    # See what data is available
    analyzer.list_available_data()
    
    # Now you can access dataframes directly:
    # analyzer.df                    # All positions
    # analyzer.all_trades_df         # All trades
    # analyzer.summary_df            # Summary statistics
    # analyzer.winning_df            # Winning positions only
    # analyzer.losing_df             # Losing positions only
    # analyzer.trade_dfs['trade1']   # Trades for specific position
    
    # Or use convenience method:
    # analyzer.get_position_trades('trade1')
    
    # Check specific position
    # analyzer.position_summary('trade1')