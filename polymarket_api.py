import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import os
import time
from functools import wraps
from typing import List, Dict, Optional, Any
import re
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


class PolymarketAPI:

    """
    Client for interacting with Polymarket's API
    Documentation: https://docs.polymarket.com/
    """
    
    # The three main API endpoints.
    def __init__(self):
        self.clob_url = "https://clob.polymarket.com"
        self.gamma_url = "https://gamma-api.polymarket.com"
        self.data_api_url = "https://data-api.polymarket.com"

    def scrape_leaderboard(self, timePeriod='month', orderBy='PNL', limit: int = 0, offset: int = 0, total=100, category='overall'):
        """Get users from Polymarket leaderboard API in batches of 20"""
        base_url = "https://data-api.polymarket.com/v1/leaderboard"
        all_users = []
        batch_size = 20
        
        # Calculate number of batches needed
        num_batches = (total + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            offset = i * batch_size
            params = {
                'timePeriod': timePeriod,
                'orderBy': orderBy,
                'limit': batch_size,
                'offset': offset,
                'category': category
            }
            
            print(f"  Fetching {category} users {offset + 1}-{min(offset + batch_size, total)}...")
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            users = response.json()
            
            all_users.extend(users)
            
            # Stop if we've received fewer users than requested (end of list)
            if len(users) < batch_size:
                break
            
            time.sleep(0.3)  # Be nice to the server
        
        return all_users

    def scrape_leaderboard_snapshot(self, limit: int = 100, timeframes: list | None = None, 
        max_retries: int = 3, retry_delay: int = 5) -> pd.DataFrame:
        """
        Scrape leaderboard snapshots across all categories and timeframes
        Appends to daily CSV file automatically
        
        Args:
            limit: Number of top traders to capture per category (default 100)
            timeframes: List of timeframes to scrape. Options: 'day', 'week', 'month'
                    Default: ['day'] for 0400 and 2000 runs, ['day', 'week', 'month'] for 1200 run
            max_retries: Number of retry attempts for failed categories (default 3)
            retry_delay: Seconds to wait between retries (default 5)
            
        Returns:
            DataFrame with columns: timestamp, timeframe, category, rank, address, pnl, volume, userName
        """
                
        if timeframes is None:
            timeframes = ['day']
        
        # Create snapshots directory if it doesn't exist
        Path('snapshots').mkdir(exist_ok=True)
        
        now = datetime.now()
        date_str = now.strftime('%Y%m%d')
        timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
        
        # Daily CSV file
        filename = f'snapshots/leaderboard_{date_str}.csv'
        
        print(f"\n{'='*60}")
        print(f"Scraping at {timestamp}")
        print(f"Timeframes: {', '.join(timeframes)}")
        print(f"{'='*60}")
        
        categories = [
            'overall',
            'politics',
            'sports',
            'crypto',
            'finance',
            'culture',
            'mentions',
            'weather',
            'economics',
            'tech'
        ]
        
        all_data = []
        failed_scrapes = []
        total_scrapes = len(categories) * len(timeframes)
        current_scrape = 0
        
        for timeframe in timeframes:
            print(f"\n--- Scraping {timeframe.upper()} leaderboards ---")
            
            for category in categories:
                current_scrape += 1
                success = False
                
                # Retry logic for each category
                for attempt in range(max_retries):
                    try:
                        if attempt > 0:
                            print(f"[{current_scrape}/{total_scrapes}] {category} ({timeframe}) - Retry {attempt}/{max_retries-1}...")
                            time.sleep(retry_delay)
                        else:
                            print(f"[{current_scrape}/{total_scrapes}] {category} ({timeframe})...")
                        
                        users = self.scrape_leaderboard(
                            timePeriod=timeframe,
                            orderBy='PNL',
                            total=limit,
                            category=category
                        )
                        
                        # Add each user with timestamp, timeframe, and category
                        for user in users:
                            all_data.append({
                                'timestamp': timestamp,
                                'timeframe': timeframe,
                                'category': category,
                                'rank': int(user.get('rank', 0)),
                                'address': user.get('proxyWallet', 'N/A'),
                                'pnl': float(user.get('pnl', 0)),
                                'volume': float(user.get('vol', 0)),
                                'userName': user.get('userName', 'Anonymous')
                            })
                        
                        print(f"  ✓ Found {len(users)} users")
                        success = True
                        break  # Success - exit retry loop
                        
                    except Exception as e:
                        print(f"  ✗ Error (attempt {attempt + 1}/{max_retries}): {e}")
                        if attempt == max_retries - 1:
                            # Final attempt failed - log it
                            failed_scrapes.append(f"{timeframe}/{category}")
        
        # Create DataFrame
        snapshot_df = pd.DataFrame(all_data)
        
        print(f"\n{'='*60}")
        print(f"Scraping complete!")
        if failed_scrapes:
            print(f"⚠ WARNING: {len(failed_scrapes)} scrapes failed after {max_retries} attempts:")
            for failed in failed_scrapes:
                print(f"  - {failed}")
        print(f"{'='*60}")
        
        # Append to daily file
        if os.path.exists(filename):
            snapshot_df.to_csv(filename, mode='a', header=False, index=False)
            print(f"✓ Appended to existing file: {filename}")
            
            # Show cumulative stats
            full_df = pd.read_csv(filename)
            num_snapshots = full_df['timestamp'].nunique()
            print(f"  Total snapshots in file: {num_snapshots}")
            print(f"  Total records: {len(full_df):,}")
        else:
            snapshot_df.to_csv(filename, mode='w', header=True, index=False)
            print(f"✓ Created new file: {filename}")
        
        print(f"  Records in this snapshot: {len(snapshot_df):,}")
        print(f"  File size: {os.path.getsize(filename) / 1024:.1f} KB")
        
        # Raise error if too many failures
        if len(failed_scrapes) > total_scrapes * 0.3:  # More than 30% failed
            raise Exception(f"Scrape quality too low: {len(failed_scrapes)}/{total_scrapes} failed")
        
        return snapshot_df
        
    def get_markets(self, limit: int = 100, offset: int = 0, closed: bool = False) -> List[Dict]:
        """
        Fetch available markets from Polymarket
        https://docs.polymarket.com/api-reference/markets/list-markets
        
        Args:
            limit: Number of markets to return
            offset: Pagination offset
            closed: Only return open markets (default: False)
        """
        url = f"{self.gamma_url}/markets"
        params = {
            'limit': limit,
            'offset': offset,
            'closed': closed
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    
    def get_user_positions(
        self, 
        address: str,
        market: Optional[str] = None,
        event_id: Optional[str] = None,
        size_threshold: Optional[float] = None,
        limit: int = 500,
        offset: int = 0,
        sort_by: str = "CURRENT",
        sort_direction: str = "DESC"
    ) -> List[Dict]:
        """
        Get current open positions for a user wallet address using Data-API
        https://docs.polymarket.com/api-reference/core/get-current-positions-for-a-user
        
        Args:
            address: Ethereum wallet address (0x...) - required
            market: Comma-separated list of condition IDs (mutually exclusive with event_id)
            event_id: Comma-separated list of event IDs (mutually exclusive with market)
            size_threshold: Minimum position size to return (>=0)
            limit: Number of positions to return (default: 500, max: 500)
            offset: Starting index for pagination (default: 0, max: 10000)
            sort_by: Sort criteria - CURRENT, INITIAL, TOKENS, CASHPNL, PERCENTPNL, 
                    TITLE, RESOLVING, PRICE, AVGPRICE (default: CURRENT)
            sort_direction: Sort direction - ASC or DESC (default: DESC)
        
        Returns:
            List of position dictionaries
        """
        url = f"{self.data_api_url}/positions"
        params = {
            'user': address,
            'limit': limit,
            'offset': offset,
            'sortBy': sort_by,
            'sortDirection': sort_direction
        }
        
        # Add optional filters if provided
        if market:
            params['market'] = market
        if event_id:
            params['eventId'] = event_id
        if size_threshold is not None:
            params['sizeThreshold'] = size_threshold
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    
    def get_all_user_positions(
        self,
        address: str,
        market: Optional[str] = None,
        event_id: Optional[str] = None,
        size_threshold: Optional[float] = None,
        sort_by: str = "CURRENT",
        sort_direction: str = "DESC",
        delay: float = 0.5
    ) -> List[Dict]:
        """
        Get ALL open positions for a user by automatically paginating through results.
        
        Args:
            address: Wallet address
            market: Optional market filter
            event_id: Optional event ID filter
            size_threshold: Optional minimum position size filter
            sort_by: Sort field (default: CURRENT)
            sort_direction: Sort direction (default: DESC)
            delay: Delay between requests in seconds
        
        Returns:
            Complete list of all open position dictionaries
        """
        all_positions = []
        offset = 0
        limit = 500
        
        while True:
            try:
                batch = self.get_user_positions(
                    address=address,
                    market=market,
                    event_id=event_id,
                    size_threshold=size_threshold,
                    limit=limit,
                    offset=offset,
                    sort_by=sort_by,
                    sort_direction=sort_direction
                )
                
                if not batch:
                    break
                
                all_positions.extend(batch)
                
                # Print progress
                print(f"Fetched {len(batch)} open positions (total: {len(all_positions)})")
                
                # Stop if we got fewer results than the limit
                if len(batch) < limit:
                    break
                    
                offset += limit
                time.sleep(delay)
            
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    print(f"Rate limited at offset {offset}. Waiting 5 seconds...")
                    time.sleep(5)
                    continue
                else:
                    raise
        
        return all_positions
    
    def get_closed_positions(
        self, 
        address: str, 
        limit: int = 50,
        offset: int = 0,
        market: Optional[str] = None,
        title: Optional[str] = None,
        event_id: Optional[str] = None,
        sort_by: str = "REALIZEDPNL",
        sort_direction: str = "DESC"
    ) -> List[Dict]:
        """
        Get closed positions for a user wallet address using Data-API
        https://docs.polymarket.com/api-reference/core/get-closed-positions-for-a-user
        
        Args:
            address: Ethereum wallet address (0x...)
            limit: Number of results to return (default: 50, max: 500)
            offset: Starting index for pagination (default: 0, max: 10000)
            market: Filter by conditionId(s), comma-separated for multiple
            title: Filter by market title
            event_id: Filter by event id(s), comma-separated for multiple
            sort_by: Sort criteria - REALIZEDPNL, TITLE, PRICE, AVGPRICE (default: REALIZEDPNL)
            sort_direction: Sort direction - ASC or DESC (default: DESC)
        
        Returns:
            List of closed position dictionaries
        """
        url = f"{self.data_api_url}/closed-positions"
        params = {
            'user': address,
            'limit': limit,
            'offset': offset,
            'sortBy': sort_by,
            'sortDirection': sort_direction
        }
        
        # Add optional filters if provided
        if market:
            params['market'] = market
        if title:
            params['title'] = title
        if event_id:
            params['eventId'] = event_id
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def get_all_closed_positions(
        self,
        address: str,
        market: Optional[str] = None,
        title: Optional[str] = None,
        event_id: Optional[str] = None,
        sort_by: str = "REALIZEDPNL",
        sort_direction: str = "DESC",
        start_timestamp: Optional[float] = None,
        end_timestamp: Optional[float] = None,
        delay: float = 0.5
    ) -> List[Dict]:
        """
        Get ALL closed positions for a user by automatically paginating through results.
        Supports date filtering via client-side filtering after fetch.
        """
        all_positions = []
        offset = 0
        limit = 25
        
        # Only use timestamp sorting and early stopping if date filtering is enabled
        use_early_stopping = bool(start_timestamp or end_timestamp)
        
        if use_early_stopping:
            sort_by = "TIMESTAMP"
            sort_direction = "DESC"
            consecutive_old_batches = 0
        
        while True:
            try:
                batch = self.get_closed_positions(
                    address=address,
                    market=market,
                    title=title,
                    event_id=event_id,
                    limit=limit,
                    offset=offset,
                    sort_by=sort_by,
                    sort_direction=sort_direction
                )
                
                if not batch:
                    break
                
                all_positions.extend(batch)
                
                # ONLY use early stopping heuristic when date filtering
                if use_early_stopping:
                    batch_has_valid = any(
                        (not start_timestamp or pos.get('timestamp', 0) >= start_timestamp) and
                        (not end_timestamp or pos.get('timestamp', 0) <= end_timestamp)
                        for pos in batch
                    )
                    
                    if not batch_has_valid:
                        consecutive_old_batches += 1 # type: ignore
                        if consecutive_old_batches >= 5:
                            print(f"No relevant positions in last 5 batches. Stopping fetch.")
                            break
                    else:
                        consecutive_old_batches = 0
                
                print(f"Fetched {len(batch)} closed positions (total fetched: {len(all_positions)})")
                
                # Stop if we got fewer results than the limit
                if len(batch) < limit:
                    break
                    
                offset += limit
                time.sleep(delay)
            
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    print(f"Rate limited at offset {offset}. Waiting 5 seconds...")
                    time.sleep(5)
                    continue
                else:
                    raise
        
        # Filter by date client-side (only if date filtering requested)
        if start_timestamp or end_timestamp:
            original_count = len(all_positions)
            filtered_positions = [
                pos for pos in all_positions
                if (not start_timestamp or pos.get('timestamp', 0) >= start_timestamp) and
                (not end_timestamp or pos.get('timestamp', 0) <= end_timestamp)
            ]
            
            print(f"Date filtering: {original_count} total → {len(filtered_positions)} in range")
            return filtered_positions
        
        return all_positions
    
    def get_user_trades(
            self, 
            address: Optional[str] = None,
            limit: int = 500,
            offset: int = 0,
            takerOnly: bool = False,
            market: Optional[str] = None,
            event_id: Optional[int] = None,
            maker_address: Optional[str] = None,
            side: Optional[str] = None,
            filter_type: Optional[str] = None,
            filter_amount: Optional[float] = None
        ) -> List[Dict]:
        """
        Get historical trades for a user or markets using Data-API.
        By default returns both maker and taker trades (takerOnly=False).
        https://docs.polymarket.com/api-reference/core/get-trades-for-a-user-or-markets
        
        Args:
            address: User wallet address (0x...) - optional if filtering by market/event
            limit: Number of trades to return (default: 500, max: 10000)
            offset: Starting index for pagination (default: 0, max: 10000)
            takerOnly: If True, only return taker trades (default: False)
            market: Comma-separated list of condition IDs (mutually exclusive with event_id)
            event_id: Comma-separated list of event IDs (mutually exclusive with market)
            maker_address: Filter by maker address (0x...)
            side: Filter by trade side - BUY or SELL
            filter_type: Filter type - CASH or TOKENS (must be provided with filter_amount)
            filter_amount: Filter amount threshold (>=0, must be provided with filter_type)
        
        Returns:
            List of trade dictionaries
        """
        url = f"{self.data_api_url}/trades"
        params: Dict[str, Any] = {
            'limit': limit,
            'offset': offset,
            'takerOnly': takerOnly
        }
        
        # Add optional parameters if provided
        if address:
            params['user'] = address
        if market:
            params['market'] = market
        if event_id:
            params['eventId'] = event_id
        if maker_address:
            params['makerAddress'] = maker_address
        if side:
            params['side'] = side
        
        # Filter type and amount must be provided together
        if filter_type and filter_amount is not None:
            params['filterType'] = filter_type
            params['filterAmount'] = filter_amount
        elif filter_type or filter_amount is not None:
            raise ValueError("filter_type and filter_amount must be provided together")
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()


    def get_all_user_trades(
            self,
            address: str,
            market: Optional[str] = None,
            event_id: Optional[int] = None,
            maker_address: Optional[str] = None,
            side: Optional[str] = None,
            filter_type: Optional[str] = None,
            filter_amount: Optional[float] = None,
            max_results: Optional[int] = None,
            rate_limit_delay: float = 0.1
        ) -> List[Dict]:
        """
        Get all user trades with automatic pagination.
        Always returns both maker and taker trades.
        
        Args:
            address: User wallet address (0x...)
            market: Comma-separated list of condition IDs
            event_id: Comma-separated list of event IDs
            maker_address: Filter by maker address
            side: Filter by trade side (BUY or SELL)
            filter_type: Filter type (CASH or TOKENS)
            filter_amount: Filter amount threshold
            max_results: Maximum total results to retrieve (None for all available)
            rate_limit_delay: Seconds to wait between batch requests (default: 0.1)
        
        Returns:
            Complete list of all trade dictionaries
        """
        import time
        
        all_trades = []
        offset = 0
        limit = 500  # Maximum per request
        
        while True:
            # Fetch batch
            batch = self.get_user_trades(
                address=address,
                limit=limit,
                offset=offset,
                market=market,
                event_id=event_id,
                maker_address=maker_address,
                side=side,
                filter_type=filter_type,
                filter_amount=filter_amount
            )
            
            # Add to results
            all_trades.extend(batch)
            
            # Check if we're done
            if len(batch) < limit:
                # Received fewer results than requested, we've hit the end
                break
            
            if max_results and len(all_trades) >= max_results:
                # Reached user-specified limit
                all_trades = all_trades[:max_results]
                break
            
            # Check offset limit (API max is 10,000)
            if offset + limit >= 10000:
                print(f"Warning: Reached API pagination limit at offset {offset + limit}")
                break
            
            # Move to next batch
            offset += limit
            
            # Rate limiting - only sleep if we're continuing the loop
            time.sleep(rate_limit_delay)
        
        return all_trades
    
    def get_user_activity(
            self, 
            address: str, 
            limit: int = 500,
            offset: int = 0,
            market: Optional[str] = None,
            event_id: Optional[str] = None,
            type: str = "TRADE",
            start: Optional[int] = None,
            end: Optional[int] = None,
            sort_by: str = "TIMESTAMP",
            sort_direction: str = "DESC",
            side: Optional[str] = None
        ) -> List[Dict]:
        """
        Get on-chain activity for a user (trades, splits, merges, redeems, rewards, conversions)
        https://docs.polymarket.com/api-reference/core/get-user-activity
        
        Args:
            address: Ethereum wallet address (0x...)
            limit: Number of activities to return (default: 500, max: 500)
            offset: Starting index for pagination (default: 0, max: 10000)
            market: Comma-separated list of condition IDs (mutually exclusive with event_id)
            event_id: Comma-separated list of event IDs (mutually exclusive with market)
            type: Activity type filter - TRADE, SPLIT, MERGE, REDEEM, REWARD, or CONVERSION
                Supports multiple comma-separated values (default: "TRADE")
            start: Start timestamp in seconds (>=0)
            end: End timestamp in seconds (>=0)
            sort_by: Sort criteria - TIMESTAMP, TOKENS, CASH (default: TIMESTAMP)
            sort_direction: Sort direction - ASC or DESC (default: DESC)
            side: Filter by trade side - BUY or SELL (only applies to TRADE type)
        
        Returns:
            List of activity dictionaries
        """
        url = f"{self.data_api_url}/activity"
        params = {
            'user': address,
            'limit': limit,
            'offset': offset,
            'sortBy': sort_by,
            'sortDirection': sort_direction,
            'type': type  # Always included now since it has a default value
        }
        
        # Add optional filters if provided
        if market:
            params['market'] = market
        if event_id:
            params['eventId'] = event_id
        if start is not None:
            params['start'] = start
        if end is not None:
            params['end'] = end
        if side:
            params['side'] = side
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    
    def get_all_user_activity(
            self,
            address: str,
            market: Optional[str] = None,
            event_id: Optional[str] = None,
            type: str = "TRADE",
            start: Optional[int] = None,
            end: Optional[int] = None,
            sort_by: str = "TIMESTAMP",
            sort_direction: str = "DESC",
            side: Optional[str] = None,
            max_results: Optional[int] = None
        ) -> List[Dict]:
        """
        Get all user activity with automatic pagination.
        
        Args:
            address: Ethereum wallet address
            max_results: Maximum total results to retrieve (None for all available)
            Other args: Same as get_user_activity()
        
        Returns:
            Complete list of all activity dictionaries
        """
        all_activities = []
        offset = 0
        limit = 500  # Maximum per request
        
        while True:
            # Fetch batch
            batch = self.get_user_activity(
                address=address,
                limit=limit,
                offset=offset,
                market=market,
                event_id=event_id,
                type=type,
                start=start,
                end=end,
                sort_by=sort_by,
                sort_direction=sort_direction,
                side=side
            )
            
            # Add to results
            all_activities.extend(batch)
            
            # Check if we're done
            if len(batch) < limit:
                # Received fewer results than requested, we've hit the end
                break
            
            if max_results and len(all_activities) >= max_results:
                # Reached user-specified limit
                all_activities = all_activities[:max_results]
                break
            
            # Check offset limit (API max is 10,000)
            if offset + limit >= 10000:
                print(f"Warning: Reached API pagination limit at offset {offset + limit}")
                break
            
            # Move to next batch
            offset += limit
            
            # Optional: Add small delay to avoid rate limiting
            # time.sleep(0.1)
        
        return all_activities
    
    def get_user_portfolio_value(self, address: str) -> Dict:
        """
        Get total portfolio value for a user
        https://docs.polymarket.com/api-reference/core/get-total-value-of-a-users-positions
        Only returns value of active positions, does not include closed position profits.

        Args:
            address: Ethereum wallet address
        """
        url = f"{self.data_api_url}/value"
        params = {'user': address}
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    
    def get_top_holders(
        self,
        market: str,
        limit: int = 100,
        min_balance: int = 1
    ) -> List[Dict]:
        """
        Get top holders for specified markets using Data-API
        https://docs.polymarket.com/api-reference/core/get-top-holders-for-markets
        
        Args:
            market: Comma-separated list of condition IDs (0x-prefixed 64-hex string) - required
            limit: Number of holders to return per market (default: 100, max: 500)
            min_balance: Minimum balance to filter holders (default: 1, range: 0-999999)
        
        Returns:
            List of dictionaries, each containing:
            - token: Market token ID
            - holders: List of holder objects with proxyWallet, amount, pseudonym, etc.
        """
        url = f"{self.data_api_url}/holders"
        params = {
            'market': market,
            'limit': limit,
            'minBalance': min_balance
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    
    def get_user_pnl(self, address: str) -> Dict:
        """
        Calculate profit/loss data for a user from their positions
        
        Args:
            address: Ethereum wallet address
        """
        positions = self.get_user_positions(address)
        
        total_pnl = 0
        total_initial_value = 0
        total_current_value = 0
        
        for position in positions:
            if 'cashPnl' in position:
                total_pnl += float(position.get('cashPnl', 0))
            if 'initialValue' in position:
                total_initial_value += float(position.get('initialValue', 0))
            if 'currentValue' in position:
                total_current_value += float(position.get('currentValue', 0))
        
        return {
            'address': address,
            'total_cash_pnl': total_pnl,
            'total_initial_value': total_initial_value,
            'total_current_value': total_current_value,
            'percent_pnl': (total_pnl / total_initial_value * 100) if total_initial_value > 0 else 0
        }
    
    def get_total_markets_traded(self, address: str) -> Dict:
        """
        Get the total number of unique markets a user has traded
        https://docs.polymarket.com/api-reference/misc/get-total-markets-a-user-has-traded
        
        Args:
            address: Ethereum wallet address (0x...) - required
        
        Returns:
            Dictionary with total market count, e.g., {"total": 123}
        """
        url = f"{self.data_api_url}/traded"
        params = {
            'user': address
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    
    def get_event_by_slug(self, slug: str) -> Dict:
        """
        Get event details by slug
        https://docs.polymarket.com/api-reference/events/get-event-by-slug
        
        Args:
            slug: Event slug (e.g., "will-trump-win-the-2024-election")
        
        Returns:
            Dictionary with event details including markets, description, etc.
        """
        url = f"{self.gamma_url}/events/slug/{slug}"
        
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    
    def get_market_by_slug(self, slug: str) -> Dict:
        """
        Get market details by slug
        https://docs.polymarket.com/api-reference/markets/get-market-by-slug
        
        Args:
            slug: Market slug (e.g., "will-trump-win-the-2024-election-yes")
        
        Returns:
            Dictionary with market details including price, volume, outcomes, etc.
        """
        url = f"{self.gamma_url}/markets/slug/{slug}"
        
        response = requests.get(url)
        response.raise_for_status()
        return response.json()