"""
IV Data Collector - Continuous fetching from Deribit
====================================================
Run this to build your historical IV database.

Usage:
    python collector.py --once          # Single snapshot
    python collector.py --interval 300  # Every 5 minutes
    python collector.py --backfill      # Fetch historical (if available)
"""

import asyncio
import argparse
from datetime import datetime, timedelta
import time
import signal
import sys
from pathlib import Path

from iv_analysis import DeribitFetcher


class DataCollector:
    """
    Continuous data collection with scheduling.
    """
    
    def __init__(self, db_path: str = "iv_data.db", currencies: list = None):
        self.fetcher = DeribitFetcher(db_path)
        self.currencies = currencies or ["BTC", "ETH"]
        self.running = True
        
        # Handle graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        print("\nShutting down gracefully...")
        self.running = False
    
    async def run_once(self):
        """Fetch single snapshot for all currencies."""
        for currency in self.currencies:
            print(f"\n[{datetime.utcnow().isoformat()}] Fetching {currency} options...")
            try:
                snapshots = await self.fetcher.run_snapshot(currency)
                print(f"  Fetched {len(snapshots)} options")
            except Exception as e:
                print(f"  Error: {e}")
    
    async def run_continuous(self, interval_seconds: int = 300):
        """
        Continuous fetching at specified interval.
        
        Args:
            interval_seconds: Time between fetches (default 5 minutes)
        """
        print(f"Starting continuous collection (interval: {interval_seconds}s)")
        print(f"Currencies: {self.currencies}")
        print("Press Ctrl+C to stop")
        print("-" * 50)
        
        while self.running:
            start_time = time.time()
            
            await self.run_once()
            
            # Calculate sleep time
            elapsed = time.time() - start_time
            sleep_time = max(0, interval_seconds - elapsed)
            
            if self.running and sleep_time > 0:
                print(f"\nNext fetch in {sleep_time:.0f} seconds...")
                await asyncio.sleep(sleep_time)
        
        print("Collection stopped.")
    
    def get_stats(self) -> dict:
        """Get database statistics."""
        import sqlite3
        
        conn = sqlite3.connect(self.fetcher.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        # Total records
        cursor.execute("SELECT COUNT(*) FROM option_snapshots")
        stats['total_records'] = cursor.fetchone()[0]
        
        # Records per currency
        cursor.execute("""
            SELECT currency, COUNT(*) 
            FROM option_snapshots 
            GROUP BY currency
        """)
        stats['by_currency'] = dict(cursor.fetchall())
        
        # Time range
        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM option_snapshots")
        row = cursor.fetchone()
        stats['first_record'] = row[0]
        stats['last_record'] = row[1]
        
        # Unique timestamps (snapshots)
        cursor.execute("SELECT COUNT(DISTINCT timestamp) FROM option_snapshots")
        stats['unique_snapshots'] = cursor.fetchone()[0]
        
        conn.close()
        return stats


def print_stats(collector: DataCollector):
    """Print database statistics."""
    stats = collector.get_stats()
    
    print("\nDatabase Statistics")
    print("=" * 40)
    print(f"Total records: {stats['total_records']:,}")
    print(f"Unique snapshots: {stats['unique_snapshots']:,}")
    print(f"First record: {stats['first_record']}")
    print(f"Last record: {stats['last_record']}")
    print("\nBy currency:")
    for currency, count in stats['by_currency'].items():
        print(f"  {currency}: {count:,}")


def main():
    parser = argparse.ArgumentParser(description="IV Data Collector")
    parser.add_argument("--db", default="iv_data.db", help="Database path")
    parser.add_argument("--currencies", nargs="+", default=["BTC", "ETH"],
                        help="Currencies to fetch")
    parser.add_argument("--once", action="store_true", help="Single snapshot only")
    parser.add_argument("--interval", type=int, default=300,
                        help="Fetch interval in seconds (default: 300)")
    parser.add_argument("--stats", action="store_true", help="Show database stats")
    
    args = parser.parse_args()
    
    collector = DataCollector(args.db, args.currencies)
    
    if args.stats:
        print_stats(collector)
        return
    
    if args.once:
        asyncio.run(collector.run_once())
        print_stats(collector)
    else:
        asyncio.run(collector.run_continuous(args.interval))


if __name__ == "__main__":
    main()
