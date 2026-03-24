import sqlite3
import csv
import logging
import threading
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class DatabaseManager:

    def __init__(self, db_type: str = "sqlite", db_path: str = "sentiment_analysis.db",
                 connection_params: Optional[Dict] = None):

        self.db_type = db_type.lower()
        self.db_path = db_path
        self.connection_params = connection_params or {}
        self.lock = threading.Lock()

        if self.db_type == "sqlite":
            self._create_tables()
        elif self.db_type == "mysql":
            self._create_tables()
        elif self.db_type == "postgresql":
            self._create_tables()
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

        logger.info(f"✓ Database initialized ({self.db_type})")

    def _get_connection(self):
        """Get a database connection (thread-safe for SQLite)"""
        if self.db_type == "sqlite":
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            return conn
        elif self.db_type == "mysql":
            import mysql.connector
            return mysql.connector.connect(
                host=self.connection_params.get('host', 'localhost'),
                user=self.connection_params.get('user', 'root'),
                password=self.connection_params.get('password', ''),
                database=self.connection_params.get('database', 'sentiment_db')
            )
        elif self.db_type == "postgresql":
            import psycopg2
            return psycopg2.connect(
                host=self.connection_params.get('host', 'localhost'),
                user=self.connection_params.get('user', 'postgres'),
                password=self.connection_params.get('password', ''),
                database=self.connection_params.get('database', 'sentiment_db')
            )
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")



    def _create_tables(self):
        """Create necessary tables if they don't exist"""
        if self.db_type == "sqlite":
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_text TEXT NOT NULL,
                    cleaned_text TEXT,
                    sentiment VARCHAR(20) NOT NULL,
                    confidence REAL NOT NULL,
                    positive_score REAL,
                    negative_score REAL,
                    model_name VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_sentiment ON analysis_results(sentiment)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_created_at ON analysis_results(created_at)
            ''')

            conn.commit()
            conn.close()
            logger.info("✓ Tables created/verified")

    def insert_result(self, result: Dict) -> bool:
        """

        Insert a single sentiment analysis result into the database.

        """
        with self.lock:
            try:
                if self.db_type == "sqlite":
                    conn = self._get_connection()
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO analysis_results
                        (original_text, cleaned_text, sentiment, confidence,
                         positive_score, negative_score, model_name)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        result.get('text', ''),
                        result.get('cleaned_text', ''),
                        result.get('sentiment', 'NEUTRAL'),
                        result.get('confidence', 0.0),
                        result.get('positive_score', 0.0),
                        result.get('negative_score', 0.0),
                        result.get('model_name', 'distilbert')
                    ))
                    conn.commit()
                    conn.close()

                return True
            except Exception as e:
                logger.error(f"Error inserting result: {e}")
                return False

    def insert_batch(self, results: List[Dict]) -> int:
        """

        Insert multiple results into the database.

        """
        inserted = 0
        for result in results:
            if self.insert_result(result):
                inserted += 1

        logger.info(f"✓ Inserted {inserted}/{len(results)} records")
        return inserted

    def get_all_results(self, limit: Optional[int] = None) -> List[Dict]:
        """

        Retrieve all analysis results from the database.

        """
        with self.lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                query = "SELECT * FROM analysis_results ORDER BY created_at DESC"

                if limit:
                    query += f" LIMIT {limit}"

                cursor.execute(query)

                if self.db_type == "sqlite":
                    rows = cursor.fetchall()
                    result = [dict(row) for row in rows]
                else:
                    columns = [desc[0] for desc in cursor.description]
                    rows = cursor.fetchall()
                    result = [dict(zip(columns, row)) for row in rows]

                conn.close()
                return result

            except Exception as e:
                logger.error(f"Error retrieving results: {e}")
                return []

    def get_by_sentiment(self, sentiment: str) -> List[Dict]:
        """

        Retrieve results filtered by sentiment.

        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            if self.db_type == "sqlite":
                cursor.execute(
                    "SELECT * FROM analysis_results WHERE sentiment = ? ORDER BY created_at DESC",
                    (sentiment.upper(),)
                )
                rows = cursor.fetchall()
                result = [dict(row) for row in rows]

            conn.close()
            return result

        except Exception as e:
            logger.error(f"Error retrieving results by sentiment: {e}")
            return []

    def get_statistics(self) -> Dict:
        """

        Get database statistics.

        """
        with self.lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                cursor.execute("SELECT COUNT(*) as count FROM analysis_results")
                total = cursor.fetchone()[0]

                cursor.execute('''
                    SELECT sentiment, COUNT(*) as count
                    FROM analysis_results
                    GROUP BY sentiment
                ''')

                sentiment_counts = {}
                for row in cursor.fetchall():
                    if self.db_type == "sqlite":
                        sentiment_counts[row[0]] = row[1]
                    else:
                        sentiment_counts[row[0]] = row[1]

                cursor.execute("SELECT AVG(confidence) as avg_conf FROM analysis_results")
                avg_confidence = cursor.fetchone()[0] or 0.0

                conn.close()

                return {
                    'total_records': total,
                    'sentiment_distribution': sentiment_counts,
                    'average_confidence': round(avg_confidence, 4)
                }

            except Exception as e:
                logger.error(f"Error getting statistics: {e}")
                return {}



class CSVExporter:
    """

    Export sentiment analysis results to CSV format.

    """

    @staticmethod
    def export_results(results: List[Dict], output_path: str = "sentiment_results.csv") -> bool:
        """

        Export results to CSV file.

        """
        if not results:
            logger.warning("No results to export")
            return False

        try:
            columns = [
                'ID', 'Original Text', 'Sentiment', 'Confidence',
                'Positive Score', 'Negative Score', 'Timestamp'
            ]

            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=columns)
                writer.writeheader()

                for idx, result in enumerate(results, 1):
                    writer.writerow({
                        'ID': idx,
                        'Original Text': result.get('text', '')[:200],
                        'Sentiment': result.get('sentiment', 'UNKNOWN'),
                        'Confidence': round(result.get('confidence', 0.0), 4),
                        'Positive Score': round(result.get('positive_score', 0.0), 4),
                        'Negative Score': round(result.get('negative_score', 0.0), 4),
                        'Timestamp': result.get('timestamp', datetime.now().isoformat())
                    })

            logger.info(f"✓ Exported {len(results)} results to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return False

    @staticmethod
    def export_statistics(stats: Dict, output_path: str = "sentiment_statistics.csv") -> bool:
        """

        Export statistics summary to CSV.

        """
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)

                writer.writerow(['Metric', 'Value'])
                writer.writerow(['Total Texts Analyzed', stats.get('total', 0)])
                writer.writerow(['Positive', stats.get('positive', 0)])
                writer.writerow(['Negative', stats.get('negative', 0)])
                writer.writerow(['Neutral', stats.get('neutral', 0)])
                writer.writerow(['Positive %', f"{stats.get('positive_pct', 0):.2f}%"])
                writer.writerow(['Negative %', f"{stats.get('negative_pct', 0):.2f}%"])
                writer.writerow(['Neutral %', f"{stats.get('neutral_pct', 0):.2f}%"])
                writer.writerow(['Average Confidence', f"{stats.get('avg_confidence', 0):.4f}"])

                writer.writerow(['Export Date', datetime.now().isoformat()])

            logger.info(f"✓ Exported statistics to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting statistics: {e}")
            return False
