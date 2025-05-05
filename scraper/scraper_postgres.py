import yfinance as yf
import pandas as pd
import numpy as np
import time
import os
import logging
import psycopg2
from psycopg2 import sql
from typing import Dict, List
from pathlib import Path
from datetime import datetime


# Configuración centralizada
class ScraperConfig:
    # Configuración de PostgreSQL
    PG_HOST = "98.85.189.191"
    PG_PORT = 5432
    PG_USER = "bvl_user"
    PG_PASSWORD = "179fae82"
    PG_DATABASE = "bvl_monitor"

    # Configuración de rutas
    SCRIPT_DIR = Path(__file__).parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    LOGS_DIR = PROJECT_ROOT / "logs"

    # Configuración de tiempo
    UPDATE_INTERVAL_MINUTES = 1
    TOTAL_RUNTIME_HOURS = 7
    TOTAL_ITERATIONS = (TOTAL_RUNTIME_HOURS * 60) // UPDATE_INTERVAL_MINUTES

    # Configuración de logging
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

    # Instrumentos a rastrear
    INSTRUMENTS = {
        "BAP": {
            "yahoo_fields": [
                "currentPrice", "previousClose", "open", "dayLow", "dayHigh",
                "bid", "dividendYield", "earningsGrowth", "revenueGrowth",
                "grossMargins", "ebitdaMargins", "operatingMargins", "financialCurrency",
                "returnOnAssets", "returnOnEquity", "bookValue", "priceToBook",
                "volume", "fiftyTwoWeekRange", "marketCap", "trailingPE"
            ]
        },
        "BRK-B": {
            "yahoo_fields": [
                "currentPrice", "previousClose", "open", "dayLow", "dayHigh",
                "bid", "dividendYield", "earningsGrowth", "revenueGrowth",
                "grossMargins", "ebitdaMargins", "operatingMargins", "financialCurrency",
                "returnOnAssets", "returnOnEquity", "bookValue", "priceToBook",
                "volume", "fiftyTwoWeekRange", "marketCap", "trailingPE"
            ]
        },
        "ILF": {
            "yahoo_fields": [
                "currentPrice", "previousClose", "open", "dayLow", "dayHigh",
                "bid", "dividendYield", "volume", "fiftyTwoWeekRange",
                "marketCap", "trailingPE"
            ],
            "is_etf": True,
            "field_mapping": {"volume": "volumen"}  # Mapeo de Yahoo Finance -> CSV
        }
    }


class DataScraper:
    def __init__(self):
        self._setup_directories()
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        self.conn = None
        self.table_columns = []

        # Conectar a PostgreSQL y verificar la tabla
        self._connect_to_postgres()
        self._check_and_update_table()
        self._get_table_columns()

    def _setup_directories(self):
        """Crea los directorios necesarios si no existen"""
        os.makedirs(ScraperConfig.LOGS_DIR, exist_ok=True)

    def _setup_logging(self):
        """Configura el sistema de logging"""
        log_filename = ScraperConfig.LOGS_DIR / f"scraper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=ScraperConfig.LOG_LEVEL,
            format=ScraperConfig.LOG_FORMAT,
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )

    def _connect_to_postgres(self):
        """Establece la conexión con la base de datos PostgreSQL"""
        try:
            self.conn = psycopg2.connect(
                host=ScraperConfig.PG_HOST,
                port=ScraperConfig.PG_PORT,
                user=ScraperConfig.PG_USER,
                password=ScraperConfig.PG_PASSWORD,
                database=ScraperConfig.PG_DATABASE
            )
            self.logger.info("Conexión establecida con PostgreSQL")
        except Exception as e:
            self.logger.error(f"Error al conectar a PostgreSQL: {str(e)}", exc_info=True)
            raise

    def _get_table_columns(self):
        """Obtiene las columnas existentes en la tabla stock_data"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                            SELECT column_name
                            FROM information_schema.columns
                            WHERE table_name = 'stock_data'
                            ORDER BY ordinal_position
                            """)
                self.table_columns = [row[0] for row in cur.fetchall()]
                self.logger.info(f"Columnas en stock_data: {self.table_columns}")
        except Exception as e:
            self.logger.error(f"Error al obtener columnas de la tabla: {str(e)}", exc_info=True)

    def _check_and_update_table(self):
        """Verifica si la tabla existe y tiene todas las columnas necesarias"""
        try:
            with self.conn.cursor() as cur:
                # Verificar si la tabla existe
                cur.execute("SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = 'stock_data')")
                table_exists = cur.fetchone()[0]

                if not table_exists:
                    # Crear la tabla si no existe
                    self.logger.info("Creando tabla stock_data")
                    cur.execute("""
                                CREATE TABLE stock_data
                                (
                                    id                   SERIAL PRIMARY KEY,
                                    symbol               VARCHAR(10) NOT NULL,
                                    timestamp            TIMESTAMP   NOT NULL,
                                    current_price        NUMERIC(10, 2),
                                    previous_close       NUMERIC(10, 2),
                                    open                 NUMERIC(10, 2),
                                    day_low              NUMERIC(10, 2),
                                    day_high             NUMERIC(10, 2),
                                    bid                  NUMERIC(10, 2),
                                    dividend_yield       NUMERIC(8, 4),
                                    volume               BIGINT,
                                    fifty_two_week_range VARCHAR(30),
                                    market_cap           BIGINT,
                                    trailing_pe          NUMERIC(10, 2),
                                    earnings_growth      NUMERIC(8, 4),
                                    revenue_growth       NUMERIC(8, 4),
                                    gross_margins        NUMERIC(8, 4),
                                    ebitda_margins       NUMERIC(8, 4),
                                    operating_margins    NUMERIC(8, 4),
                                    financial_currency   VARCHAR(10),
                                    return_on_assets     NUMERIC(8, 4),
                                    return_on_equity     NUMERIC(8, 4),
                                    book_value           NUMERIC(10, 2),
                                    price_to_book        NUMERIC(8, 4),
                                    UNIQUE (symbol, timestamp)
                                );

                                CREATE INDEX idx_stock_data_symbol ON stock_data (symbol);
                                CREATE INDEX idx_stock_data_timestamp ON stock_data (timestamp);
                                CREATE INDEX idx_stock_data_symbol_timestamp ON stock_data (symbol, timestamp);
                                """)
                    self.conn.commit()
                else:
                    # Obtener las columnas existentes
                    cur.execute("""
                                SELECT column_name
                                FROM information_schema.columns
                                WHERE table_name = 'stock_data'
                                """)
                    existing_columns = [row[0] for row in cur.fetchall()]

                    # Definir columnas que deberían existir
                    expected_columns = {
                        'id': 'SERIAL PRIMARY KEY',
                        'symbol': 'VARCHAR(10) NOT NULL',
                        'timestamp': 'TIMESTAMP NOT NULL',
                        'current_price': 'NUMERIC(10, 2)',
                        'previous_close': 'NUMERIC(10, 2)',
                        'open': 'NUMERIC(10, 2)',
                        'day_low': 'NUMERIC(10, 2)',
                        'day_high': 'NUMERIC(10, 2)',
                        'bid': 'NUMERIC(10, 2)',
                        'dividend_yield': 'NUMERIC(8, 4)',
                        'volume': 'BIGINT',
                        'fifty_two_week_range': 'VARCHAR(30)',
                        'market_cap': 'BIGINT',
                        'trailing_pe': 'NUMERIC(10, 2)',
                        'earnings_growth': 'NUMERIC(8, 4)',
                        'revenue_growth': 'NUMERIC(8, 4)',
                        'gross_margins': 'NUMERIC(8, 4)',
                        'ebitda_margins': 'NUMERIC(8, 4)',
                        'operating_margins': 'NUMERIC(8, 4)',
                        'financial_currency': 'VARCHAR(10)',
                        'return_on_assets': 'NUMERIC(8, 4)',
                        'return_on_equity': 'NUMERIC(8, 4)',
                        'book_value': 'NUMERIC(10, 2)',
                        'price_to_book': 'NUMERIC(8, 4)'
                    }

                    # Añadir columnas faltantes
                    for col_name, col_type in expected_columns.items():
                        if col_name.lower() not in [col.lower() for col in existing_columns]:
                            if col_name != 'id':  # No intentar añadir id, que debería ser la clave primaria
                                self.logger.info(f"Añadiendo columna faltante: {col_name}")
                                cur.execute(f"ALTER TABLE stock_data ADD COLUMN {col_name} {col_type}")
                                self.conn.commit()

                    # Comprobar si existe el índice de unicidad
                    cur.execute("""
                                SELECT COUNT(*)
                                FROM pg_constraint
                                WHERE conrelid = 'stock_data'::regclass 
                        AND contype = 'u' 
                        AND array_to_string(conkey, ',') = '2,3' -- Asume que symbol es la segunda columna y timestamp la tercera
                                """)
                    if cur.fetchone()[0] == 0:
                        self.logger.info("Añadiendo restricción de unicidad para (symbol, timestamp)")
                        cur.execute(
                            "ALTER TABLE stock_data ADD CONSTRAINT stock_data_symbol_timestamp_key UNIQUE (symbol, timestamp)")
                        self.conn.commit()

                    # Comprobar si existen los índices
                    cur.execute(
                        "SELECT COUNT(*) FROM pg_indexes WHERE tablename = 'stock_data' AND indexname = 'idx_stock_data_symbol'")
                    if cur.fetchone()[0] == 0:
                        self.logger.info("Creando índice idx_stock_data_symbol")
                        cur.execute("CREATE INDEX idx_stock_data_symbol ON stock_data(symbol)")
                        self.conn.commit()

                    cur.execute(
                        "SELECT COUNT(*) FROM pg_indexes WHERE tablename = 'stock_data' AND indexname = 'idx_stock_data_timestamp'")
                    if cur.fetchone()[0] == 0:
                        self.logger.info("Creando índice idx_stock_data_timestamp")
                        cur.execute("CREATE INDEX idx_stock_data_timestamp ON stock_data(timestamp)")
                        self.conn.commit()

                    cur.execute(
                        "SELECT COUNT(*) FROM pg_indexes WHERE tablename = 'stock_data' AND indexname = 'idx_stock_data_symbol_timestamp'")
                    if cur.fetchone()[0] == 0:
                        self.logger.info("Creando índice idx_stock_data_symbol_timestamp")
                        cur.execute("CREATE INDEX idx_stock_data_symbol_timestamp ON stock_data(symbol, timestamp)")
                        self.conn.commit()

                self.logger.info("Tabla stock_data verificada y actualizada correctamente")
        except Exception as e:
            self.logger.error(f"Error al verificar/actualizar tabla: {str(e)}", exc_info=True)
            self.conn.rollback()
            raise

    def _map_yahoo_to_db_field(self, yahoo_field):
        """Mapea un campo de Yahoo Finance a su equivalente en la base de datos"""
        mapping = {
            "currentPrice": "current_price",
            "previousClose": "previous_close",
            "dayLow": "day_low",
            "dayHigh": "day_high",
            "dividendYield": "dividend_yield",
            "fiftyTwoWeekRange": "fifty_two_week_range",
            "marketCap": "market_cap",
            "trailingPE": "trailing_pe",
            "earningsGrowth": "earnings_growth",
            "revenueGrowth": "revenue_growth",
            "grossMargins": "gross_margins",
            "ebitdaMargins": "ebitda_margins",
            "operatingMargins": "operating_margins",
            "financialCurrency": "financial_currency",
            "returnOnAssets": "return_on_assets",
            "returnOnEquity": "return_on_equity",
            "bookValue": "book_value",
            "priceToBook": "price_to_book"
        }

        return mapping.get(yahoo_field, yahoo_field.lower())

    def fetch_instrument_data(self, symbol: str, config: Dict) -> Dict:
        """Obtiene los datos del instrumento financiero con la misma lógica que scraper.py"""
        try:
            self.logger.info(f"Iniciando scraping para {symbol}")
            ticker = yf.Ticker(symbol)
            data = {}
            field_mapping = config.get("field_mapping", {})

            if config.get("is_etf", False):
                self.logger.debug(f"{symbol} es un ETF, usando metodo especial")
                history = ticker.history(period="1d")

                # 1. Obtener currentPrice desde el historial
                data["current_price"] = history["Close"].iloc[-1] if not history.empty else None
                self.logger.debug(f"Precio actual obtenido: {data['current_price']}")

                # 2. Procesar el volumen desde el historial (si está configurado)
                if "volume" in config["yahoo_fields"]:
                    data["volume"] = history["Volume"].iloc[-1] if not history.empty else None
                    self.logger.debug(f"Volumen obtenido: {data['volume']}")

                # 3. Procesar el resto de campos desde ticker.info
                info = ticker.info
                for yahoo_field in config["yahoo_fields"]:
                    if yahoo_field in ["currentPrice", "volume"]:  # Campos ya asignados
                        continue

                    # Mapeo CORRECTO: Yahoo Finance → CSV
                    db_field = self._map_yahoo_to_db_field(yahoo_field)  # Ej: "dividendYield" → "dividend_yield"
                    yahoo_key = field_mapping.get(yahoo_field, yahoo_field)  # Ej: "volume" → "volumen"

                    # Solo procesar si la columna existe en la tabla
                    if db_field in self.table_columns:
                        data[db_field] = info.get(yahoo_key, None)
                        self.logger.debug(f"{db_field} (desde Yahoo '{yahoo_key}'): {data[db_field]}")

            else:  # Para acciones (no-ETF)
                self.logger.debug(f"{symbol} es una accion, usando metodo estandar")
                info = ticker.info
                for yahoo_field in config["yahoo_fields"]:
                    db_field = self._map_yahoo_to_db_field(yahoo_field)
                    if db_field in self.table_columns:
                        data[db_field] = info.get(yahoo_field, None)
                        self.logger.debug(f"{db_field}: {data[db_field]}")

            # Campos comunes
            data["symbol"] = symbol
            data["timestamp"] = datetime.now()
            return data

        except Exception as e:
            self.logger.error(f"Error al obtener datos de {symbol}: {str(e)}", exc_info=True)
            return None

    def save_to_postgres(self, data: Dict) -> None:
        """Guarda los datos en la tabla stock_data de PostgreSQL"""
        if data is None:
            self.logger.warning("No hay datos para guardar")
            return

        try:
            # Filtrar solo las columnas que existen en la tabla
            filtered_data = {}
            for col in data.keys():
                if col in self.table_columns:
                    filtered_data[col] = data[col]

            if not filtered_data:
                self.logger.warning("No hay datos válidos para guardar después de filtrar por columnas existentes")
                return

            # Preparar columnas y valores para la inserción
            columns = list(filtered_data.keys())

            # Convertir valores numpy a tipos Python nativos
            values = []
            for col in columns:
                val = filtered_data[col]
                # Convertir tipos numpy a tipos Python nativos
                if hasattr(val, 'dtype') and 'numpy' in str(type(val)):
                    if np.issubdtype(val.dtype, np.integer):
                        val = int(val)
                    elif np.issubdtype(val.dtype, np.floating):
                        val = float(val)
                    elif np.issubdtype(val.dtype, np.bool_):
                        val = bool(val)
                # Manejo especial de NaN e infinitos
                if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                    val = None
                values.append(val)

            # Crear sentencia SQL para inserción
            column_str = ", ".join(columns)
            placeholder_str = ", ".join(["%s"] * len(columns))

            # Generar la parte SET de la cláusula ON CONFLICT
            update_parts = []
            for col in columns:
                if col not in ["symbol", "timestamp"]:  # No actualizar las claves
                    update_parts.append(f"{col} = EXCLUDED.{col}")

            # Completar la consulta SQL
            if update_parts:
                insert_sql = f"""
                INSERT INTO stock_data ({column_str})
                VALUES ({placeholder_str})
                ON CONFLICT (symbol, timestamp) DO UPDATE SET {', '.join(update_parts)}
                """
            else:
                insert_sql = f"""
                INSERT INTO stock_data ({column_str})
                VALUES ({placeholder_str})
                ON CONFLICT (symbol, timestamp) DO NOTHING
                """

            with self.conn.cursor() as cur:
                cur.execute(insert_sql, values)

            self.conn.commit()
            self.logger.info(f"Datos guardados en tabla stock_data para {data['symbol']}")
            self.logger.debug(f"Último registro: {data['timestamp']}")

        except Exception as e:
            self.conn.rollback()
            self.logger.error(f"Error al guardar datos: {str(e)}", exc_info=True)

    def run_scraping_cycle(self) -> None:
        """Ejecuta un ciclo completo de scraping"""
        self.logger.info("Iniciando ciclo de scraping")
        for symbol, config in ScraperConfig.INSTRUMENTS.items():
            data = self.fetch_instrument_data(symbol, config)
            if data:
                self.save_to_postgres(data)
        self.logger.info("Ciclo de scraping completado")

    def run(self) -> None:
        """Ejecuta el scraping en intervalos regulares"""
        self.logger.info("Iniciando proceso de scraping")
        self.logger.info(f"Instrumentos monitoreados: {', '.join(ScraperConfig.INSTRUMENTS.keys())}")
        self.logger.info(f"Intervalo de actualización: {ScraperConfig.UPDATE_INTERVAL_MINUTES} minutos")
        self.logger.info(f"Duración total del proceso: {ScraperConfig.TOTAL_RUNTIME_HOURS} horas")
        self.logger.info(f"Base de datos: {ScraperConfig.PG_DATABASE}")
        self.logger.info(f"Ubicación de archivos de log: {ScraperConfig.LOGS_DIR}")

        try:
            for i in range(ScraperConfig.TOTAL_ITERATIONS):
                self.logger.info(f"Ciclo {i + 1}/{ScraperConfig.TOTAL_ITERATIONS}")
                self.run_scraping_cycle()

                if i < ScraperConfig.TOTAL_ITERATIONS - 1:
                    self.logger.info(
                        f"Esperando {ScraperConfig.UPDATE_INTERVAL_MINUTES} minutos para siguiente ciclo...")
                    time.sleep(ScraperConfig.UPDATE_INTERVAL_MINUTES * 60)

            self.logger.info("Proceso completado exitosamente")

        except KeyboardInterrupt:
            self.logger.warning("Proceso interrumpido por el usuario")
        except Exception as e:
            self.logger.critical(f"Error crítico en el proceso: {str(e)}", exc_info=True)
            raise
        finally:
            # Cerrar la conexión a PostgreSQL al finalizar
            if self.conn is not None:
                self.conn.close()
                self.logger.info("Conexión a PostgreSQL cerrada")


if __name__ == "__main__":
    scraper = DataScraper()
    scraper.run()