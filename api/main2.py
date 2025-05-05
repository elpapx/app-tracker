# ---- IMPORT LIBRARIES --------
import os
import time
import threading
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import psutil
import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import sql
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel

# ---- CONFIGURATION --------
# Configuración desde variables de entorno o valores por defecto
PG_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "port": int(os.environ.get("DB_PORT", 5432)),
    "user": os.environ.get("DB_USER", "bvl_user"),
    "password": os.environ.get("DB_PASSWORD", "179fae82"),
    "database": os.environ.get("DB_NAME", "bvl_monitor")
}

# Configuración del portafolio
ORIGINAL_PRICES = {
    "BAP": 184.88,
    "BRK-B": 479.20,
    "ILF": 24.10
}

PORTFOLIO_DATA = {
    "BAP": {"description": "Credicorp Ltd.", "purchase_price": 184.88, "qty": 26},
    "BRK-B": {"description": "Berkshire Hathaway Inc. Class B", "purchase_price": 479.20, "qty": 10},
    "ILF": {"description": "iShares Latin America 40 ETF", "purchase_price": 24.10, "qty": 200}
}

# Configuración de cache
MAX_CACHE_SIZE = 5  # Número máximo de DataFrames en caché
CACHE_ITEM_TTL = 600  # 10 minutos en segundos

# Fecha de transición entre datos históricos y actuales
TRANSITION_DATE = datetime(2025, 4, 5)


# ---- LOGGING CONFIGURATION --------
def setup_logging():
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            RotatingFileHandler(
                os.path.join(log_dir, 'api.log'),
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=5,
                encoding='utf-8'
            ),
            logging.StreamHandler()
        ]
    )

    global logger, cache_logger, perf_logger, mem_logger
    logger = logging.getLogger(__name__)
    cache_logger = logging.getLogger('cache')
    perf_logger = logging.getLogger('performance')
    mem_logger = logging.getLogger('memory')

    # Log de memoria inicial
    log_memory_usage("Inicio de la aplicación")


def log_memory_usage(context=""):
    """Registra el uso de memoria actual del proceso"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_mb = mem_info.rss / (1024 * 1024)  # Convertir bytes a MB

    with cache_lock:
        cache_size_mb = cache_size / (1024 * 1024)
        cache_count = len(dataframes_cache)

    mem_logger.info(f"{context} - Memoria: {mem_mb:.2f} MB - Caché: {cache_size_mb:.2f} MB ({cache_count} DataFrames)")
    return mem_mb


# ---- DATABASE CONNECTION --------
def get_db_connection():
    """Establece y retorna una conexión a la base de datos PostgreSQL"""
    try:
        conn = psycopg2.connect(**PG_CONFIG)
        return conn
    except Exception as e:
        logger.error(f"Error al conectar a PostgreSQL: {str(e)}")
        raise


# ---- CACHE IMPLEMENTATION --------
class CacheItem:
    def __init__(self, df: pd.DataFrame, last_modified: float):
        self.df = df
        self.last_load_time = datetime.now()
        self.last_modified = last_modified
        self.size = df.memory_usage(deep=True).sum() if df is not None else 0
        self.access_count = 0

    def __lt__(self, other):
        return self.access_count < other.access_count


dataframes_cache: Dict[str, CacheItem] = {}
cache_lock = threading.Lock()
cache_size = 0


# ---- PYDANTIC MODELS --------
class StockData(BaseModel):
    symbol: str
    currentPrice: float
    previousClose: float
    open: float
    dayLow: float
    dayHigh: float
    dividendYield: Optional[float] = None
    financialCurrency: str
    volumen: Optional[int] = None
    timestamp: str


class ProfitabilityData(BaseModel):
    symbol: str
    name: str
    original_price: float
    current_price: float
    profitability_percentage: float


class TimeSeriesPoint(BaseModel):
    timestamp: str
    price: float
    return_percentage: Optional[float] = None
    volume: Optional[float] = None
    open: Optional[float] = None
    day_low: Optional[float] = None
    day_high: Optional[float] = None
    previous_close: Optional[float] = None


class SymbolTimeSeries(BaseModel):
    symbol: str
    data: List[TimeSeriesPoint]
    period: str
    current_price: Optional[float] = None
    original_price: Optional[float] = None
    current_profitability: Optional[float] = None
    average_volume: Optional[float] = None
    open_price: Optional[float] = None
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    fifty_two_week_range: Optional[str] = None
    market_cap: Optional[float] = None
    trailing_pe: Optional[float] = None
    dividend_yield: Optional[float] = None
    daily_variation: float = 0.0
    volatility: float = 0.0


class TimeSeriesResponse(BaseModel):
    series: List[SymbolTimeSeries]
    available_periods: List[str] = ["1d", "1w", "1m", "3m"]
    available_symbols: List[str] = ["BAP", "BRK-B", "ILF"]


class FinancialDataPoint(BaseModel):
    timestamp: str
    price: float
    volume: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    dividend_yield: Optional[float] = None
    profitability: Optional[float] = None


class SymbolFinancialData(BaseModel):
    symbol: str
    period: str
    data: List[FinancialDataPoint]
    stats: Dict[str, float]


class StockHolding(BaseModel):
    symbol: str
    description: str
    current_price: float
    todays_change: float
    todays_change_percent: float
    purchase_price: float
    qty: int
    total_value: float
    total_gain_loss: float
    total_gain_loss_percent: float


class PortfolioHoldings(BaseModel):
    total_value: float
    todays_change: float
    todays_change_percent: float
    total_gain_loss: float
    total_gain_loss_percent: float
    holdings: List[StockHolding]


class StockAPIException(HTTPException):
    """Excepción personalizada para errores de la API de acciones"""

    def __init__(self, status_code: int, detail: str, code: str = None):
        super().__init__(status_code=status_code, detail=detail)
        self.code = code


# ---- HELPER FUNCTIONS --------
def clear_cache_if_needed():
    """Limpia la caché si excede el tamaño máximo usando política LRU"""
    global cache_size, dataframes_cache

    if len(dataframes_cache) <= MAX_CACHE_SIZE:
        return

    cache_logger.info(f"Limpiando caché (actual: {len(dataframes_cache)} items)")
    sorted_items = sorted(dataframes_cache.items(), key=lambda x: x[1].access_count)

    while len(dataframes_cache) > MAX_CACHE_SIZE and sorted_items:
        symbol, item = sorted_items.pop(0)
        cache_size -= item.size
        del dataframes_cache[symbol]
        cache_logger.debug(f"Eliminado {symbol} de la caché (tamaño: {item.size / 1024:.2f} KB)")


def get_project_root():
    """Retorna la ruta raíz del proyecto"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_historical_data(symbol: str) -> Optional[pd.DataFrame]:
    """Carga datos históricos desde archivos CSV incluyendo volumen"""
    log_memory_usage(f"Antes de cargar datos históricos para {symbol}")

    root_path = get_project_root()
    hist_filename = f"{symbol.lower()}_historical.csv"
    hist_file_path = os.path.join(root_path, "historical_data", hist_filename)

    if not os.path.exists(hist_file_path):
        return None

    try:
        df = pd.read_csv(hist_file_path)
        df['timestamp'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)

        # Mapear columnas a nombres esperados
        column_mapping = {
            'Open': 'currentPrice',
            'High': 'dayHigh',
            'Low': 'dayLow',
            'Close': 'currentPrice',
            'Volume': 'volumen',
            'Adj Close': None
        }

        df = df.rename(columns={k: v for k, v in column_mapping.items() if v is not None})

        # Asegurar que todas las columnas requeridas existen
        required_cols = ['timestamp', 'currentPrice', 'dayHigh', 'dayLow', 'volumen']
        for col in required_cols:
            if col not in df.columns:
                df[col] = None

        log_memory_usage(f"Después de cargar datos históricos para {symbol} ({len(df)} filas)")
        return df[required_cols]
    except Exception as e:
        logger.error(f"Error al cargar datos históricos para {symbol}: {str(e)}")
        return None


def load_data_from_postgres(symbol: str, force_reload: bool = False) -> Optional[pd.DataFrame]:
    """Carga DataFrame desde PostgreSQL con manejo optimizado de caché"""
    global dataframes_cache, cache_size

    with cache_lock:
        cache_item = dataframes_cache.get(symbol)

        if cache_item and not force_reload:
            cache_expired = (datetime.now() - cache_item.last_load_time).total_seconds() > CACHE_ITEM_TTL

            if not cache_expired:
                cache_item.access_count += 1
                cache_logger.debug(f"Usando caché para {symbol} (accesos: {cache_item.access_count})")
                return cache_item.df

        log_memory_usage(f"Antes de cargar datos PostgreSQL para {symbol}")
        perf_logger.info(f"Cargando datos PostgreSQL para {symbol} {'(forzado)' if force_reload else ''}")
        start_time = time.time()

        try:
            conn = get_db_connection()
            query = """
                    SELECT
                        timestamp, current_price AS "currentPrice", previous_close AS "previousClose", open, day_low AS "dayLow", day_high AS "dayHigh", dividend_yield AS "dividendYield", volume AS "volumen", market_cap AS "marketCap", trailing_pe AS "trailingPE", fifty_two_week_range AS "fiftyTwoWeekRange"
                    FROM stock_data
                    WHERE symbol = %s
                    ORDER BY timestamp
                    """
            df = pd.read_sql(query, conn, params=(symbol,))
            conn.close()

            if df.empty:
                logger.warning(f"No se encontraron datos PostgreSQL para el símbolo: {symbol}")
                return None

            # Convertir tipos de datos
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            numeric_cols = ['currentPrice', 'previousClose', 'open', 'dayLow', 'dayHigh',
                            'dividendYield', 'volumen', 'marketCap', 'trailingPE']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Manejar caché
            df_size = df.memory_usage(deep=True).sum()
            perf_logger.info(
                f"DataFrame PostgreSQL {symbol} cargado en {time.time() - start_time:.2f}s - Tamaño: {df_size / 1024:.2f} KB")

            new_cache_item = CacheItem(df, datetime.now().timestamp())
            new_cache_item.access_count = 1

            if symbol in dataframes_cache:
                cache_size -= dataframes_cache[symbol].size

            dataframes_cache[symbol] = new_cache_item
            cache_size += df_size
            clear_cache_if_needed()

            log_memory_usage(f"Después de cargar datos PostgreSQL para {symbol} ({len(df)} filas)")
            return df

        except Exception as e:
            logger.error(f"Error al cargar datos PostgreSQL para {symbol}: {str(e)}", exc_info=True)
            return None


def load_combined_data(symbol: str, force_reload: bool = False) -> Optional[pd.DataFrame]:
    """Combina datos históricos (CSV) y actuales (PostgreSQL) en un solo DataFrame"""
    log_memory_usage(f"Antes de combinar datos para {symbol}")

    # Cargar datos históricos desde CSV
    historical_df = load_historical_data(symbol)

    # Cargar datos actuales desde PostgreSQL
    current_df = load_data_from_postgres(symbol, force_reload)

    # Combinar ambos DataFrames
    if historical_df is not None and current_df is not None:
        combined_df = pd.concat([historical_df, current_df], ignore_index=True)
    elif historical_df is not None:
        combined_df = historical_df
    elif current_df is not None:
        combined_df = current_df
    else:
        return None

    # Asegurar ordenamiento adecuado y eliminar duplicados
    combined_df = combined_df.sort_values('timestamp').drop_duplicates('timestamp')

    # Añadir columna symbol si falta
    if 'symbol' not in combined_df.columns:
        combined_df['symbol'] = symbol

    log_memory_usage(f"Después de combinar datos para {symbol} ({len(combined_df)} filas)")
    return combined_df


def get_latest_data(symbol: str) -> Dict:
    """Obtiene los datos más recientes para un símbolo desde PostgreSQL con respaldo histórico"""
    # Primero intentar con PostgreSQL
    latest = get_latest_from_postgres(symbol)

    # Si no hay datos en PostgreSQL, intentar con históricos
    if latest is None:
        latest = get_latest_from_historical(symbol)

    # Si aún no hay datos, usar valores predeterminados
    if latest is None:
        logger.warning(f"No se encontraron datos para {symbol} en ninguna fuente")
        latest = {
            'symbol': symbol,
            'currentPrice': ORIGINAL_PRICES.get(symbol, 0.0),
            'previousClose': ORIGINAL_PRICES.get(symbol, 0.0),
            'timestamp': datetime.now().isoformat()
        }

    # Asegurar que todos los campos requeridos existan
    required_fields = ['currentPrice', 'previousClose', 'open', 'dayLow', 'dayHigh',
                       'dividendYield', 'volumen', 'timestamp', 'symbol']
    for field in required_fields:
        if field not in latest:
            latest[field] = None

    # Manejar precios nulos
    if latest['currentPrice'] is None and symbol in ORIGINAL_PRICES:
        latest['currentPrice'] = ORIGINAL_PRICES[symbol]

    # Asegurar que volumen sea entero
    if latest['volumen'] is not None:
        try:
            latest['volumen'] = int(latest['volumen'])
        except (ValueError, TypeError):
            latest['volumen'] = 0

    return latest


def get_latest_from_postgres(symbol: str) -> Optional[Dict]:
    """Obtiene el registro más reciente desde PostgreSQL"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        query = """
                SELECT
                    timestamp, current_price, previous_close, open, day_low, day_high, dividend_yield, volume, market_cap, trailing_pe, fifty_two_week_range
                FROM stock_data
                WHERE symbol = %s
                ORDER BY timestamp DESC
                    LIMIT 1
                """
        cursor.execute(query, (symbol,))
        result = cursor.fetchone()
        conn.close()

        if not result:
            return None

        columns = [desc[0] for desc in cursor.description]
        data = dict(zip(columns, result))

        return {
            'symbol': symbol,
            'timestamp': data['timestamp'].isoformat(),
            'currentPrice': data['current_price'],
            'previousClose': data['previous_close'],
            'open': data['open'],
            'dayLow': data['day_low'],
            'dayHigh': data['day_high'],
            'dividendYield': data['dividend_yield'],
            'volumen': data['volume'],
            'marketCap': data['market_cap'],
            'trailingPE': data['trailing_pe'],
            'fiftyTwoWeekRange': data['fifty_two_week_range'],
            'financialCurrency': 'USD'
        }
    except Exception as e:
        logger.error(f"Error al obtener datos PostgreSQL recientes para {symbol}: {str(e)}")
        return None


def get_latest_from_historical(symbol: str) -> Optional[Dict]:
    """Obtiene el registro más reciente desde datos históricos CSV"""
    try:
        df = load_historical_data(symbol)
        if df is None or df.empty:
            return None

        latest = df.iloc[-1].to_dict()

        return {
            'symbol': symbol,
            'timestamp': latest['timestamp'].isoformat(),
            'currentPrice': latest.get('currentPrice'),
            'previousClose': latest.get('currentPrice'),  # Mismo que actual para históricos
            'open': latest.get('currentPrice'),
            'dayLow': latest.get('dayLow'),
            'dayHigh': latest.get('dayHigh'),
            'volumen': latest.get('volumen'),
            'financialCurrency': 'USD'
        }
    except Exception as e:
        logger.error(f"Error al obtener datos históricos recientes para {symbol}: {str(e)}")
        return None


def clean_json_data(data: Any) -> Any:
    """Limpia datos para asegurar serializabilidad JSON"""
    if isinstance(data, dict):
        return {k: clean_json_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_json_data(item) for item in data]
    elif pd.isna(data) or data is pd.NA:
        return None
    elif isinstance(data, (float, np.floating)) and (np.isnan(data) or np.isinf(data)):
        return None
    elif isinstance(data, pd.Timestamp):
        return data.isoformat()
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    else:
        return data


def background_update_all_dataframes():
    """Actualiza todos los dataframes en segundo plano desde PostgreSQL"""
    try:
        symbols = ["BAP", "BRK-B", "ILF"]
        for symbol in symbols:
            load_data_from_postgres(symbol, force_reload=True)
            time.sleep(1)  # Pequeño retraso entre símbolos
        logger.info(f"Actualización en segundo plano completada para {len(symbols)} símbolos")
    except Exception as e:
        logger.error(f"Error durante la actualización en segundo plano: {str(e)}")


# ---- FASTAPI SETUP --------
app = FastAPI(
    title="BVL Live Tracker API",
    description="API para datos de acciones en tiempo real",
    version="1.0.0"
)

# Archivos estáticos y configuración CORS
static_dir = os.path.join(os.path.dirname(__file__), 'static')
app.mount("/static", StaticFiles(directory=static_dir), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- MONITORING ENDPOINTS --------
@app.get("/monitor/cache")
def monitor_cache():
    """Endpoint para monitorear el estado de la caché"""
    log_memory_usage("Endpoint monitor/cache")
    with cache_lock:
        cache_info = {
            "total_items": len(dataframes_cache),
            "total_size": f"{cache_size / 1024:.2f} KB",
            "max_size": MAX_CACHE_SIZE,
            "item_ttl": CACHE_ITEM_TTL,
            "items": []
        }

        for symbol, item in dataframes_cache.items():
            cache_info["items"].append({
                "symbol": symbol,
                "size": f"{item.size / 1024:.2f} KB",
                "last_access": item.last_load_time.isoformat(),
                "access_count": item.access_count,
                "age_seconds": (datetime.now() - item.last_load_time).total_seconds()
            })

    return cache_info


@app.get("/monitor/logs")
def monitor_logs(lines: int = 100):
    """Endpoint para ver logs recientes"""
    log_memory_usage("Endpoint monitor/logs")
    log_file = os.path.join(os.path.dirname(__file__), 'logs', 'api.log')

    if not os.path.exists(log_file):
        raise HTTPException(status_code=404, detail="Archivo de log no encontrado")

    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines

        return HTMLResponse(content="<pre>" + "".join(recent_lines) + "</pre>")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al leer logs: {str(e)}")


@app.get("/monitor/memory")
def monitor_memory():
    """Endpoint para monitorear el uso de memoria"""
    mem_mb = log_memory_usage("Endpoint monitor/memory")

    # Obtener información adicional del sistema
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()

    return {
        "timestamp": datetime.now().isoformat(),
        "process_id": os.getpid(),
        "memory_usage_mb": mem_mb,
        "memory_details": {
            "rss_mb": mem_info.rss / (1024 * 1024),
            "vms_mb": mem_info.vms / (1024 * 1024),
            "shared_mb": getattr(mem_info, 'shared', 0) / (1024 * 1024),
            "percent": process.memory_percent()
        },
        "cache_info": {
            "total_items": len(dataframes_cache),
            "total_size_mb": cache_size / (1024 * 1024)
        },
        "system_memory": {
            "total_gb": psutil.virtual_memory().total / (1024 ** 3),
            "available_gb": psutil.virtual_memory().available / (1024 ** 3),
            "percent_used": psutil.virtual_memory().percent
        }
    }


# ---- MAIN ENDPOINTS --------
@app.get("/api/timeseries-with-profitability", response_model=TimeSeriesResponse)
async def get_time_series_with_profitability(
        symbol: str = Query("BAP", description="Símbolo a consultar (BAP, BRK-B, ILF)"),
        period: str = Query("1w", description="Periodo (realtime, 1d, 1w, 1m, 3m)"),
        compare_all: bool = Query(False, description="Mostrar todos los símbolos juntos")
):
    """Obtiene series temporales con información de rentabilidad"""
    log_memory_usage(f"Endpoint timeseries-with-profitability - {symbol}, {period}")
    periods_map = {
        "realtime": timedelta(days=1),
        "1d": timedelta(days=1),
        "1w": timedelta(weeks=1),
        "1m": timedelta(days=30),
        "3m": timedelta(days=90)
    }

    symbols_to_fetch = ["BAP", "BRK-B", "ILF"] if compare_all else [symbol]
    end_date = datetime.now()

    logger.info(f"Procesando símbolos: {symbols_to_fetch}, periodo: {period}")
    result = []

    for sym in symbols_to_fetch:
        try:
            # Cargar datos combinados (históricos + actuales)
            combined_df = load_combined_data(sym)

            if combined_df is None or combined_df.empty:
                logger.warning(f"No hay datos combinados para {sym}")
                continue

            if period == "realtime":
                today = end_date.replace(hour=0, minute=0, second=0, microsecond=0)
                filtered_df = combined_df[
                    (combined_df['timestamp'] >= today) &
                    (combined_df['timestamp'] <= end_date)
                    ]
                # Filtrar horas de mercado (8am a 4pm)
                filtered_df = filtered_df[
                    (filtered_df['timestamp'].dt.hour >= 8) &
                    (filtered_df['timestamp'].dt.hour <= 16)
                    ]
            else:
                start_date = end_date - periods_map[period]
                filtered_df = combined_df[
                    (combined_df['timestamp'] >= start_date) &
                    (combined_df['timestamp'] <= end_date)
                    ]

            filtered_df = filtered_df.sort_values('timestamp')

            logger.info(f"Datos filtrados para {sym} ({period}): {len(filtered_df)} registros")

            if filtered_df.empty:
                logger.warning(f"No hay datos para {sym} en el periodo {period}")
                continue

            last_current_data = get_latest_data(sym) or {}

            series_data = []
            for _, row in filtered_df.iterrows():
                if pd.notna(row['currentPrice']):
                    point = TimeSeriesPoint(
                        timestamp=row['timestamp'].isoformat(),
                        price=float(row['currentPrice']),
                        volume=float(row['volumen']) if pd.notna(row.get('volumen')) else None,
                        open=float(row['open']) if pd.notna(row.get('open')) else None,
                        day_low=float(row['dayLow']) if pd.notna(row.get('dayLow')) else None,
                        day_high=float(row['dayHigh']) if pd.notna(row.get('dayHigh')) else None,
                        previous_close=float(row['previousClose']) if pd.notna(row.get('previousClose')) else None
                    )
                    series_data.append(point)

            original_price = ORIGINAL_PRICES.get(sym)
            last_price = filtered_df['currentPrice'].iloc[-1] if not filtered_df.empty else None

            symbol_series = SymbolTimeSeries(
                symbol=sym,
                data=series_data,
                period=period,
                original_price=original_price,
                current_price=last_price,
                current_profitability=(
                    ((last_price - original_price) / original_price) * 100
                    if last_price and original_price and original_price > 0
                    else None
                ),
                market_cap=(
                    float(last_current_data['marketCap'])
                    if last_current_data and 'marketCap' in last_current_data
                       and pd.notna(last_current_data['marketCap'])
                    else None
                ),
                trailing_pe=(
                    float(last_current_data['trailingPE'])
                    if last_current_data and 'trailingPE' in last_current_data
                       and pd.notna(last_current_data['trailingPE'])
                    else None
                ),
                dividend_yield=(
                    float(last_current_data['dividendYield'])
                    if last_current_data and 'dividendYield' in last_current_data
                       and pd.notna(last_current_data['dividendYield'])
                    else None
                ),
                fifty_two_week_range=last_current_data.get('fiftyTwoWeekRange')
            )
            result.append(symbol_series)

        except Exception as e:
            logger.error(f"Error al procesar {sym}: {str(e)}", exc_info=True)

    log_memory_usage(f"Fin endpoint timeseries-with-profitability")
    return TimeSeriesResponse(series=result)


@app.get("/api/timeseries-variations", response_model=TimeSeriesResponse)
async def get_time_series_variations(
        symbol: str = Query("BAP", description="Símbolo a consultar (BAP, BRK-B, ILF)"),
        period: str = Query("1w", description="Periodo (realtime, 1d, 1w, 1m, 3m)"),
        compare_all: bool = Query(False, description="Mostrar todos los símbolos juntos")
):
    """Obtiene series temporales con información de variación de precios"""
    log_memory_usage(f"Endpoint timeseries-variations - {symbol}, {period}")
    periods_map = {
        "realtime": timedelta(hours=6),
        "1d": timedelta(days=1),
        "1w": timedelta(weeks=1),
        "1m": timedelta(days=30),
        "3m": timedelta(days=90)
    }

    symbols_to_fetch = ["BAP", "BRK-B", "ILF"] if compare_all else [symbol]
    end_date = datetime.now()

    logger.info(f"Procesando símbolos: {symbols_to_fetch}, periodo: {period}")
    result = []

    for sym in symbols_to_fetch:
        try:
            # Cargar datos combinados (históricos + actuales)
            combined_df = load_combined_data(sym)

            if combined_df is None or combined_df.empty:
                logger.warning(f"No hay datos combinados para {sym}")
                continue

            start_date = end_date - periods_map[period]
            filtered_df = combined_df[
                (combined_df['timestamp'] >= start_date) &
                (combined_df['timestamp'] <= end_date)
                ].copy()

            if period == "realtime":
                market_open = end_date.replace(hour=8, minute=0, second=0, microsecond=0)
                market_close = end_date.replace(hour=16, minute=0, second=0, microsecond=0)
                filtered_df = filtered_df[
                    (filtered_df['timestamp'] >= market_open) &
                    (filtered_df['timestamp'] <= market_close)
                    ]

            if filtered_df.empty:
                logger.warning(f"No hay datos para {sym} en el periodo {period}")
                continue

            # Calcular variaciones
            if period == "realtime":
                filtered_df['variation_pct'] = filtered_df['currentPrice'].pct_change() * 100
            else:
                first_valid_idx = filtered_df['currentPrice'].first_valid_index()
                if first_valid_idx is not None:
                    first_price = filtered_df.loc[first_valid_idx, 'currentPrice']
                    filtered_df['variation_pct'] = ((filtered_df['currentPrice'] - first_price) / first_price) * 100
                else:
                    filtered_df['variation_pct'] = 0.0

            filtered_df['variation_pct'] = filtered_df['variation_pct'].replace([np.inf, -np.inf], np.nan).fillna(0)

            last_current_data = get_latest_data(sym) or {}

            series_data = []
            for _, row in filtered_df.iterrows():
                point = TimeSeriesPoint(
                    timestamp=row['timestamp'].isoformat(),
                    price=float(row['variation_pct']),
                    volume=float(row['volumen']) if pd.notna(row.get('volumen')) else None,
                    open=float(row['open']) if pd.notna(row.get('open')) else None,
                    day_low=float(row['dayLow']) if pd.notna(row.get('dayLow')) else None,
                    day_high=float(row['dayHigh']) if pd.notna(row.get('dayHigh')) else None,
                    previous_close=float(row['previousClose']) if pd.notna(row.get('previousClose')) else None
                )
                series_data.append(point)

            original_price = ORIGINAL_PRICES.get(sym)
            last_price = filtered_df['currentPrice'].iloc[
                -1] if 'currentPrice' in filtered_df.columns and not filtered_df.empty else None
            last_variation = filtered_df['variation_pct'].iloc[-1] if not filtered_df.empty else 0.0
            volatility = filtered_df['variation_pct'].std() if len(filtered_df) > 1 else 0

            symbol_series = SymbolTimeSeries(
                symbol=sym,
                data=series_data,
                period=period,
                original_price=original_price,
                current_price=last_price,
                current_profitability=(
                    ((last_price - original_price) / original_price) * 100
                    if last_price and original_price and original_price > 0
                    else None
                ),
                market_cap=(
                    float(last_current_data.get('marketCap'))
                    if last_current_data and pd.notna(last_current_data.get('marketCap'))
                    else None
                ),
                trailing_pe=(
                    float(last_current_data.get('trailingPE'))
                    if last_current_data and pd.notna(last_current_data.get('trailingPE'))
                    else None
                ),
                dividend_yield=(
                    float(last_current_data.get('dividendYield'))
                    if last_current_data and pd.notna(last_current_data.get('dividendYield'))
                    else None
                ),
                fifty_two_week_range=last_current_data.get('fiftyTwoWeekRange'),
                daily_variation=float(last_variation),
                volatility=float(volatility)
            )
            result.append(symbol_series)

        except Exception as e:
            logger.error(f"Error al procesar {sym}: {str(e)}", exc_info=True)
            continue

    log_memory_usage(f"Fin endpoint timeseries-variations")
    return TimeSeriesResponse(series=result)


@app.get("/portfolio/holdings/live", response_model=PortfolioHoldings)
def get_portfolio_holdings_live():
    """Obtiene datos de cartera en tiempo real"""
    log_memory_usage("Endpoint portfolio/holdings/live")
    holdings = []
    portfolio_total_value = 0.0
    portfolio_todays_change_value = 0.0
    portfolio_total_gain_loss = 0.0
    portfolio_previous_value = 0.0

    for symbol, data in PORTFOLIO_DATA.items():
        current_data = get_latest_data(symbol)

        if current_data is None:
            logger.error(f"No se encontraron datos para {symbol}")
            continue

        current_price = current_data.get('currentPrice')
        previous_close = current_data.get('previousClose')

        if previous_close is None and current_price is not None:
            logger.warning(f"No hay datos de cierre anterior para {symbol}, usando precio actual")
            previous_close = current_price

        if current_price is None:
            logger.error(f"No hay precio actual para {symbol}")
            continue

        purchase_price = data["purchase_price"]
        qty = data["qty"]

        todays_change = current_price - previous_close
        todays_change_percent = (todays_change / previous_close) * 100 if previous_close > 0 else 0

        total_value = current_price * qty
        total_gain_loss = total_value - (purchase_price * qty)
        total_gain_loss_percent = (total_gain_loss / (purchase_price * qty)) * 100 if purchase_price > 0 else 0

        portfolio_total_value += total_value
        portfolio_todays_change_value += todays_change * qty
        portfolio_total_gain_loss += total_gain_loss
        portfolio_previous_value += previous_close * qty

        holding = StockHolding(
            symbol=symbol,
            description=data["description"],
            current_price=round(current_price, 2),
            todays_change=round(todays_change, 2),
            todays_change_percent=round(todays_change_percent, 2),
            purchase_price=round(purchase_price, 2),
            qty=qty,
            total_value=round(total_value, 2),
            total_gain_loss=round(total_gain_loss, 2),
            total_gain_loss_percent=round(total_gain_loss_percent, 2)
        )
        holdings.append(holding)

    portfolio_initial_value = portfolio_total_value - portfolio_total_gain_loss
    portfolio_todays_change_percent = (
        (portfolio_todays_change_value / portfolio_previous_value) * 100
        if portfolio_previous_value > 0 else 0
    )
    portfolio_total_gain_loss_percent = (
        (portfolio_total_gain_loss / portfolio_initial_value) * 100
        if portfolio_initial_value > 0 else 0
    )

    log_memory_usage(f"Fin endpoint portfolio/holdings/live")
    return PortfolioHoldings(
        total_value=round(portfolio_total_value, 2),
        todays_change=round(portfolio_todays_change_value, 2),
        todays_change_percent=round(portfolio_todays_change_percent, 2),
        total_gain_loss=round(portfolio_total_gain_loss, 2),
        total_gain_loss_percent=round(portfolio_total_gain_loss_percent, 2),
        holdings=holdings
    )


@app.post("/refresh")
def refresh_data(background_tasks: BackgroundTasks):
    """Fuerza una actualización de todos los datos en caché"""
    log_memory_usage("Endpoint refresh")
    background_tasks.add_task(background_update_all_dataframes)
    return {"message": "Actualización de datos en segundo plano iniciada"}


@app.get("/health")
def health_check():
    """Endpoint de comprobación de salud de la API"""
    log_memory_usage("Endpoint health")
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "cached_symbols": list(dataframes_cache.keys()),
        "memory_usage_mb": log_memory_usage("Health check"),
        "process_id": os.getpid(),
        "uptime": "N/A"  # Puedes implementar cálculo de tiempo de ejecución si es necesario
    }


# ---- HTML ENDPOINTS --------
@app.get("/html/timeseries-profitability", response_class=HTMLResponse)
async def get_timeseries_profitability_html():
    """Sirve la interfaz de análisis completa"""
    return FileResponse(os.path.join(static_dir, "timeseries-profitability.html"))


@app.get("/html/market/variations", response_class=HTMLResponse)
async def get_market_variations_html():
    """Sirve la página HTML de variaciones de precios"""
    return FileResponse(os.path.join(static_dir, "market_variations.html"))


@app.get("/html/portfolio/holdings", response_class=HTMLResponse)
async def get_portfolio_holdings_html():
    """Sirve la página HTML de cartera"""
    return FileResponse(os.path.join(static_dir, "portfolio_holdings.html"))


@app.get("/html")
async def get_html_index():
    """Sirve la página HTML principal"""
    return FileResponse(os.path.join(static_dir, "index.html"))


# ---- STARTUP/SHUTDOWN EVENTS --------
def start_periodic_updates():
    """Inicia hilo en segundo plano para actualizaciones periódicas de datos"""

    def update_loop():
        while True:
            try:
                logger.info("Iniciando actualización periódica de datos")
                log_memory_usage("Inicio actualización periódica")

                symbols = ["BAP", "BRK-B", "ILF"]
                for symbol in symbols:
                    try:
                        logger.info(f"Actualizando datos para {symbol}")
                        load_data_from_postgres(symbol, force_reload=True)
                        time.sleep(2)  # Pequeño retraso entre símbolos
                    except Exception as e:
                        logger.error(f"Error al actualizar {symbol}: {str(e)}")

                logger.info(f"Actualización completada para {len(symbols)} símbolos")
                log_memory_usage("Fin actualización periódica")
                time.sleep(300)  # Esperar 5 minutos entre ciclos completos

            except Exception as e:
                logger.error(f"Error en bucle de actualización: {str(e)}")
                time.sleep(30)  # Esperar antes de reintentar si hay error

    update_thread = threading.Thread(target=update_loop, daemon=True)
    update_thread.start()
    logger.info("Hilo de actualización periódica iniciado")


@app.on_event("startup")
def startup_event():
    """Ejecutar al iniciar la aplicación"""
    setup_logging()
    logger.info("Iniciando aplicación...")
    logger.info(f"Configuración de caché - Máx: {MAX_CACHE_SIZE} items, TTL: {CACHE_ITEM_TTL}s")
    logger.info(f"Configuración de PostgreSQL - Host: {PG_CONFIG['host']}, DB: {PG_CONFIG['database']}")

    try:
        # Verificar conexión a la base de datos
        conn = get_db_connection()
        conn.close()
        logger.info("Conexión a PostgreSQL verificada correctamente")
    except Exception as e:
        logger.error(f"Error al verificar conexión a PostgreSQL: {str(e)}")

    start_periodic_updates()
    logger.info("API inicializada con actualizaciones periódicas")
    log_memory_usage("Aplicación inicializada")


@app.on_event("shutdown")
def shutdown_event():
    """Ejecutar al detener la aplicación"""
    logger.info("Deteniendo aplicación...")
    log_memory_usage("Shutdown")
    with cache_lock:
        logger.info(f"Limpiando caché ({len(dataframes_cache)} items)")
        dataframes_cache.clear()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=2,
        timeout_keep_alive=65,
        log_level="info"
    )