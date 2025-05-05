import psycopg2
import pandas as pd
from tabulate import tabulate  # pip install tabulate
import sys

# Configuración de la conexión a PostgreSQL
DB_CONFIG = {
    "host": "98.85.189.191",
    "port": 5432,
    "database": "bvl_monitor",
    "user": "bvl_user",
    "password": "179fae82"
}


def connect_to_db():
    """Conectar a la base de datos"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("Conexión establecida con PostgreSQL")
        return conn
    except Exception as e:
        print(f"Error al conectar a PostgreSQL: {str(e)}")
        sys.exit(1)


def get_table_info():
    """Obtener información sobre la tabla stock_data"""
    conn = connect_to_db()
    cursor = conn.cursor()

    print("\n=== INFORMACIÓN DE LA TABLA ===")

    # Verificar existencia de la tabla
    cursor.execute("SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = 'stock_data')")
    if not cursor.fetchone()[0]:
        print("¡Error! La tabla stock_data no existe.")
        conn.close()
        return

    # Obtener columnas
    cursor.execute("""
                   SELECT column_name, data_type, is_nullable
                   FROM information_schema.columns
                   WHERE table_name = 'stock_data'
                   ORDER BY ordinal_position
                   """)
    columns = cursor.fetchall()

    print("\nEstructura de la tabla stock_data:")
    print(tabulate(columns, headers=["Columna", "Tipo de Dato", "Nullable"]))

    # Contar registros por símbolo
    cursor.execute("""
                   SELECT symbol, COUNT(*) as registros
                   FROM stock_data
                   GROUP BY symbol
                   ORDER BY registros DESC
                   """)
    counts = cursor.fetchall()

    print("\nTotal de registros por símbolo:")
    print(tabulate(counts, headers=["Símbolo", "Registros"]))

    conn.close()


def view_latest_data():
    """Ver los datos más recientes por cada símbolo"""
    conn = connect_to_db()

    print("\n=== ÚLTIMOS REGISTROS POR SÍMBOLO ===")

    # Obtener símbolos disponibles
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT symbol FROM stock_data")
    symbols = [row[0] for row in cursor.fetchall()]

    for symbol in symbols:
        # Consulta para obtener el registro más reciente para cada símbolo
        query = """
                SELECT symbol, timestamp, current_price, previous_close, open, day_low, day_high, volume, dividend_yield, market_cap, trailing_pe
                FROM stock_data
                WHERE symbol = %s
                ORDER BY timestamp DESC
                    LIMIT 1 \
                """

        # Usar pandas para una mejor visualización
        df = pd.read_sql_query(query, conn, params=(symbol,))

        print(f"\nÚltimo registro para {symbol} ({df['timestamp'].iloc[0]}):")
        # Transponemos el DataFrame para una mejor visualización
        print(df.drop('timestamp', axis=1).T)

    conn.close()


def check_data_consistency():
    """Verificar la consistencia de los datos"""
    conn = connect_to_db()
    cursor = conn.cursor()

    print("\n=== VERIFICACIÓN DE CONSISTENCIA ===")

    # Verificar valores nulos en campos importantes
    cursor.execute("""
                   SELECT symbol,
                          COUNT(*)                                               as total_registros,
                          SUM(CASE WHEN current_price IS NULL THEN 1 ELSE 0 END) as null_price,
                          SUM(CASE WHEN volume IS NULL THEN 1 ELSE 0 END)        as null_volume,
                          SUM(CASE WHEN day_low IS NULL THEN 1 ELSE 0 END)       as null_day_low,
                          SUM(CASE WHEN day_high IS NULL THEN 1 ELSE 0 END)      as null_day_high
                   FROM stock_data
                   GROUP BY symbol
                   """)
    null_check = cursor.fetchall()

    print("\nVerificación de valores nulos:")
    print(tabulate(null_check,
                   headers=["Símbolo", "Total Registros", "Precios Nulos",
                            "Volumen Nulo", "Low Nulo", "High Nulo"]))

    # Verificar rango de fechas
    cursor.execute("""
                   SELECT symbol,
                          MIN(timestamp) as primera_fecha,
                          MAX(timestamp) as ultima_fecha,
                          COUNT(*)       as total_registros
                   FROM stock_data
                   GROUP BY symbol
                   ORDER BY symbol
                   """)
    date_range = cursor.fetchall()

    print("\nRango de fechas para cada símbolo:")
    print(tabulate(date_range,
                   headers=["Símbolo", "Primera Fecha", "Última Fecha", "Total Registros"]))

    conn.close()


if __name__ == "__main__":
    print("Verificando datos almacenados en PostgreSQL...")
    get_table_info()
    view_latest_data()
    check_data_consistency()
    print("\nVerificación completada.")