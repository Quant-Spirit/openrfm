"""Predict customer churn on the shop database read from a SQL database, with the task defined in SQL.

    python examples/private/make_shop.py --out DATA
    python examples/private/predict_sql.py --data DATA                         # writes and reads DATA/shop.duckdb
    python examples/private/predict_sql.py --url postgresql://user@host/shop   # any database holding the same tables
"""
import argparse
from pathlib import Path

import duckdb
import pandas as pd
from sklearn.metrics import roc_auc_score

import openrfm

CHURN = '''
SELECT c.id, c.timestamp, COUNT(o.order_id) = 0 AS churn
FROM cases c
LEFT JOIN orders o ON o.customer_id = c.id AND o.order_time >= c.timestamp AND o.order_time < c."end"
WHERE EXISTS (SELECT 1 FROM orders p WHERE p.customer_id = c.id AND p.order_time < c.timestamp)
GROUP BY c.id, c.timestamp'''


def shop_duckdb(data):
    """DATA/shop.duckdb with the shop tables and their PRIMARY KEY and FOREIGN KEY constraints."""
    path = Path(data)/'shop.duckdb'
    path.unlink(missing_ok=True)
    con = duckdb.connect(str(path))
    root = Path(data)/'shop'/'db'
    con.execute("CREATE TABLE customers (customer_id BIGINT PRIMARY KEY, signup_time TIMESTAMP, region VARCHAR, age BIGINT)")
    con.execute("CREATE TABLE products (product_id BIGINT PRIMARY KEY, category VARCHAR, price DOUBLE)")
    con.execute("CREATE TABLE orders (order_id BIGINT PRIMARY KEY, customer_id BIGINT REFERENCES customers(customer_id), "
                "product_id BIGINT REFERENCES products(product_id), quantity BIGINT, order_time TIMESTAMP)")
    for table in ('customers', 'products', 'orders'):
        con.execute(f"INSERT INTO {table} BY NAME SELECT * FROM read_parquet('{root/table}.parquet')")
    con.close()
    return f'duckdb:///{path}'


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--data', type=Path, help='directory holding shop/ from make_shop.py')
    p.add_argument('--url', help='SQL database holding the shop tables (default: a DuckDB file built from --data)')
    p.add_argument('--recipe', choices=('r1', 'r2', 'r3'), default='r1')
    a = p.parse_args()
    db = openrfm.Database.from_sql(a.url or shop_duckdb(a.data))
    task = openrfm.Task.from_sql(db, 'customer-churn', entity='customers', label=CHURN, target='churn',
                                 timestamps=pd.date_range('2022-03-01', '2023-12-01', freq='MS'),
                                 val_start='2023-07-01', test_start='2023-10-01', horizon='30D')
    out = openrfm.predict(db, task, recipe=a.recipe, name='shop_sql')
    print(out.head())
    print('TEST AUROC', roc_auc_score(out['churn'], out['prediction']))


if __name__ == '__main__':
    main()
