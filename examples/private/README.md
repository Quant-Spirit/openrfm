# Your own database

`make_shop.py` writes a simulated shop database in the layout `openrfm predict --spec` reads:

```
DATA/shop/db/{customers,products,orders}.parquet      # Parquet metadata: pkey_col, fkey_col_to_pkey_table, time_col
DATA/shop/tasks/customer-churn/{train,val,test}.parquet
DATA/shop/tasks/customer-spend/{train,val,test}.parquet
```

Each customer orders at a rate that drifts over time; `customer-churn` asks whether a customer places no order in the
next 30 days and `customer-spend` how much it spends. `customer-churn.yaml` and `customer-spend.yaml` declare the tasks.

```bash
python examples/private/make_shop.py --out DATA
openrfm predict --spec examples/private/customer-churn.yaml --recipe r1 --data DATA
openrfm predict --spec examples/private/customer-spend.yaml --recipe r1 --data DATA
```

`predict_sql.py` does the same through the Python API: it loads the tables into a DuckDB file with their key
constraints, reads it back with `openrfm.Database.from_sql`, defines churn with `openrfm.Task.from_sql` and predicts
with `openrfm.predict`. `--url` points it at another database holding the same tables (PostgreSQL, MySQL, SQLite).

```bash
python examples/private/predict_sql.py --data DATA
```
