"""A simulated shop database in RelBench layout, as an example of one's own data for `openrfm predict --spec`.

Tables: customers, products and orders. Each customer orders at a rate that decays (or grows) over time, so its order
history predicts the next 30 days. Tasks, one row per (month start, customer with an earlier order):
`customer-churn` (no order in the next 30 days) and `customer-spend` (amount spent in the next 30 days). TRAIN months
run to 2023-06, VAL to 2023-09 and TEST to 2023-12.

    python examples/private/make_shop.py --out DATA
    openrfm predict --spec examples/private/customer-churn.yaml --data DATA --recipe r1
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

START, END = pd.Timestamp('2022-01-01'), pd.Timestamp('2024-01-31')
SPLITS = {'train': ('2022-03-01', '2023-06-01'), 'val': ('2023-07-01', '2023-09-01'), 'test': ('2023-10-01', '2023-12-01')}
WINDOW = pd.Timedelta(days=30)


def write(frame, path, pkey, fkeys, clock):
    """A Parquet file with RelBench's key-value metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    meta = {k.encode(): json.dumps(v).encode() for k, v in
            dict(pkey_col=pkey, fkey_col_to_pkey_table=fkeys, time_col=clock).items()}
    pq.write_table(table.replace_schema_metadata(meta), path)


def database(customers, products, rng):
    """(customers, products, orders) frames."""
    days = (END-START).days
    signup = START+pd.to_timedelta(rng.integers(0, days//2, customers), unit='D')
    rate = rng.gamma(2.0, 0.08, customers)                       # orders per day at signup
    trend = rng.normal(-0.002, 0.003, customers)                 # log-rate change per day
    region = rng.choice(['north', 'south', 'east', 'west'], customers)
    cust = pd.DataFrame(dict(customer_id=np.arange(customers), signup_time=signup, region=region,
                             age=rng.integers(18, 80, customers)))
    prod = pd.DataFrame(dict(product_id=np.arange(products),
                             category=rng.choice(['books', 'food', 'garden', 'tech', 'toys'], products),
                             price=np.round(rng.lognormal(3.0, 0.7, products), 2)))
    rows = []
    for c in range(customers):
        first = (signup[c]-START).days
        t = np.arange(first, days)
        n = rng.poisson(rate[c]*np.exp(trend[c]*(t-first)))
        when = np.repeat(t, n)
        rows.append(pd.DataFrame(dict(customer_id=c, day=when, product_id=rng.integers(0, products, len(when)),
                                      quantity=rng.integers(1, 4, len(when)))))
    orders = pd.concat(rows, ignore_index=True)
    orders['order_time'] = START+pd.to_timedelta(orders.pop('day'), unit='D')+pd.to_timedelta(
        rng.integers(0, 86400, len(orders)), unit='s')
    orders = orders.sort_values('order_time', ignore_index=True)
    orders.insert(0, 'order_id', np.arange(len(orders)))
    return cust, prod, orders


def task_rows(orders, prices, first, last):
    """(timestamp, customer_id, churn, spend) of each customer with an order before each month start."""
    spend = orders.quantity*prices[orders.product_id].to_numpy()
    rows = []
    for t in pd.date_range(first, last, freq='MS'):
        seen = orders.customer_id[orders.order_time < t].unique()
        window = (orders.order_time >= t) & (orders.order_time < t+WINDOW)
        amount = spend[window].groupby(orders.customer_id[window]).sum().reindex(seen, fill_value=0.0)
        rows.append(pd.DataFrame(dict(timestamp=t, customer_id=seen, churn=(amount.to_numpy() == 0).astype(np.int64),
                                      spend=amount.to_numpy())))
    return pd.concat(rows, ignore_index=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--out', type=Path, required=True, help='the database is written to OUT/shop/')
    p.add_argument('--customers', type=int, default=3000)
    p.add_argument('--products', type=int, default=200)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    cust, prod, orders = database(a.customers, a.products, np.random.default_rng(a.seed))
    root = a.out/'shop'
    write(cust, root/'db/customers.parquet', 'customer_id', {}, 'signup_time')
    write(prod, root/'db/products.parquet', 'product_id', {}, None)
    write(orders, root/'db/orders.parquet', 'order_id', {'customer_id': 'customers', 'product_id': 'products'},
          'order_time')
    for split, (first, last) in SPLITS.items():
        rows = task_rows(orders, prod.price, first, last)
        for task, target in (('customer-churn', 'churn'), ('customer-spend', 'spend')):
            write(rows[['timestamp', 'customer_id', target]], root/'tasks'/task/f'{split}.parquet', None,
                  {'customer_id': 'customers'}, 'timestamp')
    print(root, len(orders), 'orders')


if __name__ == '__main__':
    main()
