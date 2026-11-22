# RelBench and 4DBInfer tasks

`predict.py` runs a released recipe on benchmark tasks. Each database is downloaded into `DATA/<db>/` on first use and
checked by sha256; the TEST metric (AUROC for classification, MAE for regression) is printed with its standard error.

```bash
python examples/relbench/predict.py --data DATA                      # rel-f1 driver-dnf, driver-top3, driver-position
python examples/relbench/predict.py --data DATA --recipe r1 \
    --tasks rel-event/user-repeat rel-trial/study-outcome rel-amazon/item-ltv
python examples/relbench/predict.py --data DATA --tasks all          # all 32 tasks
```

The same from the command line, one task at a time:

```bash
openrfm predict --task rel-f1/driver-dnf --recipe r2 --data DATA
openrfm predict --task stackexchange-4db/user-churn --recipe r1 --data DATA
```

The task names are the keys of [`openrfm/configs/infer/tasks.yaml`](../../openrfm/configs/infer/tasks.yaml). The rel-f1
tasks take a few minutes each on one GPU; the large databases (rel-amazon, rel-hm, rel-stack, rel-avito) take longer to
download and convert on first use.
