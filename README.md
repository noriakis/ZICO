# ZICO

The repository aims to learns directed acyclic graphs (DAGs) from possibly zero‑inflated count observations (e.g., single‑cell RNA‑seq). It uses the loss of Zero‑Inflated Negative Binomial (ZINB), ZI Poisson, NB, or Poisson node‑wise likelihood and enforces acyclicity via a log‑det barrier.

## Related works

- ZiGDAG (https://github.com/junsoukchoi/ZiGDAG)
- ZiDAG (https://github.com/sqyu/ZiDAG)

## Test run


```bash
python run.py --d 10
```

This will generate simulated data with 10 variables, performs training, and produce the PNG file displaying heatmaps of three adjacency matrices: reference, learned W0 and W1. if `--save_graph` is specified, the networks at the specified threshold will be shown.
