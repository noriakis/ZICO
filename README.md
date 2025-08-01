# ZICO

The repository aims to learns directed acyclic graphs (DAGs) from possibly zero‑inflated count observations (e.g., single‑cell RNA‑seq). It uses the loss of Zero‑Inflated Negative Binomial (ZINB), ZI Poisson, or NB node‑wise likelihood and enforces acyclicity via a log‑det barrier or the standard matrix‑exponential constraint.

## Related works

- ZiGDAG (https://github.com/junsoukchoi/ZiGDAG)
- ZiDAG (https://github.com/sqyu/ZiDAG)

## Test run


```bash
python run.py
```

This will produce the PNG file displaying heatmaps of three adjacency matrices: reference, learned W0 and W1.
