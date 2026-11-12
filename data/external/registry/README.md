# Committed registry tables

Small tables the pipeline reads that no generator in this repository writes.
Each is committed, so a clone carries it and the large trees beside it under
`data/external/` are the only thing a replicator has to supply.

## `twn_stock_info.parquet`, `twn_delisting.parquet`

FinMind's `TaiwanStockInfo` and `TaiwanStockDelisting` responses, as pulled.
`TaiwanStockInfo` is the live listing registry the Taiwanese universe is drawn
from; `TaiwanStockDelisting` records which names left a board and when, which
is what `tw_data_loader` measures the panel's survivorship exposure against.
Both are one response each rather than a per-stock tree, so they are committed
here instead of being symlinked.
