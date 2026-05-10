```text
itoformer_full.zip
(updated Jan 2026, then again early May 2026) <--- This will probably be my last update. Feel free to reach out via email or LinkedIn. 
├── baselines
│   ├── arima.py
│   ├── garch.py
│   ├── har_rv.py
│   └── kalman.py
├── encoders
│   ├── calendar.py
│   ├── encoders.py
│   ├── extra_encoders.py
│   ├── graph.py
│   ├── ito_vol.py
│   ├── regime.py
│   ├── stat_factor.py
│   └── surface_curve.py
├── losses
│   ├── ito.py
│   ├── martingale.py
│   └── no_arbitrage.py
├── models
│   ├── itoformer.py
│   ├── blocks
│   │   ├── attention.py
│   │   ├── feedforward.py
│   │   ├── normalization.py
│   │   ├── state_token.py
│   │   └── utils.py
│   └── heads
│       ├── forecast.py
│       ├── martingale.py
│       ├── options_node.py
│       ├── options_svi.py
│       └── term_structure.py
├── training
│   ├── collate.py
│   ├── datamodules_options.py
│   ├── datamodules_rates.py
│   ├── datasets.py
│   ├── preprocessing.py
│   └── tokenization.py
└── utils
    ├── config.py
    ├── csvlog.py
    └── validate_config.py
```
