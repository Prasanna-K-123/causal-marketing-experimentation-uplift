# Licence scope, source rights and attribution

This project's original software and original documentation are licensed under
the [MIT licence](LICENSE), Copyright (c) 2026 Prasanna K. This includes original
Python source, tests, notebooks, workflows and dependency specification files.

The original-code grant does not relicense third-party libraries, model weights,
source reports, datasets, benchmark records or quoted source material. Existing
per-file copyright notices, upstream licences and source-provider conditions
continue to apply to that material. In a mixed file, the grant covers only the
project's original contribution. Public accessibility is not a substitute for
the applicable upstream terms, and no source-provider endorsement is implied.

The source-specific boundaries below remain separate from the MIT code licence.

## Hillstrom randomized email benchmark

The experiment is attributed to **Kevin Hillstrom / MineThatData**, [original challenge](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html). The original download URL and the separately hosted fallback are declared in [the loader](src/causal_experimentation_uplift.py); the actual successful source and payload/parsed-data hashes are recorded in [run metadata](results/run_metadata.json) and [the registered study receipt](reference/refit_uncertainty/source_receipt.json).

The fallback is a transport mirror, not proof of a redistribution licence or author endorsement. No explicit general redistribution licence for the original customer-level benchmark was verified in this review. The repository publishes research outputs and replay metadata; running the full study retrieves the benchmark separately. Review the source provider's terms before distributing the dataset or customer-level derived records. The registered fixed-split and refit results remain research evidence, not a new experiment or a client engagement.
