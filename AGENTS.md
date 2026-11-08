# Development

Keep PROLA independent of weather data in `adapters/`. Put backbone integration and correction heads in `models/`, dataset access in `data/`, and task definitions in `tasks/`.

Preserve initial function equivalence, fixed reference projectors, and freely trainable principal factors. Forward scaling and optimizer-update scaling are separate. Rescale optimizer increments without modifying Adam moments. Never mutate adapter parameters inside `forward`.

Keep data windows inside their split. Preserve task-specific metric aggregation: mean per-case RMSE for PSWD/CERRA and pooled RMSE for precipitation. Distributed evaluation must not duplicate samples.

Keep code free of comments and docstrings. Do not add server paths, credentials, datasets, weights, or run outputs. Keep README commands executable.

Run `python examples/minimal_prola.py` and `python -m unittest discover -s tests -v` after changing adapter behavior. Check the relevant synthetic weather command after changing a task or training code.
