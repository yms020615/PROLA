TASK = dict(
    name="precipitation",
    surface=("2t", "10u", "10v", "msl", "tp"),
    upper=("t", "u", "v", "r", "z"),
    levels=(50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000),
    input_lag=12,
    leads=(12, 24, 36),
)
