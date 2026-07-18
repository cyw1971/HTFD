"""Pretty-print argparse namespace."""

def print_args(args) -> None:
    print("=" * 60)
    for k, v in sorted(vars(args).items()):
        print(f"{k:>28}: {v}")
    print("=" * 60)
