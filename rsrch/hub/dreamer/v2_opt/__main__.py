import argparse

from .impl import Dreamer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--complexity-check", action="store_true")
    parser.add_argument("--per-layer-stats", action="store_true")
    args = parser.parse_args()

    dreamer = Dreamer()
    if args.complexity_check:
        dreamer.complexity_check(per_layer=args.per_layer_stats)
    else:
        dreamer.train()


if __name__ == "__main__":
    main()
