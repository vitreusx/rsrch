import io

from ruamel.yaml import YAML

ATARI_100k_3 = [
    "BattleZone",
    "DemonAttack",
    "MsPacman",
]
"""A subset of 3 games out of Atari-100k, which are most representative of the scores on the full dataset."""

ATARI_100k_5 = [
    "Assault",
    "BankHeist",
    "BattleZone",
    "DemonAttack",
    "MsPacman",
]
"""A subset of 5 games out of Atari-100k, which are most representative of the scores on the full dataset."""

yaml = YAML(typ="safe", pure=True)
yaml.default_flow_style = True
yaml.width = int(2**10)


def dumps(data):
    stream = io.StringIO()
    yaml.dump(data, stream)
    return stream.getvalue().rstrip()
