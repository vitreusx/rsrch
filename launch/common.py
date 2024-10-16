import io

from ruamel.yaml import YAML, SafeRepresenter


def represent_str(self, s):
    if "\n" in s:
        return self.represent_scalar("tag:yaml.org,2002:str", s, style="|")
    else:
        return self.represent_scalar("tag:yaml.org,2002:str", s)


yaml = YAML(typ="safe", pure=True)
yaml.representer.add_representer(str, represent_str)


class OneLineYaml:
    yaml = YAML(typ="safe", pure=True)
    yaml.default_flow_style = True
    yaml.width = int(2**10)
    yaml.representer.add_representer(str, represent_str)

    @classmethod
    def dumps(cls, data):
        stream = io.StringIO()
        cls.yaml.dump(data, stream)
        return stream.getvalue().rstrip()


format_opts = OneLineYaml.dumps


# Subsets of Atari-100k env set which are best predictors of overall performance.

ATARI_100k_1 = ["BattleZone"]
ATARI_100k_3 = ["BattleZone", "DemonAttack", "MsPacman"]
ATARI_100k_5 = ["Assault", "BankHeist", "BattleZone", "DemonAttack", "MsPacman"]

# Subsets of Atari-100k env set which are best predictors of overall performance, and for which the randomness of the final score depending on the seed is low enough

A100k_STABLE_1 = ["Assault"]
A100k_STABLE_3 = ["Assault", "BattleZone", "CrazyClimber"]
A100k_STABLE_5 = ["Assault", "BankHeist", "BattleZone", "CrazyClimber", "MsPacman"]

# Manually selected subset of games, for which the performance curve is monotonic enough
A100k_MONO = ["Pong", "CrazyClimber", "Assault", "Asterix"]
