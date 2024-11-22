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


ATARI_100k = [
    "Alien",
    "Amidar",
    "Assault",
    "Asterix",
    "BankHeist",
    "BattleZone",
    "Boxing",
    "Breakout",
    "ChopperCommand",
    "CrazyClimber",
    "DemonAttack",
    "Freeway",
    "Frostbite",
    "Gopher",
    "Hero",
    "Jamesbond",
    "Kangaroo",
    "Krull",
    "KungFuMaster",
    "MsPacman",
    "Pong",
    "PrivateEye",
    "Qbert",
    "RoadRunner",
    "Seaquest",
    "UpNDown",
]


# A subset of 3 games, which are (1) sufficiently monotonic to properly evaluate the speedup to be achieved, (2) are most predictive of the performance on Atari-100k subset of games.
A100k_MONO = ["Assault", "CrazyClimber", "MsPacman"]


PRESET_PATH = "thesis/exp/presets.yml"
