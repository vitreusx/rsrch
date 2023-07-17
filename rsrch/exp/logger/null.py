from . import api


class NullLogger(api.Logger):
    def log(self, level, msg, *args):
        ...
