class Namespace(dict):
    """A dict also accessible via attr access."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k in tuple(self.keys()):
            if isinstance(self[k], dict):
                self[k] = Namespace(self[k].items())

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __getattr__(self, key):
        if key in self:
            return self.__getitem__(key)
        else:
            raise AttributeError()
