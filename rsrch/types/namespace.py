class Namespace(dict):
    """A dict also accessible via attr access."""

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __getattr__(self, key):
        value = self.__getitem__(key)
        if isinstance(value, dict):
            value = Namespace(**value)
        return value
