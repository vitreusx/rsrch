from . import api


class NullBoard(api.Board):
    def _pass(self, *args, **kwargs):
        ...

    add_image = add_samples = add_image = add_video = _pass
