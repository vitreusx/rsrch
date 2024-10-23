import json
import pickle
import sqlite3

from .base import *


class Sqlite(StepMixin, Board):
    def __init__(self, path: str | Path, scalars: bool = True):
        super().__init__()
        self.scalars = scalars
        self.con = sqlite3.connect(path)
        self.cur = self.con.cursor()
        if self.scalars:
            self.cur.execute("create table scalars(tag, value, step)")
        self.cur.execute("create table dicts(tag, value, step)")

    def add_scalar(self, tag: str, value: Number, *, step: Step = None):
        if self.scalars:
            step = self._get_step(step)
            self.cur.execute(
                "insert into scalars values(:tag, :value, :step)",
                dict(tag=tag, value=float(value), step=step),
            )
            self.con.commit()

    def add_dict(self, tag: str, value: dict, *, step: Step = None):
        step = self._get_step(step)
        self.cur.execute(
            "insert into dicts values(:tag, :value, :step)",
            dict(tag=tag, value=pickle.dumps(value), step=step),
        )
