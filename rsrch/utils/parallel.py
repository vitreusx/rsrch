import multiprocessing as mp
from multiprocessing.connection import Connection
from multiprocessing.context import BaseContext
from threading import Thread
from typing import Any, Type, TypeVar


class _Future:
    def __init__(self, conn, args, kwargs):
        self.conn = conn
        self._args, self._kwargs = args, kwargs

    def result(self):
        return self.conn.recv()


class _FuncProxy:
    def __init__(self, conn, name: str):
        self.conn, self.name = conn, name

    def __call__(self, *args, **kwargs):
        self.conn.send((self.name, args, kwargs))
        return self.conn.recv()

    def future(self, *args, **kwargs):
        self.conn.send((self.name, args, kwargs))
        return _Future(self.conn, args, kwargs)


class _Proxy:
    def __init__(self, ctx: BaseContext, local: bool, *args, **kwargs):
        self._conn, proc_end = ctx.Pipe(duplex=True)
        self._args, self._kwargs = args, kwargs
        if local:
            self._proc = Thread(
                target=_Proxy._thr_target,
                args=(proc_end, *args),
                kwargs=kwargs,
            )
        else:
            self._proc = ctx.Process(
                target=_Proxy._proc_target,
                args=(proc_end, *args),
                kwargs=kwargs,
            )
        self._proc.start()
        self._callable = {}
        self._typeof = self._command("_typeof")

    def __getstate__(self):
        return {"_conn": self._conn, "_typeof": self._typeof}

    def __setstate__(self, state):
        super().__setattr__("_callable", {})
        self._conn = state["_conn"]
        self._typeof = state["_typeof"]

    @staticmethod
    def _proc_target(proc_end, ctor, *args, **kwargs):
        return _Proxy._target(proc_end, ctor(*args, **kwargs))

    @staticmethod
    def _thr_target(proc_end, inst):
        return _Proxy._target(proc_end, inst)

    @staticmethod
    def _target(proc_end, inst):
        while True:
            try:
                cmd, args, kwargs = proc_end.recv()
            except EOFError:
                break

            if cmd == "_kill":
                break
            elif cmd == "_typeof":
                ret = type(inst)
            elif cmd == "_callable":
                name = args
                ret = callable(getattr(inst, name, None))
            else:
                var = getattr(inst, cmd)
                if callable(var):
                    # ret = var(
                    #     *(_fix(arg) for arg in args),
                    #     **{k: _fix(v) for k, v in kwargs},
                    # )
                    ret = var(*args, **kwargs)
                else:
                    # ret = _fix(var)
                    ret = var
            proc_end.send(ret)

    def __getattr__(self, __name: str):
        if __name.startswith("_"):
            return super().__getattribute__(__name)
        else:
            return self._command(__name)

    def _command(self, __name: str):
        if __name not in self._callable:
            self._conn.send(("_callable", __name, None))
            self._callable[__name] = self._conn.recv()

        if self._callable[__name]:
            return _FuncProxy(self._conn, __name)
        else:
            self._conn.send((__name, None, None))
            return self._conn.recv()

    def __setattr__(self, __name, __value) -> None:
        if __name.startswith("_"):
            super().__setattr__(__name, __value)
        else:
            return self._command("__setattr__")(__name, __value)

    def __getitem__(self, *args, **kwargs):
        return self._command("__getitem__")(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self._command("__call__")(*args, **kwargs)

    def __del__(self):
        if hasattr(self, "_proc"):
            self._conn.send(("_kill", None, None))
            self._proc.join()


T = TypeVar("T")


class Manager:
    def __init__(self, ctx: BaseContext | None = None):
        self.ctx = ctx

    def remote(self, cls: Type[T]):
        def _ctor(*args, **kwargs) -> T:
            return _Proxy(self.ctx, False, cls, *args, **kwargs)

        return _ctor

    def local_ref(self, x: T) -> T:
        return _Proxy(self.ctx, True, x)
