import threading
from contextlib import contextmanager


class ThreadRWLock:
    def __init__(self):
        self.readers, self.waiting = 0, 0
        self.writing = False
        self.cond = threading.Condition()

    @contextmanager
    def read(self):
        with self.cond:
            while self.waiting > 0 or self.writing:
                self.cond.wait()
            self.readers += 1

        yield

        with self.cond:
            self.readers -= 1
            if self.readers == 0:
                self.cond.notify()

    @contextmanager
    def write(self):
        with self.cond:
            self.waiting += 1
            while self.readers > 0 or self.writing:
                self.cond.wait()
            self.waiting -= 1
            self.writing = True

        yield

        with self.cond:
            self.writing = False
            self.cond.notify()
