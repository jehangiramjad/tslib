import abc


class Repository(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_time_series(self, name: str, start: int, end: int) -> int:
        pass
