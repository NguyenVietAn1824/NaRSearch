from __future__ import annotations
from abc import ABC
from abc import abstractmethod
from typing import Any


class BaseService(ABC):
    @abstractmethod
    def process(self, input: Any) -> Any:
        raise NotImplementedError()

class AsyncBaseService(ABC):
    @abstractmethod
    async def aprocess(self, input: Any) -> Any:
        raise NotImplementedError()