
from typing import AsyncIterator


class AbstractDataLoader:

    async def load(self) -> AsyncIterator:
        pass