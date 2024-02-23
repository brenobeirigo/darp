from dataclasses import dataclass


@dataclass
class TimeWindow:
    earliest: int
    latest: int

    def __str__(self) -> str:
        return f"({self.earliest:<4}, {self.latest:<4})"
