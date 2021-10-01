class TimeWindow:
    def __init__(self, earliest, latest):
        self.earliest = earliest
        self.latest = latest
    
    def __str__(self) -> str:
        return f"<{self.earliest},{self.latest}>"