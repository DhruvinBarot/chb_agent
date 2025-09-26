from collections import defaultdict, deque

class ShortTermMemory:
    def __init__(self, window_size: int = 5):
        self.store = defaultdict(lambda: deque(maxlen=window_size))

    def add(self, thread_id: str, role: str, message: str):
        self.store[thread_id].append({"role": role, "message": message})

    def get(self, thread_id: str):
        return list(self.store[thread_id])

