from collections import defaultdict
import time

WINDOW_SEC = 5
MAX_REQS = 5

_bucket = defaultdict(list)

def allow_request(key: str) -> bool:
    now = time.time()
    window = _bucket[key]
    # prune old timestamps
    while window and now - window[0] > WINDOW_SEC:
        window.pop(0)
    if len(window) >= MAX_REQS:
        return False
    window.append(now)
    return True
