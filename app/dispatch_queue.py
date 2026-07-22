"""
Priority queue for non-immediate (LOW/MEDIUM severity) dispatch,
and the background worker that drains it.

Kept separate from dispatch.py on purpose: this file is generic
queueing/concurrency infrastructure (the queue, the ordering, the
worker loop) — it doesn't know or care what "dispatching an
alert" actually involves. dispatch.py is domain logic. Different
reasons to change, different files.
"""

import asyncio
import itertools

from app.routing import RoutingDecision
from app.dispatch import dispatch_alert_action
from app.metrics import record_dispatch_completion


# =========================================================
# Dispatch Priority Queue
# =========================================================
# asyncio.PriorityQueue is a min-heap: the LOWEST value comes
# out first. Severity is stored as a NEGATIVE int so CRITICAL
# (4) becomes -4 and is dequeued before LOW (1) → -1.
#
# Queue items are plain tuples: (priority, seq, event_id,
# decision, camera_id). `seq` (a monotonically increasing
# counter) exists purely as a tiebreaker — if two items have
# equal priority, PriorityQueue compares the NEXT tuple element
# to order them, and RoutingDecision isn't orderable. Without
# `seq`, two same-priority items would raise a TypeError trying
# to compare dataclasses. `seq` also happens to preserve FIFO
# order among equal-priority items, which is a nice side effect.

DISPATCH_QUEUE: asyncio.PriorityQueue = asyncio.PriorityQueue()
_dispatch_seq = itertools.count()


async def enqueue_dispatch(event_id: str, decision: RoutingDecision, camera_id: str):
    priority = -int(decision.severity)
    seq = next(_dispatch_seq)
    await DISPATCH_QUEUE.put((priority, seq, event_id, decision, camera_id))
    print(f"[QUEUE] Event={event_id} | Type={decision.alert_type} | "
          f"Severity={decision.severity.name} | priority={priority} | "
          f"qsize={DISPATCH_QUEUE.qsize()}")


# =========================================================
# Background Dispatch Worker
# =========================================================
# Consumes DISPATCH_QUEUE and calls dispatch_alert_action for
# each item, highest priority first. Runs as a single
# long-lived asyncio task started at app startup (see main.py's
# startup hook), not per-request — one consumer draining a
# shared queue.

async def dispatch_worker():
    print("[WORKER] Dispatch worker started")
    while True:
        priority, seq, event_id, decision, camera_id = await DISPATCH_QUEUE.get()
        try:
            print(f"[WORKER] Dequeued Event={event_id} | Type={decision.alert_type} | "
                  f"Severity={decision.severity.name} | qsize={DISPATCH_QUEUE.qsize()}")
            await dispatch_alert_action(event_id, decision.alert_type, camera_id)
            record_dispatch_completion(event_id, decision.alert_type, decision.severity, camera_id)
        except Exception as e:
            # A single bad dispatch must not kill the worker loop —
            # every subsequent queued event would silently stop
            # being processed if this exception propagated.
            print(f"[WORKER][ERROR] Event={event_id} failed: {e}")
        finally:
            DISPATCH_QUEUE.task_done()