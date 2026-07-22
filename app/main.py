"""
FastAPI app entrypoint.

This file should stay thin: app setup, the HTTP endpoint, and
wiring the background worker to app startup. Everything else
(severity resolution, routing decisions, dispatch side effects,
queueing, latency metrics) lives in its own module — see the
imports below for exactly what each one owns.
"""

import time
import asyncio

import uvicorn
from fastapi import FastAPI, Request

from app.severity import resolve_severity
from app.routing import route_event
from app.dispatch import dispatch_alert_action, generate_event_id
from app.dispatch_queue import enqueue_dispatch, dispatch_worker
from app.metrics import EVENT_RECEIPT_TIMES, record_dispatch_completion


app = FastAPI()


@app.on_event("startup")
async def start_background_workers():
    asyncio.create_task(dispatch_worker())


@app.post("/api/v1/event")
async def receive_event(request: Request):
    # Captured before body parsing so latency reflects true
    # end-to-end handling time, not just processing after
    # the JSON has already been read off the wire.
    receipt_time = time.monotonic()

    data = await request.json()

    event_id = generate_event_id()
    EVENT_RECEIPT_TIMES[event_id] = receipt_time
    print(f"[RECEIVE] Event={event_id} | Received at t={receipt_time:.6f}")

    alerts = data.get("alerts", [])
    camera_id = data.get("camera_id", "pc_cam")

    if not alerts:
        EVENT_RECEIPT_TIMES.pop(event_id, None)
        return {"status": "failed", "message": "No alerts provided"}

    # Each alert can be either a plain string ("loitering") or an
    # object with an optional client-supplied severity override
    # ({"type": "loitering", "severity": "HIGH"}). Normalize both
    # into (alert_type, resolved_severity) pairs up front so the
    # rest of the pipeline only ever deals with one shape.
    resolved_alerts = []
    decisions = []
    for alert in alerts:
        if isinstance(alert, dict):
            alert_type = alert.get("type")
            severity_override = alert.get("severity")
        else:
            alert_type = alert
            severity_override = None

        if not alert_type:
            print(f"[WARN] Skipping malformed alert entry: {alert}")
            continue

        severity = resolve_severity(alert_type, severity_override)
        decision = route_event(alert_type, severity)
        print(f"[ROUTE] Event={event_id} | Type={alert_type} | "
              f"Severity={decision.severity.name} | Action={decision.action} | "
              f"Immediate={decision.immediate}")

        resolved_alerts.append({"type": alert_type, "severity": severity.name})
        decisions.append(decision)

    for decision in decisions:
        if decision.immediate:
            # HIGH/CRITICAL severity: skip the queue entirely and
            # dispatch inline on this request's own coroutine, so
            # urgent alerts (e.g. "stealing") aren't stuck waiting
            # behind whatever the single background worker happens
            # to be processing at that moment.
            print(f"[BYPASS] Event={event_id} | Type={decision.alert_type} | "
                  f"Severity={decision.severity.name} — immediate dispatch, skipping queue")
            await dispatch_alert_action(event_id, decision.alert_type, camera_id)
            record_dispatch_completion(event_id, decision.alert_type, decision.severity, camera_id)
        else:
            # LOW/MEDIUM severity: don't dispatch inline. Hand off
            # to DISPATCH_QUEUE and let the background worker drain
            # it in priority order. Routine alerts no longer compete
            # with urgent ones for this request's own execution
            # time, and multiple routine alerts arriving in a burst
            # get smoothed through one consumer instead of all
            # firing dispatch side effects (beeps, camera processes)
            # simultaneously.
            print(f"[QUEUED] Event={event_id} | Type={decision.alert_type} | "
                  f"Severity={decision.severity.name} — enqueued for background dispatch")
            await enqueue_dispatch(event_id, decision, camera_id)

    return {
        "status": "success",
        "event_id": event_id,
        "alerts_received": resolved_alerts
    }


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)