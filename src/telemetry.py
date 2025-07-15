import json
import logging
import time
from typing import Any, Optional

import requests


class Telemetry:
    """Simple telemetry helper for logging and optional HTTP reporting."""

    def __init__(self, server_url: Optional[str] = None, log_file: str = "telemetry.log"):
        self.server_url = server_url
        self.logger = logging.getLogger("Telemetry")
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)

    def record(self, event: str, **kwargs: Any) -> None:
        payload = {"event": event, "timestamp": time.time(), **kwargs}
        self.logger.info(json.dumps(payload))
        if self.server_url:
            try:
                requests.post(self.server_url, json=payload, timeout=1)
            except Exception as exc:  # pragma: no cover - best effort
                self.logger.debug("Failed to send telemetry: %s", exc)
