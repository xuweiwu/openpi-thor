from __future__ import annotations

import logging
from pathlib import Path

from openpi.serving import websocket_policy_server
from openpi.training import config as _config

from openpi_thor.runtime import load_tensorrt_policy

logger = logging.getLogger(__name__)


def serve(
    config: str | _config.TrainConfig,
    bundle_dir: str | Path,
    *,
    engine_path: str | Path | None = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    require_validated: bool = True,
    default_prompt: str | None = None,
    pytorch_device: str | None = None,
) -> None:
    """Start the websocket policy server backed by a TensorRT engine."""

    logger.info("Loading TensorRT policy for bundle %s", Path(bundle_dir).expanduser().resolve())
    policy = load_tensorrt_policy(
        config,
        bundle_dir,
        engine_path=engine_path,
        require_validated=require_validated,
        default_prompt=default_prompt,
        pytorch_device=pytorch_device,
    )
    logger.info("Starting websocket policy server on %s:%s", host, port)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=host,
        port=port,
        metadata=policy.metadata,
    )
    server.serve_forever()
