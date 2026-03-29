from __future__ import annotations

from pathlib import Path

from openpi.serving import websocket_policy_server
from openpi.training import config as _config

from openpi_thor.runtime import load_tensorrt_policy


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
    policy = load_tensorrt_policy(
        config,
        bundle_dir,
        engine_path=engine_path,
        require_validated=require_validated,
        default_prompt=default_prompt,
        pytorch_device=pytorch_device,
    )
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=host,
        port=port,
        metadata=policy.metadata,
    )
    server.serve_forever()
