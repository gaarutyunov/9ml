"""
Remote QEMU API Server for Plan 9 execution.

Exposes Plan 9 VM execution as an HTTP API for remote reward computation,
enabling GRPO training from environments like Google Colab without local QEMU.

Usage:
    # Start server
    plan9-dataset serve-qemu --token SECRET --port 8080

    # With custom QEMU paths
    plan9-dataset serve-qemu --qemu-dir /path/to/qemu --token SECRET
"""

import asyncio
import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable

from .validate import FATDisk, Plan9VM
from .rewards import (
    compute_style_bonus,
    compute_reasoning_bonus,
    RewardBreakdown,
    ToolResult,
    REWARD_TOOL_SUCCESS,
    PENALTY_TOOL_FAILURE,
    REWARD_OUTPUT_MATCH,
    REWARD_PARTIAL_MATCH,
)
from .tools import parse_tool_calls


@dataclass
class ServerConfig:
    """Server configuration."""
    disk_image: str
    shared_image: str
    token: str
    host: str = "0.0.0.0"
    port: int = 8080
    rate_limit: int = 60  # requests per minute
    timeout: int = 30
    debug: bool = False


@dataclass
class RateLimiter:
    """Simple rate limiter for API protection."""
    requests_per_minute: int = 60
    _requests: dict = field(default_factory=dict)

    def check(self, client_id: str) -> bool:
        """Check if client is within rate limit."""
        now = time.time()
        minute_ago = now - 60

        # Clean old entries
        self._requests = {
            k: v for k, v in self._requests.items()
            if v > minute_ago
        }

        # Count recent requests from this client
        client_key = f"{client_id}"
        client_requests = [
            t for k, t in self._requests.items()
            if k.startswith(client_key) and t > minute_ago
        ]

        if len(client_requests) >= self.requests_per_minute:
            return False

        # Record this request
        self._requests[f"{client_key}:{now}"] = now
        return True


class QEMUServer:
    """HTTP server exposing Plan 9 VM execution."""

    def __init__(self, config: ServerConfig):
        self.config = config
        self.vm: Plan9VM | None = None
        self.disk: FATDisk | None = None
        self.rate_limiter = RateLimiter(config.rate_limit)
        self._lock = None  # Will be set in async context
        self._started = False

    def verify_token(self, provided_token: str) -> bool:
        """Securely verify bearer token."""
        return hmac.compare_digest(
            provided_token.encode(),
            self.config.token.encode()
        )

    def start_vm(self) -> bool:
        """Start the Plan 9 VM."""
        if self._started:
            return True

        # Create fresh shared disk
        self.disk = FATDisk(self.config.shared_image)
        if not self.disk.create(64):
            return False

        # Start VM
        self.vm = Plan9VM(
            self.config.disk_image,
            self.config.shared_image,
            debug=self.config.debug
        )

        if not self.vm.boot():
            return False

        self._started = True
        return True

    def stop_vm(self) -> None:
        """Stop the Plan 9 VM."""
        if self.vm:
            self.vm.shutdown()
            self.vm = None
        self.disk = None
        self._started = False

    def reset_vm(self) -> bool:
        """Reset VM state for new session."""
        if not self._started or not self.vm:
            return self.start_vm()

        # Clear output and clean up temp files
        self.vm.clear_output()
        self.vm.run_command("rm -f /mnt/host/*.c /mnt/host/*.6 /mnt/host/*.rc", timeout=5)

        # Recreate shared disk
        if self.disk:
            self.disk.create(64)

        return True

    def execute_tool(
        self,
        tool: str,
        params: dict[str, Any],
    ) -> ToolResult:
        """Execute a single tool call in the VM."""
        if not self._started or not self.vm or not self.disk:
            return ToolResult(
                name=tool,
                success=False,
                error="VM not started"
            )

        if tool == "write_file":
            path = params.get("path", "")
            content = params.get("content", "")

            if not path or not content:
                return ToolResult(
                    name=tool,
                    success=False,
                    error="Missing path or content parameter"
                )

            success = self.disk.copy_content(content, path)
            return ToolResult(
                name=tool,
                success=success,
                error="" if success else "Failed to write file"
            )

        elif tool == "read_file":
            path = params.get("path", "")

            if not path:
                return ToolResult(
                    name=tool,
                    success=False,
                    error="Missing path parameter"
                )

            if path.startswith("/"):
                # System file - read via VM
                output = self.vm.run_command(f"cat {path}", timeout=self.config.timeout)
                success = bool(output) and "error" not in output.lower()
                return ToolResult(
                    name=tool,
                    success=success,
                    output=output if success else "",
                    error="" if success else f"Failed to read {path}"
                )
            else:
                # FAT disk file
                content = self.disk.read_file(path)
                success = content is not None
                return ToolResult(
                    name=tool,
                    success=success,
                    output=content or "",
                    error="" if success else f"File not found: {path}"
                )

        elif tool == "run_command":
            command = params.get("command", "")

            if not command:
                return ToolResult(
                    name=tool,
                    success=False,
                    error="Missing command parameter"
                )

            output = self.vm.run_command(command, timeout=self.config.timeout)

            # Check for errors
            import re
            error_patterns = [
                r"error:",
                r"not found",
                r"cannot",
                r"failed",
                r"undefined:",
                r"syntax error",
                r"^rc:",
            ]

            has_error = False
            for pattern in error_patterns:
                if re.search(pattern, output, re.IGNORECASE | re.MULTILINE):
                    has_error = True
                    break

            return ToolResult(
                name=tool,
                success=not has_error,
                output=output,
                error="" if not has_error else "Command returned error"
            )

        else:
            return ToolResult(
                name=tool,
                success=False,
                error=f"Unknown tool: {tool}"
            )

    def compute_reward(
        self,
        model_output: str,
        expected_output: str | None = None,
    ) -> RewardBreakdown:
        """Compute reward for model output by executing tool calls."""
        breakdown = RewardBreakdown()

        # Parse tool calls
        tool_calls = parse_tool_calls(model_output)

        if not tool_calls:
            breakdown.reasoning_bonus = compute_reasoning_bonus(model_output)
            breakdown.compute_total()
            return breakdown

        # Execute each tool call
        results = []
        for call in tool_calls:
            result = self.execute_tool(call.get("name", ""), call.get("params", {}))
            results.append(result)
            time.sleep(0.1)

        breakdown.tool_results = results

        # Compute rewards
        for result in results:
            if result.success:
                breakdown.tool_success_reward += REWARD_TOOL_SUCCESS
            else:
                breakdown.tool_failure_penalty += PENALTY_TOOL_FAILURE

        # Check expected output
        if expected_output:
            for result in results:
                if result.output:
                    if expected_output in result.output:
                        breakdown.output_match_reward = REWARD_OUTPUT_MATCH
                        break
                    elif expected_output.lower() in result.output.lower():
                        breakdown.output_match_reward = REWARD_PARTIAL_MATCH

        # Style and reasoning bonuses
        breakdown.style_bonus = compute_style_bonus(tool_calls)
        breakdown.reasoning_bonus = compute_reasoning_bonus(model_output)

        breakdown.compute_total()
        return breakdown


def create_app(config: ServerConfig):
    """Create FastAPI application."""
    try:
        from fastapi import FastAPI, HTTPException, Depends, Header
        from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
        from pydantic import BaseModel
    except ImportError:
        raise ImportError(
            "FastAPI not installed. Install with: pip install 'plan9-dataset[server]'"
        )

    app = FastAPI(
        title="Plan 9 QEMU API",
        description="Remote execution API for Plan 9 GRPO training",
        version="0.1.0",
    )

    server = QEMUServer(config)
    security = HTTPBearer()

    class ExecuteRequest(BaseModel):
        tool: str
        params: dict[str, Any]

    class ExecuteResponse(BaseModel):
        success: bool
        output: str = ""
        error: str = ""

    class RewardRequest(BaseModel):
        model_output: str
        expected_output: str | None = None

    class RewardResponse(BaseModel):
        total: float
        tool_success_reward: float
        tool_failure_penalty: float
        output_match_reward: float
        style_bonus: float
        reasoning_bonus: float
        tool_results: list[dict]

    class HealthResponse(BaseModel):
        status: str
        vm_running: bool
        uptime: float

    start_time = time.time()

    async def verify_auth(
        credentials: HTTPAuthorizationCredentials = Depends(security)
    ) -> str:
        """Verify bearer token authentication."""
        if not server.verify_token(credentials.credentials):
            raise HTTPException(status_code=401, detail="Invalid token")
        return credentials.credentials

    async def check_rate_limit(
        client_id: str = Header(default="default", alias="X-Client-ID")
    ) -> str:
        """Check rate limit for client."""
        if not server.rate_limiter.check(client_id):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        return client_id

    @app.on_event("startup")
    async def startup():
        """Start VM on server startup."""
        if not server.start_vm():
            raise RuntimeError("Failed to start Plan 9 VM")

    @app.on_event("shutdown")
    async def shutdown():
        """Stop VM on server shutdown."""
        server.stop_vm()

    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy" if server._started else "unhealthy",
            vm_running=server._started,
            uptime=time.time() - start_time,
        )

    @app.post("/execute", response_model=ExecuteResponse)
    async def execute(
        request: ExecuteRequest,
        token: str = Depends(verify_auth),
        client_id: str = Depends(check_rate_limit),
    ):
        """Execute a single tool call in the Plan 9 VM."""
        result = server.execute_tool(request.tool, request.params)
        return ExecuteResponse(
            success=result.success,
            output=result.output,
            error=result.error,
        )

    @app.post("/reward", response_model=RewardResponse)
    async def compute_reward(
        request: RewardRequest,
        token: str = Depends(verify_auth),
        client_id: str = Depends(check_rate_limit),
    ):
        """Compute reward for model output."""
        breakdown = server.compute_reward(
            request.model_output,
            request.expected_output,
        )
        return RewardResponse(
            total=breakdown.total,
            tool_success_reward=breakdown.tool_success_reward,
            tool_failure_penalty=breakdown.tool_failure_penalty,
            output_match_reward=breakdown.output_match_reward,
            style_bonus=breakdown.style_bonus,
            reasoning_bonus=breakdown.reasoning_bonus,
            tool_results=[
                {
                    "name": r.name,
                    "success": r.success,
                    "output": r.output[:500],  # Truncate for safety
                    "error": r.error,
                }
                for r in breakdown.tool_results
            ],
        )

    @app.post("/reset")
    async def reset(
        token: str = Depends(verify_auth),
        client_id: str = Depends(check_rate_limit),
    ):
        """Reset VM state for new session."""
        success = server.reset_vm()
        if not success:
            raise HTTPException(status_code=500, detail="Failed to reset VM")
        return {"status": "reset", "success": True}

    @app.post("/batch-reward")
    async def batch_reward(
        outputs: list[str],
        expected_outputs: list[str | None] | None = None,
        token: str = Depends(verify_auth),
        client_id: str = Depends(check_rate_limit),
    ):
        """Compute rewards for multiple model outputs."""
        if expected_outputs is None:
            expected_outputs = [None] * len(outputs)

        if len(outputs) != len(expected_outputs):
            raise HTTPException(
                status_code=400,
                detail="outputs and expected_outputs must have same length"
            )

        results = []
        for output, expected in zip(outputs, expected_outputs):
            breakdown = server.compute_reward(output, expected)
            results.append({
                "total": breakdown.total,
                "tool_success_reward": breakdown.tool_success_reward,
                "tool_failure_penalty": breakdown.tool_failure_penalty,
                "output_match_reward": breakdown.output_match_reward,
                "style_bonus": breakdown.style_bonus,
                "reasoning_bonus": breakdown.reasoning_bonus,
            })
            # Reset between samples to avoid state leakage
            server.reset_vm()

        return {"rewards": results}

    return app


def run_server(
    disk_image: str,
    shared_image: str,
    token: str,
    host: str = "0.0.0.0",
    port: int = 8080,
    rate_limit: int = 60,
    timeout: int = 30,
    debug: bool = False,
):
    """Run the QEMU API server.

    Args:
        disk_image: Path to 9front QCOW2 disk image.
        shared_image: Path to FAT shared disk image.
        token: Bearer token for authentication.
        host: Host to bind to.
        port: Port to listen on.
        rate_limit: Max requests per minute per client.
        timeout: Command timeout in seconds.
        debug: Enable debug logging.
    """
    try:
        import uvicorn
    except ImportError:
        raise ImportError(
            "uvicorn not installed. Install with: pip install 'plan9-dataset[server]'"
        )

    config = ServerConfig(
        disk_image=disk_image,
        shared_image=shared_image,
        token=token,
        host=host,
        port=port,
        rate_limit=rate_limit,
        timeout=timeout,
        debug=debug,
    )

    app = create_app(config)

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="debug" if debug else "info",
    )


def generate_token() -> str:
    """Generate a secure random token."""
    return secrets.token_urlsafe(32)
