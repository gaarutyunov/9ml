"""
Remote QEMU Client for Plan 9 execution.

Client for calling the remote QEMU API server from environments like Google Colab
where local QEMU is not available.

Usage:
    from plan9_dataset.qemu_client import RemoteQEMUClient

    client = RemoteQEMUClient(
        server_url="https://your-server.com",
        token="your-token",
    )

    # Execute a tool
    result = client.execute_tool("write_file", {"path": "hello.c", "content": "..."})

    # Compute reward for model output
    reward = client.compute_reward(model_output, expected_output="Hello")

    # Use as TRL reward function
    reward_fn = client.create_reward_function()
"""

import time
from dataclasses import dataclass
from typing import Any, Callable

try:
    import requests
except ImportError:
    requests = None


@dataclass
class ToolResult:
    """Result of executing a single tool call."""
    name: str
    success: bool
    output: str = ""
    error: str = ""


@dataclass
class RewardBreakdown:
    """Detailed breakdown of reward components."""
    tool_success_reward: float = 0.0
    tool_failure_penalty: float = 0.0
    output_match_reward: float = 0.0
    style_bonus: float = 0.0
    reasoning_bonus: float = 0.0
    total: float = 0.0
    tool_results: list[ToolResult] | None = None


class RemoteQEMUClient:
    """Client for remote QEMU API server."""

    def __init__(
        self,
        server_url: str,
        token: str,
        timeout: int = 60,
        client_id: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """Initialize the client.

        Args:
            server_url: Base URL of the QEMU API server (e.g., "https://example.com").
            token: Bearer token for authentication.
            timeout: Request timeout in seconds.
            client_id: Optional client ID for rate limiting tracking.
            max_retries: Maximum number of retries for failed requests.
            retry_delay: Delay between retries in seconds.
        """
        if requests is None:
            raise ImportError(
                "requests library not installed. Install with: pip install requests"
            )

        self.server_url = server_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.client_id = client_id or f"client-{int(time.time())}"
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {token}",
            "X-Client-ID": self.client_id,
            "Content-Type": "application/json",
        })

    def _request(
        self,
        method: str,
        endpoint: str,
        json_data: dict | None = None,
        retry: bool = True,
    ) -> dict:
        """Make an HTTP request with retries.

        Args:
            method: HTTP method (GET, POST, etc.).
            endpoint: API endpoint (e.g., "/execute").
            json_data: JSON body for POST requests.
            retry: Whether to retry on failure.

        Returns:
            Response JSON data.

        Raises:
            RemoteQEMUError: If request fails after all retries.
        """
        url = f"{self.server_url}{endpoint}"
        last_error = None

        for attempt in range(self.max_retries if retry else 1):
            try:
                response = self._session.request(
                    method,
                    url,
                    json=json_data,
                    timeout=self.timeout,
                )

                if response.status_code == 429:
                    # Rate limited - wait and retry
                    wait_time = self.retry_delay * (attempt + 1) * 2
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                return response.json()

            except requests.exceptions.Timeout as e:
                last_error = RemoteQEMUError(f"Request timed out: {e}")
            except requests.exceptions.ConnectionError as e:
                last_error = RemoteQEMUError(f"Connection error: {e}")
            except requests.exceptions.HTTPError as e:
                if response.status_code == 401:
                    raise RemoteQEMUError("Authentication failed: invalid token")
                elif response.status_code == 429:
                    last_error = RemoteQEMUError("Rate limit exceeded")
                else:
                    last_error = RemoteQEMUError(f"HTTP error {response.status_code}: {e}")
            except Exception as e:
                last_error = RemoteQEMUError(f"Unexpected error: {e}")

            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay * (attempt + 1))

        raise last_error

    def health(self) -> dict:
        """Check server health.

        Returns:
            Health status dict with 'status', 'vm_running', 'uptime'.
        """
        return self._request("GET", "/health", retry=False)

    def is_healthy(self) -> bool:
        """Check if server is healthy and VM is running.

        Returns:
            True if server is healthy, False otherwise.
        """
        try:
            health = self.health()
            return health.get("status") == "healthy" and health.get("vm_running", False)
        except Exception:
            return False

    def execute_tool(self, tool: str, params: dict[str, Any]) -> ToolResult:
        """Execute a single tool call on the remote VM.

        Args:
            tool: Tool name ("write_file", "read_file", "run_command").
            params: Tool parameters.

        Returns:
            ToolResult with success status and output.
        """
        response = self._request("POST", "/execute", {
            "tool": tool,
            "params": params,
        })

        return ToolResult(
            name=tool,
            success=response.get("success", False),
            output=response.get("output", ""),
            error=response.get("error", ""),
        )

    def write_file(self, path: str, content: str) -> ToolResult:
        """Write a file on the remote VM.

        Args:
            path: File path to write to.
            content: File content.

        Returns:
            ToolResult with success status.
        """
        return self.execute_tool("write_file", {"path": path, "content": content})

    def read_file(self, path: str) -> ToolResult:
        """Read a file from the remote VM.

        Args:
            path: File path to read.

        Returns:
            ToolResult with file content in output.
        """
        return self.execute_tool("read_file", {"path": path})

    def run_command(self, command: str) -> ToolResult:
        """Run a command on the remote VM.

        Args:
            command: Command to execute in rc shell.

        Returns:
            ToolResult with command output.
        """
        return self.execute_tool("run_command", {"command": command})

    def reset(self) -> bool:
        """Reset VM state for new session.

        Returns:
            True if reset successful.
        """
        try:
            response = self._request("POST", "/reset")
            return response.get("success", False)
        except Exception:
            return False

    def compute_reward(
        self,
        model_output: str,
        expected_output: str | None = None,
    ) -> RewardBreakdown:
        """Compute reward for model output.

        Args:
            model_output: Raw model output string with tool calls.
            expected_output: Optional expected output to match.

        Returns:
            RewardBreakdown with detailed reward components.
        """
        response = self._request("POST", "/reward", {
            "model_output": model_output,
            "expected_output": expected_output,
        })

        tool_results = None
        if "tool_results" in response:
            tool_results = [
                ToolResult(
                    name=r.get("name", ""),
                    success=r.get("success", False),
                    output=r.get("output", ""),
                    error=r.get("error", ""),
                )
                for r in response["tool_results"]
            ]

        return RewardBreakdown(
            tool_success_reward=response.get("tool_success_reward", 0.0),
            tool_failure_penalty=response.get("tool_failure_penalty", 0.0),
            output_match_reward=response.get("output_match_reward", 0.0),
            style_bonus=response.get("style_bonus", 0.0),
            reasoning_bonus=response.get("reasoning_bonus", 0.0),
            total=response.get("total", 0.0),
            tool_results=tool_results,
        )

    def batch_compute_rewards(
        self,
        outputs: list[str],
        expected_outputs: list[str | None] | None = None,
    ) -> list[RewardBreakdown]:
        """Compute rewards for multiple model outputs.

        Args:
            outputs: List of model output strings.
            expected_outputs: Optional list of expected outputs.

        Returns:
            List of RewardBreakdown objects.
        """
        response = self._request("POST", "/batch-reward", {
            "outputs": outputs,
            "expected_outputs": expected_outputs,
        })

        results = []
        for r in response.get("rewards", []):
            results.append(RewardBreakdown(
                tool_success_reward=r.get("tool_success_reward", 0.0),
                tool_failure_penalty=r.get("tool_failure_penalty", 0.0),
                output_match_reward=r.get("output_match_reward", 0.0),
                style_bonus=r.get("style_bonus", 0.0),
                reasoning_bonus=r.get("reasoning_bonus", 0.0),
                total=r.get("total", 0.0),
            ))

        return results

    def create_reward_function(
        self,
        reset_between_samples: bool = False,
    ) -> Callable:
        """Create a reward function compatible with TRL GRPO trainer.

        Args:
            reset_between_samples: Whether to reset VM between samples.

        Returns:
            Reward function that takes (samples, prompts, outputs) and returns rewards.
        """
        client = self

        def reward_function(
            samples: list[str],
            prompts: list[str],
            outputs: list[str],
            **kwargs,
        ) -> list[float]:
            """Compute rewards for GRPO training.

            Args:
                samples: Full samples (prompt + output).
                prompts: Original prompts.
                outputs: Generated outputs.
                **kwargs: Additional arguments (e.g., expected_outputs).

            Returns:
                List of scalar rewards.
            """
            expected_outputs = kwargs.get("expected_outputs", [None] * len(outputs))

            if reset_between_samples:
                # Compute one at a time with resets
                rewards = []
                for output, expected in zip(outputs, expected_outputs):
                    try:
                        breakdown = client.compute_reward(output, expected)
                        rewards.append(breakdown.total)
                        client.reset()
                    except Exception:
                        rewards.append(0.0)
                return rewards
            else:
                # Batch compute
                try:
                    breakdowns = client.batch_compute_rewards(outputs, expected_outputs)
                    return [b.total for b in breakdowns]
                except Exception:
                    return [0.0] * len(outputs)

        return reward_function

    def close(self) -> None:
        """Close the client session."""
        self._session.close()

    def __enter__(self) -> "RemoteQEMUClient":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()


class RemoteQEMUError(Exception):
    """Error from remote QEMU API."""
    pass


def create_client_from_env() -> RemoteQEMUClient:
    """Create client from environment variables.

    Looks for:
    - QEMU_SERVER_URL: Server URL
    - QEMU_TOKEN: Authentication token

    Returns:
        Configured RemoteQEMUClient.

    Raises:
        ValueError: If environment variables not set.
    """
    import os

    server_url = os.environ.get("QEMU_SERVER_URL")
    token = os.environ.get("QEMU_TOKEN")

    if not server_url:
        raise ValueError("QEMU_SERVER_URL environment variable not set")
    if not token:
        raise ValueError("QEMU_TOKEN environment variable not set")

    return RemoteQEMUClient(server_url=server_url, token=token)


def create_remote_reward_function(
    server_url: str | None = None,
    token: str | None = None,
    reset_between_samples: bool = False,
) -> Callable:
    """Create a remote reward function for TRL GRPO trainer.

    Args:
        server_url: Server URL. If None, reads from QEMU_SERVER_URL env var.
        token: Auth token. If None, reads from QEMU_TOKEN env var.
        reset_between_samples: Whether to reset VM between samples.

    Returns:
        Reward function compatible with TRL GRPOTrainer.
    """
    import os

    if server_url is None:
        server_url = os.environ.get("QEMU_SERVER_URL")
    if token is None:
        token = os.environ.get("QEMU_TOKEN")

    if not server_url or not token:
        raise ValueError(
            "Server URL and token required. Set QEMU_SERVER_URL and QEMU_TOKEN "
            "environment variables or pass them as arguments."
        )

    client = RemoteQEMUClient(server_url=server_url, token=token)
    return client.create_reward_function(reset_between_samples=reset_between_samples)
