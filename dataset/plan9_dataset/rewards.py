"""
GRPO reward functions for Plan 9 function calling agents.

Computes rewards based on:
1. Tool call execution success
2. Expected output matching
3. Plan 9 style/idiom adherence
4. Reasoning quality (optional bonus)
"""

import re
import time
from dataclasses import dataclass, field
from typing import Any

from .tools import (
    THINK_START,
    THINK_END,
    parse_tool_calls,
    parse_thinking,
)
from .validate import FATDisk, Plan9VM


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
    tool_results: list[ToolResult] = field(default_factory=list)

    def compute_total(self) -> float:
        """Compute total reward from components."""
        self.total = (
            self.tool_success_reward
            + self.tool_failure_penalty
            + self.output_match_reward
            + self.style_bonus
            + self.reasoning_bonus
        )
        return self.total


# Reward constants
REWARD_TOOL_SUCCESS = 1.0
PENALTY_TOOL_FAILURE = -0.5
REWARD_OUTPUT_MATCH = 2.0
REWARD_PARTIAL_MATCH = 1.0
REWARD_STYLE_NIL = 0.3
REWARD_STYLE_HEADERS = 0.3
REWARD_STYLE_PRINT = 0.2
REWARD_STYLE_EXITS = 0.2
REWARD_REASONING = 0.5


def execute_tool_call(
    call: dict[str, Any],
    vm: Plan9VM,
    disk: FATDisk,
    timeout: int = 30
) -> ToolResult:
    """Execute a single tool call and return result.

    Args:
        call: Parsed tool call with 'name' and 'params'.
        vm: Running Plan 9 VM instance.
        disk: FAT disk for file operations.
        timeout: Command timeout in seconds.

    Returns:
        ToolResult with success status and output.
    """
    name = call.get("name", "")
    params = call.get("params", {})

    if name == "write_file":
        path = params.get("path", "")
        content = params.get("content", "")

        if not path or not content:
            return ToolResult(
                name=name,
                success=False,
                error="Missing path or content parameter"
            )

        # Write to FAT disk (VM will read from /mnt/host/)
        success = disk.copy_content(content, path)
        return ToolResult(
            name=name,
            success=success,
            output="" if success else "",
            error="" if success else "Failed to write file"
        )

    elif name == "read_file":
        path = params.get("path", "")

        if not path:
            return ToolResult(
                name=name,
                success=False,
                error="Missing path parameter"
            )

        # Check if it's a system file or a FAT disk file
        if path.startswith("/"):
            # System file - read via VM
            output = vm.run_command(f"cat {path}", timeout=timeout)
            success = bool(output) and "error" not in output.lower()
            return ToolResult(
                name=name,
                success=success,
                output=output if success else "",
                error="" if success else f"Failed to read {path}"
            )
        else:
            # FAT disk file
            content = disk.read_file(path)
            success = content is not None
            return ToolResult(
                name=name,
                success=success,
                output=content or "",
                error="" if success else f"File not found: {path}"
            )

    elif name == "run_command":
        command = params.get("command", "")

        if not command:
            return ToolResult(
                name=name,
                success=False,
                error="Missing command parameter"
            )

        # Execute command in VM
        output = vm.run_command(command, timeout=timeout)

        # Determine success based on output
        # Common error patterns in Plan 9
        error_patterns = [
            r"error:",
            r"not found",
            r"cannot",
            r"failed",
            r"undefined:",
            r"syntax error",
            r"^rc:",  # rc shell errors
        ]

        has_error = False
        for pattern in error_patterns:
            if re.search(pattern, output, re.IGNORECASE | re.MULTILINE):
                has_error = True
                break

        return ToolResult(
            name=name,
            success=not has_error,
            output=output,
            error="" if not has_error else "Command returned error"
        )

    else:
        return ToolResult(
            name=name,
            success=False,
            error=f"Unknown tool: {name}"
        )


def compute_style_bonus(tool_calls: list[dict[str, Any]]) -> float:
    """Compute bonus for Plan 9 style/idiom adherence.

    Checks written files for:
    - Use of nil instead of NULL
    - Proper Plan 9 headers (#include <u.h>, #include <libc.h>)
    - Use of print() instead of printf()
    - Use of exits() instead of exit()
    """
    bonus = 0.0

    for call in tool_calls:
        if call.get("name") != "write_file":
            continue

        content = call.get("params", {}).get("content", "")
        if not content:
            continue

        # Check for Plan 9 idioms in C code
        if "#include" in content or "void\nmain" in content or "void main" in content:
            # It's C code
            if "nil" in content and "NULL" not in content:
                bonus += REWARD_STYLE_NIL

            if "#include <u.h>" in content and "#include <libc.h>" in content:
                bonus += REWARD_STYLE_HEADERS

            if "print(" in content and "printf(" not in content:
                bonus += REWARD_STYLE_PRINT

            if "exits(" in content and "exit(" not in content:
                bonus += REWARD_STYLE_EXITS

    return bonus


def compute_reasoning_bonus(model_output: str) -> float:
    """Compute bonus for including reasoning before tool calls.

    Rewards the model for:
    - Including <think>...</think> tags
    - Non-trivial reasoning content
    """
    thinking = parse_thinking(model_output)

    if thinking is None:
        return 0.0

    # Require some minimum content
    if len(thinking) < 20:
        return 0.0

    return REWARD_REASONING


def compute_reward(
    model_output: str,
    tool_calls: list[dict[str, Any]],
    results: list[ToolResult],
    expected_output: str | None = None,
) -> RewardBreakdown:
    """Compute total reward from tool execution results.

    Args:
        model_output: Raw model output (for reasoning analysis).
        tool_calls: List of parsed tool calls.
        results: List of execution results.
        expected_output: Optional expected output string to match.

    Returns:
        RewardBreakdown with detailed reward components.
    """
    breakdown = RewardBreakdown()
    breakdown.tool_results = results

    # Reward for successful tool calls
    for result in results:
        if result.success:
            breakdown.tool_success_reward += REWARD_TOOL_SUCCESS
        else:
            breakdown.tool_failure_penalty += PENALTY_TOOL_FAILURE

    # Bonus for expected output match
    if expected_output:
        for result in results:
            if result.output:
                if expected_output in result.output:
                    breakdown.output_match_reward = REWARD_OUTPUT_MATCH
                    break
                # Partial match (case-insensitive)
                elif expected_output.lower() in result.output.lower():
                    breakdown.output_match_reward = REWARD_PARTIAL_MATCH

    # Style bonus for Plan 9 idioms
    breakdown.style_bonus = compute_style_bonus(tool_calls)

    # Reasoning bonus
    breakdown.reasoning_bonus = compute_reasoning_bonus(model_output)

    breakdown.compute_total()
    return breakdown


def execute_and_reward(
    model_output: str,
    vm: Plan9VM,
    disk: FATDisk,
    expected_output: str | None = None,
    timeout: int = 30,
) -> RewardBreakdown:
    """Parse, execute, and compute reward for model output.

    This is the main entry point for GRPO reward computation.

    Args:
        model_output: Raw model output string.
        vm: Running Plan 9 VM instance.
        disk: FAT disk for file operations.
        expected_output: Optional expected output to match.
        timeout: Command timeout in seconds.

    Returns:
        RewardBreakdown with execution results and rewards.
    """
    # Parse tool calls from model output
    tool_calls = parse_tool_calls(model_output)

    if not tool_calls:
        # No tool calls - return zero reward
        return RewardBreakdown(
            reasoning_bonus=compute_reasoning_bonus(model_output)
        )

    # Execute each tool call
    results = []
    for call in tool_calls:
        result = execute_tool_call(call, vm, disk, timeout)
        results.append(result)

        # Small delay between calls
        time.sleep(0.1)

    # Compute reward
    return compute_reward(model_output, tool_calls, results, expected_output)


class RewardEnvironment:
    """Environment for computing rewards with persistent VM session.

    Use this class to batch multiple reward computations in a single
    VM session, avoiding VM boot overhead for each sample.
    """

    def __init__(self, disk_image: str, shared_image: str, debug: bool = False):
        """Initialize environment.

        Args:
            disk_image: Path to 9front QCOW2 disk image.
            shared_image: Path to FAT shared disk image.
            debug: Enable debug logging.
        """
        self.disk_image = disk_image
        self.shared_image = shared_image
        self.debug = debug
        self.vm: Plan9VM | None = None
        self.disk: FATDisk | None = None
        self._started = False

    def start(self) -> bool:
        """Start VM and prepare disk.

        Returns:
            True if successful, False otherwise.
        """
        if self._started:
            return True

        # Create fresh shared disk
        self.disk = FATDisk(self.shared_image)
        if not self.disk.create(64):
            return False

        # Start VM
        self.vm = Plan9VM(self.disk_image, self.shared_image, debug=self.debug)
        if not self.vm.boot():
            return False

        self._started = True
        return True

    def stop(self) -> None:
        """Stop VM and clean up."""
        if self.vm:
            self.vm.shutdown()
            self.vm = None
        self.disk = None
        self._started = False

    def compute_reward(
        self,
        model_output: str,
        expected_output: str | None = None,
        timeout: int = 30,
    ) -> RewardBreakdown:
        """Compute reward for a single model output.

        Args:
            model_output: Raw model output string.
            expected_output: Optional expected output to match.
            timeout: Command timeout in seconds.

        Returns:
            RewardBreakdown with execution results and rewards.

        Raises:
            RuntimeError: If environment not started.
        """
        if not self._started or not self.vm or not self.disk:
            raise RuntimeError("Environment not started. Call start() first.")

        # Clear VM state before each reward computation
        self.vm.clear_output()

        return execute_and_reward(
            model_output, self.vm, self.disk, expected_output, timeout
        )

    def batch_compute_rewards(
        self,
        outputs: list[str],
        expected_outputs: list[str | None] | None = None,
        timeout: int = 30,
    ) -> list[RewardBreakdown]:
        """Compute rewards for multiple model outputs.

        Args:
            outputs: List of model output strings.
            expected_outputs: Optional list of expected outputs.
            timeout: Command timeout in seconds.

        Returns:
            List of RewardBreakdown objects.
        """
        if expected_outputs is None:
            expected_outputs = [None] * len(outputs)

        results = []
        for output, expected in zip(outputs, expected_outputs):
            reward = self.compute_reward(output, expected, timeout)
            results.append(reward)

        return results

    def __enter__(self) -> "RewardEnvironment":
        """Context manager entry."""
        if not self.start():
            raise RuntimeError("Failed to start reward environment")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()


def create_reward_function(
    disk_image: str,
    shared_image: str,
    debug: bool = False,
):
    """Create a reward function for TRL GRPO trainer.

    This returns a function compatible with TRL's reward_funcs parameter.

    Args:
        disk_image: Path to 9front QCOW2 disk image.
        shared_image: Path to FAT shared disk image.
        debug: Enable debug logging.

    Returns:
        Reward function that takes (samples, prompts, outputs) and returns rewards.
    """
    env = RewardEnvironment(disk_image, shared_image, debug)

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
        # Start environment if not already started
        if not env._started:
            if not env.start():
                # Return zero rewards if VM fails
                return [0.0] * len(outputs)

        expected_outputs = kwargs.get("expected_outputs", [None] * len(outputs))

        rewards = []
        for output, expected in zip(outputs, expected_outputs):
            breakdown = env.compute_reward(output, expected, timeout=30)
            rewards.append(breakdown.total)

        return rewards

    # Attach cleanup method
    reward_function.cleanup = env.stop

    return reward_function
