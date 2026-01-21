"""
Plan 9 tool definitions for function calling agents.

Only 3 simple tools - everything else is done via run_command:
- write_file: Write content to a file
- read_file: Read a file
- run_command: Execute a command in rc shell
"""

import json
from typing import Any

# Special tokens for function calling (FunctionGemma-style)
FUNCTION_CALL_START = "<start_function_call>"
FUNCTION_CALL_END = "<end_function_call>"
FUNCTION_RESPONSE_START = "<start_function_response>"
FUNCTION_RESPONSE_END = "<end_function_response>"

# Reasoning tokens
THINK_START = "<think>"
THINK_END = "</think>"

# Gemma turn tokens
START_OF_TURN = "<start_of_turn>"
END_OF_TURN = "<end_of_turn>"


PLAN9_TOOLS = [
    {
        "name": "write_file",
        "description": "Write content to a file in the Plan 9 filesystem",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path to write to (relative or absolute)"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["path", "content"]
        }
    },
    {
        "name": "read_file",
        "description": "Read content from a file in the Plan 9 filesystem",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path to read from"
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "run_command",
        "description": "Execute a command in the rc shell. Use for compiling (6c, 6l), "
                       "running binaries, executing rc scripts, mounting filesystems, etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Command to execute in rc shell"
                }
            },
            "required": ["command"]
        }
    }
]


def get_tools_json() -> str:
    """Return tools definition as JSON string."""
    return json.dumps(PLAN9_TOOLS, indent=2)


def format_system_prompt(tools: list[dict] | None = None) -> str:
    """Format system prompt with tool definitions.

    Args:
        tools: List of tool definitions. Defaults to PLAN9_TOOLS.

    Returns:
        Formatted system prompt string.
    """
    if tools is None:
        tools = PLAN9_TOOLS

    tools_str = json.dumps(tools, indent=2)

    return f"""{START_OF_TURN}system
You are a Plan 9 programming assistant. You can use the following tools to interact with the Plan 9 operating system:

{tools_str}

When using tools, wrap your call in special tokens:
{FUNCTION_CALL_START}call:tool_name{{"param": "value"}}{FUNCTION_CALL_END}

You will receive responses in:
{FUNCTION_RESPONSE_START}response:tool_name{{"result": "value"}}{FUNCTION_RESPONSE_END}

Before making tool calls, you may reason about the task using <think>...</think> tags.

Plan 9 conventions:
- Use #include <u.h> and #include <libc.h> for C programs
- Use print() instead of printf(), exits(nil) instead of exit(0)
- Compile C with: 6c file.c && 6l -o file file.6
- rc is the shell; scripts start with #!/bin/rc
{END_OF_TURN}
"""


def format_user_prompt(message: str) -> str:
    """Format a user message."""
    return f"{START_OF_TURN}user\n{message}\n{END_OF_TURN}\n"


def format_model_turn(content: str) -> str:
    """Format a model turn with optional content."""
    return f"{START_OF_TURN}model\n{content}\n{END_OF_TURN}\n"


def format_tool_call(tool_name: str, params: dict[str, Any]) -> str:
    """Format a tool call for the model to output.

    Args:
        tool_name: Name of the tool to call.
        params: Parameters to pass to the tool.

    Returns:
        Formatted tool call string.
    """
    params_json = json.dumps(params)
    return f"{FUNCTION_CALL_START}call:{tool_name}{params_json}{FUNCTION_CALL_END}"


def format_tool_response(tool_name: str, result: dict[str, Any]) -> str:
    """Format a tool response from execution.

    Args:
        tool_name: Name of the tool that was called.
        result: Result from tool execution.

    Returns:
        Formatted tool response string.
    """
    result_json = json.dumps(result)
    return f"{FUNCTION_RESPONSE_START}response:{tool_name}{result_json}{FUNCTION_RESPONSE_END}"


def format_thinking(thought: str) -> str:
    """Format a thinking block."""
    return f"{THINK_START}\n{thought}\n{THINK_END}"


def parse_tool_calls(model_output: str) -> list[dict[str, Any]]:
    """Parse tool calls from model output.

    Args:
        model_output: Raw model output string.

    Returns:
        List of parsed tool calls, each with 'name' and 'params' keys.
    """
    calls = []

    # Find all function call blocks
    start_idx = 0
    while True:
        start = model_output.find(FUNCTION_CALL_START, start_idx)
        if start == -1:
            break

        end = model_output.find(FUNCTION_CALL_END, start)
        if end == -1:
            break

        # Extract the call content
        call_content = model_output[start + len(FUNCTION_CALL_START):end]

        # Parse call:tool_name{params}
        if call_content.startswith("call:"):
            call_content = call_content[5:]  # Remove "call:" prefix

            # Find where JSON starts
            brace_idx = call_content.find("{")
            if brace_idx != -1:
                tool_name = call_content[:brace_idx]
                params_json = call_content[brace_idx:]

                try:
                    params = json.loads(params_json)
                    calls.append({
                        "name": tool_name,
                        "params": params
                    })
                except json.JSONDecodeError:
                    # Invalid JSON, skip this call
                    pass

        start_idx = end + len(FUNCTION_CALL_END)

    return calls


def parse_thinking(model_output: str) -> str | None:
    """Extract thinking content from model output.

    Args:
        model_output: Raw model output string.

    Returns:
        Thinking content if found, None otherwise.
    """
    start = model_output.find(THINK_START)
    if start == -1:
        return None

    end = model_output.find(THINK_END, start)
    if end == -1:
        return None

    return model_output[start + len(THINK_START):end].strip()


def validate_tool_call(call: dict[str, Any]) -> tuple[bool, str | None]:
    """Validate a parsed tool call against tool definitions.

    Args:
        call: Parsed tool call with 'name' and 'params'.

    Returns:
        Tuple of (is_valid, error_message).
    """
    tool_name = call.get("name")
    params = call.get("params", {})

    # Find tool definition
    tool_def = None
    for tool in PLAN9_TOOLS:
        if tool["name"] == tool_name:
            tool_def = tool
            break

    if tool_def is None:
        return False, f"Unknown tool: {tool_name}"

    # Check required parameters
    schema = tool_def.get("parameters", {})
    required = schema.get("required", [])
    properties = schema.get("properties", {})

    for param in required:
        if param not in params:
            return False, f"Missing required parameter: {param}"

    # Check parameter types
    for param, value in params.items():
        if param in properties:
            expected_type = properties[param].get("type")
            if expected_type == "string" and not isinstance(value, str):
                return False, f"Parameter {param} must be a string"

    return True, None


# Example usage patterns for documentation
EXAMPLE_PATTERNS = {
    "compile_c": {
        "description": "Compile and run a C program",
        "calls": [
            {"name": "write_file", "params": {"path": "hello.c", "content": '#include <u.h>\n#include <libc.h>\n\nvoid\nmain(int argc, char *argv[])\n{\n\tprint("Hello!\\n");\n\texits(nil);\n}\n'}},
            {"name": "run_command", "params": {"command": "6c hello.c && 6l -o hello hello.6"}},
            {"name": "run_command", "params": {"command": "./hello"}}
        ]
    },
    "rc_script": {
        "description": "Write and run an rc script",
        "calls": [
            {"name": "write_file", "params": {"path": "script.rc", "content": "#!/bin/rc\necho hello world\n"}},
            {"name": "run_command", "params": {"command": "chmod +x script.rc && ./script.rc"}}
        ]
    },
    "read_system": {
        "description": "Read system information",
        "calls": [
            {"name": "read_file", "params": {"path": "/dev/user"}},
            {"name": "run_command", "params": {"command": "cat /dev/sysname"}}
        ]
    }
}
