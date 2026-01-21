"""
Multi-turn conversation dataset with real tool execution.

Three dataset types:
1. knowledge - Raw source code and docs for LoRA post-training
2. sft - Multi-turn conversations with tool calls for SFT
3. grpo - Same format as sft, used with execution-based rewards
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .tools import (
    PLAN9_TOOLS,
    format_system_prompt,
    format_tool_call,
    format_tool_response,
    format_thinking,
    START_OF_TURN,
    END_OF_TURN,
    FUNCTION_CALL_START,
    FUNCTION_CALL_END,
)


@dataclass
class ToolCall:
    """A single tool call with its response."""
    name: str
    params: dict[str, Any]
    response: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "params": self.params,
            "response": self.response,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ToolCall":
        return cls(
            name=data["name"],
            params=data["params"],
            response=data.get("response"),
        )


@dataclass
class Turn:
    """A single turn in a conversation."""
    role: str  # "user", "model", "tool"
    content: str
    thinking: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {"role": self.role, "content": self.content}
        if self.thinking:
            d["thinking"] = self.thinking
        if self.tool_calls:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Turn":
        tool_calls = [ToolCall.from_dict(tc) for tc in data.get("tool_calls", [])]
        return cls(
            role=data["role"],
            content=data["content"],
            thinking=data.get("thinking"),
            tool_calls=tool_calls,
        )


@dataclass
class Conversation:
    """A multi-turn conversation with tool calls."""
    turns: list[Turn] = field(default_factory=list)
    category: str = ""
    source: str = ""
    validated: bool = False
    validation_output: str = ""

    def to_dict(self) -> dict:
        return {
            "turns": [t.to_dict() for t in self.turns],
            "category": self.category,
            "source": self.source,
            "validated": self.validated,
            "validation_output": self.validation_output,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Conversation":
        turns = [Turn.from_dict(t) for t in data.get("turns", [])]
        return cls(
            turns=turns,
            category=data.get("category", ""),
            source=data.get("source", ""),
            validated=data.get("validated", False),
            validation_output=data.get("validation_output", ""),
        )

    def to_text(self, include_system: bool = True) -> str:
        """Convert to training text format."""
        parts = []

        if include_system:
            parts.append(format_system_prompt())

        for turn in self.turns:
            if turn.role == "user":
                parts.append(f"{START_OF_TURN}user\n{turn.content}\n{END_OF_TURN}")
            elif turn.role == "model":
                model_content = ""
                if turn.thinking:
                    model_content += format_thinking(turn.thinking) + "\n"
                if turn.tool_calls:
                    for tc in turn.tool_calls:
                        model_content += format_tool_call(tc.name, tc.params) + "\n"
                if turn.content:
                    if model_content:
                        model_content += turn.content
                    else:
                        model_content = turn.content
                parts.append(f"{START_OF_TURN}model\n{model_content.strip()}\n{END_OF_TURN}")
            elif turn.role == "tool":
                # Tool responses are presented as user turns with special format
                parts.append(f"{START_OF_TURN}user\n{turn.content}\n{END_OF_TURN}")

        return "\n".join(parts)

    def to_sharegpt(self) -> dict:
        """Convert to ShareGPT format for Axolotl."""
        conversations = []
        for turn in self.turns:
            if turn.role == "user":
                conversations.append({"from": "human", "value": turn.content})
            elif turn.role == "model":
                content = ""
                if turn.thinking:
                    content += format_thinking(turn.thinking) + "\n"
                if turn.tool_calls:
                    for tc in turn.tool_calls:
                        content += format_tool_call(tc.name, tc.params) + "\n"
                if turn.content:
                    content += turn.content
                conversations.append({"from": "gpt", "value": content.strip()})
            elif turn.role == "tool":
                # Tool responses as function_response
                conversations.append({"from": "function_response", "value": turn.content})
        return {"conversations": conversations}


@dataclass
class KnowledgeItem:
    """A raw text item for knowledge injection."""
    text: str
    source: str = ""
    file_type: str = ""  # "c", "rc", "mkfile", "doc", "man"

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source": self.source,
            "file_type": self.file_type,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgeItem":
        return cls(
            text=data["text"],
            source=data.get("source", ""),
            file_type=data.get("file_type", ""),
        )


def load_conversations(path: Path) -> list[Conversation]:
    """Load conversations from JSONL file."""
    if not path.exists():
        return []

    conversations = []
    with open(path) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                conversations.append(Conversation.from_dict(data))
    return conversations


def save_conversations(conversations: list[Conversation], path: Path) -> int:
    """Save conversations to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for conv in conversations:
            f.write(json.dumps(conv.to_dict()) + "\n")
    return len(conversations)


def load_knowledge(path: Path) -> list[KnowledgeItem]:
    """Load knowledge items from JSONL file."""
    if not path.exists():
        return []

    items = []
    with open(path) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                items.append(KnowledgeItem.from_dict(data))
    return items


def save_knowledge(items: list[KnowledgeItem], path: Path) -> int:
    """Save knowledge items to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item.to_dict()) + "\n")
    return len(items)


def export_sft_text(conversations: list[Conversation], path: Path) -> int:
    """Export conversations as plain text for SFT training."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for conv in conversations:
            f.write(conv.to_text(include_system=True))
            f.write("\n\n")
    return len(conversations)


def export_sft_sharegpt(conversations: list[Conversation], path: Path) -> int:
    """Export conversations in ShareGPT format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [conv.to_sharegpt() for conv in conversations]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return len(conversations)


def export_knowledge_text(items: list[KnowledgeItem], path: Path) -> int:
    """Export knowledge as plain text for continued pretraining."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in items:
            # Add file type header
            if item.file_type == "c":
                f.write(f"/* Plan 9 C - {item.source} */\n")
            elif item.file_type == "rc":
                f.write(f"# Plan 9 rc - {item.source}\n")
            elif item.file_type == "mkfile":
                f.write(f"# Plan 9 mkfile - {item.source}\n")
            elif item.file_type == "doc":
                f.write(f"# Plan 9 documentation - {item.source}\n\n")

            f.write(item.text)
            f.write("\n\n---\n\n")
    return len(items)


def create_conversation_from_task(
    user_request: str,
    thinking: str,
    tool_sequence: list[tuple[str, dict, dict]],  # (name, params, response)
    final_response: str,
    category: str = "",
    source: str = "",
) -> Conversation:
    """Create a conversation from a task specification.

    Args:
        user_request: What the user asks for
        thinking: Model's reasoning
        tool_sequence: List of (tool_name, params, response) tuples
        final_response: Model's final message after tools complete
        category: Category for the conversation
        source: Source attribution

    Returns:
        Conversation object
    """
    turns = []

    # User turn
    turns.append(Turn(role="user", content=user_request))

    # Model turn with thinking and first tool call(s)
    # Group consecutive tool calls into one model turn
    model_turn = Turn(role="model", content="", thinking=thinking, tool_calls=[])

    for i, (name, params, response) in enumerate(tool_sequence):
        tool_call = ToolCall(name=name, params=params, response=response)
        model_turn.tool_calls.append(tool_call)

        # After adding tool call, add tool response turn
        turns.append(model_turn)
        turns.append(Turn(
            role="tool",
            content=format_tool_response(name, response)
        ))

        # Prepare next model turn (empty unless more tool calls)
        model_turn = Turn(role="model", content="", tool_calls=[])

    # Final model response
    turns.append(Turn(role="model", content=final_response))

    return Conversation(
        turns=turns,
        category=category,
        source=source,
    )


def get_dataset_stats(
    conversations_path: Path | None = None,
    knowledge_path: Path | None = None,
) -> dict[str, Any]:
    """Get statistics about the datasets."""
    stats = {}

    if conversations_path and conversations_path.exists():
        convs = load_conversations(conversations_path)
        stats["conversations"] = {
            "total": len(convs),
            "validated": len([c for c in convs if c.validated]),
            "by_category": {},
            "total_turns": sum(len(c.turns) for c in convs),
            "total_tool_calls": sum(
                sum(len(t.tool_calls) for t in c.turns)
                for c in convs
            ),
        }
        for c in convs:
            cat = c.category or "uncategorized"
            stats["conversations"]["by_category"][cat] = (
                stats["conversations"]["by_category"].get(cat, 0) + 1
            )

    if knowledge_path and knowledge_path.exists():
        items = load_knowledge(knowledge_path)
        stats["knowledge"] = {
            "total": len(items),
            "by_type": {},
            "total_chars": sum(len(item.text) for item in items),
        }
        for item in items:
            ft = item.file_type or "unknown"
            stats["knowledge"]["by_type"][ft] = (
                stats["knowledge"]["by_type"].get(ft, 0) + 1
            )

    return stats
