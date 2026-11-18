import json
import re
from dataclasses import dataclass

from agents.llm_client import LLMClient
from storage.models.research import ToolDefinition


@dataclass
class ToolBuildRequest:
    """Describes a capability that AURA wants to build."""
    capability: str
    reason: str


@dataclass
class GeneratedTool:
    """A declarative tool specification; never executable source code."""
    name: str
    capability: str
    description: str
    purpose: str
    inputs: list[str]
    outputs: list[str]
    dependencies: list[str]
    validation_requirements: list[str]


class ToolBuilder:
    """
    Designs declarative tool specifications.

    This component never generates or executes source code, installs
    dependencies, or modifies the host environment.
    """

    REQUIRED_FIELDS = {
        "name", "capability", "description", "purpose",
        "inputs", "outputs", "dependencies", "validation_requirements",
    }
    MAX_GENERATION_ATTEMPTS = 3
    TOOL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,63}$")

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        max_generation_attempts: int = MAX_GENERATION_ATTEMPTS,
    ):
        if max_generation_attempts < 1:
            raise ValueError("max_generation_attempts must be at least 1.")
        self.llm_client = llm_client if llm_client is not None else LLMClient()
        self.max_generation_attempts = max_generation_attempts

    def build(self, request: ToolBuildRequest) -> GeneratedTool:
        if not isinstance(request.capability, str) or not request.capability.strip():
            raise ValueError("Capability cannot be empty.")
        if not isinstance(request.reason, str) or not request.reason.strip():
            raise ValueError("Reason cannot be empty.")

        capability = request.capability.strip()
        prompt = f"""
You are the Tool Architect of AURA. Design a declarative tool specification
for the missing capability below.

Capability: {capability}
Reason: {request.reason}

Return ONLY a JSON object with exactly these fields:
{{
  "name": "lowercase_snake_case_name",
  "capability": "{capability}",
  "description": "what the tool does",
  "purpose": "why AURA needs it",
  "inputs": ["input description"],
  "outputs": ["output description"],
  "dependencies": ["package, API, model, or system requirement"],
  "validation_requirements": ["how the tool can be tested"]
}}

Use a lowercase snake_case name. All list values must be non-empty strings.
Prefer deterministic and reproducible designs. Do not include source code,
commands to execute, markdown, or explanatory prose. Dependencies are metadata
only; AURA will not install them automatically.
"""
        current_prompt = prompt
        last_error = None
        for attempt in range(1, self.max_generation_attempts + 1):
            try:
                response = self.llm_client.generate(current_prompt)
                data = self._parse_json(response)
                tool = self._tool_from_data(data)
                self._validate_tool(tool)
                if tool.capability != capability:
                    raise ValueError(
                        "Generated tool capability does not match the requested capability."
                    )
                return tool
            except Exception as error:
                last_error = error
                if attempt < self.max_generation_attempts:
                    current_prompt = (
                        prompt
                        + "\nThe previous response was invalid: "
                        + str(error)
                        + ". Regenerate the complete specification as valid JSON."
                    )
        raise RuntimeError(
            "Failed to create a valid tool specification after "
            f"{self.max_generation_attempts} attempts: {last_error}"
        ) from last_error

    def _tool_from_data(self, data: dict) -> GeneratedTool:
        if not isinstance(data, dict):
            raise ValueError("Tool specification must be a JSON object.")
        missing = self.REQUIRED_FIELDS - set(data)
        extra = set(data) - self.REQUIRED_FIELDS
        if missing:
            raise ValueError(f"Missing tool specification fields: {sorted(missing)}")
        if extra:
            raise ValueError(f"Unexpected tool specification fields: {sorted(extra)}")
        return GeneratedTool(**data)

    def to_tool_definition(self, tool: GeneratedTool) -> ToolDefinition:
        self._validate_tool(tool)
        return ToolDefinition(
            tool_id=tool.name,
            name=tool.name,
            capability=tool.capability,
            description=tool.description,
            purpose=tool.purpose,
            input_schema={"inputs": list(tool.inputs)},
            output_schema={"outputs": list(tool.outputs)},
            dependencies=list(tool.dependencies),
            validation_requirements=list(tool.validation_requirements),
            status="draft",
        )

    def _parse_json(self, response: str) -> dict:
        """Extract one JSON object from plain, fenced, or prose-wrapped output."""
        if not isinstance(response, str) or not response.strip():
            raise ValueError("LLM response is empty.")
        text = response.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            lines = lines[1:] if lines and lines[0].lstrip().startswith("```") else lines
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        start = text.find("{")
        if start < 0:
            raise ValueError("LLM response does not contain a JSON object.")
        value, _ = json.JSONDecoder().raw_decode(text[start:])
        if not isinstance(value, dict):
            raise ValueError("Tool specification must be a JSON object.")
        return value

    def _validate_tool(self, tool: GeneratedTool) -> None:
        for field in ("name", "capability", "description", "purpose"):
            value = getattr(tool, field, None)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Generated tool {field} cannot be empty.")
        if not self.TOOL_NAME_PATTERN.fullmatch(tool.name):
            raise ValueError(
                "Generated tool name must be 2-64 lowercase snake_case characters."
            )

        for field in (
            "inputs", "outputs", "dependencies", "validation_requirements"
        ):
            values = getattr(tool, field, None)
            if not isinstance(values, list):
                raise ValueError(f"Generated tool {field} must be a list.")
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"Generated tool {field} must contain non-empty strings.")
            normalized = [value.strip() for value in values]
            if len(normalized) != len(set(normalized)):
                raise ValueError(f"Generated tool {field} cannot contain duplicates.")

        if not tool.inputs:
            raise ValueError("Generated tool must define at least one input.")
        if not tool.outputs:
            raise ValueError("Generated tool must define at least one output.")
        if not tool.validation_requirements:
            raise ValueError("Generated tool must define validation requirements.")
