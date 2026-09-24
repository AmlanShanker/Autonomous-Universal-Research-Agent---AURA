import json
from dataclasses import dataclass

from agents.llm_client import LLMClient
from storage.models.research import ToolDefinition


@dataclass
class ToolBuildRequest:
    """
    Describes a capability that AURA wants to build.
    """

    capability: str
    reason: str


@dataclass
class GeneratedTool:
    """
    Structured specification for a tool designed by AURA.

    This object describes the tool only.
    It does not contain executable code.
    """

    name: str
    description: str
    purpose: str

    inputs: list[str]
    outputs: list[str]

    dependencies: list[str]
    validation_requirements: list[str]


class ToolBuilder:
  
    def __init__(
        self,
        llm_client: LLMClient | None = None,
    ):
        self.llm_client = (
            llm_client
            if llm_client is not None
            else LLMClient()
        )

    def build(
        self,
        request: ToolBuildRequest,
    ) -> GeneratedTool:
        """
        Ask the LLM to design a tool for the
        requested capability.
        """

        capability = request.capability.strip()

        if not capability:
            raise ValueError(
                "Capability cannot be empty."
            )

        prompt = f"""
You are the Tool Architect of an autonomous
research system called AURA.

AURA has detected a missing capability.

Design a tool that can provide this capability.

Missing capability:
{capability}

Why the capability is required:
{request.reason}

Return ONLY valid JSON.

The JSON must contain exactly these fields:

{{
    "name": "tool_name",
    "description": "short description",
    "purpose": "what the tool accomplishes",
    "inputs": [
        "input 1",
        "input 2"
    ],
    "outputs": [
        "output 1",
        "output 2"
    ],
    "dependencies": [
        "dependency 1"
    ],
    "validation_requirements": [
        "validation requirement 1"
    ]
}}

Requirements:

- The name must represent the capability.
- The description must explain what the tool does.
- The purpose must explain why AURA needs it.
- Inputs must describe information the tool needs.
- Outputs must describe structured information the tool produces.
- Dependencies may include Python packages, external APIs,
  models, or system requirements.
- Validation requirements must describe how AURA could
  verify that the tool works correctly.
- Prefer deterministic and reproducible designs.
- Do not include executable source code.
- Do not include markdown.
- Do not include explanations outside the JSON.
"""

        response = self.llm_client.generate(
            prompt
        )

        try:
            data = self._parse_json(
                response
            )

            tool = GeneratedTool(
                name=data["name"],
                description=data["description"],
                purpose=data["purpose"],
                inputs=data["inputs"],
                outputs=data["outputs"],
                dependencies=data["dependencies"],
                validation_requirements=(
                    data["validation_requirements"]
                ),
            )

        except Exception as error:
            raise RuntimeError(
                "Failed to create a valid tool "
                f"specification: {error}"
            ) from error

        self._validate_tool(
            tool
        )

        return tool

    def to_tool_definition(
        self,
        tool: GeneratedTool,
    ) -> ToolDefinition:
        """
        Convert a generated tool specification into
        the persistent AURA ToolDefinition model.
        """

        return ToolDefinition(
            tool_id=tool.name,
            name=tool.name,
            description=tool.description,
            purpose=tool.purpose,
            input_schema={
                "inputs": tool.inputs,
            },
            output_schema={
                "outputs": tool.outputs,
            },
            dependencies=tool.dependencies,
            validation_requirements=(
                tool.validation_requirements
            ),
            status="draft",
        )

    def _parse_json(
        self,
        response: str,
    ) -> dict:
        """
        Parse JSON returned by the LLM.

        Handles JSON accidentally wrapped
        in markdown code fences.
        """

        response = response.strip()

        if response.startswith("```"):
            lines = response.splitlines()

            if lines[0].startswith("```"):
                lines = lines[1:]

            if (
                lines
                and lines[-1].strip() == "```"
            ):
                lines = lines[:-1]

            response = "\n".join(
                lines
            ).strip()

        return json.loads(
            response
        )

    def _validate_tool(
        self,
        tool: GeneratedTool,
    ) -> None:
        """
        Validate the generated tool specification.
        """

        if not tool.name.strip():
            raise ValueError(
                "Generated tool name cannot be empty."
            )

        if not tool.description.strip():
            raise ValueError(
                "Generated tool description "
                "cannot be empty."
            )

        if not tool.purpose.strip():
            raise ValueError(
                "Generated tool purpose "
                "cannot be empty."
            )

        if not tool.inputs:
            raise ValueError(
                "Generated tool must define "
                "at least one input."
            )

        if not tool.outputs:
            raise ValueError(
                "Generated tool must define "
                "at least one output."
            )

        if not tool.validation_requirements:
            raise ValueError(
                "Generated tool must define "
                "validation requirements."
            )
