import json

from storage.models.execution import ExecutionResult


class ResearchFindingExtractor:
    """
    Uses an LLM to identify reusable research findings from
    execution results.

    This component extracts knowledge but does not persist it.
    Persistence is handled by ResearchMemoryWriter.
    """

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def extract(
        self,
        research_question: str,
        execution_result: ExecutionResult,
    ) -> list[dict]:
        """
        Extract reusable findings from an execution result.

        Returns a list of dictionaries containing:

        - content
        - memory_type
        - tags
        - metadata
        """

        prompt = f"""
You are a research-memory extraction system.

Identify only reusable research knowledge from the execution
result below.

Research question:
{research_question}

Execution result:
{json.dumps(execution_result.output, indent=2)}

Return ONLY valid JSON in this exact structure:

{{
    "findings": [
        {{
            "content": "A concise reusable research finding.",
            "memory_type": "finding",
            "tags": ["tag1", "tag2"],
            "metadata": {{}}
        }}
    ]
}}

Rules:
- Extract only information supported by the execution result.
- Do not invent facts.
- Do not include raw execution logs.
- Do not include generic statements such as "the task completed".
- Keep each finding concise and reusable.
- If there is no meaningful reusable knowledge, return:
  {{"findings": []}}
"""

        response = self.llm_client.generate(prompt)

        data = self._parse_json(response)

        findings = data.get("findings")

        if not isinstance(findings, list):
            raise ValueError(
                "Finding extractor response must contain "
                "a 'findings' list."
            )

        validated_findings = []

        for finding in findings:
            if not isinstance(finding, dict):
                raise ValueError(
                    "Each extracted finding must be an object."
                )

            content = finding.get("content")

            if not isinstance(content, str) or not content.strip():
                raise ValueError(
                    "Each extracted finding must contain "
                    "non-empty string content."
                )

            memory_type = finding.get(
                "memory_type",
                "finding",
            )

            tags = finding.get("tags", [])
            metadata = finding.get("metadata", {})

            if not isinstance(memory_type, str):
                raise ValueError(
                    "Finding 'memory_type' must be a string."
                )

            if not isinstance(tags, list):
                raise ValueError(
                    "Finding 'tags' must be a list."
                )

            if not isinstance(metadata, dict):
                raise ValueError(
                    "Finding 'metadata' must be an object."
                )

            validated_findings.append(
                {
                    "content": content.strip(),
                    "memory_type": memory_type,
                    "tags": tags,
                    "metadata": metadata,
                }
            )

        return validated_findings

    def _parse_json(self, response: str) -> dict:
        response = response.strip()

        if response.startswith("```"):
            lines = response.splitlines()

            if lines and lines[0].startswith("```"):
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            response = "\n".join(lines).strip()

        try:
            return json.loads(response)
        except json.JSONDecodeError as error:
            raise ValueError(
                "Finding extractor returned invalid JSON."
            ) from error