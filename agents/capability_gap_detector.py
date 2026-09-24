from dataclasses import dataclass


@dataclass
class CapabilityGap:
    """
    Represents a capability that AURA currently lacks.
    """

    task_id: str
    capability: str
    reason: str


class CapabilityGapDetector:
    """
    Detects missing capabilities required by research tasks.
    """

    def __init__(self, tool_registry):
        self.tool_registry = tool_registry

    def detect(
        self,
        task_id: str,
        required_tools: list[str],
    ) -> list[CapabilityGap]:
        """
        Detect tools required by a task that are not
        currently available in the Tool Registry.
        """

        gaps = []

        for tool_name in required_tools:

            if not self.tool_registry.has(tool_name):

                gaps.append(
                    CapabilityGap(
                        task_id=task_id,
                        capability=tool_name,
                        reason=(
                            f"Task requires '{tool_name}', "
                            "but no registered tool provides "
                            "this capability."
                        ),
                    )
                )

        return gaps
