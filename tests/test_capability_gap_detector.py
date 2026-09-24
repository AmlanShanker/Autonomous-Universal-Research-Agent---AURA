from agents.capability_gap_detector import CapabilityGapDetector
from tools.registry import ToolRegistry


def main():
    registry = ToolRegistry()

    detector = CapabilityGapDetector(
        tool_registry=registry
    )

    gaps = detector.detect(
        task_id="task_002",
        required_tools=[
            "cloud_compute",
            "literature_search",
        ],
    )

    print("Detected capability gaps:")

    for gap in gaps:
        print(f"- Task: {gap.task_id}")
        print(f"  Capability: {gap.capability}")
        print(f"  Reason: {gap.reason}")


if __name__ == "__main__":
    main()
