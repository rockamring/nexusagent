"""NexusAgent Gradio Web UI。

启动方式:
    python -m nexus.ui
    # 或
    from nexus.ui import launch
    launch()
"""

from nexus.ui.app import launch

__all__ = ["launch"]
