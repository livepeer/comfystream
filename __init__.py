"""Setup expose Comfystream UI and native nodes to ComfyUI."""

import os
import pathlib
import logging
import importlib
import traceback

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def load_nodes() -> tuple[dict[str, type], dict[str, str]]:
    """Load all nodes in the Stream Pack."""
    nodes_dir = pathlib.Path(__file__).parent / "nodes"
    node_class_mappings, node_display_name_mappings = {}, {}

    # Dynamically import all Python modules in the nodes directory.
    for module_path in nodes_dir.iterdir():
        if (
            module_path.is_file()
            and module_path.suffix == ".py"
            and module_path.stem != "__init__"
        ):
            try:
                module_name = f"{__package__}.nodes.{module_path.stem}"
                module = importlib.import_module(module_name)

                # Update mappings if defined in the module.
                if hasattr(module, "NODE_CLASS_MAPPINGS"):
                    node_class_mappings.update(module.NODE_CLASS_MAPPINGS)
                if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                    node_display_name_mappings.update(module.NODE_DISPLAY_NAME_MAPPINGS)
            except Exception:
                format_exc = traceback.format_exc()
                log.error(f"Failed to load module {module_name}:\n{format_exc}")

    return node_class_mappings, node_display_name_mappings


def ensure_init_files():
    """Ensures that the directories 'comfy/' and 'comfy_extras/' and their subdirectories
    are recognized by ComfyUI as Python packages by creating empty __init__.py files
    where needed.
    """
    # Go up two levels from custom_nodes/comfystream_inside to reach ComfyUI root.
    comfy_root = pathlib.Path(__file__).resolve().parents[2]
    base_dirs = ["comfy", "comfy_extras"]

    for base_dir in base_dirs:
        base_path = comfy_root / base_dir
        if not base_path.exists():
            continue

        # Create __init__.py in the root of base_dir first.
        root_init = base_path / "__init__.py"
        if not root_init.exists():
            root_init.touch()

        # Then walk subdirectories.
        for subdir in base_path.rglob("*"):
            if subdir.is_dir():
                init_path = subdir / "__init__.py"
                if not init_path.exists():
                    init_path.touch()


# Create __init__.py files in ComfyUI directories.
ensure_init_files()

# Point to the directory containing our web files.
os.path.join(os.path.dirname(os.path.realpath(__file__)), "web")

# Import and expose native nodes.
NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = load_nodes()
NODE_DISPLAY_NAME_MAPPINGS["Comfystream"] = "Comfystream Native Nodes"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
