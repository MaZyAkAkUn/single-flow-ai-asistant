"""
Tool Set Management System
Provides comprehensive management of LLM tools for different use cases.
"""

from typing import List, Dict, Optional, Set
from langchain_core.tools import BaseTool
from ..core.logging_config import get_logger

logger = get_logger(__name__)

class ToolSet:
    """
    Represents a collection of tools for a specific use case.

    Attributes:
        name: Name of the tool set
        tools: List of tools in this set
        tool_names: Set of tool names for quick lookup
    """

    def __init__(self, name: str, tools: List[BaseTool] = None):
        """
        Initialize a tool set.

        Args:
            name: Name of the tool set
            tools: Initial list of tools (optional)
        """
        self.name = name
        self.tools = tools or []
        self.tool_names: Set[str] = set()

        # Initialize tool names set
        for tool in self.tools:
            if hasattr(tool, 'name') and tool.name:
                self.tool_names.add(tool.name)

    def add_tool(self, tool: BaseTool) -> bool:
        """
        Add a tool to this set.

        Args:
            tool: Tool to add

        Returns:
            True if tool was added, False if it already exists
        """
        if not hasattr(tool, 'name') or not tool.name:
            logger.warning(f"Cannot add tool without name to tool set '{self.name}'")
            return False

        if tool.name in self.tool_names:
            logger.debug(f"Tool '{tool.name}' already exists in tool set '{self.name}'")
            return False

        self.tools.append(tool)
        self.tool_names.add(tool.name)
        logger.info(f"Added tool '{tool.name}' to tool set '{self.name}'")
        return True

    def remove_tool(self, tool_name: str) -> bool:
        """
        Remove a tool from this set.

        Args:
            tool_name: Name of tool to remove

        Returns:
            True if tool was removed, False if it didn't exist
        """
        if tool_name not in self.tool_names:
            logger.debug(f"Tool '{tool_name}' not found in tool set '{self.name}'")
            return False

        # Remove from tools list
        self.tools = [tool for tool in self.tools
                     if hasattr(tool, 'name') and tool.name != tool_name]

        # Remove from tool names set
        self.tool_names.remove(tool_name)
        logger.info(f"Removed tool '{tool_name}' from tool set '{self.name}'")
        return True

    def has_tool(self, tool_name: str) -> bool:
        """
        Check if tool exists in this set.

        Args:
            tool_name: Name of tool to check

        Returns:
            True if tool exists, False otherwise
        """
        return tool_name in self.tool_names

    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """
        Get a tool by name.

        Args:
            tool_name: Name of tool to get

        Returns:
            Tool instance or None if not found
        """
        for tool in self.tools:
            if hasattr(tool, 'name') and tool.name == tool_name:
                return tool
        return None

    def clear(self):
        """Clear all tools from this set."""
        self.tools.clear()
        self.tool_names.clear()
        logger.info(f"Cleared all tools from tool set '{self.name}'")

    def copy(self) -> 'ToolSet':
        """
        Create a copy of this tool set.

        Returns:
            New ToolSet instance with copied tools
        """
        new_tool_set = ToolSet(self.name, self.tools.copy())
        new_tool_set.tool_names = self.tool_names.copy()
        return new_tool_set

    def __len__(self) -> int:
        """Get number of tools in this set."""
        return len(self.tools)

    def __str__(self) -> str:
        """String representation of tool set."""
        tool_count = len(self.tools)
        tool_names = ', '.join(sorted(self.tool_names)) if self.tool_names else 'none'
        return f"ToolSet(name='{self.name}', tools={tool_count}, tool_names=[{tool_names}])"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()

class ToolSetManager:
    """
    Manages multiple tool sets for different use cases.

    Attributes:
        tool_sets: Dictionary of tool sets by name
        default_tool_set: Name of the default tool set
    """

    def __init__(self):
        """Initialize tool set manager."""
        self.tool_sets: Dict[str, ToolSet] = {}
        self.default_tool_set: Optional[str] = None

    def create_tool_set(self, name: str, tools: List[BaseTool] = None) -> ToolSet:
        """
        Create a new tool set.

        Args:
            name: Name of the tool set
            tools: Initial list of tools (optional)

        Returns:
            Created ToolSet instance
        """
        if name in self.tool_sets:
            logger.warning(f"Tool set '{name}' already exists")
            return self.tool_sets[name]

        tool_set = ToolSet(name, tools)
        self.tool_sets[name] = tool_set

        # Set as default if no default exists
        if self.default_tool_set is None:
            self.default_tool_set = name

        logger.info(f"Created tool set '{name}' with {len(tools or [])} tools")
        return tool_set

    def get_tool_set(self, name: str) -> Optional[ToolSet]:
        """
        Get a tool set by name.

        Args:
            name: Name of tool set to get

        Returns:
            ToolSet instance or None if not found
        """
        return self.tool_sets.get(name)

    def set_default_tool_set(self, name: str) -> bool:
        """
        Set the default tool set.

        Args:
            name: Name of tool set to set as default

        Returns:
            True if successful, False if tool set doesn't exist
        """
        if name not in self.tool_sets:
            logger.warning(f"Cannot set default tool set: '{name}' not found")
            return False

        self.default_tool_set = name
        logger.info(f"Set default tool set to '{name}'")
        return True

    def get_default_tool_set(self) -> Optional[ToolSet]:
        """
        Get the default tool set.

        Returns:
            Default ToolSet instance or None if not set
        """
        if self.default_tool_set:
            return self.get_tool_set(self.default_tool_set)
        return None

    def delete_tool_set(self, name: str) -> bool:
        """
        Delete a tool set.

        Args:
            name: Name of tool set to delete

        Returns:
            True if deleted, False if not found or is default
        """
        if name not in self.tool_sets:
            logger.warning(f"Tool set '{name}' not found")
            return False

        if name == self.default_tool_set:
            logger.warning(f"Cannot delete default tool set '{name}'")
            return False

        del self.tool_sets[name]
        logger.info(f"Deleted tool set '{name}'")
        return True

    def list_tool_sets(self) -> List[str]:
        """
        List all available tool set names.

        Returns:
            List of tool set names
        """
        return list(self.tool_sets.keys())

    def get_all_tool_sets(self) -> Dict[str, ToolSet]:
        """
        Get all tool sets.

        Returns:
            Dictionary of all tool sets
        """
        return self.tool_sets.copy()

    def clear_all_tool_sets(self):
        """Clear all tool sets (except default if it exists)."""
        if self.default_tool_set:
            # Keep default tool set, just clear its tools
            default_set = self.get_tool_set(self.default_tool_set)
            if default_set:
                default_set.clear()
        else:
            self.tool_sets.clear()
            self.default_tool_set = None

        logger.info("Cleared all tool sets")

    def copy_tool_set(self, source_name: str, target_name: str) -> Optional[ToolSet]:
        """
        Copy a tool set to a new name.

        Args:
            source_name: Name of source tool set
            target_name: Name for new tool set

        Returns:
            New ToolSet instance or None if source not found
        """
        source_set = self.get_tool_set(source_name)
        if not source_set:
            logger.warning(f"Source tool set '{source_name}' not found")
            return None

        if target_name in self.tool_sets:
            logger.warning(f"Target tool set '{target_name}' already exists")
            return self.tool_sets[target_name]

        new_set = source_set.copy()
        new_set.name = target_name
        self.tool_sets[target_name] = new_set

        logger.info(f"Copied tool set '{source_name}' to '{target_name}'")
        return new_set

    def __str__(self) -> str:
        """String representation of tool set manager."""
        tool_set_count = len(self.tool_sets)
        default_name = self.default_tool_set or 'none'
        return f"ToolSetManager(tool_sets={tool_set_count}, default='{default_name}')"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()
