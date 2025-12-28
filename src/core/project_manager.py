"""
Project context manager for handling project state and full context loading.
"""
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import os

from ..data.schemas import ProjectContext
from ..core.logging_config import get_logger

logger = get_logger(__name__)


class ProjectContextManager:
    """
    Manages project contexts with lazy loading and token optimization.
    Handles project state, context caching, and full context retrieval.
    """
    
    def __init__(self, data_dir: str = "./data/projects"):
        """Initialize project context manager."""
        self.data_dir = data_dir
        self._ensure_data_dir()
        self._project_cache: Dict[str, ProjectContext] = {}
        self._full_context_cache: Dict[str, str] = {}
        self._current_project_id: Optional[str] = None
        self._max_cached_projects = 10
        
    def _ensure_data_dir(self):
        """Ensure project data directory exists."""
        os.makedirs(self.data_dir, exist_ok=True)
        
    def add_project(
        self,
        project_id: str,
        project_name: str,
        project_description: str,
        project_status: str = "active",
        tags: List[str] = None,
        context_summary: str = "",
        context_full: str = "",
        metadata: Dict[str, Any] = None
    ) -> bool:
        """
        Add or update a project.
        
        Args:
            project_id: Unique project identifier
            project_name: Human-readable project name
            project_description: Brief project description
            project_status: Project status (active, paused, completed, etc.)
            tags: List of project tags
            context_summary: Brief context summary for prompts
            context_full: Full project context (cached separately for performance)
            metadata: Additional project metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            project = ProjectContext(
                project_id=project_id,
                project_name=project_name,
                project_description=project_description,
                project_status=project_status,
                tags=tags or [],
                context_summary=context_summary,
                context_full=context_full if context_full else None,
                metadata=metadata or {},
                last_updated=datetime.now()
            )
            
            # Store in cache
            self._project_cache[project_id] = project
            
            # Store full context separately if provided
            if context_full:
                self._full_context_cache[project_id] = context_full
                
            # Save to disk
            self._save_project(project_id)
            
            logger.info(f"Added/updated project: {project_name} ({project_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add project {project_id}: {e}")
            return False
            
    def set_current_project(self, project_id: str) -> bool:
        """Set current active project."""
        if project_id in self._project_cache:
            # Mark all projects as not current
            for proj in self._project_cache.values():
                proj.is_current = False
                
            # Mark current project
            self._project_cache[project_id].is_current = True
            self._current_project_id = project_id
            
            logger.info(f"Set current project to: {self._project_cache[project_id].project_name}")
            return True
        else:
            logger.warning(f"Project not found: {project_id}")
            return False
            
    def get_current_project(self) -> Optional[ProjectContext]:
        """Get currently active project."""
        if self._current_project_id:
            return self._project_cache.get(self._current_project_id)
        return None
        
    def get_project_by_id(self, project_id: str) -> Optional[ProjectContext]:
        """Get project by ID with lazy loading."""
        if project_id in self._project_cache:
            return self._project_cache[project_id]
            
        # Try to load from disk
        project = self._load_project(project_id)
        if project:
            self._project_cache[project_id] = project
            return project
            
        return None
        
    def get_full_project_context(self, project_id: str) -> Optional[str]:
        """
        Get full project context for LLM tool calls.
        This implements the lazy loading concept from the user requirements.
        """
        project = self.get_project_by_id(project_id)
        if not project:
            return None
            
        # Check cache first
        if project_id in self._full_context_cache:
            return self._full_context_cache[project_id]
            
        # Try to load from separate context file
        context_file = os.path.join(self.data_dir, f"{project_id}_context.json")
        if os.path.exists(context_file):
            try:
                with open(context_file, 'r', encoding='utf-8') as f:
                    context_data = json.load(f)
                    full_context = context_data.get('context_full', '')
                    self._full_context_cache[project_id] = full_context
                    return full_context
            except Exception as e:
                logger.error(f"Failed to load full context for {project_id}: {e}")
                
        # Fallback to project description if no full context
        return project.project_description
        
    def get_active_projects(self, limit: int = 5) -> List[ProjectContext]:
        """Get list of active projects (minimal info for prompts)."""
        active_projects = [
            proj for proj in self._project_cache.values()
            if proj.project_status == "active"
        ]
        
        # Sort by last_updated
        active_projects.sort(key=lambda p: p.last_updated, reverse=True)
        
        # Apply limit
        return active_projects[:limit]
        
    def search_projects(self, query: str) -> List[ProjectContext]:
        """Search projects by name, description, or tags."""
        query_lower = query.lower()
        matching_projects = []
        
        for project in self._project_cache.values():
            # Search in name
            if query_lower in project.project_name.lower():
                matching_projects.append(project)
                continue
                
            # Search in description
            if query_lower in project.project_description.lower():
                matching_projects.append(project)
                continue
                
            # Search in tags
            for tag in project.tags:
                if query_lower in tag.lower():
                    matching_projects.append(project)
                    break
                    
        return matching_projects
        
    def update_project_status(self, project_id: str, status: str) -> bool:
        """Update project status."""
        project = self.get_project_by_id(project_id)
        if project:
            project.project_status = status
            project.last_updated = datetime.now()
            self._save_project(project_id)
            logger.info(f"Updated project {project_id} status to: {status}")
            return True
        return False
        
    def update_project_context(
        self, 
        project_id: str, 
        context_summary: str = "",
        context_full: str = ""
    ) -> bool:
        """Update project context (summary and/or full context)."""
        project = self.get_project_by_id(project_id)
        if project:
            if context_summary:
                project.context_summary = context_summary
            if context_full:
                project.context_full = context_full
                self._full_context_cache[project_id] = context_full
                
            project.last_updated = datetime.now()
            self._save_project(project_id)
            
            # Save full context separately if provided
            if context_full:
                self._save_full_context(project_id, context_full)
                
            logger.info(f"Updated project {project_id} context")
            return True
        return False
        
    def delete_project(self, project_id: str) -> bool:
        """Delete a project."""
        if project_id in self._project_cache:
            del self._project_cache[project_id]
            
        if project_id in self._full_context_cache:
            del self._full_context_cache[project_id]
            
        # Remove from disk
        project_file = os.path.join(self.data_dir, f"{project_id}.json")
        context_file = os.path.join(self.data_dir, f"{project_id}_context.json")
        
        try:
            if os.path.exists(project_file):
                os.remove(project_file)
            if os.path.exists(context_file):
                os.remove(context_file)
                
            if project_id == self._current_project_id:
                self._current_project_id = None
                
            logger.info(f"Deleted project: {project_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete project {project_id}: {e}")
            return False
            
    def get_projects_summary(self) -> Dict[str, Any]:
        """Get summary of all projects."""
        total_projects = len(self._project_cache)
        active_projects = len([p for p in self._project_cache.values() if p.project_status == "active"])
        
        return {
            "total_projects": total_projects,
            "active_projects": active_projects,
            "current_project": self._current_project_id,
            "cached_projects": len(self._project_cache),
            "cached_contexts": len(self._full_context_cache)
        }
        
    def _save_project(self, project_id: str):
        """Save project metadata to disk."""
        project = self._project_cache.get(project_id)
        if not project:
            return
            
        project_file = os.path.join(self.data_dir, f"{project_id}.json")
        
        try:
            project_data = {
                "project_id": project.project_id,
                "project_name": project.project_name,
                "project_description": project.project_description,
                "project_status": project.project_status,
                "tags": project.tags,
                "context_summary": project.context_summary,
                "metadata": project.metadata,
                "last_updated": project.last_updated.isoformat(),
                "is_current": project.is_current
            }
            
            with open(project_file, 'w', encoding='utf-8') as f:
                json.dump(project_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to save project {project_id}: {e}")
            
    def _save_full_context(self, project_id: str, full_context: str):
        """Save full project context to separate file."""
        context_file = os.path.join(self.data_dir, f"{project_id}_context.json")
        
        try:
            context_data = {
                "project_id": project_id,
                "context_full": full_context,
                "updated_at": datetime.now().isoformat()
            }
            
            with open(context_file, 'w', encoding='utf-8') as f:
                json.dump(context_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to save full context for {project_id}: {e}")
            
    def _load_project(self, project_id: str) -> Optional[ProjectContext]:
        """Load project from disk."""
        project_file = os.path.join(self.data_dir, f"{project_id}.json")
        
        if not os.path.exists(project_file):
            return None
            
        try:
            with open(project_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            project = ProjectContext(
                project_id=data["project_id"],
                project_name=data["project_name"],
                project_description=data["project_description"],
                project_status=data["project_status"],
                tags=data.get("tags", []),
                context_summary=data.get("context_summary", ""),
                metadata=data.get("metadata", {}),
                last_updated=datetime.fromisoformat(data["last_updated"]),
                is_current=data.get("is_current", False)
            )
            
            # Load full context if exists
            context_file = os.path.join(self.data_dir, f"{project_id}_context.json")
            if os.path.exists(context_file):
                try:
                    with open(context_file, 'r', encoding='utf-8') as f:
                        context_data = json.load(f)
                        project.context_full = context_data.get("context_full")
                        self._full_context_cache[project_id] = project.context_full
                except Exception as e:
                    logger.warning(f"Failed to load full context for {project_id}: {e}")
                    
            return project
            
        except Exception as e:
            logger.error(f"Failed to load project {project_id}: {e}")
            return None
            
    def cleanup_old_projects(self, days_old: int = 90):
        """Clean up old inactive projects."""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        projects_to_remove = []
        
        for project_id, project in self._project_cache.items():
            if (project.project_status in ["completed", "archived"] and 
                project.last_updated < cutoff_date):
                projects_to_remove.append(project_id)
                
        for project_id in projects_to_remove:
            self.delete_project(project_id)
            
        if projects_to_remove:
            logger.info(f"Cleaned up {len(projects_to_remove)} old projects")
