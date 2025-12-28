HEre i will store all info about gug fixing cases, what problem was and what soluion we found for it.
- BUG: Main chat wil not got a orret toolset for model. It got only topic-related tools but not web search.
-- FIX: Problem was in not corret check of availability of main chat toolsset in enchanced_llm_adapter.py, row 139:
Correct code:
===
current_tool_set = self.tool_set_manager.get_tool_set("main_chat")
        if  current_tool_set is None:
            logger.error("No main_chat tool set available")
            return
===
WAS:
===
current_tool_set = self.tool_set_manager.get_tool_set("main_chat")
        if  not current_tool_set:
            logger.error("No main_chat tool set available")
            return
===
