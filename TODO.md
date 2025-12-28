## Current:
- [ ]  fetch conversation`s messages  on the same topic
        - load corresponding related files, docs adn so on.
## Next
- add Email integration, new email - asitant tell about it(account, sender, summary)
- [ ] implement projects and project context fro prompts
- [ ] for this project prompts context, make a gui overview and mangment. SO user will be able to seee and manage projects and their content.



## Maybe tasks

## DONE 
- [x] fix tool`s use cases
- [x] FIx message dupication with image galleries or when web_search tools executed.
UPD: errors appears while we execute web_search tools
- [x] Fix image rendering in web_search responses - implementing automatic gallery system

- [x] store conversations on the same topic

- [x] Topic detection, now works well 
- [x] Fix problem with drafts - when we load draft adn send it successfully, we need to mark it as sended, nad dont insert it on app restart.
- [x] Fix problem: "Streaming error: Maximum tool iterations (3) reached. Please check your API configuration." happens when asistant use tools for topic managment and web_search, we should adjust limits, maybe fix problem whihc occurs it. Im think we can got problem cos main chat model try to use topic tools, but it should not do it, cos it task for Topic analizer/tracker model.
We can try to change main model prompt, im think proble in it.
Fixed by corret toolset managment.
- [x] Fix problem with intent detecto - looks like we have duplicated functional again, with LLM_adapter/EnchancedLLmadapter, maybe also for intent analysys methods to.
Im checked logs and see that 
        logger.info(f"Using LLM provider: {self.llm_adapter.provider}, model: {self.llm_adapter.config.get('model', 'unknown')}")
this string was not called during intent analysis, even it placed in intent_analyzer.py in method     def _analyze_with_llm_fewshot(self, user_message: str, conversation_context: List[Dict[str, Any]] = None) -> UserIntent:
SO lloks like we again have dplication of functional.
- [x] topic tracking - make middleweare for detecting topic of current messages, based on topic we ned to save mesages in corresponding table/add crresnding field with topic. For example, user ask for programming question, than about military equipment - so we can clearly see two diferent topics, and csave tham to corresnding toopics.
Same for projects , if we ee that conversation flows into project related topic, we save it as prokect conversation.
conversation in tis case - few continus mesages fro muser and assitant wihc corelates to same topic.
if topic changes or conversation eded, we save its messages to corresponding topics/mark tahm.

- [x] provider/model combos. We need abiity to create such cobo from avialable providers and theiir modes.
add corresponding functioanl and ui in hte Settings tab, so we can create new/edit/delete combos.
for example combo_id_1: provider: openrouter, model: openai/gpt4o, and so on.
this combos e can later use as models for diferent usecases instead of entering it for each case separately.
For ex: Use case: main chat, we can set ccombo we want, and dont need to select seaparate provider/model.

- [x] Fix combo tab implementation - we ned to use created combos for usecases.
- [x] Fix combo tab implementation - it broken UI, not fit into tab

- [x] Fix combo tab implementation - its not displying in Settigns menu!

- [x] maybe change web retriver for better one, cos current returns too much non usefull text
- [x] fix memori sql errors
 - [x] for Shift-Enter combo make inserting new line not sending message
- [x] integrate memori lib tool to existing structured prompt buildings.
- [x] restore support of memori momory tool for persistant memory.
- [x] make beautifull markdown rendering in our chat.

- [x] make better response handling, now looks like we dont stream response, cos we add an animation for this process
we need to fix 
-- [x] streaming process
- [x] FIx audi playback problem - when we finish playing TTSed answer, app can crash or freeze.
We need to solve this problem(use gemini)
-- [x] animation process
- [x] better prompt structuring - divide with XML tags, separate all blocks sys prompt, user query, message history and so on.
- [x] Fix TTS problems

- [x] organize conversation history, not only put all messages into prompt (in combination with memory/memori)
here we need to change concept, nowo we have  a one conv for all needs, it is good.
But ew need tocahnge al little this system, now we just put all mesage history in request, but we need to make smth like ** topic based context genetration: **
Agent have all mesage saved in DB, also memory system provided by memori.
To keep clear one conversation we need to control what we aps into prompt/context.
for example user ask for latest news about ai, asistant respones, than like pafter one hur user come and tell: "hm waht current situation in Sudan, how conflict going?" os at this moment wwe dont need a info from "latest news about ai" user`s qury, so we should not pas it into context. im think we can make smth like that:
user make question/we hve conversation on specific topic, this messages saves into DB, and if user ask another question not related to this topic, we mark this topic as finished/paused, and clear context for new, this paused topic messages saved in DB, marks smth like topic_name_id_date or smth liek this. Whr nuser ask question/tat convrsation, we can search for topics related if any and just load corersponding message from DB. All topicss ids, with short description can be pased to LLM context, so it can deceide when to load specific topic, or we can just make vector search call, and load related to user`s query topic/s. ye it can be similar to waht memori do, but i think such approach is more effective.
For example when user ask smth about sepcific project(for example PMC in Sahel ergion), we can load corresponding topics from our DB, with all mesage we have in it. so user will receive specialized mesage history on requested topic.
isnt it just conversation/sesion system??? - maybe but if it, our approach allow to do all context managment automaticly? maybe
Also we can manage projects using this system, and load corespnding files/docs related to specific topic/project.
so conversation will be focused on current project while user just have single conversation flow.
Hm in such system we maybe not need a memori? or it can be integrated too. 
Just in separate block <memories/> for example.
SO we will have comprehencive and structured prompt, smth like:
<SystemPrompt>
<SystemInfo>
<UserInfo/Personalizzation/>
<RelatedMemories>
<UserProjects>
for each project
<ProjectName/ID/>
<ProjectDescription/>- name/desc used to find related project/s to current conversation
<ProjectCurrent/> - if working n current project?
<ProjectContextFull> - if we work on this project
<UserQuery>
<MesageHistory>  - maybe recent 10-20
<AdditionalContextRAG> - ftched from VectorDB/websearch

Idea: simple personal assistant for personal usag. PyQt6, Langchain, -based.
We need:
- usual chat interface for comunication wit LLMs(mutiple support using OpenRouter)
- vector database integration with langchain retriver(for chat with files feature)
- text to speech for replies(using Diferent providers(OpenAI by default, can be selected in settigns))
- voice transcription using diferent providers(same as in tts)
- memory system (longterm/episodical/and os on, langchain based) preffered - https://github.com/GibsonAI/memori with langchain integration
- single chat conversation - not separate chats, only one flow of conversation with assistant.
- advance prompt system, we can customize system prompt, integrate into it memories from memori, add predefined prompts and so on.
Imoprtant, all AI interaction made with Langchain, not using openai or other providers libs/sdks, langchain aready have it builin(like langchain-openai module)

## NOTES
### Very Fast mdoels for middleware taks:
- google/gemini-2.5-flash-lite: google-ai-studio latency 0,58s 105 tps 
- google/gemini-2.5-flash : google-vertex/global. 0,47 115tps
- minimax/minimax-m2 deepinfra/fp8 0,26, 88,4
- meta-llama/llama-3.2-1b-instruct 0,3 358tps
meta-llama/llama-3.2-3b-instruct cloudflare 0,24 212tps
- [PREFERED for INENT] meta-llama/llama-4-scout groq  0,12 1000tps
- openai/gpt-oss-20b groq 0,2 1 646tps
- openai/gpt-oss-120b 0,11s 572,7tps

### list of models to use
- qwen/qwen3-8b
- qwen/qwen3-next-80b-a3b-thinking
