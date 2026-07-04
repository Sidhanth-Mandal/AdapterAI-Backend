from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
import os 

# -----------------------------
# Main orchestrator config 
#______________________________



GROQ_MODEL = 'openai/gpt-oss-120b'
groq_llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0.0,
        max_tokens=2048,
        api_key=os.environ["GROQ_API_KEY"],
    )

CLAUDE_MODEL = 'claude-sonnet-4-6'
claude_llm = ChatAnthropic(
        model_name= CLAUDE_MODEL ,
        max_tokens=4096 ,
        api_key=os.environ["ANTHROPIC_API_KEY"])

#=======================================
# Change here

MAX_TOOL_ITERATIONS = 10
main_llm = groq_llm
_MAIN_MODEL = GROQ_MODEL
_MAX_TOOL_RESULT_CHARS = 3000  # Max chars kept from a tool result before truncating (prevents 413 errors)



# ------------------------------
# Custom Tool Subagent Model
# ______________________________


CustomToolSubagentMODEL = "openai/gpt-oss-120b"  #"claude-haiku-4-5"

groq_llm_custom_tool = ChatGroq(
    model=CustomToolSubagentMODEL,
    temperature=0.0,
    max_tokens=4096,
    api_key=os.environ["GROQ_API_KEY"],
)

claude_llm_custom_tool = ChatAnthropic(
    model = CustomToolSubagentMODEL,
    temperature=0.0,
    max_tokens=4096,
    api_key=os.environ["ANTHROPIC_API_KEY"]
)

#=======================================
# Change here

MAX_CUSTOM_TOOL_ITERATIONS = 6 # Maximum agentic iterations to prevent infinite loops
custom_tool_llm = groq_llm_custom_tool




# Template Creator CHATBOT Model

groq_template_creator_chatbot_llm  = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.7,
        max_tokens=1024,
        streaming=False,
        api_key=os.environ["GROQ_API_KEY"])

claude_template_creator_chatbot_llm =ChatAnthropic(
        model = 'claude-sonnet-4-6',
        temperature= 0.7,
        max_tokens =1024,
        streaming= False,
        api_key=os.environ["ANTHROPIC_API_KEY"])

Template_Chat_llm = groq_template_creator_chatbot_llm

# Template Creator Planner Model

claude_planner_llm = ChatAnthropic(
        model="claude-haiku-4-5",
        temperature=0.4,
        max_tokens=8096,
        streaming=False,
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )

# Tool Generator Model
CLAUDE_TOOL_GEN_MODEL_NAME = "claude-sonnet-4-6"
CLAUDE_TOOL_GEN_LLM = ChatAnthropic(model=CLAUDE_TOOL_GEN_MODEL_NAME)