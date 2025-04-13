from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START,MessagesState,StateGraph
from langchain_community.chat_models.tongyi import ChatTongyi
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage


load_dotenv()

# 使用通义千问模型
llm = ChatTongyi(
    model_name="qwen-max",  # 可选模型：qwen-turbo, qwen-plus, qwen-max等
    temperature=0
)

# 定义Graph
workflow = StateGraph(state_schema=MessagesState)

# 定义模型函数调用
def call_model(state:MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages":response}

# 定义节点
workflow.add_edge(START,"model")
workflow.add_node("model",call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

config = {"configurable":{"thread_id":"abc123"}}
multi_turn_prompts = [
    "你好，我正在学习太空知识，你能告诉有关行星的知识吗？",
    "我们的太阳系中最大的行星是什么？",
    "与地球相比，它有多大",
]

for prompt in multi_turn_prompts:
    input_message = [HumanMessage(prompt)]
    output = app.invoke({"messages":input_message},config)
    output["messages"][-1].pretty_print()


