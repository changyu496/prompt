from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import PromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START,MessagesState,StateGraph
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

# 使用通义千问模型
llm = ChatTongyi(
    model_name="qwen-max",  # 可选模型：qwen-turbo, qwen-plus, qwen-max等
    temperature=0
)

prompts = ["日本首都在哪里？","它有多少人口","这个城市最著名的建筑是什么"]

print("单轮回复")

for prompt in prompts:
    print(f"Q:{prompt}")
    print(f"A:{llm.invoke(prompt).content}\n")



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
print("多轮回复")
config = {"configurable":{"thread_id":"abc456"}}
for prompt in prompts:
    print(f"Q:{prompt}")
    output = app.invoke({"messages":[HumanMessage(prompt)]},config)
    print(f"A:{output['messages'][-1].content}\n")