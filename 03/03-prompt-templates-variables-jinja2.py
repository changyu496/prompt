from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# 使用通义千问
llm = ChatTongyi(
    model_name="qwen-max",
    temperature=0
)

# 提示词模版配合一个变量
simple_prompt = PromptTemplate(
    template="对{topic}做出简要说明",input_variables=["topic"]
)

prompt = simple_prompt.format(topic="光合作用")
response = llm.invoke(prompt)

print(f"提示:\n{prompt}")
print("\n"+"="*100+"\n")
print(f"回复:\n{response.content}")

# 更多的键值对
complex_template = PromptTemplate(
    template="简明扼要的向{audience}受众解释{field}领域中的{concept}概念",
    input_variables=["concept","field","audience"]
)
prompt = complex_template.format(
    concept="梯度",field="人工智能",audience="初学者"
)
response = llm.invoke(prompt)
print(f"提示:\n{prompt}")
print("\n"+"="*100+"\n")
print(f"回复:\n{response.content}")