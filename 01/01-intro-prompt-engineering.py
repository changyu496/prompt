from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# 使用通义千问模型
llm = ChatTongyi(
    model_name="qwen-max",  # 可选模型：qwen-turbo, qwen-plus, qwen-max等
    temperature=0
)

# 简单的例子入门
basic_prompt = "用一句话解释提示工程的概念。"
print(llm.invoke(basic_prompt).content)

# 更结构化的提示得到更详细的回应

structed_prompt = PromptTemplate(
    input_variables=["topic"],
    template="提供{topic}的定义，解释其重要性，并列出三个主要优点。",
)

chain = structed_prompt | llm # 管道符的应用
input_variables = {"topic","提示工程"}
output = chain.invoke(input_variables).content
print(output)

# 克服语言模型的局限性

fact_check_prompt = PromptTemplate(
    input_variables=["statement"],
    template= """评估以下陈述的事实准确性，如果不正确，请提供正确的信息；
    陈述:{statement}
    评估:
    """,
)
chain = fact_check_prompt|llm
print(chain.invoke("法国的首都是伦敦").content)