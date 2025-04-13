from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# 使用通义千问模型
llm = ChatTongyi(
    model_name="qwen-max",  # 可选模型：qwen-turbo, qwen-plus, qwen-max等
    temperature=0
)

# 单轮提示
single_turn_prompt = "三原色是哪三种颜色"
print(llm.invoke(single_turn_prompt).content)   

# 结构化
structed_prompt = PromptTemplate(
    input_variables=["topic"],
    template="简要说明{topic}并列出三个主要组成部分"
)
chain = structed_prompt|llm
print(chain.invoke({"topic","颜色的原理"}).content)