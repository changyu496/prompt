from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import PromptTemplate
from IPython.display import display, Markdown
from dotenv import load_dotenv

load_dotenv()


# 使用通义千问
llm = ChatTongyi(
    model_name="qwen-max",
    temperature=0
)
# 基本思维链提示
standard_prompt = PromptTemplate(
    input_variables=["question"],
    template="简洁的回答以下问题：{question}",
)

cot_prompt = PromptTemplate(
    input_variables=["question"],
    template="简明扼要的逐步回答以下问题：{question}",
)

standard_chain = standard_prompt|llm
cot_chain = cot_prompt|llm

# 示例问题
question = "如果一列火车在 2 小时内行驶 120 公里，那么它的平均速度是多是公里/小时？"

standard_response = standard_chain.invoke(question).content
cot_response = cot_chain.invoke(question).content

print("标准回复")
print(standard_response)
print("\n"+"="*50+"\n")
print("思维链回复:")
print(cot_response)

advanced_cot_prompt = PromptTemplate(
    input_variables = ["question"],
    template="""逐步解决以下问题。对于每个步骤：
1. 说明你要计算的内容
2. 写下你将使用的公式（如果适用）
3. 执行计算
4. 解释结果

问题：{question}

解决方案："""
)
advanced_cot_chain = advanced_cot_prompt|llm
complex_question = "一辆汽车以 60 公里/小时的速度行驶 150 公里，然后以 50 公里/小时的速度行驶 100 公里。整个行程的平均速度是多少？"

advanced_cot_response = advanced_cot_chain.invoke(complex_question).content
standard_response = standard_chain.invoke(complex_question).content
print(standard_response)
print(advanced_cot_response)


# 更复杂的问题
challenging_question = """
一个圆柱形水箱，半径为 1.5 米，高为 4 米，水箱已装满 2/3。
如果以每分钟 10 升的速度加水，水箱需要多长时间才能溢出？
以小时和分钟为单位给出答案，四舍五入到最接近的分钟。
（π 使用 3.14159，1000 升 = 1 立方米）"""

standard_response = standard_chain.invoke(challenging_question).content
cot_response = advanced_cot_chain.invoke(challenging_question).content

print("标准回复:")
print(standard_response)
print("\n" + "=" * 50 + "\n")
print("\n思维链回复:")
print(cot_response)


logical_reasoning_prompt = PromptTemplate(
    input_variables=["scenario"],
    template="""彻底分析以下逻辑谜题。在分析中遵循以下步骤：

列出事实：

清楚地总结所有给定的信息和陈述。
识别所涉及的所有角色或元素。
识别可能的角色或条件：

确定适用于角色或元素的所有可能的角色、行为或状态（例如，说真话的人、撒谎者、两种都有）。
注意约束：

概述谜题中指定的任何规则、约束或关系。
生成可能的场景：

系统地考虑角色或元素的所有可能的角色或条件组合。
确保所有排列都得到考虑。
测试每个场景：

对于每个可能的场景：
假设你分配的角色或条件。
根据这些假设分析每个语句。
检查场景中的一致性或矛盾性。
消除不一致的场景：

丢弃任何导致矛盾或违反约束的场景。
跟踪排除每个场景的理由。
总结解决方案：

确定测试后保持一致的场景。
总结调查结果。
提供明确的答案：

明确说明每个角色或元素的作用或条件。
根据您的分析，解释为什么这是唯一可能的解决方案。
场景：

{scenario}

分析：""",
)

logical_reasoning_chain = logical_reasoning_prompt | llm

logical_puzzle = """房间里有三个人：艾米、鲍勃和查理。
其中一个人总是说真话，另一个人总是撒谎，还有一个人说真话和撒谎。
艾米说：“鲍勃是个骗子。”
鲍勃说：“查理说真话和撒谎。”
查理说：“艾米和我都是骗子。”
确定每个人的本性（说真话的人、撒谎的人或说假话的人）。"""

logical_reasoning_response = logical_reasoning_chain.invoke(logical_puzzle).content
print(logical_reasoning_response)