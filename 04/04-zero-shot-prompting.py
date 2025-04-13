from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


# 使用通义千问
llm = ChatTongyi(
    model_name="qwen-max",
    temperature=0
)

def create_chain(prompt_template):
    prompt = PromptTemplate.from_template(prompt_template)
    return prompt|llm

# 直接任务规范
direct_task_prompt = """将以下文本的情绪分类为积极，消极或中性。不要解释你的理由，只需提供分类即可。
文本:{text}
情绪：
"""
direct_task_chain = create_chain(direct_task_prompt)

texts = [
    "我非常喜欢这部电影！演员的表演非常精彩。",
    "对于一年中的这个时候来说，今天的天气相当典型。",
    "我对这家餐厅的服务感到失望。",
]

for text in texts:
    result = direct_task_chain.invoke({"text":text}).content
    print(f"文本:{text}")
    print(f"情绪:{result}")

# 格式规范
format_spec_prompt = """生成一篇{topic}的简短新闻文章，
按照以下格式组织你的回复：

标题：[文章的吸引人的标题]

引言：[总结要点的简短介绍段落]

正文：[提供更多细节的 2-3 个简短段落]

结论：[结论句或行动号召]"""

format_spec_chain = create_chain(format_spec_prompt)

topic = "哪吒2票房超过一百亿"
result =  format_spec_chain.invoke({"topic",topic}).content
print(result)

# 多步骤推理
multi_step_prompt = """分析以下文本的主要论点、支持证据和潜在反驳。
按以下步骤提供分析：
1. 主要论点：确定并陈述抓哟主张或论点。
2. 支持证据：列出用于支持主要论点的关键点或证据。
3. 潜在反驳：提出主要论点的可能反对意见或其他观点。

文本：{text}

分析：
"""

multi_step_chan = create_chain(multi_step_prompt)

text = """虽然电动汽车经常被吹捧为解决气候变化的方案，但它们对环境的影响并不像看上去那么简单。
生产电动汽车电池需要大量的采矿作业，这可能导致栖息地破坏和水污染。
此外，如果用于为这些车辆充电的电力来自化石燃料，总体碳足迹可能不会显著减少。
然而，随着可再生能源变得越来越普遍和电池技术的进步，电动汽车确实可以在应对气候变化方面发挥关键作用。"""

result = multi_step_chan.invoke({"text":text}).content
print(result)

# 比较分析
def compare_prompts(task,prompts_templates):
    """
    Compare different prompt templates for the same task.

    Args:
        task (str): The task description or input.
        prompt_templates (dict): A dictionary of prompt templates with their names as keys.
    """
    print(f"任务：{task}\n")
    for name,template in prompts_templates.items():
        chain = create_chain(template)
        result = chain.invoke({"task":task}).content
        print(f"{name} 提示结果：")
        print(result)
        print("\n"+"-"*50+"\n")

task = "简明阐述元宇宙的概念"
prompts_templates = {
    "基础":"解释{task}",
    "结构化":"""通过解决以下几点来解释{task}:
    1.定义
    2.主要特征
    3.实际应用
    4.对行业的潜在影响
    """
}       
compare_prompts(task,prompts_templates)