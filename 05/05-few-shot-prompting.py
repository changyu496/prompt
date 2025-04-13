from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


# 使用通义千问
llm = ChatTongyi(
    model_name="qwen-max",
    temperature=0
)

def few_shot_sentiment_classification(input_text):
    few_shot_prompt = PromptTemplate(
        input_variables=["input_text"],
        template="""
            将情绪分类为正面、负面或中性。
            示例:
            文本:我喜欢这个产品！它太棒了。
            情绪:正面

            文本:这部电影很糟糕，我讨厌它。
            情绪:负面

            文本:今天的天气还可以。
            情绪:中性

            现在，对以下内容进行分类，只返回正面、负面或中性，不需要理由：
            文本:{input_text}
            情绪:
""" 
    )

    chain = few_shot_prompt|llm
    result = chain.invoke({"input_text":input_text}).content
    
    # 整理结果
    result = result.strip()

    if ":" in result:    
        result = result.split(":")[1].strip()
    
    return result

def multi_task_few_shot(input_text,task):
    few_shot_prompt = PromptTemplate(
        input_variables=["input_text","task"],
        template="""
对给定的文本执行指定的任务,只返回结果，不需要解释理由。

        示例：
        文本：我喜欢这个产品！它太棒了。
        任务：情绪
        结果：积极

        文本：你好，你好吗？
        任务：语言
        结果：中文

        现在，执行以下任务：
        文本：{input_text}
        任务：{task}
        结果：
"""
    )
    chain = few_shot_prompt|llm
    return chain.invoke({"input_text":input_text,"task":task}).content

test_text = "这家餐厅这么棒！你敢信？"
result = few_shot_sentiment_classification(test_text)
print(f"输入:{test_text}")
print(f"预测情绪:{result}")

print(multi_task_few_shot("我真不敢相信这有多棒！", "情绪"))
print(multi_task_few_shot("I can't believe how great this is!", "语言"))