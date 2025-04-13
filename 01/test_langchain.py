from langchain.llms import OpenAI

# 替换为你的OpenAI API密钥
llm = OpenAI(openai_api_key="sk-bf5629c558e44b3cbb2329b0cc30d143")

response = llm("请用一句话解释人工智能是什么?")
print(response)