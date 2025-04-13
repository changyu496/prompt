from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from jinja2 import Template

load_dotenv()

# 使用通义千问
llm = ChatTongyi(
    model_name="qwen-max",
    temperature=0
)

class Jinja2PromptTemplate:
    """A class to represent a template for generating prompts with variables
    Attributes:
        template (str): The template string with variables
        input_variables (list): A list of the variable names in the template
    """
    def __init__(self,template,input_variables):
        self.template = Template(template)
        self.input_variables = input_variables
    def format(self,**kwargs):
        return self.template.render(**kwargs)
    
conditional_template = Jinja2PromptTemplate(
    template="我叫 {{ name }}，今年 {{ age }} 岁。"
    "{% if profession %}我现在的职业是{{ profession }}.{% else %}我现在还没有工作.{% endif %} "
    "你能根据这些信息给我一些职业建议吗？请简明扼要地回答。",
    input_variables=["name","age","profession"],)
prompt = conditional_template.format(name="常雨",age=39,profession="程序员")
response = llm.invoke(prompt)
print(f"提示:\n{prompt}")
print("\n"+"="*100+"\n")
print(f"回复:\n{response.content}")

# 模版中使用列表
list_template = PromptTemplate(
    template="将这些项目分门别类：{items}。提供类别以及每个类别中的项目。",
    input_variables=["items"]
)
prompt = list_template.format(items="苹果，香蕉，胡萝卜，锤子，螺丝刀，钳子，小说，教科书，杂志")
response = llm.invoke(prompt)
print(f"提示:\n{prompt}")
print("\n"+"="*100+"\n")
print(f"回复:\n{response.content}")

# 模版中使用循环