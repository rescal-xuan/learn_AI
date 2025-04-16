from llm.base_model import LanguageModel
from memory.memory import Memory
from tool.tool  import Tool
from callback.callback  import Callback
from rag.rag import RAG
from agent.simple_agent  import SimpleAgent

class SimpleAgent:
    def __init__(self, name):
        self.name = name
        self.language_model = LanguageModel("DeepSeek -R1")
        self.memory = Memory()
        self.tool = Tool("Calculator")
        self.callback = Callback("OnComplete")

    def greet(self):
        print(f"我是{self.name},正在生成回答,请稍后……")

    def add(self, a, b):
        result = self.tool.use()
        return a + b

    def generate_response(self, prompt):
        response = self.language_model.generate_text(prompt)
        self.memory.remember(response)
        return response

    def execute_callback(self):
        self.callback.execute()

if __name__ == "__main__":
    agent = SimpleAgent("Agent007")
    agent.greet()
    result = agent.add(5, 3)
    print(f"The result of addition is: {result}")
    response = agent.generate_response("What is AI?")
    print(f"Generated response: {response}")
    agent.execute_callback()