from dotenv  import load_dotenv
from openai import OpenAI
import os
load_dotenv()

class LanguageModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.api_key =os.getenv("API-KEY")
        self.llm = OpenAI(api_key=self.api_key,base_url="https://api.deepseek.com")
    def generate_text(self, prompt):

        response =self.llm.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content":prompt},
                    ],
                        stream=True
                                        )   
        return response.choices[0].message.content
        
        
if  __name__ =="__main__"   :    
    agent =LanguageModel('agent')
    print(agent.generate_text("你可以干什么"))

