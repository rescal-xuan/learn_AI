import asyncio
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from browser_use import Agent

# dotenv
load_dotenv()

api_key = os.getenv('DEEPSEEK_API_KEY', 'sk-7deefd6776fd4550bd82c5d4b814eb7f')
if not api_key:
	raise ValueError('DEEPSEEK_API_KEY is not set')


async def run_search():
	agent = Agent(
		task=(
			
			"1. 在搜索框中输入抖音并搜索"
			'2. 点击搜索结果中的第一个链接'
			'3. 关闭扫码登录'	
            '3. 返回第一个视频的内容'
   
		),
		llm=ChatOpenAI(
			base_url='https://api.deepseek.com/v1',
			model='deepseek-chat',
			api_key=SecretStr(api_key),
		),
		use_vision=False,
	)

	await agent.run()


if __name__ == '__main__':
	asyncio.run(run_search())
