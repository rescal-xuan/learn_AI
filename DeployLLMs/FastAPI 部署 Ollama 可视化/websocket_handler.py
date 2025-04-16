import ollama
from fastapi import WebSocket
from openai import OpenAI
import os


async def get_input(websocket:WebSocket):
    await websocket.accept()  # 接受WebSocket连接
    user_input = await websocket.receive_text()
    return user_input
async def websocket_endpoint(websocket: WebSocket,model):
    if model =='API':
        
        stream =chat_model(WebSocket)
    else :
        stream=chat_ollama()
    try:
        for chunk in stream:  # 遍历流式传输的结果
            model_output = chunk['message']['content']  # 获取模型输出的内容
            await websocket.send_text(model_output)  # 通过WebSocket发送模型输出的内容
    except Exception as e:  # 捕获异常
        await websocket.send_text(f"Error: {e}")  # 通过WebSocket发送错误信息
    finally:
        await websocket.close()  # 关闭WebSocket连接

    
async def  chat_model(websocket: WebSocket):
    client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),  # This is the default and can be omitted
)
    model='deepseek-chat',  
    openai_api_base='https://api.deepseek.com',
    await websocket.accept()  # 接受WebSocket连接
    user_input = await websocket.receive_text()  # 接收用户输入的文本消息
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content":  user_input ,
            }
        ],
        # model="gpt-4o",
        stream=True  # 启用流式传输
    ) 
    
    
    return chat_completion
    

def chat_ollama(): 
    user_input =get_input(websocket=WebSocket)
    stream =ollama.chat(
        
         messages=[
            {
                "role": "user",
                "content":  user_input ,
            }
        ],
        model='llama3.1',
        stream=True  # 启用流式传输
    )
    return stream