import requests
from mcp.server.fastmcp import FastMCP
from typing import Dict, Union
import asyncio
# 创建MCP服务器实例
mcp = FastMCP("WeatherQueryServer")

@mcp.tool()
async def get_weather(city):
    url =f"http://api.tangdouz.com/tq.php?dz={city}"
    response =requests.get(url)
    return response.text.replace(r"\r","\n")



if __name__ == "__main__":
   
    print("天气查询服务正在启动，等待连接...")
    mcp.run(transport='stdio')
