class Tool:
    def __init__(self, tool_name):
        self.tool_name = tool_name

    def use(self):
        print(f"Using tool: {self.tool_name}")