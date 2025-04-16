class Callback:
    def __init__(self, callback_name):
        self.callback_name = callback_name

    def execute(self):
        print(f"Executing callback: {self.callback_name}")