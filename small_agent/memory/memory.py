class Memory:
    def __init__(self):
        self.storage = []

    def remember(self, item):
        self.storage.append(item)

    def recall(self):
        return self.storage