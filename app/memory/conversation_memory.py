class ConversationMemory:
    def __init__(self, max_history=5):
        self.max_history = max_history
        self.history = []

    def add_user_message(self, message: str):
        self.history.append({
            "role": "user",
            "content": message,
        })
        self._trim()

    def add_assistant_message(self, message: str):
        self.history.append({
            "role": "assistant",
            "content": message,
        })
        self._trim()

    def get_history(self):
        return self.history

    def clear(self):
        self.history = []

    def _trim(self):
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-self.max_history * 2:]