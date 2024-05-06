from huggingface_hub import login, notebook_login

class HuggingFaceNotebookLogin:
    def __init__(self, toekn ):
        self.token = toekn

    def login(self):
        print("Logging in...")
        login(self.token)
        #notebook_login()