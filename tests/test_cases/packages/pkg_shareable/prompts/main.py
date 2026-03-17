from lllm import Prompt

researcher = Prompt(path="researcher", prompt="You are a researcher. Analyze {topic}.")
writer = Prompt(path="writer", prompt="You are a writer. Write about {topic}.")
