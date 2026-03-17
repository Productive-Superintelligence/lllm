"""Chat prompts for pkg_alpha — simple and template-variable prompts."""
from lllm.core.prompt import Prompt

greet = Prompt(
    path="greet",
    prompt="Hello! How can I help you today?",
    metadata={"category": "greeting", "version": "1.0"},
)

farewell = Prompt(
    path="farewell",
    prompt="Goodbye! It was great chatting with you.",
    metadata={"category": "greeting", "version": "1.0"},
)

ask_topic = Prompt(
    path="ask_topic",
    prompt="What is your question about {topic}? Please be specific.",
    metadata={"requires": ["topic"]},
)

multi_var = Prompt(
    path="multi_var",
    prompt="Discuss {subject} from the perspective of {viewpoint} in {language}.",
    metadata={"requires": ["subject", "viewpoint", "language"]},
)

# Prompt with double-brace literal (should NOT be treated as a variable)
literal_braces = Prompt(
    path="literal_braces",
    prompt="Use {{double braces}} to escape. The value is: {value}.",
    metadata={"note": "tests escaped braces"},
)
