import pytest
from lllm.core.models import Prompt, Function

def test_prompt_initialization():
    p = Prompt(path="test", prompt="Hello {name}")
    assert p(name="World") == "Hello World"

def test_prompt_with_functions():
    f = Function(name="test_func", description="desc", properties={})
    p = Prompt(path="test_func", prompt="Hi", function_list=[f])
    assert "test_func" in p.functions
    assert p.functions["test_func"] == f
