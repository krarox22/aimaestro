from langchain_core.messages import AIMessage
from langgraph.graph import END

from aimaestro.graph import route_message
from tests.conftest import update_memory_call


def test_no_tool_calls_ends():
    state = {"messages": [AIMessage(content="hello", tool_calls=[])]}
    assert route_message(state, None, None) == END


def test_user_routes_to_profile():
    state = {"messages": [update_memory_call("user")]}
    assert route_message(state, None, None) == "update_profile"


def test_todo_routes_to_todos():
    state = {"messages": [update_memory_call("todo")]}
    assert route_message(state, None, None) == "update_todos"


def test_instructions_routes_to_instructions():
    state = {"messages": [update_memory_call("instructions")]}
    assert route_message(state, None, None) == "update_instructions"


def test_unknown_update_type_ends_instead_of_raising():
    """Regression: the original raised ValueError, which killed the chat session
    on a single malformed tool argument."""
    state = {"messages": [update_memory_call("nonsense")]}
    assert route_message(state, None, None) == END
