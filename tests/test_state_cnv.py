# test_state_cnv.py

import pytest
import statescnv.state_cnv 
import inspect

def get_callable_functions(module):
    """
    Return a list of (name, function) pairs for all user-defined
    callable functions in the module.
    """
    return [
        (name, func)
        for name, func in inspect.getmembers(module, inspect.isfunction)
        if inspect.getmodule(func) == module
    ]

@pytest.mark.parametrize("func_name, func", get_callable_functions(state_cnv))
def test_function_runs(func_name, func):
    """
    Smoke test: just check that each function runs when called with dummy arguments if possible.
    If function requires args, it is skipped unless a default or dummy-safe call is possible.
    """
    try:
        sig = inspect.signature(func)
        params = sig.parameters

        # Try calling with no arguments if all have defaults
        if all(p.default != inspect.Parameter.empty or p.kind == inspect.Parameter.VAR_POSITIONAL or p.kind == inspect.Parameter.VAR_KEYWORD
               for p in params.values()):
            func()  # call with no args
        else:
            pytest.skip(f"Skipping {func_name} because it requires arguments")
    except Exception as e:
        pytest.fail(f"Function {func_name} raised an exception: {e}")

