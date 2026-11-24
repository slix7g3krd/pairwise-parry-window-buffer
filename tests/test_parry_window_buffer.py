import pytest
import gameplay.parry_window_buffer as mod

CASES = [
    ("window_by_event_ts", (1,)),
    ("reorder_input", (1,)),
    ("unbind_on_swap", (1,)),
    ("timer_round_tag", (1,)),
    ("rollback_window", (1,)),
    ("on_input", (1,)),
    ("on_anim_notify", (1,)),
    ("open_window", (1,)),
    ("parry", (1,)),
    ("confirm", (1,)),
    ("validate_window", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
