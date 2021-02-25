import pytest

from evalutils.cli import AbbreviatedChoice


def test_abbreviated_choice_not_unique():
    with pytest.raises(ValueError):
        AbbreviatedChoice(choices=["Same", "same", "other"])


@pytest.mark.parametrize(
    "value, expected",
    (
        ("s", "same"),
        ("S", "same"),
        ("Same", "same"),
        ("same", "same"),
        ("other", "oTHER"),
        ("O", "oTHER"),
        ("o", "oTHER"),
        ("OtHER", "oTHER"),
    ),
)
def test_abbreviated_choice_convert(value: str, expected: str):
    ac = AbbreviatedChoice(choices=["same", "oTHER"])
    assert ac.convert(value=value, param=None, ctx=None) == expected
