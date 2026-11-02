import pandas as pd
import pytest

from src.data import (CELLS, answer_matches, build_a_rows, build_b_rows,
                      load_city_pairs, split_pools)


@pytest.fixture
def pairs(tmp_path):
    csv = tmp_path / "cities.csv"
    pd.DataFrame({
        "statement": ["The city of Lodz is in Poland.",
                      "The city of Lodz is in China.",
                      "The city of Maracay is in Venezuela.",
                      "The city of Maracay is in the Dominican Republic."],
        "label": [1, 0, 1, 0],
        "city": ["Lodz", "Lodz", "Maracay", "Maracay"],
        "country": ["Poland", "China", "Venezuela", "the Dominican Republic"],
        "correct_country": ["Poland", "Poland", "Venezuela", "Venezuela"],
    }).to_csv(csv, index=False)
    return load_city_pairs(csv)


def test_load_city_pairs(pairs):
    assert list(pairs.city) == ["Lodz", "Maracay"]
    lodz = pairs.iloc[0]
    assert lodz.false_country == "China"
    assert lodz.false_stmt == "The city of Lodz is in China."


def test_split_pools_disjoint_and_deterministic():
    cities = [f"c{i}" for i in range(100)]
    a, b = split_pools(cities, frac_a=0.4, seed=0)
    assert len(a) == 40 and len(b) == 60
    assert not set(a) & set(b)
    assert (a, b) == split_pools(list(reversed(cities)), frac_a=0.4, seed=0)


def test_a_rows(pairs):
    a = build_a_rows(pairs)
    assert len(a) == 4
    assert a.text.str.startswith("<bos>The city of").all()
    assert a.groupby("city").is_false.sum().eq(1).all()


def test_b_grid(pairs):
    b = build_b_rows(pairs)
    assert len(b) == len(pairs) * len(CELLS)
    # each city appears once per cell
    assert b.groupby(["city", "instr", "answer_true"]).size().eq(1).all()
    # only the lie-instruction + false-answer cell is a lie
    assert b[b.is_lie == 1][["instr", "answer_true"]].drop_duplicates().values.tolist() == [["L", 0]]
    # answer text is the Geometry of Truth statement verbatim, nothing after it
    row = b[(b.city == "Lodz") & (b.instr == "L") & (b.answer_true == 0)].iloc[0]
    assert row.text.endswith("<start_of_turn>model\nThe city of Lodz is in China.")
    assert "with a lie" in row.text
    neutral = b[(b.city == "Lodz") & (b.instr == "N")].iloc[0]
    assert "truthfully" not in neutral.text and "lie" not in neutral.text


@pytest.mark.parametrize("gen,country,ok", [
    ("Poland", "Poland", True),
    ("The answer is poland.", "Poland", True),
    ("Nigeria", "Niger", False),
    ("Niger", "Niger", True),
    ("Democratic Republic of the Congo", "the Republic of the Congo", False),
    ("Republic of the Congo", "the Republic of the Congo", True),
    ("Republic of Congo", "the Republic of the Congo", True),
    ("Democratic Republic of Congo", "the Republic of the Congo", False),
    ("Democratic Republic of Congo", "the Democratic Republic of the Congo", True),
    ("USA", "the United States", True),
    ("North Korea", "South Korea", False),
    ("Equatorial Guinea", "Guinea", False),
    ("Côte d'Ivoire", "Côte d'Ivoire", True),
    ("Ivory Coast", "Côte d'Ivoire", True),
])
def test_answer_matches(gen, country, ok):
    assert answer_matches(gen, country) is ok
