"""Tests for adaptive_classifier core components."""

from adaptive_classifier.taxonomy import Taxonomy
from adaptive_classifier.normalizer import create_normalizer
from adaptive_classifier.types import Classification, ClassificationSource


def test_taxonomy_from_dict():
    t = Taxonomy.from_dict({
        "Food": {
            "Burgers": ["Hamburger", "Cheeseburger"],
            "Pizza": ["Pepperoni", "Margherita"],
        },
        "Drinks": ["Coffee", "Tea"],
    })

    paths = t.leaf_paths
    assert "Food > Burgers > Hamburger" in paths
    assert "Food > Burgers > Cheeseburger" in paths
    assert "Food > Pizza > Pepperoni" in paths
    assert "Drinks > Coffee" in paths
    assert len(paths) == 6


def test_taxonomy_from_flat():
    paths = [
        "A > B > C",
        "A > B > D",
        "A > E > F",
    ]
    t = Taxonomy.from_flat(paths)
    assert t.leaf_paths == paths
    assert t.top_level_categories() == ["A"]
    assert t.depth() == 3


def test_taxonomy_render():
    t = Taxonomy.from_dict({"Food": {"Burgers": ["Hamburger"]}})
    rendered = t.render_for_prompt()
    assert "Food" in rendered
    assert "Burgers" in rendered
    assert "Hamburger" in rendered


def test_taxonomy_subtree():
    t = Taxonomy.from_dict({
        "Food": {"Burgers": ["Hamburger"]},
        "Drinks": {"Hot": ["Coffee"]},
    })
    sub = t.subtree("Food")
    assert "Burgers > Hamburger" in sub.leaf_paths


def test_normalizer_abbreviations():
    norm = create_normalizer()
    assert "cheese" in norm("chz")
    assert "burger" in norm("brgr")
    assert "cheese" in norm("chz brgr")


def test_normalizer_strip_amounts():
    norm = create_normalizer(strip_amounts=True)
    assert "$" not in norm("bookshelf - $249.99")
    assert "bookshelf" in norm("bookshelf - $249.99")


def test_normalizer_strip_codes():
    norm = create_normalizer(strip_codes=True)
    result = norm("WALMART SUPERCENTER #4532")
    assert "#4532" not in result
    assert "walmart" in result.lower()


def test_normalizer_custom_abbreviations():
    norm = create_normalizer(abbreviations={"wmt": "walmart"})
    assert "walmart" in norm("wmt")


def test_classification_type():
    c = Classification(
        input_text="chz brgr",
        category_path="Food > Burgers > Cheeseburger",
        confidence=0.92,
        source=ClassificationSource.EMBEDDING,
    )
    assert c.leaf_label == "Cheeseburger"
    assert c.path_parts == ["Food", "Burgers", "Cheeseburger"]
    d = c.to_dict()
    assert d["source"] == "embedding"
    assert d["confidence"] == 0.92


if __name__ == "__main__":
    test_taxonomy_from_dict()
    test_taxonomy_from_flat()
    test_taxonomy_render()
    test_taxonomy_subtree()
    test_normalizer_abbreviations()
    test_normalizer_strip_amounts()
    test_normalizer_strip_codes()
    test_normalizer_custom_abbreviations()
    test_classification_type()
    print("All tests passed!")
