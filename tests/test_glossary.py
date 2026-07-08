"""Tests for code/utils/createGlossary.py.

The glossary sizes are fully determined by the category taxonomies, so they make
good regression anchors: if a category list drifts, these counts change.
"""
import createGlossary as glossary


# 21 age groups x 2 sexes = 42 age-sex columns
AGE_SEX = 42


def test_individuals_glossary_counts():
    gl = glossary.create_individuals_glossary()
    assert len(gl["Ethnicity_by_Sex_by_Age"]) == 18 * AGE_SEX   # 756
    assert len(gl["Religion_by_Sex_by_Age"]) == 9 * AGE_SEX     # 378
    assert len(gl["Marital_Status_by_Sex_by_Age"]) == 6 * AGE_SEX  # 252


def test_households_glossary_counts():
    gl = glossary.create_households_glossary()
    assert len(gl["HH_Composition_by_Ethnicity"]) == 15 * 18    # 270
    assert len(gl["HH_Composition_by_Religion"]) == 15 * 9      # 135


def test_sequential_index_is_one_based_and_contiguous():
    gl = glossary.create_individuals_glossary()
    idx = list(gl["Ethnicity_by_Sex_by_Age"]["Sequential_Index"])
    assert idx == list(range(1, len(idx) + 1))


def test_combined_glossary_spans_largest_table():
    ind = glossary.create_individuals_glossary()
    hh = glossary.create_households_glossary()
    combined, all_glossaries = glossary.create_combined_glossary(ind, hh)

    assert len(combined) == 756  # largest constituent table
    assert set(all_glossaries) == {
        "Ethnicity_by_Sex_by_Age",
        "Religion_by_Sex_by_Age",
        "Marital_Status_by_Sex_by_Age",
        "HH_Composition_by_Ethnicity",
        "HH_Composition_by_Religion",
    }
