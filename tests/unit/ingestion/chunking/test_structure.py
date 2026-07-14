from backend.ingestion.chunking.structure import detect_headings, split_into_sections


def test_detects_numbered_article_headings():
    text = "Article 1 - Definitions\nSome text here.\n\nArticle 2 - The Facility\nMore text."
    headings = detect_headings(text)
    titles = [h.title for h in headings]
    assert any("Article 1" in t for t in titles)
    assert any("Article 2" in t for t in titles)


def test_nested_numbering_gets_deeper_level():
    text = "Article 5 - Covenants\nText.\n\nArticle 5.2 - Information Covenants\nMore text."
    headings = detect_headings(text)
    article_5 = next(h for h in headings if h.title.startswith("Article 5 "))
    article_5_2 = next(h for h in headings if h.title.startswith("Article 5.2"))
    assert article_5_2.level > article_5.level


def test_detects_roman_numeral_chapter():
    text = "CHAPTER II\nSome regulatory text follows here to fill the section body."
    headings = detect_headings(text)
    assert any("CHAPTER II" in h.title.upper() or "II" in h.title for h in headings)


def test_detects_markdown_headings():
    text = "# Title\nIntro text.\n\n## Section One\nBody text."
    headings = detect_headings(text)
    assert headings[0].level == 0
    assert headings[0].title == "Title"
    assert headings[1].level == 1
    assert headings[1].title == "Section One"


def test_detects_all_caps_fallback_heading():
    text = "RISK FACTORS\nCredit risk on the portfolio remains elevated this year."
    headings = detect_headings(text)
    assert any(h.title == "RISK FACTORS" for h in headings)


def test_does_not_treat_short_lowercase_line_as_heading():
    text = "just a normal short line\nMore normal text follows here for good measure."
    headings = detect_headings(text)
    assert headings == []


def test_detects_note_and_item_headings_for_financial_reports():
    text = (
        "Item 7. Management's Discussion\nBody.\n\nNote 14 — Segment Reporting\nBody."
    )
    headings = detect_headings(text)
    titles = [h.title for h in headings]
    assert any(t.startswith("Item 7") for t in titles)
    assert any(t.startswith("Note 14") for t in titles)


class TestSplitIntoSections:
    def test_no_headings_returns_single_root_section(self):
        text = "Just plain prose with no structure at all."
        sections = split_into_sections(text)
        assert len(sections) == 1
        assert sections[0].breadcrumb == ""
        assert sections[0].text == text

    def test_preamble_before_first_heading_is_its_own_section(self):
        text = "Preamble text.\n\nArticle 1 - Definitions\nBody one."
        sections = split_into_sections(text)
        assert sections[0].breadcrumb == ""
        assert "Preamble text." in sections[0].text
        assert sections[1].breadcrumb.startswith("Article 1")

    def test_breadcrumb_nests_under_parent_heading(self):
        text = (
            "Article 5 - Covenants\nIntro.\n\n"
            "Article 5.1 - Financial Covenants\nBody one.\n\n"
            "Article 5.2 - Information Covenants\nBody two."
        )
        sections = split_into_sections(text)
        breadcrumbs = [s.breadcrumb for s in sections]

        assert breadcrumbs[0].startswith("Article 5 ")
        assert breadcrumbs[1].startswith(breadcrumbs[0] + " > ")
        assert "Article 5.1" in breadcrumbs[1]
        assert breadcrumbs[2].startswith(breadcrumbs[0] + " > ")
        assert "Article 5.2" in breadcrumbs[2]
        assert breadcrumbs[1] != breadcrumbs[2]

    def test_sibling_heading_pops_previous_sibling_from_breadcrumb(self):
        text = (
            "Article 1 - Definitions\nBody.\n\n"
            "Article 1.1 - General Terms\nBody.\n\n"
            "Article 2 - The Facility\nBody."
        )
        sections = split_into_sections(text)
        last_breadcrumb = sections[-1].breadcrumb
        assert "Article 1.1" not in last_breadcrumb
        assert last_breadcrumb.startswith("Article 2")

    def test_sections_cover_the_entire_document_contiguously(self):
        text = "Preamble.\n\nArticle 1 - Definitions\nBody one.\n\nArticle 2 - Facility\nBody two."
        sections = split_into_sections(text)
        assert sections[0].start_char == 0
        assert sections[-1].end_char == len(text)
        for prev, nxt in zip(sections, sections[1:]):
            assert prev.end_char == nxt.start_char
