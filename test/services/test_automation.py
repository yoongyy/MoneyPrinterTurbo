import json

from app.services import automation


def test_link_extractor_collects_absolute_links():
    parser = automation.LinkExtractor("https://example.com/news/")
    parser.feed('<a href="/news/one" title="First story">Ignored</a>')
    parser.feed('<a href="https://example.com/news/two">Second <b>story</b></a>')
    parser.close()

    assert parser.links == [
        {"url": "https://example.com/news/one", "title": "First story"},
        {"url": "https://example.com/news/two", "title": "Second story"},
    ]


def test_parse_json_object_accepts_fenced_json():
    payload = {"index": 1, "reason": "best item"}
    assert automation.parse_json_object(f"```json\n{json.dumps(payload)}\n```") == payload


def test_normalize_url_drops_fragment_and_normalizes_host():
    assert (
        automation.normalize_url("HTTPS://Example.COM/path?a=1#section")
        == "https://example.com/path?a=1"
    )


def test_dedupe_candidates_preserves_first_url():
    candidates = [
        automation.SourceCandidate("A", "article", "https://example.com/a#x", "First"),
        automation.SourceCandidate("B", "article", "https://example.com/a", "Second"),
    ]

    result = automation._dedupe_candidates(candidates)

    assert len(result) == 1
    assert result[0].source_name == "A"
    assert result[0].title == "First"
