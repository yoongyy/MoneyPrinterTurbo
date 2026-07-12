import json

from app.services import automation
from app.services import content_source


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
        automation.normalize_url("HTTPS://Example.COM/path?a=1&utm_source=x#section")
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


def test_seen_store_reserves_url_until_complete(tmp_path):
    store = automation.SeenStore(str(tmp_path / "seen.sqlite3"))

    assert store.reserve("https://example.com/a?utm_source=x", "Story")
    assert not store.reserve("https://example.com/a", "Story")
    assert store.should_skip("https://example.com/a")


def test_seen_store_tracks_url_aliases(tmp_path):
    store = automation.SeenStore(str(tmp_path / "seen.sqlite3"))
    store.upsert_many(
        ["https://example.com/a", "https://example.com/a?utm_medium=social"],
        "complete",
        "Story",
    )

    assert store.should_skip("https://example.com/a?utm_source=newsletter")


def test_source_content_hash_detects_duplicate_content(tmp_path):
    store = automation.SeenStore(str(tmp_path / "seen.sqlite3"))
    source = content_source.SourceContent(
        source_type="article",
        source_url="https://example.com/a",
        final_url="https://example.com/a",
        title="Story",
        content="Same source content with enough detail.",
        extraction_method="article",
    )
    content_hash = automation.source_content_hash(source)

    store.upsert("https://example.com/a", "complete", "Story", content_hash=content_hash)

    assert store.has_content_hash(content_hash)


def test_source_content_hash_uses_content_before_title():
    first = content_source.SourceContent(
        source_type="article",
        source_url="https://example.com/a",
        final_url="https://example.com/a",
        title="First headline",
        content="The article body is the stable duplicate signal.",
        extraction_method="article",
    )
    second = content_source.SourceContent(
        source_type="article",
        source_url="https://example.com/b",
        final_url="https://example.com/b",
        title="Second headline",
        content="The article body is the stable duplicate signal.",
        extraction_method="article",
    )

    assert automation.source_content_hash(first) == automation.source_content_hash(second)


def test_ready_video_filename_uses_title_for_single_video():
    assert (
        automation._ready_video_filename(
            "印尼女子利用盲人乞讨被捕，收入揭露",
            index=1,
            total=1,
            suffix=".mp4",
        )
        == "印尼女子利用盲人乞讨被捕-收入揭露.mp4"
    )


def test_ready_video_filename_adds_index_for_multiple_videos():
    assert (
        automation._ready_video_filename(
            "News Title",
            index=2,
            total=3,
            suffix=".mp4",
        )
        == "News-Title-2.mp4"
    )
