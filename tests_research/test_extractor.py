from tdca_research.models import Passage, RetrievalHit
from tdca_research.reasoning.extractor import ClaimExtractor
from tdca_research.utils import normalize_text, query_compacted_passages


def test_span_recovery_requires_answer_in_declared_visible_source():
    hits = [RetrievalHit(Passage("p", "Biography", "Ada Lovelace was born in London."), 1.0, 1, "bm25", "q")]
    visible = normalize_text("[p] Biography\nAda Lovelace was born in London.")
    spans = ClaimExtractor._recover_answer_span("London", ["p"], hits, visible)
    assert spans == ["Ada Lovelace was born in London."]
    assert ClaimExtractor._recover_answer_span("Paris", ["p"], hits, visible) == []
    assert ClaimExtractor._recover_answer_span("London", ["other"], hits, visible) == []


def test_query_sentence_compaction_is_generic_verbatim_and_provenanced():
    text = "Unrelated opening sentence. Ada Lovelace was born in London. A final unrelated sentence."
    hits = [RetrievalHit(Passage("p7", "Biography", text), 1.0, 1, "bm25", "Where born?")]
    compacted = query_compacted_passages(hits, "Where was Ada Lovelace born?", 500, sentences_per_passage=1)
    assert compacted == "[p7] Biography\nAda Lovelace was born in London."
    assert "Unrelated opening" not in compacted
    assert compacted.split("\n", 1)[1] in text


def test_query_sentence_compaction_falls_back_deterministically_without_overlap():
    text = "First sentence. Second sentence. Third sentence."
    hits = [RetrievalHit(Passage("x", "Title", text), 1.0, 1, "bm25", "q")]
    compacted = query_compacted_passages(hits, "no lexical overlap", 500, sentences_per_passage=2)
    assert compacted == "[x] Title\nFirst sentence. Second sentence."
