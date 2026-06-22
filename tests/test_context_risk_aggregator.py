from src.context_risk_aggregator import ContextRiskAggregator, ContextRiskInput


def test_context_aggregator_compresses_metadata_to_one_score():
    aggregator = ContextRiskAggregator()
    trusted = aggregator.aggregate(ContextRiskInput(0.9, "none", False, False, 0.0))
    untrusted_tool = aggregator.aggregate(ContextRiskInput(0.2, "high", True, True, 0.8))

    assert 0 <= trusted <= 1
    assert 0 <= untrusted_tool <= 1
    assert untrusted_tool > trusted
