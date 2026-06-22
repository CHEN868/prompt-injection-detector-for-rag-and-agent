from src.decision_engine import decision_from_probability


def test_v3_decision_thresholds_use_warn():
    assert decision_from_probability(0.44) == "ALLOW"
    assert decision_from_probability(0.45) == "WARN"
    assert decision_from_probability(0.74) == "WARN"
    assert decision_from_probability(0.75) == "BLOCK"
    assert decision_from_probability(0.10, rule_block=True) == "BLOCK"
