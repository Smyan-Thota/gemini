from decision_factory.training.repeated_fit import memorization_gate


def test_repeated_fit_requires_all_rows_and_ce_threshold():
    assert memorization_gate(24, 0.05)
    assert not memorization_gate(23, 0.0)
    assert not memorization_gate(24, 0.050001)
