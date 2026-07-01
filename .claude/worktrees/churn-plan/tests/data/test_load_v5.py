from floodit import config
from floodit.data.load_v5 import load_v5


def test_v5_join_keeps_split_and_labels():
    tr = load_v5("train", verify=False)
    assert len(tr) == 7190
    assert round(tr["churned"].mean(), 3) == 0.231


def test_v5_markov_features_present_no_nulls():
    tr = load_v5("train", verify=False)
    for col in config.V5_NUMERICAL_COLUMNS:
        assert col in tr.columns, f"{col} missing"
    assert not tr[config.V5_NUMERICAL_COLUMNS].isna().any().any()


def test_transition_probabilities_in_unit_interval():
    tr = load_v5("train", verify=False)
    for col in config.V5_PROB_MARKOV:
        assert tr[col].min() >= 0.0, f"{col} below 0"
        assert tr[col].max() <= 1.0, f"{col} above 1"


def test_outgoing_probabilities_sum_to_at_most_one():
    # P(cur|prev) summed over the emitted edges from a source must be <= 1
    # (we emit a subset of edges per source, so it can be < 1, never > 1).
    tr = load_v5("train", verify=False)
    game_edges = [c for c in config.V5_PROB_MARKOV if c.startswith("p_game__")]
    s = tr[game_edges].sum(axis=1)
    assert (s <= 1.0 + 1e-9).all()


def test_bigram_flags_binary():
    tr = load_v5("train", verify=False)
    for col in config.V5_BIGRAM_FLAGS:
        assert set(tr[col].unique()) <= {0, 1}


def test_users_without_transitions_zero_filled():
    tr = load_v5("train", verify=False)
    # users with no transitions in window -> all probabilities 0
    assert (tr["mk_total_transitions"] == 0).any()
    no_trans = tr[tr["mk_total_transitions"] == 0]
    assert (no_trans[config.V5_PROB_MARKOV] == 0).all().all()
