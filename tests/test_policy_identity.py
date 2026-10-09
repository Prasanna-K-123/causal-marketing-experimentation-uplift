import pytest
from audit_policy import same_budget_gain


def test_policy_value_matches_explicit_randomized_budget():
    top,bottom,q=.08,.03,.3
    targeted=q*top
    random=q*(q*top+(1-q)*bottom)
    assert same_budget_gain(top,bottom,q)==pytest.approx(targeted-random)


def test_zero_information_or_full_budget_has_no_targeting_advantage():
    assert same_budget_gain(.1,.1,.3)==0
    assert same_budget_gain(.2,.1,0)==0
    assert same_budget_gain(.2,.1,1)==0
