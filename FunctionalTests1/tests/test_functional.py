"""Each instructor case is an independently named pytest item."""
import pytest
from functional_cases import CASE_IDS, calculate, compare, output_name, read


@pytest.mark.parametrize("case", CASE_IDS)
def test_instructor_case(case):
    compare(case, calculate(case), read(output_name(case)))
