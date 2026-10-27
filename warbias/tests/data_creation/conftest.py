import pytest
from data_creation_support import case_inputs


@pytest.fixture
def inputs(tmp_path):
    return case_inputs(tmp_path)
