import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest

from repositories.job_repository import (
    SuggestionStatus, validate_suggestion_status,
    JobStatus, validate_job_status,
)


class TestSuggestionStatus:
    def test_valid_passes(self):
        for s in SuggestionStatus:
            validate_suggestion_status(s)

    def test_invalid_rejected(self):
        with pytest.raises(ValueError):
            validate_suggestion_status("bogus")


class TestJobStatus:
    def test_valid_passes(self):
        for s in JobStatus:
            validate_job_status(s)

    def test_invalid_rejected(self):
        with pytest.raises(ValueError):
            validate_job_status("bogus")
