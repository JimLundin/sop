"""Settings shared by every test module.

The Hypothesis profiles live here rather than in one test file so that any
subset of the suite can be run with any of them -- CI runs the soundness
properties a second time under `--hypothesis-profile=deep`, which is the one
place the property tests are given enough examples to reach the corners of the
shape language.
"""

from hypothesis import HealthCheck, settings

settings.register_profile(
    "sop",
    max_examples=400,
    deadline=None,  # the first call pays for the extension's import
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
settings.register_profile(
    "deep",
    max_examples=2000,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
settings.load_profile("sop")
