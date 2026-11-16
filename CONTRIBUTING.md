# Contributing to Quorum

Bug reports, questions and pull requests for the open core are welcome at
https://github.com/Bodhi8/quorum-swarm/issues.

## Code

1. Fork, branch from `main`, and install with `pip install -e ".[dev]"`.
2. Add or update tests in `tests/`. Run `pytest -q` (a few seconds).
3. A change to a default (field knobs, bandit settings, gate thresholds) needs a result
   from `quorum ablate` or `quorum compare` in the pull request: defaults are set by
   how they score on identical scenario sequences, not by taste.
4. Every new source file starts with the Apache 2.0 header (a test checks it).
5. Sign off your commits (`git commit -s`).

## Licensing of contributions

The open core is licensed under the Apache License 2.0. By submitting a contribution
you license it on the same terms (Section 5 of the license) and certify, under the
Developer Certificate of Origin, that you have the right to submit it. Quorum Pro is a
separate proprietary product; contributions here never become part of it without a
separate written agreement. See [TRADEMARKS.md](TRADEMARKS.md).
