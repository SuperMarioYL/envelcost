# envelcost — 3-line happy path
#
# From a cold start (after `uv tool install git+https://github.com/SuperMarioYL/envelcost`,
# `uvx --from git+https://github.com/SuperMarioYL/envelcost envelcost`, or a
# checkout with `uv pip install -e .` — the package is not published on PyPI,
# so an unqualified tool invocation cannot resolve it):

envelcost run                                      # replay 5 tasks x 2 envelopes, assert the >2x variance gate
envelcost project --gpus 8xH100 --seats 50        # read the multipliers, project seat-capacity + cost/seat
envelcost report                                  # print the per-envelope token table, write .envelcost/envelcost-report.{json,md}
