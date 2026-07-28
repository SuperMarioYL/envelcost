# envelcost — 3-line happy path
#
# From a cold checkout (after `uv pip install -e .` or `uvx envelcost`):

envelcost run                                      # replay 5 tasks x 2 envelopes, assert the >2x variance gate
envelcost project --gpus 8xH100 --seats 50        # read the multipliers, project seat-capacity + cost/seat
envelcost report                                  # print the per-envelope token table, write .envelcost/envelcost-report.{json,md}
