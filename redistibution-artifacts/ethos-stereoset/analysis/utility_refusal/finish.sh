#!/bin/bash
set -eo pipefail
work=$1
track=$2
if [[ $# -ge 3 ]]; then
    python3 "$work/reconcile.py" "$work" "$track" "$3" "${4:-0}"
elif [[ "$track" == utility ]]; then
    python3 "$work/collect.py" "$work" "$work/retain_utility.csv"
else
    python3 "$work/review_sample.py" "$work" "$work/review"
fi
