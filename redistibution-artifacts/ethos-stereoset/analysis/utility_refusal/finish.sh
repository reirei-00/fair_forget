#!/bin/bash
set -eo pipefail
work=$1
track=$2
case "$track" in
    utility) python3 "$work/collect.py" "$work" "$work/retain_utility.csv" ;;
    bbq) python3 "$work/review_sample.py" "$work" "$work/review" ;;
    *) exit 2 ;;
esac
