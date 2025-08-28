#! /usr/bin/env bash
mkdir -p print

NB_TO=html

filter_tasks() {
    jq '. | .cells |= map(select((.source | if type == "array" then join("") else . end) | contains("##TASK>")) | if .cell_type == "code" then .outputs = [] | .execution_count = null else . end)' solutions/$1 | uv run jupyter nbconvert --stdin --to $NB_TO --output=print/$1
    open print/$1.$NB_TO
}

filter_tasks 01_QAOA.ipynb
filter_tasks 02_PenaltyTerms.ipynb
filter_tasks 03_XYMixers.ipynb
filter_tasks 04_Aqarios_Luna.ipynb
filter_tasks 05_Benchmarking.ipynb
filter_tasks 06_ModelExtension.ipynb
