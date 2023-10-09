#!/usr/bin/env bash

#TEST_SET="$1"
WORK_DIR="$1"
#CSV_FILE="$2"

python $(dirname "$0")/test.py \
$(dirname "$0")/../configs/csra/resnet101-csra_1xb16_celeba-448px.py \
~/Desktop/data/saved_models/mmpretrain/resnet101-csra_1xb16_celeba-448px/epoch_20.pth \
--out "$(dirname "$0")/../work_dirs/${WORK_DIR}/res.pkl" \
--out-item pred \
--work-dir "$(dirname "$0")/../work_dirs/${WORK_DIR}" \
--show-dir "$(dirname "$0")/../work_dirs/${WORK_DIR}" \
--interval 10

python $(dirname "$0")/read_pkl.py \
"$(dirname "$0")/../work_dirs/${WORK_DIR}/res.pkl" \
"$(dirname "$0")/../work_dirs/${WORK_DIR}/res.csv"