#!/usr/bin/env bash
set -e

MOUSE_ID="m21"
DATE="261006"
# All sessions included in concat/sorting.
SESSION_LIST=(1 2 3 4 5 6 7 8 9 10 11)
# Sessions that also receive Motive and tuning-curve analysis.
FREE_MOVING_SESSIONS=(1 5 6 8 9 10)
# Motive CSV rotation representation: true for Quaternion, false for Euler XYZ.
MOTIVE_IS_QUATERNION=true
# Sessions recorded with the Basler camera and analyzed from YOLO pose tracking.
# Keep this disjoint from FREE_MOVING_SESSIONS; both produce head-direction outputs.
BASLER_SESSIONS=(11)

SESSION_TAG="${SESSION_LIST[*]}"
for SESSION_ID in "${SESSION_LIST[@]}"; do
    if ((SESSION_ID < 1 || SESSION_ID > 9)); then
        SESSION_TAG="sessions-${SESSION_TAG// /-}"
        break
    fi
done
SESSION_TAG="${SESSION_TAG// /}"
SESSION_GROUP="${DATE}_${SESSION_TAG}"

# Entry modules come from this checkout; runtimes and data locations are configurable.
RFMAPPING_DIR="${RFMAPPING_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
PIPELINE_DIR="${PIPELINE_DIR:-$RFMAPPING_DIR/preprocessing/pipeline}"
SPIKEINTERFACE_DIR="$RFMAPPING_DIR/preprocessing/spikeinterface"
RFMAPPING_PYTHON="${RFMAPPING_PYTHON:-$HOME/.virtualenvs/rfmapping/bin/python}"
PIPELINE_PYTHON="${PIPELINE_PYTHON:-$PIPELINE_DIR/.venv/bin/python}"
SPIKEINTERFACE_PYTHON="${SPIKEINTERFACE_PYTHON:-$SPIKEINTERFACE_DIR/venv/bin/python}"
ANALYZER_OUTPUT_DIR="${ANALYZER_OUTPUT_DIR:-$SPIKEINTERFACE_DIR/sorting_analyzer}"
# Motive and YOLO are separate applications, not bundled copies of this repository.
MOTIVE_ANALYSIS_DIR="${MOTIVE_ANALYSIS_DIR:-/mnt/ssd4.1/apps/motive_analysis}"
MOTIVE_PYTHON="${MOTIVE_PYTHON:-/mnt/ssd4.1/apps/Analysis/.venv/bin/python}"
YOLO_DIR="${YOLO_DIR:-$HOME/yolov26}"
YOLO_PYTHON="${YOLO_PYTHON:-$HOME/.virtualenvs/yolov26/bin/python}"
LOCAL_RAW_BASE="${LOCAL_RAW_BASE:-/mnt/ssd4.1}"
RECORDING_ROOT="${RECORDING_ROOT:-/mnt/senzailab/Kai/#Recording}"
RECORDING_BASE="$RECORDING_ROOT/$MOUSE_ID"

(
    for SESSION_ID in "${SESSION_LIST[@]}"; do
        LOCAL_SESSION_DIR="$LOCAL_RAW_BASE/${DATE}_${SESSION_ID}"
        REMOTE_RAW_DIR="$RECORDING_BASE/$DATE/${DATE}_${SESSION_ID}/$DATE"

        mkdir -p "$REMOTE_RAW_DIR"
        rsync -rlt \
            --no-whole-file \
            --partial-dir=.rsync-partial \
            --human-readable \
            --info=progress2 \
            "$LOCAL_SESSION_DIR/" \
            "$REMOTE_RAW_DIR/"
    done
) &
SYNC_PID=$!

cd "$RFMAPPING_DIR"
"$PIPELINE_PYTHON" -m scripts.generate_pipeline_config \
    --mouse-id "$MOUSE_ID" \
    --date "$DATE" \
    --sessions "${SESSION_LIST[@]}" \
    --local-raw-base "$LOCAL_RAW_BASE" \
    --recording-root "$RECORDING_ROOT" \
    --output "$PIPELINE_DIR/config_result.yaml"

cd "$PIPELINE_DIR"
"$PIPELINE_PYTHON" -m pipeline.run_script "$PIPELINE_DIR/config_result.yaml"

PIPELINE_LOCAL_DIR="$LOCAL_RAW_BASE/Data/$SESSION_GROUP"
PIPELINE_ARCHIVE_DIR="$RECORDING_BASE/$DATE/$SESSION_GROUP/pipeline/$SESSION_GROUP"
mkdir -p "$PIPELINE_ARCHIVE_DIR"

sync_pipeline_archive() {
    rsync -rlt \
        --no-whole-file \
        --partial-dir=.rsync-partial \
        --human-readable \
        --info=progress2 \
        "$@" \
        "$PIPELINE_LOCAL_DIR/" \
        "$PIPELINE_ARCHIVE_DIR/"
}

# Upload stable pipe outputs while app.py builds analyzer and split artifacts.
# Split folders are replaced by app.py, so only the final pass includes them.
sync_pipeline_archive --exclude='Probe*/kilosort_[0-9]*/' &
PIPELINE_UPLOAD_PID=$!

wait "$SYNC_PID"

cd "$RFMAPPING_DIR"
"$SPIKEINTERFACE_PYTHON" -m preprocessing.spikeinterface.app \
    --mouse-id "$MOUSE_ID" \
    --date "$DATE" \
    --sessions "${SESSION_LIST[@]}" \
    --local-raw-base "$LOCAL_RAW_BASE" \
    --recording-root "$RECORDING_ROOT" \
    --analyzer-output-dir "$ANALYZER_OUTPUT_DIR"

# Analysis writes directly to NAS and continues while the archive upload finishes.
(
PROBE_LIST=""
for probe_dir in "$PIPELINE_LOCAL_DIR"/Probe*; do
    probe_name="${probe_dir##*/}"
    PROBE_LIST+="${probe_name#Probe}"
done

cd "$RFMAPPING_DIR"
for SESSION_ID in "${SESSION_LIST[@]}"; do
    "$SPIKEINTERFACE_PYTHON" -m preprocessing.spikeinterface.generate_adc_spike_time \
        "$RECORDING_BASE" \
        "$DATE" \
        "$SESSION_ID" \
        "$PROBE_LIST"
done

for SESSION_ID in "${FREE_MOVING_SESSIONS[@]}"; do
    SESSION_DIR="$RECORDING_BASE/$DATE/${DATE}_${SESSION_ID}"
    CSV_PATH="$SESSION_DIR/$DATE.csv"
    TAK_PATH="$SESSION_DIR/$DATE.tak"

    if [[ ! -f "$CSV_PATH" || ! -f "$TAK_PATH" ]]; then
        echo "Skipping Motive session $SESSION_ID: CSV/TAK pair incomplete"
        continue
    fi

    mkdir -p "$SESSION_DIR/data/config"
    MOTIVE_CONFIG="$SESSION_DIR/data/config/motive_analysis.yaml"
    sed \
        -e "s|^  path: .*|  path: \"$CSV_PATH\"|" \
        -e "s|^  base_dir: .*|  base_dir: input|" \
        -e "s|^  is_quaternion: .*|  is_quaternion: $MOTIVE_IS_QUATERNION|" \
        "$MOTIVE_ANALYSIS_DIR/config/analysis.yaml" > "$MOTIVE_CONFIG"

    ANALYSIS_CONFIG="$MOTIVE_CONFIG" \
        "$MOTIVE_PYTHON" "$MOTIVE_ANALYSIS_DIR/scripts/run_analysis.py"

    HEAD_DIRECTION_PATH="$SESSION_DIR/data/processed/head_direction.json"
    if [[ ! -f "$HEAD_DIRECTION_PATH" ]]; then
        echo "Skipping tuning curves session $SESSION_ID: Motive head_direction.json missing"
        continue
    fi

    cd "$RFMAPPING_DIR"
    "$RFMAPPING_PYTHON" -m scripts.run_tuning_curves \
        "$RECORDING_BASE" \
        "$DATE" \
        "$SESSION_ID" \
        "$PROBE_LIST"

    "$RFMAPPING_PYTHON" -m scripts.plot_hd_cell_distribution \
        "$SESSION_DIR" "$PROBE_LIST"

    for ((PROBE_INDEX = 0; PROBE_INDEX < ${#PROBE_LIST}; PROBE_INDEX++)); do
        PROBE_NAME="${PROBE_LIST:PROBE_INDEX:1}"

        "$RFMAPPING_PYTHON" -m scripts.run_cylinder_spatial \
            "$SESSION_DIR" \
            --probe "$PROBE_NAME" \
            --phase baseline

        "$RFMAPPING_PYTHON" -m scripts.ebc_video \
            "$SESSION_DIR" \
            --probe "$PROBE_NAME" \
            --phase baseline \
            --arena-type cylinder
    done
done

for SESSION_ID in "${BASLER_SESSIONS[@]}"; do
    SESSION_DIR="$RECORDING_BASE/$DATE/${DATE}_${SESSION_ID}"
    VIDEO_PATH="$SESSION_DIR/$DATE.avi"

    if [[ ! -f "$VIDEO_PATH" ]]; then
        echo "Skipping Basler session $SESSION_ID: $VIDEO_PATH missing"
        continue
    fi

    cd "$YOLO_DIR"
    "$YOLO_PYTHON" "$YOLO_DIR/detect_stream.py" \
        "$RECORDING_BASE" \
        "$DATE" \
        "$SESSION_ID"

    cd "$RFMAPPING_DIR"
    "$RFMAPPING_PYTHON" -m scripts.run_tuning_curves \
        "$RECORDING_BASE" \
        "$DATE" \
        "$SESSION_ID" \
        "$PROBE_LIST" \
        --basler

    "$RFMAPPING_PYTHON" -m scripts.plot_hd_cell_distribution \
        "$SESSION_DIR" "$PROBE_LIST"

    for ((PROBE_INDEX = 0; PROBE_INDEX < ${#PROBE_LIST}; PROBE_INDEX++)); do
        PROBE_NAME="${PROBE_LIST:PROBE_INDEX:1}"

        "$RFMAPPING_PYTHON" -m scripts.spatial_cell_analysis \
            --recording-root "$RECORDING_BASE" \
            --date "$DATE" \
            --recording-number "$SESSION_ID" \
            --probe "$PROBE_NAME" \
            --phase baseline

        "$RFMAPPING_PYTHON" -m scripts.ebc_video \
            "$SESSION_DIR" \
            --probe "$PROBE_NAME" \
            --phase baseline \
            --arena-type rectangle
    done
done
) &
ANALYSIS_PID=$!

# Keep the archive transfers sequential, independently of the analysis above.
RUN_STATUS=0
if wait "$PIPELINE_UPLOAD_PID"; then
    sync_pipeline_archive || RUN_STATUS=$?
else
    RUN_STATUS=$?
fi
# Join both lines even if an upload failed, preserving a nonzero exit status.
wait "$ANALYSIS_PID" || RUN_STATUS=$?
exit "$RUN_STATUS"
