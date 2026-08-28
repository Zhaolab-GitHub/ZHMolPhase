#!/usr/bin/env bash


set -Eeuo pipefail
IFS=$'\n\t'

###############################################################################
# Locate the project
###############################################################################

CALLER_DIR="$(pwd)"
PIPELINE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd -- "${PIPELINE_DIR}/../.." >/dev/null 2>&1 && pwd)"

###############################################################################
# Default options
###############################################################################

ESM2_MODEL=""
CKPT_DIR="${PROJECT_ROOT}/ckpts"

ESM2_ENV="esm2_env"
ZHMOLPHASE_ENV="try_ZHMolPhase"

CLEAN_WORK=0

###############################################################################
# Read-only project paths
###############################################################################

LLPS_SOURCE="${PROJECT_ROOT}/dataset/LLPS_test.txt"
NON_LLPS_SOURCE="${PROJECT_ROOT}/dataset/non_LLPS_test.txt"

FASTA_SOURCE_DIR="${PIPELINE_DIR}/fasta"

GET_PDB_SCRIPT="${PROJECT_ROOT}/scripts/get_pdb.py"
COMPUTE_ESM2_SCRIPT="${PROJECT_ROOT}/scripts/compute_esm2.py"
PREDICT_SCRIPT="${PROJECT_ROOT}/predict.py"

ORIGINAL_PLOT_DIR="${PROJECT_ROOT}/data_figure/test"
ORIGINAL_PLOT_SCRIPT="${ORIGINAL_PLOT_DIR}/plot_AUC.py"

###############################################################################
# Temporary working paths
###############################################################################

WORK_ROOT="${PIPELINE_DIR}/work"

INPUT_DIR="${WORK_ROOT}/input"
FASTA_DIR="${FASTA_SOURCE_DIR}"
PDB_DIR="${WORK_ROOT}/pdb"
LLM_DIR="${WORK_ROOT}/LLM"
RAW_SCORE_DIR="${WORK_ROOT}/raw_scores"
CLEAN_SCORE_DIR="${WORK_ROOT}/clean_scores"
FIGURE_STAGE="${WORK_ROOT}/generated_figures"

LLPS_LIST="${INPUT_DIR}/LLPS_test.txt"
NON_LLPS_LIST="${INPUT_DIR}/non_LLPS_test.txt"
ALL_NAMES="${INPUT_DIR}/all_test_names.txt"

RAW_LLPS_SCORE="${RAW_SCORE_DIR}/LLPS_score.raw.txt"
RAW_NON_LLPS_SCORE="${RAW_SCORE_DIR}/non_LLPS_score.raw.txt"

CLEAN_LLPS_SCORE="${CLEAN_SCORE_DIR}/LLPS_score.txt"
CLEAN_NON_LLPS_SCORE="${CLEAN_SCORE_DIR}/non_LLPS_score.txt"

###############################################################################
# Final retained output
###############################################################################

OUTPUT_ROOT="${PIPELINE_DIR}/output"
FINAL_SCORE_DIR="${OUTPUT_ROOT}/ZHMolPhase"
FINAL_FIGURE_DIR="${OUTPUT_ROOT}/figure"

###############################################################################
# Helpers
###############################################################################

usage() {
    cat <<EOF
Usage:
  bash run_test_pipeline.sh \\
      --esm2-model /path/to/esm2_t33_650M_UR50D.pt \\
      [--ckpt-dir /path/to/ckpts] \\
      [--esm2-env esm2_env] \\
      [--zhmolphase-env try_ZHMolPhase] \\
      [--clean]

Required:
  --esm2-model PATH
      Local ESM-2 checkpoint:
      esm2_t33_650M_UR50D.pt

Optional:
  --ckpt-dir PATH
      ZHMolPhase checkpoint directory.
      Default:
      ${PROJECT_ROOT}/ckpts

  --esm2-env NAME
      Conda environment for ESM-2 embedding generation.
      Default: esm2_env

  --zhmolphase-env NAME
      Conda environment for ZHMolPhase inference and plotting.
      Default: try_ZHMolPhase

  --clean
      Delete any unfinished temporary workspace and previous final output
      before starting.

  -h, --help
      Show this help message.

Example:
  bash run_test_pipeline.sh \\
      --esm2-model /home/user/models/esm2_t33_650M_UR50D.pt \\
      --clean

Required FASTA layout:
  ${FASTA_SOURCE_DIR}/<protein_id>.fasta

For plotting, only this file is copied:
  ${ORIGINAL_PLOT_SCRIPT}

It is executed directly inside:
  ${OUTPUT_ROOT}/

After successful completion, only these generated outputs are retained:
  ${OUTPUT_ROOT}/ZHMolPhase/LLPS_score.txt
  ${OUTPUT_ROOT}/ZHMolPhase/non_LLPS_score.txt
  ${OUTPUT_ROOT}/figure/

The copied plot_AUC.py is deleted after plotting.

The input FASTA directory is preserved:
  ${FASTA_SOURCE_DIR}/
EOF
}


die() {
    echo "[ERROR] $*" >&2
    exit 1
}


info() {
    echo "[INFO] $*"
}


section() {
    echo
    echo "============================================================================"
    echo "$*"
    echo "============================================================================"
}


resolve_user_path() {
    local value="$1"

    if [[ "${value}" = /* ]]; then
        realpath -m -- "${value}"
    else
        realpath -m -- "${CALLER_DIR}/${value}"
    fi
}


require_file() {
    local path="$1"
    local description="$2"

    [[ -s "${path}" ]] || die "${description} not found or empty: ${path}"
}


require_dir() {
    local path="$1"
    local description="$2"

    [[ -d "${path}" ]] || die "${description} not found: ${path}"
}


count_nonempty_lines() {
    local path="$1"

    awk 'NF > 0 {n++} END {print n + 0}' "${path}"
}


normalize_name_list() {
    local source_file="$1"
    local output_file="$2"

    awk '
        {
            gsub(/\r/, "", $0)
        }

        /^[[:space:]]*#/ {
            next
        }

        /^[[:space:]]*$/ {
            next
        }

        {
            protein_id = $1

            if (!(protein_id in seen)) {
                print protein_id
                seen[protein_id] = 1
            }
        }
    ' "${source_file}" > "${output_file}"
}


run_in_env() {
    local environment_name="$1"
    shift

    conda run --no-capture-output -n "${environment_name}" "$@"
}


check_expected_files() {
    local names_file="$1"
    local directory="$2"
    local suffix="$3"
    local label="$4"
    local missing_file="$5"

    : > "${missing_file}"

    while IFS= read -r protein_id; do
        [[ -n "${protein_id}" ]] || continue

        if [[ ! -s "${directory}/${protein_id}${suffix}" ]]; then
            echo "${protein_id}" >> "${missing_file}"
        fi
    done < "${names_file}"

    local missing_count
    missing_count="$(count_nonempty_lines "${missing_file}")"

    if [[ "${missing_count}" -gt 0 ]]; then
        echo "[ERROR] Missing ${label} files: ${missing_count}" >&2
        echo "[ERROR] Missing protein IDs:" >&2
        cat "${missing_file}" >&2
        return 1
    fi

    rm -f -- "${missing_file}"
    info "${label} files are complete."
}


clean_score_file() {
    local raw_file="$1"
    local clean_file="$2"
    local expected_rows="$3"
    local label="$4"

    require_file "${raw_file}" "${label} raw prediction file"

    # Keep only the first two whitespace-delimited columns:
    #
    #   protein_id    prediction_score
    #
    # For example:
    #   A    0.8    1
    #
    # becomes:
    #   A    0.8
    awk '
        NF >= 2 {
            print $1 "\t" $2
        }
    ' "${raw_file}" > "${clean_file}"

    require_file "${clean_file}" "${label} two-column score file"

    local clean_rows
    clean_rows="$(count_nonempty_lines "${clean_file}")"

    if [[ "${clean_rows}" -ne "${expected_rows}" ]]; then
        die "${label} score file contains ${clean_rows} rows; expected ${expected_rows}. Raw file: ${raw_file}"
    fi

    if ! awk '
        NF != 2 {
            exit 1
        }
    ' "${clean_file}"; then
        die "${label} cleaned score file does not contain exactly two columns."
    fi

    info "${label} score file converted to two columns: ${clean_file}"
}


collect_generated_figures() {
    local source_root="$1"
    local staging_root="$2"

    local figure_count=0
    local source_file
    local relative_path
    local destination_path

    rm -rf -- "${staging_root}"
    mkdir -p -- "${staging_root}"

    while IFS= read -r -d '' source_file; do
        relative_path="${source_file#${source_root}/}"

        if [[ "${relative_path}" == figure/* ]]; then
            relative_path="${relative_path#figure/}"
        fi

        destination_path="${staging_root}/${relative_path}"
        mkdir -p -- "$(dirname -- "${destination_path}")"
        cp -a -- "${source_file}" "${destination_path}"

        figure_count=$((figure_count + 1))
    done < <(
        find "${source_root}" -type f \
            \( \
                -iname '*.png'  -o \
                -iname '*.pdf'  -o \
                -iname '*.svg'  -o \
                -iname '*.eps'  -o \
                -iname '*.jpg'  -o \
                -iname '*.jpeg' -o \
                -iname '*.tif'  -o \
                -iname '*.tiff' \
            \) \
            -print0
    )

    if [[ "${figure_count}" -eq 0 ]]; then
        die "plot_AUC.py completed, but no figure files were generated."
    fi

    info "Generated figure files found: ${figure_count}"
}


###############################################################################
# Parse command-line arguments
###############################################################################

while [[ $# -gt 0 ]]; do
    case "$1" in
        --esm2-model)
            [[ $# -ge 2 ]] || die "Missing value after --esm2-model"
            ESM2_MODEL="$2"
            shift 2
            ;;

        --ckpt-dir)
            [[ $# -ge 2 ]] || die "Missing value after --ckpt-dir"
            CKPT_DIR="$2"
            shift 2
            ;;

        --esm2-env)
            [[ $# -ge 2 ]] || die "Missing value after --esm2-env"
            ESM2_ENV="$2"
            shift 2
            ;;

        --zhmolphase-env)
            [[ $# -ge 2 ]] || die "Missing value after --zhmolphase-env"
            ZHMOLPHASE_ENV="$2"
            shift 2
            ;;

        --clean)
            CLEAN_WORK=1
            shift
            ;;

        -h|--help)
            usage
            exit 0
            ;;

        *)
            die "Unknown argument: $1. Use --help for supported options."
            ;;
    esac
done

[[ -n "${ESM2_MODEL}" ]] || {
    usage
    die "--esm2-model is required."
}

ESM2_MODEL="$(resolve_user_path "${ESM2_MODEL}")"
CKPT_DIR="$(resolve_user_path "${CKPT_DIR}")"

###############################################################################
# Safe cleanup requested by the user
###############################################################################

if [[ "${CLEAN_WORK}" -eq 1 ]]; then
    if [[ -d "${WORK_ROOT}" ]]; then
        info "Removing unfinished temporary workspace: ${WORK_ROOT}"
        rm -rf -- "${WORK_ROOT}"
    fi

    if [[ -d "${OUTPUT_ROOT}" ]]; then
        info "Removing previous final output: ${OUTPUT_ROOT}"
        rm -rf -- "${OUTPUT_ROOT}"
    fi
fi

###############################################################################
# Create or resume the temporary workspace
###############################################################################

mkdir -p \
    "${INPUT_DIR}" \
    "${PDB_DIR}" \
    "${LLM_DIR}" \
    "${RAW_SCORE_DIR}" \
    "${CLEAN_SCORE_DIR}"

trap '
    status=$?
    echo
    echo "[FAILED] Pipeline stopped at line ${BASH_LINENO[0]} with exit code ${status}."
    echo "[FAILED] Temporary files were retained for inspection or resumption:"
    echo "         '"${WORK_ROOT}"'"
    exit "${status}"
' ERR

###############################################################################
# Introduction
###############################################################################

section "ZHMolPhase independent test-set reproduction"

info "Project root       : ${PROJECT_ROOT}"
info "Pipeline folder    : ${PIPELINE_DIR}"
info "FASTA input dir   : ${FASTA_SOURCE_DIR}"
info "Temporary workspace: ${WORK_ROOT}"
info "Final output       : ${OUTPUT_ROOT}"
info "ESM-2 environment  : ${ESM2_ENV}"
info "Model environment  : ${ZHMOLPHASE_ENV}"
info "ESM-2 checkpoint   : ${ESM2_MODEL}"
info "Model checkpoints  : ${CKPT_DIR}"

###############################################################################
# Step 1: validate project files and Conda environments
###############################################################################

section "[Step 1/8] Validating files, directories, and environments"

command -v conda >/dev/null 2>&1 || die "conda was not found in PATH."
command -v realpath >/dev/null 2>&1 || die "realpath was not found."

require_file "${LLPS_SOURCE}" "dataset/LLPS_test.txt"
require_file "${NON_LLPS_SOURCE}" "dataset/non_LLPS_test.txt"

require_dir "${FASTA_SOURCE_DIR}" "Local FASTA input directory"
require_file "${GET_PDB_SCRIPT}" "scripts/get_pdb.py"
require_file "${COMPUTE_ESM2_SCRIPT}" "scripts/compute_esm2.py"
require_file "${PREDICT_SCRIPT}" "predict.py"

require_file "${ESM2_MODEL}" "ESM-2 checkpoint"
require_dir "${CKPT_DIR}" "ZHMolPhase checkpoint directory"

require_dir "${ORIGINAL_PLOT_DIR}" "data_figure/test directory"
require_file "${ORIGINAL_PLOT_SCRIPT}" "data_figure/test/plot_AUC.py"

info "Checking Conda environment: ${ESM2_ENV}"
run_in_env "${ESM2_ENV}" python -c \
    "import sys, torch, esm; print('python:', sys.version.split()[0]); print('torch:', torch.__version__); print('esm: available'); print('cuda:', torch.cuda.is_available())"

info "Checking Conda environment: ${ZHMOLPHASE_ENV}"
run_in_env "${ZHMOLPHASE_ENV}" python -c \
    "import sys, numpy, pandas, torch, sklearn, matplotlib; print('python:', sys.version.split()[0]); print('numpy:', numpy.__version__); print('pandas:', pandas.__version__); print('torch:', torch.__version__); print('sklearn:', sklearn.__version__); print('matplotlib:', matplotlib.__version__)"

###############################################################################
# Step 2: prepare isolated test-set lists
###############################################################################

section "[Step 2/8] Preparing test-set protein lists"

normalize_name_list "${LLPS_SOURCE}" "${LLPS_LIST}"
normalize_name_list "${NON_LLPS_SOURCE}" "${NON_LLPS_LIST}"

N_POSITIVE="$(count_nonempty_lines "${LLPS_LIST}")"
N_NEGATIVE="$(count_nonempty_lines "${NON_LLPS_LIST}")"

[[ "${N_POSITIVE}" -gt 0 ]] || die "The positive test-set list is empty."
[[ "${N_NEGATIVE}" -gt 0 ]] || die "The negative test-set list is empty."

OVERLAP_FILE="${INPUT_DIR}/overlapping_test_ids.txt"

comm -12 \
    <(LC_ALL=C sort -u "${LLPS_LIST}") \
    <(LC_ALL=C sort -u "${NON_LLPS_LIST}") \
    > "${OVERLAP_FILE}"

N_OVERLAP="$(count_nonempty_lines "${OVERLAP_FILE}")"

if [[ "${N_OVERLAP}" -gt 0 ]]; then
    die "Positive and negative test sets overlap for ${N_OVERLAP} proteins. See ${OVERLAP_FILE}"
fi

rm -f -- "${OVERLAP_FILE}"

cat "${LLPS_LIST}" "${NON_LLPS_LIST}" \
    | awk 'NF > 0 && !seen[$1]++ {print $1}' \
    > "${ALL_NAMES}"

N_TOTAL="$(count_nonempty_lines "${ALL_NAMES}")"

if [[ "${N_TOTAL}" -ne $((N_POSITIVE + N_NEGATIVE)) ]]; then
    die "Unexpected combined test-set size: ${N_TOTAL}"
fi

info "Positive proteins: ${N_POSITIVE}"
info "Negative proteins: ${N_NEGATIVE}"
info "Total proteins   : ${N_TOTAL}"

###############################################################################
# Step 3: validate the provided FASTA files
###############################################################################

section "[Step 3/8] Validating provided FASTA files"

check_expected_files \
    "${ALL_NAMES}" \
    "${FASTA_DIR}" \
    ".fasta" \
    "FASTA" \
    "${INPUT_DIR}/missing_fasta_ids.txt"

info "Using pre-provided FASTA files from: ${FASTA_DIR}"

###############################################################################
# Step 4: download AlphaFold PDB files
###############################################################################

section "[Step 4/8] Downloading AlphaFold structures"

run_in_env "${ZHMOLPHASE_ENV}" \
    python "${GET_PDB_SCRIPT}" \
    "${ALL_NAMES}" \
    "${PDB_DIR}"

check_expected_files \
    "${ALL_NAMES}" \
    "${PDB_DIR}" \
    ".pdb" \
    "PDB" \
    "${INPUT_DIR}/missing_pdb_ids.txt"

###############################################################################
# Step 5: generate ESM-2 embeddings
###############################################################################

section "[Step 5/8] Generating ESM-2 residue-level embeddings"

run_in_env "${ESM2_ENV}" \
    python "${COMPUTE_ESM2_SCRIPT}" \
    --names_txt "${ALL_NAMES}" \
    --fasta_dir "${FASTA_DIR}" \
    --out_dir "${LLM_DIR}" \
    --model_path "${ESM2_MODEL}"

check_expected_files \
    "${ALL_NAMES}" \
    "${LLM_DIR}" \
    ".rep_1280.npy" \
    "ESM-2 embedding" \
    "${INPUT_DIR}/missing_embedding_ids.txt"

###############################################################################
# Step 6: predict LLPS-positive proteins
###############################################################################

section "[Step 6/8] Predicting dataset/LLPS_test.txt"

run_in_env "${ZHMOLPHASE_ENV}" \
    python "${PREDICT_SCRIPT}" \
    --names_txt "${LLPS_LIST}" \
    --fasta_dir "${FASTA_DIR}" \
    --pdb_dir "${PDB_DIR}" \
    --llm_dir "${LLM_DIR}" \
    --ckpt_dir "${CKPT_DIR}" \
    --out "${RAW_LLPS_SCORE}"

clean_score_file \
    "${RAW_LLPS_SCORE}" \
    "${CLEAN_LLPS_SCORE}" \
    "${N_POSITIVE}" \
    "LLPS"

###############################################################################
# Step 7: predict non-LLPS proteins
###############################################################################

section "[Step 7/8] Predicting dataset/non_LLPS_test.txt"

run_in_env "${ZHMOLPHASE_ENV}" \
    python "${PREDICT_SCRIPT}" \
    --names_txt "${NON_LLPS_LIST}" \
    --fasta_dir "${FASTA_DIR}" \
    --pdb_dir "${PDB_DIR}" \
    --llm_dir "${LLM_DIR}" \
    --ckpt_dir "${CKPT_DIR}" \
    --out "${RAW_NON_LLPS_SCORE}"

clean_score_file \
    "${RAW_NON_LLPS_SCORE}" \
    "${CLEAN_NON_LLPS_SCORE}" \
    "${N_NEGATIVE}" \
    "non-LLPS"

###############################################################################
# Step 8: copy only plot_AUC.py into output/ and plot the new scores
###############################################################################

section "[Step 8/8] Running output/plot_AUC.py with newly generated scores"

rm -rf -- "${OUTPUT_ROOT}"
mkdir -p \
    "${OUTPUT_ROOT}/ZHMolPhase" \
    "${OUTPUT_ROOT}/figure"

# Copy only the plotting script. No other method folders or previous results are
# copied from data_figure/test/.
cp -a -- "${ORIGINAL_PLOT_SCRIPT}" "${OUTPUT_ROOT}/plot_AUC.py"

# Place the newly generated two-column scores where plot_AUC.py expects them.
cp -a -- "${CLEAN_LLPS_SCORE}" \
    "${OUTPUT_ROOT}/ZHMolPhase/LLPS_score.txt"

cp -a -- "${CLEAN_NON_LLPS_SCORE}" \
    "${OUTPUT_ROOT}/ZHMolPhase/non_LLPS_score.txt"

require_file \
    "${OUTPUT_ROOT}/ZHMolPhase/LLPS_score.txt" \
    "output/ZHMolPhase/LLPS_score.txt"

require_file \
    "${OUTPUT_ROOT}/ZHMolPhase/non_LLPS_score.txt" \
    "output/ZHMolPhase/non_LLPS_score.txt"

info "plot_AUC.py will read only these newly generated files:"
info "  ${OUTPUT_ROOT}/ZHMolPhase/LLPS_score.txt"
info "  ${OUTPUT_ROOT}/ZHMolPhase/non_LLPS_score.txt"

(
    cd "${OUTPUT_ROOT}"
    run_in_env "${ZHMOLPHASE_ENV}" python plot_AUC.py
)

###############################################################################
# Finalize output: keep only scores and figures
###############################################################################

section "Keeping final scores and figures; deleting all other generated files"

collect_generated_figures \
    "${OUTPUT_ROOT}" \
    "${FIGURE_STAGE}"

cp -a -- \
    "${OUTPUT_ROOT}/ZHMolPhase/LLPS_score.txt" \
    "${CLEAN_SCORE_DIR}/LLPS_score.final.txt"

cp -a -- \
    "${OUTPUT_ROOT}/ZHMolPhase/non_LLPS_score.txt" \
    "${CLEAN_SCORE_DIR}/non_LLPS_score.final.txt"

# Rebuild output/ so plot_AUC.py, __pycache__, text files, and any other
# plotting by-products are removed.
rm -rf -- "${OUTPUT_ROOT}"
mkdir -p \
    "${OUTPUT_ROOT}/ZHMolPhase" \
    "${OUTPUT_ROOT}/figure"

mv -- \
    "${CLEAN_SCORE_DIR}/LLPS_score.final.txt" \
    "${OUTPUT_ROOT}/ZHMolPhase/LLPS_score.txt"

mv -- \
    "${CLEAN_SCORE_DIR}/non_LLPS_score.final.txt" \
    "${OUTPUT_ROOT}/ZHMolPhase/non_LLPS_score.txt"

cp -a -- "${FIGURE_STAGE}/." "${OUTPUT_ROOT}/figure/"

awk 'NF != 2 {exit 1}' \
    "${OUTPUT_ROOT}/ZHMolPhase/LLPS_score.txt" \
    || die "Final LLPS_score.txt does not contain exactly two columns."

awk 'NF != 2 {exit 1}' \
    "${OUTPUT_ROOT}/ZHMolPhase/non_LLPS_score.txt" \
    || die "Final non_LLPS_score.txt does not contain exactly two columns."

UNEXPECTED_TOP_LEVEL="$(
    find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 \
        ! -name 'ZHMolPhase' \
        ! -name 'figure' \
        -print
)"

if [[ -n "${UNEXPECTED_TOP_LEVEL}" ]]; then
    echo "[ERROR] Unexpected final output entries:" >&2
    echo "${UNEXPECTED_TOP_LEVEL}" >&2
    exit 1
fi

# fasta/ is outside work/ and is preserved.
rm -rf -- "${WORK_ROOT}"

###############################################################################
# Finished
###############################################################################

section "[DONE] Independent test-set reproduction completed"

echo "Only the following final results were retained:"
echo
echo "  ${FINAL_SCORE_DIR}/LLPS_score.txt"
echo "  ${FINAL_SCORE_DIR}/non_LLPS_score.txt"
echo "  ${FINAL_FIGURE_DIR}/"
echo
echo "Both score files contain exactly two columns:"
echo "  protein_id    prediction_score"
echo
echo "Only data_figure/test/plot_AUC.py was copied for plotting."
echo "It ran directly inside output/ and read the newly generated score files."
echo "The copied plot_AUC.py, downloaded PDB files, ESM-2 embeddings, raw"
echo "predictions, and all other temporary files were deleted."
echo
echo "The provided FASTA input directory was preserved:"
echo "  ${FASTA_SOURCE_DIR}"
echo
echo "Original reference results remain unchanged in:"
echo "  ${ORIGINAL_PLOT_DIR}"
