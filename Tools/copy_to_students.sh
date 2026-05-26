#!/bin/bash

# ── Configuration (update once a year) ────────────────────────────────────────
BASE_DIR=~/hupsel-2026/hupsel
COURSE_DIR=/lustre/shared/Courses/HWM23306-2026/Students
STUDENT_LIST=~/hupsel-2026/scripts/stud_list.csv
TEST_STUDENT=user001
# ──────────────────────────────────────────────────────────────────────────────

usage() {
    echo "Usage: $0 <folder_name> [--dry-run] [--test]"
    echo ""
    echo "  <folder_name>   Name of the folder inside BASE_DIR to copy"
    echo "  --dry-run       Show what would happen, without copying anything"
    echo "  --test          Only run for ${TEST_STUDENT}"
    exit 1
}

# ── Parse arguments ────────────────────────────────────────────────────────────
DRY_RUN=false
TEST_MODE=false
FOLDER=""

for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=true ;;
        --test)    TEST_MODE=true ;;
        -*)        echo "Unknown option: $arg"; usage ;;
        *)         FOLDER="$arg" ;;
    esac
done

[ -z "$FOLDER" ] && usage

# ── Validate source ────────────────────────────────────────────────────────────
SOURCE="${BASE_DIR}/${FOLDER}"

if [ ! -d "$SOURCE" ]; then
    echo "ERROR: Source folder not found: $SOURCE"
    exit 1
fi

# ── Build student list ─────────────────────────────────────────────────────────
if $TEST_MODE; then
    STUDENTS=("$TEST_STUDENT")
else
    mapfile -t STUDENTS < <(grep -v '^\s*$' "$STUDENT_LIST")
fi

# ── Summary header ─────────────────────────────────────────────────────────────
echo "Source : $SOURCE"
echo "Target : $COURSE_DIR/<student>/atm_fluxes/$FOLDER"
echo "Students: ${#STUDENTS[@]}"
$DRY_RUN  && echo "Mode   : DRY-RUN (no files will be copied)"
$TEST_MODE && echo "Mode   : TEST (${TEST_STUDENT} only)"
echo "──────────────────────────────────────────────────"

# ── Main loop ──────────────────────────────────────────────────────────────────
SKIPPED=0
COPIED=0
ERRORS=0

for student in "${STUDENTS[@]}"; do
    ATM_DIR="${COURSE_DIR}/${student}/atm_fluxes"
    DEST="${ATM_DIR}/${FOLDER}"

    # Check that the student's atm_fluxes directory exists
    if [ ! -d "$ATM_DIR" ]; then
        echo "WARNING [$student]: atm_fluxes directory not found — skipping student"
        (( ERRORS++ ))
        continue
    fi

    echo "[$student]"

    # Walk source files
    while IFS= read -r -d '' src_file; do
        rel="${src_file#${SOURCE}/}"          # path relative to source root
        dest_file="${DEST}/${rel}"

        if [ -f "$dest_file" ]; then
            echo "  SKIP (exists): $rel"
            (( SKIPPED++ ))
        else
            if $DRY_RUN; then
                echo "  COPY (dry-run): $rel"
            else
                dest_dir="$(dirname "$dest_file")"
                if [ ! -d "$dest_dir" ]; then
                    mkdir -p "$dest_dir"
                    setfacl -m "user:${student}:rwx" "$dest_dir"
                fi
                cp "$src_file" "$dest_file"
                setfacl -m "user:${student}:rw-" "$dest_file"
                echo "  COPY: $rel"
            fi
            (( COPIED++ ))
        fi
    done < <(find "$SOURCE" \
                -path '*/.ipynb_checkpoints' -prune -o \
                -path '*/__pycache__'        -prune -o \
                -type f -print0 | sort -z)

done

# ── Summary footer ─────────────────────────────────────────────────────────────
echo "──────────────────────────────────────────────────"
$DRY_RUN && echo "Files that would be copied : $COPIED"
$DRY_RUN || echo "Files copied               : $COPIED"
echo "Files skipped (existed)    : $SKIPPED"
echo "Students with errors       : $ERRORS"
