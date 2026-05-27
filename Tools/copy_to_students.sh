#!/bin/bash

# ── Configuration (update once a year) ────────────────────────────────────────
BASE_DIR=~/hupsel-2026/hupsel
COURSE_DIR=/lustre/shared/Courses/HWM23306-2026/Students
STUDENT_LIST=~/hupsel-2026/scripts/stud_list.csv
TEST_STUDENT=user001
# ──────────────────────────────────────────────────────────────────────────────

usage() {
    echo "Usage: $0 <path> [--dry-run] [--test]"
    echo ""
    echo "  <path>      Folder name or relative file path inside BASE_DIR"
    echo "              e.g. 'Step-0' or 'Step-0/hupsel_helper.py'"
    echo "  --dry-run   Show what would happen, without copying anything"
    echo "  --test      Only run for ${TEST_STUDENT}"
    exit 1
}

# ── Helper: find next available backup name ────────────────────────────────────
next_backup() {
    local file="$1"
    local n=1
    local backup
    while true; do
        backup="$(printf '%s.bck.%02d' "$file" $n)"
        [ ! -f "$backup" ] && echo "$backup" && return
        (( n++ ))
    done
}

# ── Helper: copy one file with setfacl ────────────────────────────────────────
deploy_file() {
    local src="$1"
    local dest="$2"
    local student="$3"

    local dest_dir
    dest_dir="$(dirname "$dest")"

    # Ensure parent directory exists and student can write to it
    if [ ! -d "$dest_dir" ]; then
        mkdir -p "$dest_dir"
        setfacl -m "user:${student}:rwx" "$dest_dir"
    fi

    cp "$src" "$dest"
    setfacl -m "user:${student}:rw-" "$dest"
}

# ── Parse arguments ────────────────────────────────────────────────────────────
DRY_RUN=false
TEST_MODE=false
ARG=""

for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=true ;;
        --test)    TEST_MODE=true ;;
        -*)        echo "Unknown option: $arg"; usage ;;
        *)         ARG="$arg" ;;
    esac
done

[ -z "$ARG" ] && usage

# ── Detect file vs folder mode ─────────────────────────────────────────────────
SOURCE="${BASE_DIR}/${ARG}"

if [ -f "$SOURCE" ]; then
    MODE="file"
elif [ -d "$SOURCE" ]; then
    MODE="folder"
else
    echo "ERROR: Not found: $SOURCE"
    exit 1
fi

# ── Build student list ─────────────────────────────────────────────────────────
if $TEST_MODE; then
    STUDENTS=("$TEST_STUDENT")
else
    mapfile -t STUDENTS < <(grep -v '^\s*$' "$STUDENT_LIST")
fi

# ── Summary header ─────────────────────────────────────────────────────────────
echo "Mode   : ${MODE^}"
echo "Source : $SOURCE"
echo "Target : $COURSE_DIR/<student>/atm_fluxes/$ARG"
echo "Students: ${#STUDENTS[@]}"
$DRY_RUN  && echo "Run    : DRY-RUN (no files will be copied)"
$TEST_MODE && echo "Run    : TEST (${TEST_STUDENT} only)"
echo "──────────────────────────────────────────────────"

# ── Main loop ──────────────────────────────────────────────────────────────────
BACKED_UP=0
SKIPPED=0
COPIED=0
ERRORS=0

for student in "${STUDENTS[@]}"; do
    ATM_DIR="${COURSE_DIR}/${student}/atm_fluxes"
    DEST="${ATM_DIR}/${ARG}"

    if [ ! -d "$ATM_DIR" ]; then
        echo "WARNING [$student]: atm_fluxes directory not found — skipping student"
        (( ERRORS++ ))
        continue
    fi

    echo "[$student]"

    if [ "$MODE" = "file" ]; then
        # ── Single file ──────────────────────────────────────────────────────
        if [ -f "$DEST" ]; then
            backup="$(next_backup "$DEST")"
            if $DRY_RUN; then
                echo "  BACKUP (dry-run): $(basename "$DEST") -> $(basename "$backup")"
                echo "  COPY   (dry-run): $(basename "$ARG")"
            else
                cp "$DEST" "$backup"
                setfacl -m "user:${student}:rw-" "$backup"
                echo "  BACKUP: $(basename "$DEST") -> $(basename "$backup")"
                deploy_file "$SOURCE" "$DEST" "$student"
                echo "  COPY: $(basename "$ARG")"
                (( BACKED_UP++ ))
            fi
        else
            if $DRY_RUN; then
                echo "  COPY (dry-run): $(basename "$ARG")"
            else
                deploy_file "$SOURCE" "$DEST" "$student"
                echo "  COPY: $(basename "$ARG")"
            fi
        fi
        (( COPIED++ ))

    else
        # ── Folder ───────────────────────────────────────────────────────────
        while IFS= read -r -d '' src_file; do
            rel="${src_file#${SOURCE}/}"
            dest_file="${DEST}/${rel}"

            if [ -f "$dest_file" ]; then
                echo "  SKIP (exists): $rel"
                (( SKIPPED++ ))
            else
                if $DRY_RUN; then
                    echo "  COPY (dry-run): $rel"
                else
                    deploy_file "$src_file" "$dest_file" "$student"
                    echo "  COPY: $rel"
                fi
                (( COPIED++ ))
            fi
        done < <(find "$SOURCE" \
                    -path '*/.ipynb_checkpoints' -prune -o \
                    -path '*/__pycache__'        -prune -o \
                    -type f -print0 | sort -z)
    fi

done

# ── Summary footer ─────────────────────────────────────────────────────────────
echo "──────────────────────────────────────────────────"
$DRY_RUN && echo "Files that would be copied : $COPIED"
$DRY_RUN || echo "Files copied               : $COPIED"
[ "$MODE" = "file" ] && echo "Files backed up            : $BACKED_UP"
echo "Files skipped (existed)    : $SKIPPED"
echo "Students with errors       : $ERRORS"
