#!/bin/bash

# -----------------------------
# CONFIGURATION
# -----------------------------
START_DATE="2022-06-01"    # Older start date
END_DATE="2022-06-30"      # Older end date
TOTAL_COMMITS=12           # Number of commits to spread across the range
LOCAL_DIR="/F/Projects/dolphins/"
GITHUB_USERNAME="SiddiqueDataEng"
REPO_NAME="dolphin.git"
BRANCH_NAME="main"

cd "$LOCAL_DIR" || exit 1

# Init if needed
git init
git checkout -B $BRANCH_NAME
git remote remove origin 2>/dev/null
git remote add origin "https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"

# -----------------------------
# Generate Random Backdated Timestamps
# -----------------------------
echo "📆 Generating realistic backdated commit timestamps..."

SECONDS_START=$(date -d "$START_DATE" +%s)
SECONDS_END=$(date -d "$END_DATE +1 day" +%s)

# Array of possible commit messages for realism
declare -a COMMIT_MESSAGES=(
    "Fixed bug in data processing"
    "Updated data extraction logic"
    "Refactored ETL pipeline"
    "Optimized SQL queries for sales data"
    "Fixed missing sales records"
    "Improved data cleaning process"
    "Added logging to monitor data flow"
    "Corrected data types in final output"
    "Refined data aggregation methods"
    "Updated test cases for new data format"
    "Resolved performance bottleneck in data pipeline"
    "Enhanced error handling in ETL scripts"
)

for i in $(seq 1 $TOTAL_COMMITS); do
    # Random date in the range
    RANDOM_DAY=$(shuf -i $SECONDS_START-$SECONDS_END -n 1)
    COMMIT_DATE=$(date -d "@$RANDOM_DAY" +%Y-%m-%d)

    # Random time between 09:00:00 and 18:00:00
    HOUR=$(shuf -i 9-18 -n 1)
    MINUTE=$(shuf -i 0-59 -n 1)
    SECOND=$(shuf -i 0-59 -n 1)
    TIME=$(printf "%02d:%02d:%02d" $HOUR $MINUTE $SECOND)

    FULL_DATETIME="$COMMIT_DATE $TIME"

    # Get a random commit message from the array
    COMMIT_MSG="${COMMIT_MESSAGES[$((RANDOM % ${#COMMIT_MESSAGES[@]}))]}"

    echo "⏱️ Commit $i @ $FULL_DATETIME: $COMMIT_MSG"

    # Simulate file edit (dummy file)
    FILE="log_day_$i.txt"
    echo "Commit $i: $COMMIT_MSG on $FULL_DATETIME" >> "$FILE"

    # Backdate commit with specific message and timestamp
    GIT_COMMITTER_DATE="$FULL_DATETIME" git commit -m "$COMMIT_MSG" --date="$FULL_DATETIME"

    # Optionally add a delay to simulate a time difference between commits
    sleep 1
done

# -----------------------------
# Push all
# -----------------------------
git push --set-upstream origin $BRANCH_NAME --force
echo "✅ Done: $TOTAL_COMMITS backdated commits with contextual messages pushed!"

# -----------------------------
# Clean up dummy files (optional)
# -----------------------------
echo "🧹 Cleaning up dummy log files..."
rm log_day_*.txt

# Commit clean-up
git add -A
git commit -m ""
git push origin $BRANCH_NAME
echo "✅ Clean-up complete: "
