# Check label counts across all files
echo "=== Label counts per file (sample) ==="
for f in $(ls /Users/anirudhvyas/cyan-yolo-dataset/labels/train/*.txt | head -30); do
    count=$(wc -l < "$f" | tr -d ' ')
    if [ "$count" -lt 3 ]; then
        echo "$(basename $f): $count labels ⚠️"
    fi
done

echo ""
echo "=== Files with only 1 label ==="
for f in /Users/anirudhvyas/cyan-yolo-dataset/labels/train/*.txt; do
    count=$(wc -l < "$f" | tr -d ' ')
    if [ "$count" -eq 1 ]; then
        echo "$(basename $f)"
    fi
done | head -20

echo ""
echo "=== Distribution of label counts ==="
for f in /Users/anirudhvyas/cyan-yolo-dataset/labels/train/*.txt; do
    wc -l < "$f"
done | sort | uniq -c | sort -rn
