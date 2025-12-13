cd /Users/anirudhvyas/Downloads/Cyan
mkdir jpeg
for f in *.HEIC *.heic; do
  sips -s format jpeg "$f" --out "jpeg/${f%.*}.jpg"
done
