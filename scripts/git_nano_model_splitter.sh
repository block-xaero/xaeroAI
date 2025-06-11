#!/bin/bash
# Safer approach - Remove models/nano from tracking without rewriting history

set -e

echo "🧹 Safely removing models/nano from git tracking..."

# Backup current nano directory
echo "💾 Creating backup of current nano models..."
if [ -d "models/nano" ]; then
    cp -r models/nano models/nano_backup_$(date +%Y%m%d_%H%M%S)
    echo "✅ Backup created"
fi

# Remove from git tracking but keep files locally
echo "📤 Removing models/nano from git tracking..."
git rm -r --cached models/nano/ 2>/dev/null || echo "models/nano not in git yet"

# Add to gitignore temporarily
echo "🚫 Adding models/nano to .gitignore..."
echo "" >> .gitignore
echo "# Temporarily ignore nano models during transition" >> .gitignore
echo "models/nano/*.safetensors" >> .gitignore
echo "models/nano/*_tokenizer/" >> .gitignore

# Commit the removal
echo "💾 Committing removal of nano models..."
git add .gitignore
git commit -m "Remove large nano models from tracking

- Preparing to add compressed/split versions
- Original models backed up locally"

# Push the removal
echo "📤 Pushing removal to remote..."
git push origin main

echo "✅ Nano models removed from git tracking"
echo ""
echo "📁 Files still exist locally:"
if [ -d "models/nano" ]; then
    ls -la models/nano/ | head -10
    echo "   ... (showing first 10 entries)"
fi

echo ""
echo "🚀 Next steps:"
echo "   1. Run ./scripts/compress_split_models.sh"
echo "   2. Update .gitignore to allow parts:"
echo "      # Remove the temporary ignore lines and add:"
echo "      !models/nano/*_part_*"
echo "      !models/nano/*_manifest.txt"
echo "   3. git add models/nano/*_part_* models/nano/*_manifest.txt"
echo "   4. git commit -m 'Add compressed nano model parts'"
echo "   5. git push origin main"