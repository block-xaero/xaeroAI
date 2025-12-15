export CC=/usr/bin/clang
export CXX=/usr/bin/clang++
export MACOSX_DEPLOYMENT_TARGET=14.0
export LIBCLANG_PATH="/opt/homebrew/opt/llvm/lib"

cd /Users/anirudhvyas/xaeroai
cargo clean
cargo build --release 2>&1 | tail -50