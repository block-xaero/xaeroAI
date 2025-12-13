# Set compiler env
export CC=/usr/bin/clang
export CXX=/usr/bin/clang++
export MACOSX_DEPLOYMENT_TARGET=14.0

# Run unit tests (no models needed)
cargo test --release 2>&1