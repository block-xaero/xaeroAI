export CC=/usr/bin/clang
export CXX=/usr/bin/clang++
export MACOSX_DEPLOYMENT_TARGET=14.0

cargo run --release --bin simple_test -- --models-dir ./scripts/models