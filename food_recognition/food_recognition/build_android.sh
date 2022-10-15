#!/bin/bash
export ANDROID_NDK=/home/haishi/android-ndk-r16b

if [ -z "$ANDROID_NDK" ]; then
  echo "ANDROID_NDK not set; please set it to the Android NDK directory"
  exit 1
fi

if [ ! -d "$ANDROID_NDK" ]; then
  echo "ANDROID_NDK not a directory; did you install it under $ANDROID_NDK?"
  exit 1
fi

if [ ! -d "$ANDROID_ABI" ]; then
	ANDROID_ABI=arm64-v8a
fi

mkdir -p build_android
cd build_android

cmake .. \
    -G "Unix Makefiles" \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK\build\cmake\android.toolchain.cmake" \
    -DANDROID_NDK="$ANDROID_NDK" \
    -DANDROID_ABI="$ANDROID_ABI" \
    -DANDROID_NATIVE_API_LEVEL=24 \
    -DANDROID_TOOLCHAIN="clang" \
    -DANDROID_STL="c++_static" \
    -DCMAKE_BUILD_TYPE="Release" \
    || exit 1

cmake --build . -- -j8 || exit 1
