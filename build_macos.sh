#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
TMP_ROOT=/private/tmp

export BAZELISK_HOME="$TMP_ROOT/tf-bazelisk"
export CCACHE_DIR="$TMP_ROOT/tf-ccache"
export CCACHE_BASEDIR="$ROOT_DIR"
export CCACHE_NOHASHDIR=true
export CCACHE_COMPILERCHECK=content
export TF_PYTHON_VERSION=3.11
export WHEEL_NAME=tensorflow_cpu
export USE_PYWRAP_RULES=1
export DEVELOPER_DIR=/Library/Developer/CommandLineTools

# Pinned toolchain/build assumptions for this macOS build flow.
BAZEL_BIN=/opt/homebrew/bin/bazel
OUTPUT_USER_ROOT="$TMP_ROOT/tf-bazel-root"
MACOS_MINIMUM_OS=12.0
MACOS_SDK_VERSION=26.5
JOBS=12
TARGET=//tensorflow/tools/pip_package:wheel

if command -v ccache >/dev/null 2>&1; then
  CCACHE_PREFIX_DIR="$(dirname "$(command -v ccache)")/libexec"
  if [ -d "$CCACHE_PREFIX_DIR" ]; then
    export PATH="$CCACHE_PREFIX_DIR:$PATH"
  fi
fi

if [ ! -x "$BAZEL_BIN" ]; then
  echo "bazel not found at $BAZEL_BIN" >&2
  exit 1
fi

"$BAZEL_BIN" --output_user_root="$OUTPUT_USER_ROOT" build \
  --cpu=darwin_arm64 \
  --macos_minimum_os="$MACOS_MINIMUM_OS" \
  --macos_sdk_version="$MACOS_SDK_VERSION" \
  --copt=-Wno-invalid-specialization \
  --host_copt=-Wno-invalid-specialization \
  --repo_env=TF_PYTHON_VERSION="$TF_PYTHON_VERSION" \
  --repo_env=WHEEL_NAME="$WHEEL_NAME" \
  --repo_env=USE_PYWRAP_RULES="$USE_PYWRAP_RULES" \
  --action_env=CCACHE_DIR \
  --host_action_env=CCACHE_DIR \
  --action_env=CCACHE_BASEDIR \
  --host_action_env=CCACHE_BASEDIR \
  --action_env=CCACHE_NOHASHDIR \
  --host_action_env=CCACHE_NOHASHDIR \
  --action_env=CCACHE_COMPILERCHECK \
  --host_action_env=CCACHE_COMPILERCHECK \
  --action_env=DEVELOPER_DIR="$DEVELOPER_DIR" \
  --host_action_env=DEVELOPER_DIR="$DEVELOPER_DIR" \
  --strategy=CppCompile=local \
  --jobs="$JOBS" \
  --verbose_failures \
  --show_progress_rate_limit=30 \
  "$TARGET"
