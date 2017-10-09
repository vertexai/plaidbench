# Copyright Vertex.AI.

package(default_visibility = ["//visibility:public"])

load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

pkg_tar(
    name = "pkg",
    files = glob(["**/*"]),
    package_dir = "plaidbench",
    strip_prefix = ".",
)

py_library(
    name = "plaidbench",
    srcs = ["plaidbench.py"],
    data = glob(["networks/**"]),
)
