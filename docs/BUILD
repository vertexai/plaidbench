# Copyright Vertex.AI.

package(default_visibility = ["//visibility:public"])

genrule(
    name = "html",
    srcs = glob([
        "**/*.rst",
        "_static/**",
    ]) + [
        "conf.py",
        "//:plaidbench.py",
    ],
    outs = ["_build"],
    cmd = " ".join([
        "sphinx-build -b html",
        "$$(dirname $(location conf.py))",
        "$(@D)/_build"
    ]),
)

