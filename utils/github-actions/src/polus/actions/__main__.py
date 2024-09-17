"""CLI for the utility."""

import json

import typer
from polus.actions import Comparison
from polus.actions import Filters

app = typer.Typer()


@app.command()
def main(
    branch: bool = typer.Option(
        False,
        "--branch",
        "-b",
        help="Branch to compare against main/master for the tool search.",
    ),
    filters: Filters = typer.Option(
        ...,
        "--filter",
        "-f",
        help="Filter for the tools.",
    ),
) -> None:
    """Main function for the CLI."""
    tools = Comparison.apply(branch, filters)

    # Raise github warnings if there are any
    for t in tools:
        for w in t.warnings:
            print(f"::warning::{t.pkg_dir} {w}")  # noqa: T201

    # Create the json matrix for github actions
    gh_json = {"include": []}  # type: ignore[var-annotated]
    for tool in tools:
        gh_json["include"].append(
            {
                "package_dir": str(tool.abs_pkg_dir),
                "package_name": tool.pkg_name,
                "python_version": str(tool.py_version),
            },
        )

    # Print the outputs for github actions
    print(  # noqa: T201
        f"::set-output name=matrix::{json.dumps(gh_json)}",
    )
    print(  # noqa: T201
        "::set-output name=list::"
        + " ".join(v["package_dir"] for v in gh_json["include"]),
    )
    print(  # noqa: T201
        f"::set-output name=num_packages::{len(gh_json['include'])}",
    )


if __name__ == "__main__":
    app()
