"""Comparison between git commits for finding relevant tools."""

import enum
import pathlib
import subprocess

from .tool_spec import ToolSpec

BROKEN_TOOLS = [
    "binary-operations",  # Not fully updated to new tool standards
    "precompute-slide",  # (Najib): Single failing test: 1023_1024_Segmentation_Zarr
    "microjson-to-ome",  # (Hamdah): AttributeError: type object 'SerializerConfig' has no attribute '__slots__'  # noqa: E501
    "region-segmentation-eval",  # Installation error with numba.
    "basic-flatfield-estimation",  # Jax installation error.
    "ftl-label",  # Requires Rust installation. Also, has not been updated to new tool standards.  # noqa: E501
    "cell-border-segmentation",  # (Hamdah): keras.models.load_model() is getting incorrect model format error.  # noqa: E501
]


class Filters(str, enum.Enum):
    """Filters on tools."""

    All = "all"
    """Find all tools."""

    Dev = "dev"
    """Only tools with dev versions."""

    def apply(self, tool: ToolSpec) -> bool:
        """Apply the comparison to the tool."""
        answer = False
        if self == Filters.All:
            answer = True
        elif self == Filters.Dev:
            answer = tool.is_pre_release
        else:
            msg = f"Filter {self} not implemented."
            raise NotImplementedError(msg)
        return answer


class Comparison(str, enum.Enum):
    """Comparison between git commits for finding relevant tools."""

    All = "all"
    """Find all tools in the repository."""

    Branch = "branch"
    """Find all changed tools in the current branch compared to the default branch."""

    @staticmethod
    def apply(  # noqa: PLR0912, C901
        branch: bool,
        filters: Filters,
    ) -> list[ToolSpec]:
        """Returns all tools that are relevant.

        Args:
            branch: The branch to compare against. If None, then all tools are returned.
            filters: The filter to apply to the tools.

        Returns:
            List of tools that are relevant.
        """
        repo_root = pathlib.Path(
            subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],  # noqa: S603, S607
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip(),
        ).resolve()

        possible_tools: list[pathlib.Path] = []

        if branch:
            # Get the default branch of the repository
            # git remote show [your_remote] | sed -n '/HEAD branch/s/.*: //p'
            git_lines = (
                subprocess.run(
                    [  # noqa: S603, S607
                        "git",
                        "remote",
                        "show",
                        "origin",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                .stdout.strip()
                .split("\n")
            )

            # Find the line that contains the default branch
            default_branch = None
            for line in git_lines:
                if "HEAD branch" in line:
                    default_branch = line.split(": ")[1]
                    break
            else:
                msg = "Could not find default branch."
                raise ValueError(msg)

            # Check if the default branch is `main` or `master`
            if default_branch not in ["main", "master"]:
                msg = (
                    f"Default branch {default_branch} not supported. "
                    "Must be `main` or `master`."
                )
                raise ValueError(
                    msg,
                )

            # Find all files that have changed between the current branch and
            # the default branch
            changed_files = (
                subprocess.run(
                    [  # noqa: S603, S607
                        "git",
                        "diff",
                        "--diff-filter=ACMR",
                        "--name-only",
                        f"origin/{default_branch}...",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                .stdout.strip()
                .split("\n")
            )

            # Find all directories that contain a `pyproject.toml` file and
            # whose name ends in `-tool`
            changed_dirs = set()
            for file in changed_files:
                if file.endswith("pyproject.toml"):
                    parent = pathlib.Path(file).resolve().parent
                    if parent.name.endswith("-tool"):
                        changed_dirs.add(parent)

            possible_tools = list(changed_dirs)
        else:
            # Find all tools on current branch

            # Recursively search for all directories whose name ends in `-tool`
            # and which contain a `pyproject.toml` file
            for tool_dir in repo_root.rglob("*-tool"):
                if not (tool_dir / "pyproject.toml").exists():
                    continue
                possible_tools.append(tool_dir)

        # Apply the filter to the tools we found.
        tools = []
        for tool_dir in possible_tools:
            name = tool_dir.name.split("-tool")[0]
            if name in BROKEN_TOOLS:
                continue
            tool = ToolSpec(tool_dir)
            if filters.apply(tool):
                tools.append(tool)

        return tools
