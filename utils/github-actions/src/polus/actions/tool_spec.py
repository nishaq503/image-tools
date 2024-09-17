"""Tool specification for dealing with tools in GitHub Actions."""

import datetime
import pathlib
import subprocess

import packaging.requirements
import packaging.specifiers
import packaging.version
import requests
import toml  # type: ignore[import]

POLUS_MIN_PY_VERSION = packaging.version.parse("3.9")


class ToolSpec:
    """Tool specification for dealing with tools in GitHub Actions.

    Attributes:
        pkg_dir (Path): Path to the directory containing the pyproject.toml file.
        pkg_name (str): Name of the tool from the pyproject.toml file.
        import_name (str): Name used to import the tool in Python.
        version (Version): Version of the tool from the pyproject.toml file.
        py_version (Version): Minimum Python version required to run the tool.
        has_bfio (bool): Whether the tool has a bfio dependency.
        has_filepattern (bool): Whether the tool has a filepattern dependency.
        is_pre_release (bool): Whether the tool is a pre-release version.
        warnings (list[str]): Warnings about the tool.
    """

    def __init__(self, pkg_dir: pathlib.Path) -> None:
        """Initialize a ToolSpec object.

        Successful initialization requires the presence of a pyproject.toml file
        and that the tool follows the Polus Tool standards.
        """
        # Get the root of the git repository
        git_root = pathlib.Path(
            subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],  # noqa: S603, S607
                capture_output=True,
                text=True,
                check=True,
                cwd=pkg_dir,
            ).stdout.strip(),
        ).resolve()

        self.warnings = validate_tool_files(pkg_dir)
        self.abs_pkg_dir = pkg_dir
        self.pkg_dir = pkg_dir.relative_to(git_root)

        # Check to see if the directory contains a pyproject.toml file
        pyproject_path = pkg_dir / "pyproject.toml"
        if not pyproject_path.exists():
            msg = f"Could not find pyproject.toml file in {pkg_dir}"
            raise ValueError(msg)

        # Read the pyproject.toml file
        pyproject = toml.load(pyproject_path)

        self.pkg_name: str = pyproject["tool"]["poetry"]["name"]

        # Check that the parent directory of the pyproject.toml file is the package name
        pkg_parent = str(self.pkg_dir.parent).replace("/", "-")
        if pkg_parent not in self.pkg_name:
            msg = f"Package name {self.pkg_name} does not match directory {pkg_parent}"
            raise ValueError(
                msg,
            )
        import_parts = self.pkg_name.split(f"-{pkg_parent}-")
        if len(import_parts) != 2:  # noqa: PLR2004
            msg = f"Package name {self.pkg_name} does not match directory {pkg_parent}"
            raise ValueError(
                msg,
            )
        polus_part = import_parts[0].replace("-", ".")
        parent_part = pkg_parent.replace("-", ".")
        pkg_part = import_parts[1].replace("-", "_")
        self.import_name = f"{polus_part}.{parent_part}.{pkg_part}"

        self.version = packaging.version.parse(pyproject["tool"]["poetry"]["version"])

        try:
            py_version = packaging.version.parse(
                pyproject["tool"]["poetry"]["dependencies"]["python"],
            )
            if py_version < POLUS_MIN_PY_VERSION:
                msg = f"Unsupported Python: {py_version}"
                raise ValueError(msg)
        except packaging.version.InvalidVersion:
            try:
                py_spec = packaging.specifiers.SpecifierSet(
                    pyproject["tool"]["poetry"]["dependencies"]["python"],
                )
                for py in parse_py_life_cycle():
                    if py in py_spec:
                        py_version = py
                        break
                else:
                    msg = f"Unsupported Python: {py_spec}"
                    raise ValueError(msg)
            except packaging.specifiers.InvalidSpecifier:
                raise

        self.py_version = py_version
        self.has_bfio = "bfio" in pyproject["tool"]["poetry"]["dependencies"]
        self.has_filepattern = (
            "filepattern" in pyproject["tool"]["poetry"]["dependencies"]
        )
        self.is_pre_release = self.version.is_prerelease

    def __str__(self) -> str:
        """Return the package name of the ToolSpec object."""
        return self.pkg_name

    def __repr__(self) -> str:
        """Return a string representation of the ToolSpec object."""
        return "\n".join(
            [
                f"Package Dir: {self.pkg_dir}",
                f"Package Name: {self.pkg_name}",
                f"Import Name: {self.import_name}",
                f"Version: {self.version}",
                f"Python Version: {self.py_version}",
                f"Has bfio: {self.has_bfio}",
                f"Has filepattern: {self.has_filepattern}",
                f"Is Pre-Release: {self.is_pre_release}",
            ],
        )


def validate_tool_files(
    tool_dir: pathlib.Path,
) -> list[str]:
    """Checks that the tool directory contains the necessary files for a Polus Tool."""
    warnings = []

    required_files = [
        "pyproject.toml",
        "Dockerfile",
        "README.md",
        "VERSION",
        ".bumpversion.cfg",
    ]
    for f in required_files:
        if not (tool_dir / f).exists():
            warnings.append(f"Missing required file: {f}")

    # Check that there is an `ict.yaml` or `ict.yml` file
    for f in ["ict.yaml", "ict.yml"]:
        if (tool_dir / f).exists():
            break
    else:
        warnings.append("Missing required file: ICT file")

    # Check that there is a with with the `cwl` extension
    for file in tool_dir.glob("*.cwl"):
        if file.is_file():
            break
    else:
        warnings.append("Missing required file: CWL file")

    return warnings


def parse_py_life_cycle(
    url: str = "https://raw.githubusercontent.com/python/devguide/main/include/release-cycle.json",
) -> list[packaging.version.Version]:
    """Parse the release-cycle.json file to get the list of usable python versions.

    This will be a sorted list of python versions that:
    - have been released,
    - have not reached end of life, and
    - are greater than or equal to the minimum version required by PolusAI.
    """
    # Download the release-cycle.json file
    with requests.get(url, timeout=30) as response:
        response.raise_for_status()
        data: dict = response.json()

    py_versions = []
    for k, v in data.items():
        # Parse the date of first release and end of life
        first_release = parse_month_or_day(v["first_release"])
        end_of_life = parse_month_or_day(v["end_of_life"])
        now = datetime.datetime.now()

        # Still in alpha or beta
        if now <= first_release:
            continue

        # Past end of life
        if end_of_life <= now:
            continue

        py_ver = packaging.version.parse(k)

        # Only include versions that are valid under Polus standards
        if py_ver < POLUS_MIN_PY_VERSION:
            continue

        py_versions.append(py_ver)

    return sorted(py_versions)


def parse_month_or_day(date: str) -> datetime.datetime:
    """Parse a date string that may only have a month or day."""
    try:
        return datetime.datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        try:
            return datetime.datetime.strptime(date, "%Y-%m")
        except ValueError as e:
            msg = f"Could not parse date: {date}"
            raise ValueError(msg) from e
