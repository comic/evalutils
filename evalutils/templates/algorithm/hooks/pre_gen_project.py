import re

FORBIDDEN_NAMES = ["evalutils", "pandas", "Evaluation", "Algorithm"]

MODULE_REGEX = r"^[_a-zA-Z][_a-zA-Z0-9]+$"

package_name = "{{ cookiecutter.package_name }}"

if not re.match(MODULE_REGEX, package_name) or package_name in FORBIDDEN_NAMES:
    print(f"ERROR: '{package_name}' is not a valid Python module name!")
    exit(1)
