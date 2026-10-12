import os
import argparse
import warnings

BASE_VARIABLES = ["PATH", "PYTHONPATH", "LD_LIBRARY_PATH"]

# Additional variables to include if set
ADDITIONAL_VARIABLES = [
    "DAF_BUTLER_REPOSITORY_INDEX",
    "PGUSER",
    "PGPASSFILE",
    "AWS_SHARED_CREDENTIALS_FILE",
    "LSST_RESOURCES_S3_PROFILE_embargo",
]

# Add variables to BASE_VARIABLES if they are set; warn if not
for var in ADDITIONAL_VARIABLES:
    if var in os.environ and os.environ[var]:
        BASE_VARIABLES.append(var)
    else:
        warnings.warn(f"Environment variable {var} is not set or is empty.")


def main(filename, variables):
    variables = list(variables)
    variables.extend(
        var
        for var in os.environ
        if (var.endswith("DIR") and f"SETUP_{var[:-4]}" in os.environ)
    )
    with open(filename, "w") as f:
        for var in variables:
            f.write(f"{var}={os.environ[var]}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Write selected variables from the current environment "
            "into a Visual Studio Code environemnt files."
        )
    )
    parser.add_argument("-f", "--filename", default=".env", help="Filename to write")
    parser.add_argument(
        "-v",
        "--variable",
        default=list(BASE_VARIABLES),
        action="append",
        dest="variables",
        help=("An additional variables to export; may be " "provided multiple times."),
    )
    args = parser.parse_args()
    main(args.filename, args.variables)
