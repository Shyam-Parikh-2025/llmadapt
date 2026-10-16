"""`python -m llmadapt ...` - the same as the `llmadapt` command."""

import sys

from .cli import main

# Guarded so that merely importing this module (pkgutil.walk_packages in the
# test suite does exactly that) does not run the CLI and exit the process.
if __name__ == "__main__":
    sys.exit(main())
