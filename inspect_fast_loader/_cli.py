"""CLI wrapper that patches inspect_ai before running the standard inspect CLI."""

import inspect_fast_loader

inspect_fast_loader.patch()

from inspect_ai._cli.main import main  # noqa: E402

if __name__ == "__main__":
    main()
