# Import CoralCLI for programmatic use
# Note: We intentionally don't import the 'main' function here as it would
# shadow the 'main' module and break mock.patch("coral.cli.main.Repository")
# For CLI entry point, use coral.cli.main:main directly
from .main import CoralCLI

__all__ = ["CoralCLI"]
