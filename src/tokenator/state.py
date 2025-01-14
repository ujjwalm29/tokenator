"""Global state for tokenator."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Global flag to track if tokenator is properly initialized
is_tokenator_enabled = True

# Store the database path
db_path: Optional[str] = None
