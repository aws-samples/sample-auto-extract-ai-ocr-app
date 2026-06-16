"""
Routers package
"""

# Import all routers to make them available
from . import health
from . import ocr
from . import upload
from . import extraction
from . import schema
from . import s3_sync
from . import agent
from . import admin
from . import user
from . import sharing
from . import usecase_tools

__all__ = [
    'health',
    'ocr',
    'upload',
    'extraction',
    'schema',
    's3_sync',
    'agent',
    'admin',
    'user',
    'sharing',
    'usecase_tools',
]
