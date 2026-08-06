"""
Routers package
"""

# Import all routers to make them available
from . import health
from . import images
from . import jobs
from . import system
from . import tools
from . import apps
from . import admin
from . import user
from . import sharing

__all__ = [
    'health',
    'images',
    'jobs',
    'system',
    'tools',
    'apps',
    'admin',
    'user',
    'sharing',
]
