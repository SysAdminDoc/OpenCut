"""Compatibility facade for routes delivered under the historic Wave L blueprint.

Implementations live in purpose-named modules; this facade preserves the
wave_l blueprint and import surface used by existing clients.
"""

from .advanced_video_generation_routes import *  # noqa: F401,F403
from .avatar_generation_routes import *  # noqa: F401,F403
from .multimodal_image_routes import *  # noqa: F401,F403
from .music_generation_routes import *  # noqa: F401,F403
from .speech_generation_routes import *  # noqa: F401,F403
from .video_enhancement_routes import *  # noqa: F401,F403
from .video_generation_routes import *  # noqa: F401,F403
