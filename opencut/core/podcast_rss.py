"""
OpenCut Podcast RSS Feed Generator

Generates a fully compliant RSS 2.0 feed with iTunes podcast namespace
extensions and Podlove Simple Chapters.
"""

import logging
import os
import time
from typing import Dict, List, Optional
from xml.dom.minidom import parseString
from xml.etree.ElementTree import Element, SubElement, tostring

logger = logging.getLogger("opencut")

# Namespace URIs
ITUNES_NS = "http://www.itunes.com/dtds/podcast-1.0.dtd"
PSC_NS = "http://podlove.org/simple-chapters"
ATOM_NS = "http://www.w3.org/2005/Atom"

# Required feed metadata keys
_REQUIRED_FEED_KEYS = ("title", "description", "author", "language")
# Required episode keys
_REQUIRED_EPISODE_KEYS = ("title", "audio_path")


def _format_duration(seconds: float) -> str:
    """Convert seconds to HH:MM:SS duration string for iTunes."""
    seconds = int(max(0, seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


def _format_rfc2822(timestamp: float) -> str:
    """Format a Unix timestamp as RFC-2822 date string for RSS."""
    return time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime(timestamp))


def _validate_feed_metadata(meta: dict) -> None:
    """Raise ValueError if required feed metadata fields are missing."""
    missing = [k for k in _REQUIRED_FEED_KEYS if not meta.get(k)]
    if missing:
        raise ValueError(f"Missing required feed metadata: {', '.join(missing)}")


def _validate_episode(ep: dict, idx: int) -> None:
    """Raise ValueError if required episode fields are missing."""
    missing = [k for k in _REQUIRED_EPISODE_KEYS if not ep.get(k)]
    if missing:
        raise ValueError(
            f"Episode {idx}: missing required fields: {', '.join(missing)}"
        )


def _guess_mime_type(path: str) -> str:
    """Guess MIME type from audio file extension."""
    ext = os.path.splitext(path)[1].lower()
    return {
        ".mp3": "audio/mpeg",
        ".m4a": "audio/x-m4a",
        ".mp4": "audio/mp4",
        ".ogg": "audio/ogg",
        ".opus": "audio/opus",
        ".wav": "audio/wav",
        ".flac": "audio/flac",
        ".aac": "audio/aac",
    }.get(ext, "audio/mpeg")


def _get_file_size(path: str) -> int:
    """Return file size in bytes, or 0 if the file doesn't exist."""
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def generate_podcast_rss(
    episodes: List[Dict],
    feed_metadata: Dict,
    output_path: Optional[str] = None,
) -> str:
    """
    Generate a podcast RSS 2.0 feed with iTunes extensions.

    Args:
        episodes:      List of episode dicts.  Each must have at least
                       ``title`` and ``audio_path``.  Optional keys:
                       ``description``, ``duration_seconds``, ``pub_date``
                       (Unix timestamp), ``chapter_markers`` (list of
                       ``{start, title}`` dicts).
        feed_metadata: Dict with ``title``, ``description``, ``author``,
                       ``language``.  Optional: ``email``, ``category``,
                       ``image_url``, ``website_url``.
        output_path:   If given, write the XML to this file.

    Returns:
        The RSS XML as a string.
    """
    # --- validation ---
    _validate_feed_metadata(feed_metadata)
    for i, ep in enumerate(episodes):
        _validate_episode(ep, i)

    # --- build XML tree ---
    rss = Element("rss")
    rss.set("version", "2.0")
    rss.set("xmlns:itunes", ITUNES_NS)
    rss.set("xmlns:psc", PSC_NS)
    rss.set("xmlns:atom", ATOM_NS)

    channel = SubElement(rss, "channel")

    # Channel metadata
    SubElement(channel, "title").text = feed_metadata["title"]
    SubElement(channel, "description").text = feed_metadata["description"]
    SubElement(channel, "language").text = feed_metadata.get("language", "en")
    SubElement(channel, "generator").text = "OpenCut Podcast RSS Generator"
    SubElement(channel, "lastBuildDate").text = _format_rfc2822(time.time())

    if feed_metadata.get("website_url"):
        SubElement(channel, "link").text = feed_metadata["website_url"]

    # iTunes channel tags
    SubElement(channel, "itunes:author").text = feed_metadata["author"]
    SubElement(channel, "itunes:summary").text = feed_metadata["description"]

    if feed_metadata.get("email"):
        owner = SubElement(channel, "itunes:owner")
        SubElement(owner, "itunes:name").text = feed_metadata["author"]
        SubElement(owner, "itunes:email").text = feed_metadata["email"]

    if feed_metadata.get("category"):
        cat = SubElement(channel, "itunes:category")
        cat.set("text", feed_metadata["category"])

    if feed_metadata.get("image_url"):
        img = SubElement(channel, "itunes:image")
        img.set("href", feed_metadata["image_url"])
        # Standard RSS image block
        image_el = SubElement(channel, "image")
        SubElement(image_el, "url").text = feed_metadata["image_url"]
        SubElement(image_el, "title").text = feed_metadata["title"]
        if feed_metadata.get("website_url"):
            SubElement(image_el, "link").text = feed_metadata["website_url"]

    SubElement(channel, "itunes:explicit").text = "false"

    # --- episodes ---
    for ep in episodes:
        item = SubElement(channel, "item")
        SubElement(item, "title").text = ep["title"]

        desc = ep.get("description", "")
        SubElement(item, "description").text = desc
        SubElement(item, "itunes:summary").text = desc

        # Publication date
        pub_ts = ep.get("pub_date", time.time())
        if isinstance(pub_ts, (int, float)):
            SubElement(item, "pubDate").text = _format_rfc2822(pub_ts)
        else:
            SubElement(item, "pubDate").text = str(pub_ts)

        # Enclosure (audio file)
        audio_path = ep["audio_path"]
        enc = SubElement(item, "enclosure")
        enc.set("url", audio_path)
        enc.set("type", _guess_mime_type(audio_path))
        enc.set("length", str(_get_file_size(audio_path)))

        # Duration
        duration_sec = ep.get("duration_seconds", 0)
        if duration_sec:
            SubElement(item, "itunes:duration").text = _format_duration(duration_sec)

        SubElement(item, "itunes:explicit").text = "false"

        # Podlove Simple Chapters
        chapters = ep.get("chapter_markers", [])
        if chapters:
            psc_chapters = SubElement(item, "psc:chapters")
            psc_chapters.set("version", "1.2")
            for ch in chapters:
                ch_el = SubElement(psc_chapters, "psc:chapter")
                start_sec = ch.get("start", 0)
                ch_el.set("start", _format_duration(start_sec))
                ch_el.set("title", ch.get("title", ""))
                if ch.get("href"):
                    ch_el.set("href", ch["href"])
                if ch.get("image"):
                    ch_el.set("image", ch["image"])

    # --- serialise ---
    rough_xml = tostring(rss, encoding="unicode", xml_declaration=False)
    xml_declaration = '<?xml version="1.0" encoding="UTF-8"?>\n'
    try:
        pretty = parseString(rough_xml).toprettyxml(indent="  ", encoding=None)
        # minidom adds its own declaration; strip it and use ours
        lines = pretty.split("\n")
        if lines and lines[0].startswith("<?xml"):
            lines = lines[1:]
        xml_str = xml_declaration + "\n".join(lines)
    except Exception:
        xml_str = xml_declaration + rough_xml

    # --- write to file if requested ---
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(xml_str)
        logger.info("Podcast RSS feed written to %s", output_path)

    return xml_str
