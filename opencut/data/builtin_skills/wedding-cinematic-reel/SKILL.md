---
id: wedding-cinematic-reel
name: wedding-cinematic-reel
title: Wedding Cinematic Reel
description: Color-match event footage to a hero/reference clip, sync highlight cuts to music beats, and build a four-minute wedding reel.
version: 1.0.0
author: opencut-community
license: MIT
category: event
applicable_to: ["sequence", "project"]
required_features: ["editing.color-match", "audio.beat-markers", "video.highlights", "video.merge", "export.video"]
estimated_seconds: 240
---

# Wedding Cinematic Reel

Use this skill when the user asks for a wedding highlight film, reception recap, ceremony teaser, or a four-minute cinematic reel from event footage.

The skill expects three explicit inputs:

1. `primary_sequence_path`: the current event sequence or flattened select reel.
2. `reference_clip_path`: the hero/reference clip whose grade should drive color matching.
3. `music_path`: the licensed music bed used for beat detection.

Execution policy:

1. Color-match the primary sequence to the reference clip at moderate strength so ceremony skin tones remain natural.
2. Detect beat markers from the music bed with subdivisions for cut points.
3. Extract highlight candidates with wedding pacing constraints and a 240-second target.
4. Assemble beat-aligned highlight clips with soft transitions.
5. Export a review master, leaving final delivery approval to the user.

Never invent media paths or silently choose an unlicensed song. If the user has not supplied a reference clip or music bed, ask for those inputs before execution.
