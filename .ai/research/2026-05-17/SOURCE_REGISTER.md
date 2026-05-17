# OpenCut — Source Register (2026-05-17 research run)

Every source consulted during this research pass. Local sources have absolute paths; external sources have URLs. Source IDs prefixed `R##` (this run) to distinguish from ROADMAP.md `L##` / `S##` / `V43-L##` / `V43-S##`.

---

## Local evidence

| ID | Source | Inspected |
|---|---|---|
| R-L01 | `Z:\repos\OpenCut\README.md` | 2026-05-17 |
| R-L02 | `Z:\repos\OpenCut\CLAUDE.md` (1,509 lines) | 2026-05-17 |
| R-L03 | `Z:\repos\OpenCut\AGENTS.md` | 2026-05-17 |
| R-L04 | `Z:\repos\OpenCut\CLAUDE-HANDOFF-PROMPT.md` | 2026-05-17 |
| R-L05 | `Z:\repos\OpenCut\ROADMAP.md` v4.3 (2,703 lines) | 2026-05-17 |
| R-L06 | `Z:\repos\OpenCut\ROADMAP-NEXT.md` (843 lines) | 2026-05-17 |
| R-L07 | `Z:\repos\OpenCut\ROADMAP-COMPLETED.md` | 2026-05-17 |
| R-L08 | `Z:\repos\OpenCut\CHANGELOG.md` (1,197 lines) | 2026-05-17 |
| R-L09 | `Z:\repos\OpenCut\CODEX-CHANGELOG.md` | 2026-05-17 |
| R-L10 | `Z:\repos\OpenCut\MODERNIZATION.md` | 2026-05-17 |
| R-L11 | `Z:\repos\OpenCut\AUDIT.md` (624 lines) | 2026-05-17 |
| R-L12 | `Z:\repos\OpenCut\research.md` (329 lines) | 2026-05-17 |
| R-L13 | `Z:\repos\OpenCut\features.md` (2,932 lines, sampled lines 1-100) | 2026-05-17 |
| R-L14 | `Z:\repos\OpenCut\DEVELOPMENT.md` | 2026-05-17 |
| R-L15 | `Z:\repos\OpenCut\SECURITY.md` | 2026-05-17 |
| R-L16 | `Z:\repos\OpenCut\CONTRIBUTING.md` | 2026-05-17 |
| R-L17 | `Z:\repos\OpenCut\pyproject.toml` | 2026-05-17 |
| R-L18 | `Z:\repos\OpenCut\requirements.txt`, `requirements-lock.txt` | 2026-05-17 |
| R-L19 | `Z:\repos\OpenCut\opencut\routes` listing (101 blueprints) | 2026-05-17 |
| R-L20 | `Z:\repos\OpenCut\opencut\core` listing (523 files) | 2026-05-17 |
| R-L21 | `Z:\repos\OpenCut\tests` listing (131 files) | 2026-05-17 |
| R-L22 | `Z:\repos\OpenCut\opencut\_generated\route_manifest.json` (1,359 routes / 101 blueprints) | 2026-05-17 |
| R-L23 | `Z:\repos\OpenCut\opencut\_generated\model_cards.json` (47 model cards) | 2026-05-17 |
| R-L24 | `Z:\repos\OpenCut\opencut\registry.py` (F100 feature registry, 514 lines, 29 documented records) | 2026-05-17 |
| R-L25 | `Z:\repos\OpenCut\opencut\model_cards.py` (721 lines) | 2026-05-17 |
| R-L26 | `Z:\repos\OpenCut\opencut\preflight.py` (180 lines) | 2026-05-17 |
| R-L27 | `Z:\repos\OpenCut\opencut\openapi.py` (222 lines) | 2026-05-17 |
| R-L28 | `Z:\repos\OpenCut\opencut\mcp_server.py` (930 lines, 27 tools) | 2026-05-17 |
| R-L29 | `Z:\repos\OpenCut\opencut\auth.py` (215 lines) | 2026-05-17 |
| R-L30 | `Z:\repos\OpenCut\opencut\security.py` | 2026-05-17 |
| R-L31 | `Z:\repos\OpenCut\opencut\user_data.py` | 2026-05-17 |
| R-L32 | `Z:\repos\OpenCut\opencut\helpers.py` | 2026-05-17 |
| R-L33 | `Z:\repos\OpenCut\opencut\routes\captions.py` | 2026-05-17 (diff only) |
| R-L34 | `Z:\repos\OpenCut\opencut\routes\system.py` | 2026-05-17 (diff only) |
| R-L35 | `Z:\repos\OpenCut\opencut\routes\timeline.py` | 2026-05-17 (diff only) |
| R-L36 | `git log --oneline -60` on `main`, 2026-05-17 — shows F006/F010/F011/F066/F095/F097/F098/F099/F100/F101/F102/F103/F104/F105/F106/F109/F110/F111/F112/F115/F116/F117/F118/F120 commits | 2026-05-17 |
| R-L37 | `git status`, `git diff --stat HEAD`, `git diff` for 7 modified files | 2026-05-17 |
| R-L38 | `git branch --show-current && git remote -v` (branch=main, ahead 25, remote=SysAdminDoc/OpenCut) | 2026-05-17 |
| R-L39 | `Z:\repos\OpenCut\docs\` listing — MODELS.md, NODE_ADVISORIES.md, RESEARCH.md, ROADMAP.md, ROADMAP-COMPLETED.md, UXP_MIGRATION.md, WINDOWS_ARM64_PACKAGING.md | 2026-05-17 |
| R-L40 | `Z:\repos\OpenCut\scripts\` listing — bootstrap_check.py, release_smoke.py, sbom.py, seed_github_issues.py, sync_version.py | 2026-05-17 |
| R-L41 | `Z:\repos\OpenCut\.github\` — ISSUE_TEMPLATE/, issue-seeds.yml, labels.yml, copilot-instructions.md, workflows/ | 2026-05-17 |

---

## External evidence — Premiere / UXP / CEP

| ID | Source |
|---|---|
| R-P01 | https://helpx.adobe.com/premiere/desktop/whats-new/release-notes.html — Premiere 26.0 / 26.0.1 / 26.0.2 / 26.2 / 26.2.2 release notes |
| R-P02 | https://helpx.adobe.com/premiere/desktop/whats-new/whats-new.html — Premiere What's New |
| R-P03 | https://www.newsshooter.com/2026/01/20/whats-new-in-adobe-premiere-pro-26-0/ — Premiere 26.0 launch (Sundance) |
| R-P04 | https://petapixel.com/2026/01/20/rebranded-adobe-premiere-26-arrives-with-one-click-object-tracking/ — rename to "Premiere" + AI Object Mask |
| R-P05 | https://nofilmschool.com/adobe-updates-nab-2026 — NAB 2026 update coverage |
| R-P06 | https://www.provideocoalition.com/new-ai-and-masking-tools-in-premiere-plus-major-upgrade-to-after-effects/ |
| R-P07 | https://developer.adobe.com/premiere-pro/uxp/ — Premiere UXP docs hub |
| R-P08 | https://developer.adobe.com/premiere-pro/uxp/ppro_reference/ — UXP API reference |
| R-P09 | https://github.com/AdobeDocs/uxp-premiere-pro-samples — Adobe UXP Premiere samples |
| R-P10 | https://blog.developer.adobe.com/en/publish/2026/04/uxp-hybrid-plugins-now-available-for-premiere — UXP Hybrid Plugins (Apr 2026) |
| R-P11 | https://forums.creativeclouddeveloper.com/t/premierepro-migrating-from-cep-to-uxp-community-api-gap-issue-summary/11373 — CEP→UXP API gap summary |
| R-P12 | https://medium.com/adobetech/updates-for-creative-cloud-desktop-extensibility-0dd5c663563e — Adobe CEP EOL announcement |
| R-P13 | https://developer.adobe.com/indesign/uxp/resources/migration-guides/cep/ — CEP to UXP migration guide |
| R-P14 | https://www.kreativecore.store/blog/cep-to-uxp-migration.html — third-party CEP EOL summary |
| R-P15 | https://digitalanarchy.com/blog/video-editing-plugins/transcriptive-end-of-life-web-services-will-be-ending-in-may-2026/ — Transcriptive EOL May 2026 |
| R-P16 | https://blog.developer.adobe.com/en/publish/2026/03/introducing-webview-ui-in-bolt-uxp-build-richer-adobe-plugins-faster — Bolt UXP WebView UI announcement |
| R-P17 | https://github.com/hyperbrew/bolt-uxp — Bolt UXP repo (MIT) |
| R-P18 | https://github.com/hyperbrew/bolt-cep — Bolt CEP repo |

---

## External evidence — MCP / agent ecosystem

| ID | Source |
|---|---|
| R-M01 | https://github.com/ayushozha/AdobePremiereProMCP — 1,060 PPro RPC tools, MIT, polyglot |
| R-M02 | https://github.com/leancoderkavy/premiere-pro-mcp — 269 tools, stdio + HTTP/SSE |
| R-M03 | https://github.com/hetpatel-11/Adobe_Premiere_Pro_MCP — 97 tools, March 2026 validation |
| R-M04 | https://github.com/vericontext/vibeframe — CLI-first agentic video, MIT, v0.104.3 |
| R-M05 | https://github.com/FireRedTeam/FireRed-OpenStoryline — Apache-2, MCP + Claude Code Skills |
| R-M06 | https://github.com/HKUDS/VideoAgent — all-in-one agentic framework |
| R-M07 | https://github.com/HKUDS/ViMax — script-to-video agent |
| R-M08 | https://a16z.com/its-time-for-agentic-video-editing/ — a16z analysis |
| R-M09 | https://github.com/modelcontextprotocol/servers/issues/3646 — MCP video-editing demand |
| R-M10 | https://help.descript.com/hc/en-us/articles/36803785502221-Underlord-beta-Your-AI-co-editor-in-Descript |
| R-M11 | https://www.descript.com/underlord — Underlord product page |
| R-M12 | https://captions.ai/ — Captions.ai |
| R-M13 | https://agent.odysser.com/ — Odysser AI Video Editing Agent |
| R-M14 | https://crayo.ai/ — short-form template tool |
| R-M15 | https://crepal.ai/ — "AI Video Creation Agent" |

---

## External evidence — competitive products

| ID | Source |
|---|---|
| R-C01 | https://www.adobe.com/products/premiere.html |
| R-C02 | https://www.blackmagicdesign.com/products/davinciresolve/whatsnew — Resolve 21 |
| R-C03 | https://documents.blackmagicdesign.com/SupportNotes/DaVinci_Resolve_21_New_Features_Guide.pdf — Resolve 21 features PDF |
| R-C04 | https://www.newsshooter.com/2026/04/21/blackmagic-design-davinci-resolve-21-photo-page-preview-nab-2026/ |
| R-C05 | https://postperspective.com/blackmagics-davinci-resolve-21-new-photo-page-ai-tools-more/ |
| R-C06 | https://www.redsharknews.com/davinci-resolve-21-nab-2026-photo-page-ai-tools |
| R-C07 | https://www.broadcastnow.co.uk/tech-innovation/blackmagic-adds-ai-search-and-voice-creation-in-davinci-resolve-21/5216037.article |
| R-C08 | https://www.capcut.com/ |
| R-C09 | https://www.descript.com/features |
| R-C10 | https://www.opus.pro/ |
| R-C11 | https://www.submagic.co/ |
| R-C12 | https://runwayml.com/product |
| R-C13 | https://filmora.wondershare.com/ |
| R-C14 | https://www.veed.io/ |
| R-C15 | https://www.kapwing.com/ |
| R-C16 | https://www.topazlabs.com/topaz-video-ai |
| R-C17 | https://www.heygen.com/features |
| R-C18 | https://elevenlabs.io/docs/capabilities/text-to-speech |
| R-C19 | https://www.newsfilecorp.com/release/297461/Descript-Launches-API-in-Open-Beta-with-Editing-and-Workflow-Updates — Descript API open beta (14 May 2026) |

---

## External evidence — AI models (post-2026-04 / 2026-05)

| ID | Source |
|---|---|
| R-A01 | https://github.com/GAIR-NLP/daVinci-MagiHuman — daVinci-MagiHuman 15B, Apache-2 |
| R-A02 | https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5 — HunyuanVideo-1.5 (Tencent licence) |
| R-A03 | https://fal.ai/ltx-2.3 — LTX-Video 2.3 (Apache-2) |
| R-A04 | https://github.com/SandAI-org/MAGI-1 — MAGI-1 + MagiCompiler (Apache-2) |
| R-A05 | https://wavespeed.ai/blog/posts/wan-2-7-coming-soon-major-upgrade/ — Wan 2.7 announcement |
| R-A06 | https://arxiv.org/html/2511.07399v2 — StreamDiffusionV2 |
| R-A07 | https://decart.ai/publications/mirage — MirageLSD |
| R-A08 | https://github.com/modelscope/DiffSynth-Studio — DiffSynth-Studio Diffusion Templates (Apache-2) |
| R-A09 | https://github.com/OpenBMB/VoxCPM — VoxCPM2 (Apache-2) |
| R-A10 | https://github.com/k2-fsa/OmniVoice — OmniVoice (Apache-2) |
| R-A11 | https://mistral.ai/news/voxtral-tts — Voxtral TTS (CC-BY-NC) |
| R-A12 | https://github.com/QwenLM/Qwen3-TTS — Qwen3-TTS (Apache-2) |
| R-A13 | https://github.com/fishaudio/fish-speech — Fish Speech S2 Pro |
| R-A14 | https://github.com/index-tts/index-tts — IndexTTS2 |
| R-A15 | https://github.com/Tencent-Hunyuan/HunyuanVideo-Foley — HunyuanVideo-Foley (Tencent licence) |
| R-A16 | https://github.com/FunAudioLLM/ThinkSound/tree/prismaudio — PrismAudio (code Apache, weights research) |
| R-A17 | https://huggingface.co/kyutai/mimi — Mimi codec (Apache-2) |
| R-A18 | https://arxiv.org/html/2604.24199v2 — DriftSE speech enhancement |
| R-A19 | https://github.com/ByteDance-Seed/SeedVR — SeedVR / v2.5 (Apache-2) |
| R-A20 | https://github.com/OpenImagingLab/FlashVSR — FlashVSR (Apache-2) |
| R-A21 | https://github.com/pq-yang/MatAnyone2 — MatAnyone 2 (NTU S-Lab 1.0, non-commercial) |
| R-A22 | https://ai.meta.com/blog/segment-anything-model-3/ — SAM 3 / 3.1 |
| R-A23 | https://github.com/ByteDance-Seed/depth-anything-3 — Depth Anything 3 (Apache-2) |
| R-A24 | https://arxiv.org/html/2510.09182v1 — Online Video Depth Anything |
| R-A25 | https://c2pa.org/the-c2pa-launches-content-credentials-2-3-and-celebrates-5-years-of-impact-across-the-digital-ecosystem/ — C2PA 2.3 launch |
| R-A26 | https://www.w3.org/news/2026/proposed-advancement-of-imsc-text-profile-1-3-to-w3c-recommendation/ — IMSC Text Profile 1.3 CR (3 Apr 2026) |
| R-A27 | https://opencolorio.org/ — OpenColorIO 2.5 + ACES 2.0 |

---

## External evidence — dependency upgrades & security advisories

| ID | Source |
|---|---|
| R-D01 | https://www.sentinelone.com/vulnerability-database/cve-2026-40192/ — Pillow CVE-2026-40192 (FITS GZIP bomb) |
| R-D02 | https://advisories.gitlab.com/pkg/pypi/pillow/CVE-2026-40192 |
| R-D03 | https://www.appsecure.security/vulnerability-database/cve-2026-25990/ — Pillow CVE-2026-25990 (PSD overflow) |
| R-D04 | https://www.debian.org/lts/security/dla-4197-1 — flask-cors CVE-2024-1681 / 6839 / 6844 / 6866 / 6221 |
| R-D05 | https://pypi.org/project/flask-cors/ |
| R-D06 | https://github.com/advisories/GHSA-67mh-4wv8-2f99 — esbuild dev-server CORS |
| R-D07 | https://www.wiz.io/vulnerability-database/cve/ghsa-67mh-4wv8-2f99 |
| R-D08 | https://vite.dev/blog/announcing-vite8 — Vite 8 (Mar 12 2026, Rolldown-backed) |
| R-D09 | https://blog.miguelgrinberg.com/post/a-year-in-review-flask-in-2025 — Flask 2025 review |
| R-D10 | https://github.com/pallets/flask/releases |
| R-D11 | https://click.palletsprojects.com/en/stable/changes/ |
| R-D12 | https://github.com/Textualize/rich/releases |
| R-D13 | https://pypi.org/project/python-json-logger/ |
| R-D14 | https://pypi.org/project/psutil/ |
| R-D15 | https://numpy.org/news/ |
| R-D16 | https://github.com/jiaaro/pydub/pull/816 — pydub audioop-lts PR (closed) |
| R-D17 | https://github.com/jiaaro/pydub/issues/863 |
| R-D18 | https://pypi.org/project/audioop-lts/ |
| R-D19 | https://librosa.org/doc/main/changelog.html |
| R-D20 | https://github.com/spotify/pedalboard/releases |
| R-D21 | https://github.com/facebookresearch/demucs — repo (dormant) |
| R-D22 | https://github.com/adefossez/demucs — Defossez fork (active) |
| R-D23 | https://aicenter.ai/products/kokoro-tts |
| R-D24 | https://github.com/resemble-ai/resemble-enhance |
| R-D25 | https://pypi.org/project/faster-whisper/ |
| R-D26 | https://github.com/m-bain/whisperX/releases — 3.8.5 Apr 1 2026 |
| R-D27 | https://pypi.org/project/whisperx/ |
| R-D28 | https://modal.com/blog/choosing-whisper-variants |
| R-D29 | https://opennmt.net/CTranslate2/installation.html — CTranslate2 4.7.1 |
| R-D30 | https://github.com/speaches-ai/speaches/issues/620 — CTranslate2 arm64 cuda wheel |
| R-D31 | https://www.pyannote.ai/changelog — pyannote 4.0 (Feb 7 2026) |
| R-D32 | https://github.com/pyannote/pyannote-audio/blob/main/CHANGELOG.md |
| R-D33 | https://github.com/XPixelGroup/BasicSR/pull/659 — basicsr functional_tensor PR |
| R-D34 | https://github.com/xinntao/Real-ESRGAN/issues/859 |
| R-D35 | https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/13985 |
| R-D36 | https://pypi.org/project/realesrgan/ |
| R-D37 | https://github.com/replicate/GFPGAN — replicate active GFPGAN fork |
| R-D38 | https://github.com/microsoft/onnxruntime/releases/tag/v1.25.0 |
| R-D39 | https://github.com/facebookresearch/audiocraft/issues/423 — torch==2.1.0 pin |
| R-D40 | https://github.com/huggingface/transformers/issues/38464 — transformers v5 cascade |
| R-D41 | https://huggingface.co/blog/transformers-v5 |
| R-D42 | https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md |
| R-D43 | https://socket.dev/pypi/package/transnetv2-pytorch |
| R-D44 | https://auto-editor.com/installing — Nim auto-editor |
| R-D45 | https://basswood-io.com/blog/nim-auto-editor-is-now-in-beta |
| R-D46 | https://pytorch.org/blog/pytorch-2-11-release-blog/ |
| R-D47 | https://www.phoronix.com/news/PyTorch-2.10-Released |
| R-D48 | https://tokenmix.ai/blog/mcp-updates-changelog-every-protocol-change-2026 |
| R-D49 | https://medium.com/the-ai-language/mcp-is-migrating-from-version-1-to-version-2-07f4cc7624fb |
| R-D50 | https://github.com/OpenTimelineIO/otio-aaf-adapter — AAF adapter (post-split) |
| R-D51 | https://opentimelineio.readthedocs.io/en/latest/tutorials/adapters.html |
| R-D52 | https://pypi.org/project/scenedetect/ — PySceneDetect 0.7 (May 3 2026) |
| R-D53 | https://security.snyk.io/package/pip/deep-translator — PYSEC-2022-252 |
| R-D54 | https://vulert.com/vuln-db/pypi-deep-translator-30671 |
| R-D55 | https://github.com/opencv/opencv-python/issues/1212 — bundled ffmpeg CVEs |
| R-D56 | https://github.com/opencv/opencv-python/issues/1186 |
| R-D57 | https://github.com/opencv/opencv/wiki/OE-5.-OpenCV-5 |
| R-D58 | https://9to5linux.com/ffmpeg-8-1-hoare-multimedia-framework-brings-d3d12-h-264-av1-encoding |
| R-D59 | https://linuxiac.com/ffmpeg-8-1-brings-vulkan-compute-codecs-and-new-decoder-support/ |
| R-D60 | https://www.phoronix.com/news/FFmpeg-8.0-Released |

---

## Community / forum signal

| ID | Source |
|---|---|
| R-S01 | https://community.adobe.com/ — Adobe Premiere forums (UXP gap reports) |
| R-S02 | https://forums.creativeclouddeveloper.com/ — Creative Cloud developer forum (UXP API gaps) |
| R-S03 | https://www.indiehackers.com/post/i-analyzed-500-reddit-complaints-about-ai-tools-the-1-frustration-isnt-hallucination-0066da0b1c — AI-tool complaint analysis |
| R-S04 | https://www.trustpilot.com/review/ai.invideo.io — InVideo Trustpilot |
| R-S05 | https://artlist.io/blog/ai-hallucinations/ — AI hallucinations write-up |
| R-S06 | https://news.ycombinator.com/item?id=46832384 — HN OpenShot pain-points discussion |
| R-S07 | https://github.com/mifi/lossless-cut/issues — LosslessCut open issues (signal: ARM64 packaging, CSV import, segment colour, cursor ergonomics, overwrite warnings) |
| R-S08 | https://github.com/AcademySoftwareFoundation/OpenTimelineIO/issues — OTIO data-preservation issues |

---

## Citation index

Wherever this research run cites a fact, it should reference the relevant ID above. The three subagent reports (in `RESEARCH_LOG.md`) include their own inline links — `SOURCE_REGISTER.md` deduplicates them here for ease of audit.

**Source coverage assessment (Pass 1):** every claim in `STATE_OF_REPO.md`, `SECURITY_AND_DEPENDENCY_REVIEW.md`, `COMPETITOR_MATRIX.md`, `DATASET_MODEL_INTEGRATION_REVIEW.md`, `FEATURE_BACKLOG.md`, and `PRIORITIZATION_MATRIX.md` (Pass 1 sections) traces to one of the R-prefixed IDs above, or to ROADMAP.md v4.3's existing `L##`/`S##`/`V43-L##`/`V43-S##` appendix.

---

## Pass 2 — Additional sources (2026-05-17 second pass)

### Local evidence (additional)

| ID | Source |
|---|---|
| R-L42 | `Z:\repos\OpenCut\opencut\checks.py` full read (965 lines, 105 check functions) |
| R-L43 | `Z:\repos\OpenCut\opencut\registry.py` full read (514 lines, 29 FeatureRecord entries) |
| R-L44 | `Z:\repos\OpenCut\opencut\openapi.py` full read (222 lines, 30 endpoints + 35 job endpoints schema-mapped) |
| R-L45 | `Z:\repos\OpenCut\opencut\mcp_server.py` first 200 lines + full `MCP_TOOLS` list (27 tools) |
| R-L46 | `Z:\repos\OpenCut\opencut\_generated\route_manifest.json` full inspection (per-blueprint route counts) |
| R-L47 | `Z:\repos\OpenCut\opencut\_generated\model_cards.json` full inspection (47 cards, full license breakdown) |
| R-L48 | `Z:\repos\OpenCut\installer\src\OpenCut.Installer\` walk (6 XAML pages + 10 services + 3 models; 2,326 lines C#) |
| R-L49 | `Z:\repos\OpenCut\.github\workflows\build.yml` full read (148 lines) |
| R-L50 | `Z:\repos\OpenCut\.github\issue-seeds.yml` walk (15 seeded roadmap items) |
| R-L51 | `Z:\repos\OpenCut\.github\ISSUE_TEMPLATE\` listing (bug_report, config, feature_request, good_first_issue) |
| R-L52 | `Z:\repos\OpenCut\features.md` sample walk (40 entries across 12 categories) |
| R-L53 | `Z:\repos\OpenCut\CLAUDE.md` lines 300-500 read (extra gotchas from v1.18-v1.25 + Wave H confirmations) |
| R-L54 | `Z:\repos\OpenCut\tests\` listing (126 test_*.py files + conftest + fuzz/ + jsx_mock.js); 36 skip markers across 10 files |
| R-L55 | `Z:\repos\OpenCut\opencut\_generated\` listing (`__init__.py`, `model_cards.json`, `route_manifest.json`) |

### Pass-2 external — Premiere / UXP / Hybrid Plugins

| ID | Source |
|---|---|
| R-P19 | https://github.com/AdobeDocs/uxp-premiere-pro-samples — actual sample-panels walk, 3 panels |
| R-P20 | https://www.npmjs.com/package/@adobe/premierepro — dist-tags latest 26.2.0 (2026-04-23) / beta 26.3.0-beta.67 (2026-05-07) |
| R-P21 | https://developer.adobe.com/premiere-pro/uxp/plugins/hybrid-plugins/build/ — Hybrid Plugin build docs |
| R-P22 | https://developer.adobe.com/photoshop/uxp/2022/guides/hybrid-plugins/ — Photoshop hybrid precedent (structurally identical) |
| R-P23 | https://developer.adobe.com/photoshop/uxp/2022/uxp-api/reference-js/Modules/uxp/Persistent%20File%20Storage/FileSystemProvider/ — UXP file system API |
| R-P24 | https://developer.adobe.com/photoshop/uxp/2022/uxp-api/reference-js/Modules/uxp/Key-Value%20Storage/SecureStorage/ — UXP secureStorage |
| R-P25 | https://developer.adobe.com/premiere-pro/uxp/resources/recipes/filesystem-operations/ — UXP filesystem recipes |
| R-P26 | https://developer.adobe.com/premiere-pro/uxp/resources/recipes/html-events/ — UXP HTML event handling |
| R-P27 | https://developer.adobe.com/premiere-pro/uxp/resources/fundamentals/apis/ — UXP fundamentals |
| R-P28 | https://community.adobe.com/t5/premiere-pro-bugs/premiere-pro-25-6-3-uxp-issue-unable-to-make-requests-with-http-urls-vs-https/idi-p/15622477 — known macOS HTTP bug |
| R-P29 | https://github.com/AdobeDocs/uxp/issues/36 — drag-and-drop file support ticket (promised CY2026) |
| R-P30 | https://community.adobe.com/questions-729/extendscript-to-uxp-for-premiere-pro-1553924 — Sept 2026 ExtendScript EOL community confirmation |
| R-P31 | https://github.com/Adobe-CEP/CEP-Resources/tree/master/UXP-Migration-Guide — Adobe CEP migration guide |
| R-P32 | https://hyperbrew.co/blog/premiere-pro-uxp-beta/ — Hyper Brew Premiere UXP beta blog |
| R-P33 | https://skywork.ai/skypage/en/unlocking-video-editing-adobe-premiere/1981241710777434112 — Skywork Premiere Pro MCP architecture deep-dive |

### Pass-2 external — Frame.io / review platforms

| ID | Source |
|---|---|
| R-F01 | https://blog.adobe.com — Frame.io V4 announcement |
| R-F02 | https://frame.io/v4 — Frame.io V4 product page |
| R-F03 | https://frame.io/pricing — Frame.io 2026 pricing tiers |
| R-F04 | https://frame.io/features/c2c — Camera-to-Cloud |
| R-F05 | https://helpx.adobe.com/legal/product-descriptions/frameio.html — Frame.io legal/product description |
| R-F06 | https://support.frame.io — C2C compatibility articles (RED V-Raptor, Teradek Cube/Prism/Serv) |
| R-F07 | https://redsharknews.com — Frame.io Drive NAB 2026 |
| R-F08 | https://blog.frame.io — 2026/04/15 Drive announcement; 2022/07/11 "Reinventing timecode for the cloud" |
| R-F09 | https://developer.frame.io — webhook docs (signed payload envelope) |
| R-F10 | https://github.com/elonen/clapshot — Rust + Svelte review tool, GPLv2 (proto3/plugins MIT) |
| R-F11 | https://github.com/Techiebutler/freeframe — HLS adaptive streaming + frame-accurate comments |
| R-F12 | https://github.com/yusufipk/OpenFrame — "Fair-source" not OSI |
| R-F13 | https://github.com/davidguva/OpenVidReview — timestamped + EDL export to DaVinci |
| R-F14 | https://github.com/KirisameMarisa/video-review — on-prem timeline comments |
| R-F15 | https://krock.io/pricing — commercial reference |
| R-F16 | https://iconik.io/pricing — commercial reference |
| R-F17 | https://picflow.com — commercial reference |
| R-F18 | https://filestage.io — commercial reference |
| R-F19 | https://github.com/juanfont/headscale — self-hosted Tailscale control plane |
| R-F20 | https://headscale.net/stable/usage/getting-started — Headscale docs |
| R-F21 | https://tailscale.com/docs/features/tailscale-funnel — Tailscale Funnel |
| R-F22 | https://github.com/schollz/croc — MIT P2P transfer |
| R-F23 | https://github.com/syncthing/syncthing — MPL-2.0 continuous P2P sync |
| R-F24 | https://github.com/rclone/rclone — MIT cloud storage tool |
| R-F25 | https://github.com/magic-wormhole/magic-wormhole — MIT spiritual ancestor of croc |
| R-F26 | https://opentimelineio.readthedocs.io — OTIO Marker schema for review-bundle interchange |
| R-F27 | https://teradek.com/blogs/articles/frameio-c2c — Teradek C2C hardware |
| R-F28 | https://cined.com — Frame.io C2C articles |
| R-F29 | https://pro.sony — Sony Ci + Teradek release |
| R-F30 | https://developer.vimeo.com/api/reference/response/comment — Vimeo Comment object shape |

### Pass-2 external — Niche AI + accessibility + standards

| ID | Source |
|---|---|
| R-N01 | https://github.com/HKUDS/ViMax — Director/Screenwriter/Producer/Generator pipeline |
| R-N02 | OpenMontage GitHub (per "awesome-ai-agents-2026" index) — 11 pipelines, 49 tools, 400+ agent skills |
| R-N03 | https://github.com/FireRedTeam/FireRed-OpenStoryline — intention-driven editing |
| R-N04 | Toonflow GitHub — novel/script-to-animation cross-platform desktop |
| R-N05 | Timeline Studio GitHub — AI-augmented NLE (April 2026) |
| R-N06 | https://www.cgchannel.com/2026/05/sneak-peek-nexus-for-blender/ — NeXus for Blender preview |
| R-N07 | EditYourself arXiv Jan 2026 — DiT audio-driven video-to-video for talking-head transcript edits |
| R-N08 | Causal Forcing++ arXiv May 2026 — few-step AR diffusion distillation for real-time interactive video |
| R-N09 | https://www.phoronix.com/news/AOMedia-OAC-Open-Audio-Codec — AOM AOC successor to Opus |
| R-N10 | https://w3c.github.io/wcag3/guidelines/ — WCAG 3.0 draft (extended AD + descriptive transcript) |
| R-N11 | https://www.fcc.gov/consumer-governmental-affairs/commission-announces-effective-date-closed-captioning-display-settings-rule — Aug 17 2026 effective date |
| R-N12 | https://www.itu.int/rec/R-REC-BT.1702/ — ITU-R BT.1702 PSE (2023 latest) |
| R-N13 | https://dl.acm.org/doi/10.1145/3663547.3759422 — SignStreamNet ACM ASSETS 2025 |
| R-N14 | https://github.com/microsoft/ai-audio-descriptions — Microsoft AI Audio Descriptions |
| R-N15 | https://broadcastwriter.com/2024/12/12/bbc-subtitle-style-guide-2024/ — BBC subtitle style guide (160-180 wpm) |
| R-N16 | https://github.com/Shopify/react-native-skia/issues/643 — RTL rendering broken in Skia default |
| R-N17 | https://artificialanalysis.ai/speech-to-text/models/whisper — Whisper Hindi/Arabic accuracy data |
| R-N18 | https://partnerhelp.netflixstudios.com/hc/en-us/articles/115000614752 — Netflix IMF delivery spec |
| R-N19 | https://tech.ebu.ch/imf — DPP IMF EBU |
| R-N20 | https://github.com/quietvoid/dovi_tool — Dolby Vision OSS chain |
| R-N21 | https://tech.ebu.ch/docs/techreports/tr045.pdf — EBU TR 045 ADM BWF workflow |
| R-N22 | https://itecsonline.com/post/homebrew-on-macos-complete-package-manager-guide — Homebrew Cask 2026 |
| R-N23 | https://web.dev/blog/webgpu-supported-major-browsers — WebGPU baseline Jan 2026 |
| R-N24 | https://aptabase.com — privacy-first desktop telemetry |
| R-N25 | https://openobserve.ai — single-binary observability |
| R-N26 | https://signoz.io — OSS Datadog alternative |
| R-N27 | https://github.com/posthog/posthog — autocapture-default complaint surface |

**Source coverage assessment (Pass 2 cumulative):** every Pass-2 claim in `ROUTE_READINESS_AUDIT.md`, `INSTALLER_AUDIT.md`, `TEST_COVERAGE_GAPS.md`, `FEATURES_RECONCILIATION.md`, `FEATURE_BACKLOG_ADDENDUM.md`, and the PRIORITIZATION_MATRIX §6.5 update traces to a Pass-1 or Pass-2 R-prefixed ID.

**Total source count this run:** 41 local (R-L01-55, gaps where I sampled) + 60 external dependencies (R-D01-60) + 30 Premiere/UXP (R-P01-33) + 30 Frame.io (R-F01-30) + 27 AI models / standards (R-A01-27 + R-N01-27) + 27 niche AI / accessibility (R-N01-27) + 15 MCP/agents (R-M01-15) + 19 commercial products (R-C01-19) + 8 community signal (R-S01-08) ≈ **~250 unique sources**.

---

## Pass 4 — Release-smoke and source refresh (2026-05-17 fourth pass)

### Local command evidence

| ID | Source |
|---|---|
| R-P4-L01 | `python -m opencut.tools.dump_route_manifest --check` — PASS, 1,359 routes / 101 blueprints |
| R-P4-L02 | `python scripts/sync_version.py --check` — PASS, all version surfaces at v1.32.0 |
| R-P4-L03 | `python scripts/bootstrap_check.py` — PASS, 6 sub-checks |
| R-P4-L04 | `python -m pip_audit -r requirements-lock.txt` — PASS, no known vulnerabilities |
| R-P4-L05 | Raw `npm audit --json` in `extension/com.opencut.panel` — one known moderate Vite `.map` advisory, matching F095 |
| R-P4-L06 | `ruff check opencut --select E,F,I --ignore E501,E402` — PASS after safe Ruff fixes |
| R-P4-L07 | Targeted pytest slice (`test_local_auth.py`, `test_hardening.py`, `test_config_and_userdata.py`, `test_boolean_coercion.py`, `test_crash_packet.py`, `test_review_bundle.py`, `test_route_manifest.py`) — PASS, `119 passed` |
| R-P4-L08 | `python scripts/release_smoke.py --json` — PASS end-to-end; pytest-fast `232 passed` |
| R-P4-L09 | `npm view @adobe/premierepro version dist-tags --json` — `latest=26.2.0`, `beta=26.3.0-beta.67` |

### External sources refreshed in Pass 4

| ID | Source |
|---|---|
| R-P4-E01 | https://developer.adobe.com/premiere-pro/uxp/ — official Premiere UXP API overview; confirms UXP is the Premiere extensibility platform and links DOM APIs / Hybrid Plugins / distribution |
| R-P4-E02 | https://developer.adobe.com/premiere-pro/uxp/plugins/hybrid-plugins/build/ — official Hybrid Plugin build guide; confirms `.uxpaddon` packaging, manifest requirements, platform/architecture directory layout |
| R-P4-E03 | https://developer.adobe.com/premiere-pro/uxp/resources/fundamentals/typescript-support/ — official note that Premiere DOM API TypeScript definitions are shipped as `@adobe/premierepro` |
| R-P4-E04 | https://github.com/AcademySoftwareFoundation/OpenTimelineIO/releases — OTIO adapter split note; full adapter set now belongs in `OpenTimelineIO-Plugins`, including AAF/FCP XML/CMX3600 |
| R-P4-E05 | https://advisories.gitlab.com/npm/vite/GHSA-4w7w-66w2-5vf9/ — Vite `.map` path traversal advisory; affected/fixed versions and CVSS 5.3 medium |

**Source coverage assessment (Pass 4):** Pass 4 claims in `LIVE_VERIFICATION.md`, `PROJECT_CONTEXT.md`, `ROADMAP.md` v4.7, `CHANGESET_SUMMARY.md`, and `CONTINUE_FROM_HERE.md` trace to R-P4-L01 through R-P4-L09 plus R-P4-E01 through R-P4-E05.

---

## Pass 9 — F195 MCP curated tool expansion (2026-05-17 ninth pass)

### Local source evidence

| ID | Source |
|---|---|
| R-P9-L01 | `opencut/mcp_server.py` — MCP tool definitions, `_TOOL_ROUTES`, path validation, and `handle_tool_call` dispatch |
| R-P9-L02 | `opencut/routes/wave_l_routes.py` — source routes for face reshape, skin retouch, smart upscale, and ElevenLabs TTS |
| R-P9-L03 | `opencut/routes/captions.py` — source route for `POST /captions/qc` |
| R-P9-L04 | `opencut/routes/timeline.py` — source routes for marker import, review bundle, and C2PA provenance |
| R-P9-L05 | `opencut/routes/system.py` — source route for `GET /system/capabilities` |
| R-P9-L06 | `opencut/routes/wave_k_routes.py` — source routes for Brand Kit, semantic search, and spectral match |
| R-P9-L07 | `python -m py_compile opencut/mcp_server.py scripts/release_smoke.py tests/test_mcp_server.py` — PASS |
| R-P9-L08 | `python -m pytest tests/test_mcp_server.py tests/test_release_smoke.py -q` — PASS, `17 passed` |
| R-P9-L09 | `ruff check opencut/mcp_server.py scripts/release_smoke.py tests/test_mcp_server.py --select E,F,I --ignore E501,E402` — PASS |
| R-P9-L10 | `python scripts/release_smoke.py --json` — PASS, all 13 steps green; pytest-fast `246 passed` |

**Source coverage assessment (Pass 9):** Pass 9 claims in `ROADMAP.md` v4.12, `PROJECT_CONTEXT.md`, `ROUTE_READINESS_AUDIT.md`, `FEATURE_BACKLOG_ADDENDUM.md`, `PRIORITIZATION_MATRIX.md`, `CHANGESET_SUMMARY.md`, and `CONTINUE_FROM_HERE.md` trace to R-P9-L01 through R-P9-L10. No new external source was used.

---

## Pass 10 — F202 macOS notarization tooling (2026-05-17 tenth pass)

### Local source evidence

| ID | Source |
|---|---|
| R-P10-L01 | `.github/workflows/build.yml` — existing PyInstaller release workflow; Pass 10 added macOS notarization and ZIP release upload wiring |
| R-P10-L02 | `opencut_server.spec` — PyInstaller bundle source for `dist/OpenCut-Server` |
| R-P10-L03 | `scripts/notarize_macos.sh` — Developer ID signing + notarytool submission script |
| R-P10-L04 | `docs/MACOS_NOTARIZATION.md` — required secrets and local release commands |
| R-P10-L05 | `python -m pytest tests/test_macos_notarization.py tests/test_release_smoke.py -q` — PASS, `15 passed` |
| R-P10-L06 | `ruff check tests/test_macos_notarization.py scripts/release_smoke.py --select E,F,I --ignore E501,E402` — PASS |
| R-P10-L07 | `C:\Program Files\Git\bin\bash.exe -n scripts/notarize_macos.sh` — PASS |
| R-P10-L08 | `python scripts/release_smoke.py --json` — PASS, all 13 steps green; pytest-fast `249 passed` |

### External source evidence

| ID | Source |
|---|---|
| R-P10-E01 | https://developer.apple.com/documentation/security/notarizing-macos-software-before-distribution — Apple notarization overview; confirms notary service behavior and altool retirement |
| R-P10-E02 | https://developer.apple.com/documentation/security/notarizing_macos_software_before_distribution/customizing_the_notarization_workflow — Apple custom workflow; confirms `xcrun notarytool submit --wait` and ZIP submission caveat |
| R-P10-E03 | https://developer.apple.com/documentation/technotes/tn3147-migrating-to-the-latest-notarization-tool — Apple migration note for the latest notarization tool |

**Source coverage assessment (Pass 10):** Pass 10 claims in `ROADMAP.md` v4.13, `PROJECT_CONTEXT.md`, `INSTALLER_AUDIT.md`, `FEATURE_BACKLOG_ADDENDUM.md`, `PRIORITIZATION_MATRIX.md`, `CHANGESET_SUMMARY.md`, and `CONTINUE_FROM_HERE.md` trace to R-P10-L01 through R-P10-L08 plus R-P10-E01 through R-P10-E03.

---

## Pass 11 — F204 release SBOM attachment (2026-05-17 eleventh pass)

### Local source evidence

| ID | Source |
|---|---|
| R-P11-L01 | `scripts/sbom.py` — existing CycloneDX 1.5 SBOM generator |
| R-P11-L02 | `.github/workflows/build.yml` — release workflow updated to generate/archive/upload `dist/opencut-sbom.cyclonedx.json` |
| R-P11-L03 | `tests/test_release_sbom.py` — generator and workflow wiring tests |
| R-P11-L04 | `python -m pytest tests/test_release_sbom.py tests/test_release_smoke.py -q` — PASS, `14 passed` |
| R-P11-L05 | `ruff check tests/test_release_sbom.py scripts/release_smoke.py --select E,F,I --ignore E501,E402` — PASS |
| R-P11-L06 | Workflow YAML parse command — PASS |
| R-P11-L07 | `python scripts/sbom.py --format json --output dist/opencut-sbom.cyclonedx.json` — PASS; generated CycloneDX 1.5 JSON |
| R-P11-L08 | `python scripts/release_smoke.py --json` — PASS, all 13 steps green; pytest-fast `251 passed` |

**Source coverage assessment (Pass 11):** Pass 11 claims in `ROADMAP.md` v4.14, `PROJECT_CONTEXT.md`, `INSTALLER_AUDIT.md`, `FEATURE_BACKLOG_ADDENDUM.md`, `PRIORITIZATION_MATRIX.md`, `CHANGESET_SUMMARY.md`, and `CONTINUE_FROM_HERE.md` trace to R-P11-L01 through R-P11-L08. No new external source was used.

---

## Pass 12 — F205 attempt + F207 installer FFmpeg manifest (2026-05-17 twelfth pass)

### Local source evidence

| ID | Source |
|---|---|
| R-P12-L01 | `ffmpeg/ffmpeg.exe -version` — bundled FFmpeg reports `8.0.1-essentials_build-www.gyan.dev` |
| R-P12-L02 | `installer/src/OpenCut.Installer/Models/AppConstants.cs` — added FFmpeg/ffprobe version constants |
| R-P12-L03 | `installer/src/OpenCut.Installer/Services/InstallEngine.cs` — WPF manifest writer |
| R-P12-L04 | `OpenCut.iss` — Inno manifest writer |
| R-P12-L05 | `tests/test_ffmpeg_installer_manifest.py` — static F207 contract test |
| R-P12-L06 | `python -m pytest tests/test_ffmpeg_installer_manifest.py tests/test_release_smoke.py -q` — PASS, `15 passed` |
| R-P12-L07 | `ruff check tests/test_ffmpeg_installer_manifest.py scripts/release_smoke.py --select E,F,I --ignore E501,E402` — PASS |
| R-P12-L08 | `python -m py_compile scripts/release_smoke.py` — PASS |
| R-P12-L09 | `python scripts/release_smoke.py --json` — PASS, all 13 steps green; pytest-fast `254 passed` |
| R-P12-L10 | `dotnet build installer/src/OpenCut.Installer/OpenCut.Installer.csproj --no-restore` — BLOCKED, no .NET SDK installed |
| R-P12-L11 | F205 coverage measurement command — timed out after 20 minutes; no `dist/coverage-f205.json` output |

**Source coverage assessment (Pass 12):** Pass 12 claims in `ROADMAP.md` v4.15, `PROJECT_CONTEXT.md`, `INSTALLER_AUDIT.md`, `FEATURE_BACKLOG_ADDENDUM.md`, `PRIORITIZATION_MATRIX.md`, `CHANGESET_SUMMARY.md`, and `CONTINUE_FROM_HERE.md` trace to R-P12-L01 through R-P12-L11. No new external source was used.

---

## Pass 13 — F208 OpenAPI contract gate (2026-05-17 thirteenth pass)

### Local source evidence

| ID | Source |
|---|---|
| R-P13-L01 | `opencut/openapi.py` — legacy OpenAPI 3.0.3 generator updated to convert Flask `<param>` syntax, emit path parameters, unique operation IDs, and mutating-method 400/403 responses |
| R-P13-L02 | `opencut/core/openapi_spec.py` — existing `/api/openapi.json` OpenAPI 3.1 generator already converts Flask path syntax |
| R-P13-L03 | `tests/test_openapi_contract.py` — new F208 route-parity and response-shape contract tests |
| R-P13-L04 | `scripts/release_smoke.py` — added `tests/test_openapi_contract.py` to the `pytest-fast` release gate |
| R-P13-L05 | `python -m pytest tests/test_openapi_contract.py tests/test_release_smoke.py -q` — PASS, `16 passed` |
| R-P13-L06 | `ruff check opencut/openapi.py tests/test_openapi_contract.py scripts/release_smoke.py --select E,F,I --ignore E501,E402` — PASS |
| R-P13-L07 | `python -m py_compile opencut/openapi.py scripts/release_smoke.py tests/test_openapi_contract.py` — PASS |
| R-P13-L08 | `python scripts/release_smoke.py --json` — PASS, all 13 steps green; pytest-fast `258 passed` |

**Source coverage assessment (Pass 13):** Pass 13 claims in `ROADMAP.md` v4.16, `PROJECT_CONTEXT.md`, `TEST_COVERAGE_GAPS.md`, `FEATURE_BACKLOG_ADDENDUM.md`, `PRIORITIZATION_MATRIX.md`, `CHANGESET_SUMMARY.md`, `RESEARCH_LOG.md`, and `CONTINUE_FROM_HERE.md` trace to R-P13-L01 through R-P13-L08. No new external source was used.

---

## Pass 14 — F209 MCP route consistency gate (2026-05-17 fourteenth pass)

### Local source evidence

| ID | Source |
|---|---|
| R-P14-L01 | `opencut/mcp_server.py` — `_TOOL_ROUTES` corrected so `opencut_chat_edit` points to shipped `POST /chat` |
| R-P14-L02 | `tests/test_mcp_server.py` — added live Flask route consistency test for all MCP default routes and special action routes |
| R-P14-L03 | `opencut/routes/system.py` — shipped `POST /chat` route for the chat editing assistant |
| R-P14-L04 | `python -m pytest tests/test_mcp_server.py tests/test_release_smoke.py -q` — PASS, `18 passed` |
| R-P14-L05 | `ruff check opencut/mcp_server.py tests/test_mcp_server.py --select E,F,I --ignore E501,E402` — PASS |
| R-P14-L06 | `python -m py_compile opencut/mcp_server.py tests/test_mcp_server.py` — PASS |
| R-P14-L07 | live route-table probe — PASS, `39` MCP tools / `39` route mappings / `0` missing backend routes |
| R-P14-L08 | `python scripts/release_smoke.py --json` — PASS, all 13 steps green; pytest-fast `259 passed` |

**Source coverage assessment (Pass 14):** Pass 14 claims in `ROADMAP.md` v4.17, `PROJECT_CONTEXT.md`, `TEST_COVERAGE_GAPS.md`, `FEATURE_BACKLOG_ADDENDUM.md`, `PRIORITIZATION_MATRIX.md`, `CHANGESET_SUMMARY.md`, `RESEARCH_LOG.md`, and `CONTINUE_FROM_HERE.md` trace to R-P14-L01 through R-P14-L08. No new external source was used.

---

## Pass 15 — F218 blueprint import-order stability (2026-05-17 fifteenth pass)

### Local source evidence

| ID | Source |
|---|---|
| R-P15-L01 | `opencut/routes/__init__.py` — explicit `get_core_blueprints()` order and `motion_design_api` alias registration |
| R-P15-L02 | `tests/test_route_collisions.py` — added stable blueprint order and alias-registration assertions |
| R-P15-L03 | `scripts/release_smoke.py` — added `tests/test_route_collisions.py` to `pytest-fast` |
| R-P15-L04 | `python -m pytest tests/test_route_collisions.py tests/test_release_smoke.py -q` — PASS, `19 passed` |
| R-P15-L05 | `ruff check tests/test_route_collisions.py scripts/release_smoke.py --select E,F,I --ignore E501,E402` — PASS after import-order formatting |
| R-P15-L06 | `python -m py_compile tests/test_route_collisions.py scripts/release_smoke.py` — PASS |
| R-P15-L07 | `python scripts/release_smoke.py --json` — PASS, all 13 steps green; pytest-fast `266 passed` |

**Source coverage assessment (Pass 15):** Pass 15 claims in `ROADMAP.md` v4.18, `PROJECT_CONTEXT.md`, `TEST_COVERAGE_GAPS.md`, `FEATURE_BACKLOG_ADDENDUM.md`, `PRIORITIZATION_MATRIX.md`, `CHANGESET_SUMMARY.md`, `RESEARCH_LOG.md`, and `CONTINUE_FROM_HERE.md` trace to R-P15-L01 through R-P15-L07. No new external source was used.

---

## Pass 16 — F219 SBOM completeness gate (2026-05-17 sixteenth pass)

### Local source evidence

| ID | Source |
|---|---|
| R-P16-L01 | `scripts/sbom.py` — added unique dependency component assembly, model-card components, JSON/XML dependency graph output, and SBOM CLI counts |
| R-P16-L02 | `tests/test_sbom_completeness.py` — new F219 regression tests for declared dependency coverage, 47 model-card components, unique `bom-ref` values, and dependency graph references |
| R-P16-L03 | `scripts/release_smoke.py` — added `tests/test_sbom_completeness.py` to `pytest-fast` |
| R-P16-L04 | `python -m pytest tests/test_sbom_completeness.py tests/test_release_sbom.py tests/test_release_smoke.py -q` — PASS, `17 passed` |
| R-P16-L05 | `ruff check scripts/sbom.py tests/test_sbom_completeness.py scripts/release_smoke.py --select E,F,I --ignore E501,E402` — PASS |
| R-P16-L06 | `python -m py_compile scripts/sbom.py tests/test_sbom_completeness.py scripts/release_smoke.py` — PASS |
| R-P16-L07 | `python scripts/sbom.py --format json --output dist/opencut-sbom-f219.cyclonedx.json` and `python scripts/sbom.py --format xml --output dist/opencut-sbom-f219.cyclonedx.xml` — PASS, 14 required components / 73 optional components / 47 model-card components |
| R-P16-L08 | `python scripts/release_smoke.py --json` — PASS, all 13 steps green; pytest-fast `269 passed` |

**Source coverage assessment (Pass 16):** Pass 16 claims in `ROADMAP.md` v4.19, `PROJECT_CONTEXT.md`, `TEST_COVERAGE_GAPS.md`, `FEATURE_BACKLOG_ADDENDUM.md`, `PRIORITIZATION_MATRIX.md`, `CHANGESET_SUMMARY.md`, `RESEARCH_LOG.md`, and `CONTINUE_FROM_HERE.md` trace to R-P16-L01 through R-P16-L08. No new external source was used.

---

## Pass 17 — F236 FCC caption display-settings tokens (2026-05-17 seventeenth pass)

### External source evidence

| ID | Source |
|---|---|
| R-P17-E01 | 47 CFR § 79.103(e), eCFR/Cornell mirror — https://www.law.cornell.edu/cfr/text/47/79.103 — readily-accessible caption display-setting factors: proximity, discoverability, previewability, consistency/persistence; compliance required for next-generation operating systems deployed after August 17, 2026 |
| R-P17-E02 | Federal Register final rule/compliance-date notice, FR Doc. 2025-02816 — https://www.federalregister.gov/d/2025-02816 — effective February 21, 2025; compliance date for 47 CFR 79.103(e) is August 17, 2026 |
| R-P17-E03 | FCC 24-79 Third Report and Order PDF — https://docs.fcc.gov/public/attachments/FCC-24-79A1_Rcd.pdf — display-setting surface includes caption presentation, color, opacity, size, font, caption background color/opacity, character edge attributes, and caption window color |

### Local source evidence

| ID | Source |
|---|---|
| R-P17-L01 | `opencut/core/caption_display_settings.py` — new canonical display-setting token schema, normalization, CSS preview values, and ASS `force_style` conversion |
| R-P17-L02 | `opencut/routes/captions.py` — added `/captions/display-settings/tokens`, `/captions/display-settings/preview`, and `display_settings` handling for `/captions/burnin/file` |
| R-P17-L03 | `tests/test_caption_display_settings.py` — new F236 regression tests for FCC factors, token coverage, normalization, preview payloads, and routes |
| R-P17-L04 | `scripts/release_smoke.py` — added `tests/test_caption_display_settings.py` to `pytest-fast` |
| R-P17-L05 | `python -m opencut.tools.dump_route_manifest` — regenerated `opencut/_generated/route_manifest.json`, now 1,361 routes / 101 blueprints |
| R-P17-L06 | `python -m opencut.tools.dump_api_aliases --check` — PASS, 15 aliases / 218 canonical `/api` routes |
| R-P17-L07 | `python -m opencut.tools.dump_feature_readiness --check` — PASS, 58 generated records / 67 route bindings |
| R-P17-L08 | `python -m pytest tests/test_caption_display_settings.py tests/test_route_manifest.py tests/test_release_smoke.py -q` — PASS, `21 passed` |
| R-P17-L09 | `ruff check opencut/core/caption_display_settings.py opencut/routes/captions.py tests/test_caption_display_settings.py scripts/release_smoke.py --select E,F,I --ignore E501,E402` — PASS |
| R-P17-L10 | `python -m py_compile opencut/core/caption_display_settings.py opencut/routes/captions.py tests/test_caption_display_settings.py scripts/release_smoke.py` — PASS |
| R-P17-L11 | `python scripts/release_smoke.py --json` — PASS, all 13 steps green; route manifest `1,361` routes / `101` blueprints; pytest-fast `273 passed` |

**Source coverage assessment (Pass 17):** Pass 17 claims in `ROADMAP.md` v4.20, `PROJECT_CONTEXT.md`, `FEATURE_BACKLOG_ADDENDUM.md`, `PRIORITIZATION_MATRIX.md`, `CHANGESET_SUMMARY.md`, `RESEARCH_LOG.md`, and `CONTINUE_FROM_HERE.md` trace to R-P17-E01 through R-P17-E03 and R-P17-L01 through R-P17-L11.

---

## Pass 18 — F237 loudness standards registry (2026-05-17 eighteenth pass)

### External source evidence

| ID | Source |
|---|---|
| R-P18-E01 | ITU-R BS.1770 recommendation page — https://www.itu.int/rec/R-REC-BS.1770-5-202311-I/en — confirms Recommendation BS.1770-5 (11/2023), approved 2023-11-22, status In force, free download |
| R-P18-E02 | ITU-R BS.1770 version listing — https://www.itu.int/rec/r-rec-bs.1770/_page.print — confirms BS.1770-5 is Main/In force and BS.1770-4 (10/2015) is Superseded |
| R-P18-E03 | ITU-R BS.1770-5 PDF — https://www.itu.int/dms_pubrec/itu-r/rec/bs/R-REC-BS.1770-5-202311-I!!PDF-E.pdf — defines algorithms for programme loudness and true-peak signal level |
| R-P18-E04 | EBU R 128 publication page — https://tech.ebu.ch/fr/publications/r128 — confirms Version 5.0 (November 2023), target average programme loudness -23 LUFS, and maximum true peak descriptor |
| R-P18-E05 | FFmpeg filters documentation, loudnorm — https://ffmpeg.org/ffmpeg-filters.html#loudnorm — confirms `loudnorm` is EBU R128 normalization that can target integrated loudness, LRA, and maximum true peak |
| R-P18-E06 | Spotify loudness normalization guidance — https://support.spotify.com/ee-en/artists/article/loudness-normalization/ — confirms Spotify adjusts tracks to -14 dB LUFS according to ITU 1770 and recommends masters below -1 dBTP |

### Local source evidence

| ID | Source |
|---|---|
| R-P18-L01 | `opencut/core/loudness_standards.py` — new canonical source-backed loudness standards, preset, and platform target registry |
| R-P18-L02 | `opencut/core/audio_suite.py` — now imports canonical loudness presets while preserving the historical `LOUDNESS_PRESETS` export |
| R-P18-L03 | `opencut/core/audio_analysis.py` — now imports the shared platform target map, preserving existing `broadcast = -24 LUFS` behavior and adding `ebu_broadcast = -23 LUFS` |
| R-P18-L04 | `opencut/core/broadcast_qc.py` — EBU R128 QC metadata now names ITU-R BS.1770-5, EBU R128 v5.0, and the EBU source URL |
| R-P18-L05 | `opencut/routes/audio.py` — `/audio/loudness-presets` now exposes presets, standards, and correction metadata; `/audio/normalize` response includes target/source metadata |
| R-P18-L06 | `tests/test_loudness_standards.py` — new F237 tests for current ITU/EBU facts, preset targets, compatibility exports, platform target semantics, and route payload |
| R-P18-L07 | `scripts/release_smoke.py` — added `tests/test_loudness_standards.py` to `pytest-fast` |
| R-P18-L08 | `python -m pytest tests/test_loudness_standards.py tests/test_release_smoke.py -q` — PASS, `17 passed` |
| R-P18-L09 | focused compatibility route slice — PASS, `9 passed` for loudness standards, legacy preset export, platform target tests, and route smoke |
| R-P18-L10 | `ruff check opencut/core/loudness_standards.py opencut/core/audio_suite.py opencut/core/audio_analysis.py opencut/core/broadcast_qc.py opencut/routes/audio.py tests/test_loudness_standards.py scripts/release_smoke.py --select E,F,I --ignore E501,E402` — PASS |
| R-P18-L11 | `python -m py_compile opencut/core/loudness_standards.py opencut/core/audio_suite.py opencut/core/audio_analysis.py opencut/core/broadcast_qc.py opencut/routes/audio.py tests/test_loudness_standards.py scripts/release_smoke.py` — PASS |
| R-P18-L12 | `python scripts/release_smoke.py --json` — PASS, all 13 steps green; route manifest `1,361` routes / `101` blueprints; pytest-fast `278 passed` |

**Source coverage assessment (Pass 18):** Pass 18 claims in `ROADMAP.md` v4.21, `PROJECT_CONTEXT.md`, `FEATURE_BACKLOG_ADDENDUM.md`, `PRIORITIZATION_MATRIX.md`, `CHANGESET_SUMMARY.md`, `RESEARCH_LOG.md`, and `CONTINUE_FROM_HERE.md` trace to R-P18-E01 through R-P18-E06 and R-P18-L01 through R-P18-L12.

---

## Pass 19 — F240 caption reading-speed profiles (2026-05-17 nineteenth pass)

### External source evidence

| ID | Source |
|---|---|
| R-P19-E01 | Netflix English (USA) Timed Text Style Guide — https://partnerhelp.netflixstudios.com/hc/en-us/articles/217350977-English-USA-Timed-Text-Style-Guide — current public guide lists adult programs at up to 20 characters per second and children's programs at up to 17 characters per second |
| R-P19-E02 | BBC Subtitle Guidelines archived official page — https://archive.ph/2026.01.08-135056/https%3A/www.bbc.co.uk/accessibility/forproducts/guides/subtitles/%23Spelling-out — version 1.2.4a recommends 160-180 words per minute and notes editorial adjustment by programme |
| R-P19-E03 | DCMP Captioning Key print page — https://dcmp.org/captioningkey/print — presentation-rate section says lower/middle/upper educational media should not exceed 130/140/160 words per minute |
| R-P19-E04 | 47 CFR § 79.1 eCFR/Cornell mirror — https://www.law.cornell.edu/cfr/text/47/79.1 — FCC caption quality rules require offline captions to display with enough time to be read completely but do not set a fixed WPM cap |
| R-P19-E05 | YouTube Help, Add subtitles & captions — https://support.google.com/youtube/answer/2734796?hl=en — official help documents caption text and timestamps but does not publish a hard reading-speed limit |

### Local source evidence

| ID | Source |
|---|---|
| R-P19-L01 | `opencut/core/caption_reading_profiles.py` — new source-backed registry for Netflix adult/children, BBC editorial, DCMP upper-level, FCC qualitative, and YouTube advisory reading-speed profiles |
| R-P19-L02 | `opencut/core/caption_compliance.py` — added per-call rule overrides so QC can apply reading-speed profiles without mutating global standards |
| R-P19-L03 | `opencut/core/caption_qc.py` — added `reading_profile` overlay support, profile metadata in API dictionaries, and fixed advisory-mode downgrades for actual violation names |
| R-P19-L04 | `opencut/routes/captions.py` — added `GET /captions/qc/reading-profiles` and `reading_profile` / `profile` / `speed_profile` support on `POST /captions/qc` |
| R-P19-L05 | `tests/test_caption_reading_profiles.py` — new F240 regression tests for source facts, aliases, adult-vs-children Netflix CPS, BBC WPM warnings, and route payloads |
| R-P19-L06 | `scripts/release_smoke.py` — added `tests/test_caption_reading_profiles.py` to `pytest-fast` |
| R-P19-L07 | `python -m opencut.tools.dump_route_manifest` — regenerated `opencut/_generated/route_manifest.json`, now 1,362 routes / 101 blueprints |
| R-P19-L08 | `python -m pytest tests/test_caption_reading_profiles.py tests/test_caption_qc.py tests/test_analysis.py::TestCaptionCompliance -q --tb=short` — PASS, `31 passed` |
| R-P19-L09 | `ruff check opencut/core/caption_reading_profiles.py opencut/core/caption_compliance.py opencut/core/caption_qc.py opencut/routes/captions.py scripts/release_smoke.py tests/test_caption_reading_profiles.py --select E,F,I --ignore E501,E402` — PASS |
| R-P19-L10 | `python -m py_compile opencut/core/caption_reading_profiles.py opencut/core/caption_compliance.py opencut/core/caption_qc.py opencut/routes/captions.py scripts/release_smoke.py tests/test_caption_reading_profiles.py` — PASS |
| R-P19-L11 | `python -m opencut.tools.dump_route_manifest --check --quiet`, `python -m opencut.tools.dump_api_aliases --check`, `python -m opencut.tools.dump_feature_readiness --check` — PASS; route manifest 1,362 routes / 101 blueprints, aliases 15 / 218, readiness 58 records / 67 bindings |
| R-P19-L12 | `python scripts/release_smoke.py --json` — PASS, all 13 steps green; pytest-fast `284 passed`; pip-audit no vulnerabilities; npm advisory gate reports only the documented Vite waiver |

**Source coverage assessment (Pass 19):** Pass 19 claims in `ROADMAP.md` v4.22, `PROJECT_CONTEXT.md`, `FEATURE_BACKLOG_ADDENDUM.md`, `PRIORITIZATION_MATRIX.md`, `CHANGESET_SUMMARY.md`, `RESEARCH_LOG.md`, and `CONTINUE_FROM_HERE.md` trace to R-P19-E01 through R-P19-E05 and R-P19-L01 through R-P19-L12.

---

## Pass 20 — F241 text-shaping gate (2026-05-17 twentieth pass)

### Local source evidence

| ID | Source |
|---|---|
| R-P20-L01 | `opencut/core/caption_burnin.py` — burn-in path uses FFmpeg `ass` / `subtitles` filters for caption hardcoding |
| R-P20-L02 | `opencut/core/styled_captions.py` — styled caption overlay path uses Pillow by default and optional `skia-python` when installed |
| R-P20-L03 | bundled `ffmpeg/ffmpeg.exe -hide_banner -version` — reports `8.0.1-essentials_build-www.gyan.dev` with `--enable-libass`, `--enable-libharfbuzz`, and `--enable-libfribidi` in configuration |
| R-P20-L04 | bundled `ffmpeg/ffmpeg.exe -hide_banner -filters` — exposes exact `ass` and `subtitles` video filters; `greyedge assumption` confirmed why substring matches are unsafe |
| R-P20-L05 | local Pillow feature probe — Pillow `12.2.0` reports `raqm=false`, `harfbuzz=false`, `fribidi=false`, `freetype2=true` |
| R-P20-L06 | `opencut/tools/text_shaping_gate.py` — new machine-readable gate resolving FFmpeg, checking libass/HarfBuzz/FriBidi/filter support, and reporting Pillow/Skia shaping capability |
| R-P20-L07 | `scripts/release_smoke.py` — added `text-shaping` step and included `tests/test_text_shaping_gate.py` in `pytest-fast` |
| R-P20-L08 | `.github/workflows/build.yml` — CI now runs `python -m opencut.tools.text_shaping_gate --json` after standard dependency installation |
| R-P20-L09 | `tests/test_text_shaping_gate.py` — new F241 regression tests for exact FFmpeg parsing, missing-HarfBuzz failure, strict Pillow promotion, release-smoke wiring, and workflow wiring |
| R-P20-L10 | `python -m opencut.tools.text_shaping_gate --json` — PASS; FFmpeg/libass hard gate OK, Pillow RAQM advisory warning, Skia skipped |
| R-P20-L11 | `python -m pytest tests/test_text_shaping_gate.py tests/test_release_smoke.py -q --tb=short` — PASS, `17 passed` |
| R-P20-L12 | `ruff check opencut/tools/text_shaping_gate.py scripts/release_smoke.py tests/test_text_shaping_gate.py --select E,F,I --ignore E501,E402` — PASS |
| R-P20-L13 | `python -m py_compile opencut/tools/text_shaping_gate.py scripts/release_smoke.py tests/test_text_shaping_gate.py` — PASS |
| R-P20-L14 | `python scripts/release_smoke.py --json` — PASS, all 14 steps green; `text-shaping` reports one advisory Pillow RAQM warning; pytest-fast `289 passed`; pip-audit no vulnerabilities; npm advisory gate reports only the documented Vite waiver |

**Source coverage assessment (Pass 20):** Pass 20 claims in `ROADMAP.md` v4.23, `PROJECT_CONTEXT.md`, `FEATURE_BACKLOG_ADDENDUM.md`, `PRIORITIZATION_MATRIX.md`, `CHANGESET_SUMMARY.md`, `RESEARCH_LOG.md`, and `CONTINUE_FROM_HERE.md` trace to R-P20-L01 through R-P20-L14.
