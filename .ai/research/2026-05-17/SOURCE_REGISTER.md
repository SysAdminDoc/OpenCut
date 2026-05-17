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
