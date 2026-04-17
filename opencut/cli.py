"""
OpenCut CLI - Command line interface for video editing automation.

Usage:
    opencut silence  input.mp4 -o output.xml       # Remove silences, export Premiere XML
    opencut captions input.mp4 -o output.srt        # Generate captions
    opencut podcast  input.mp4 -o output.xml        # Diarize + multicam switch
    opencut full     input.mp4 -o output.xml        # All features combined
    opencut info     input.mp4                      # Show media info
"""

import os
import sys
import time

import click
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from opencut import __version__

console = Console()

BANNER = r"""
  ___                    ____      _
 / _ \ _ __   ___ _ __  / ___|   _| |_
| | | | '_ \ / _ \ '_ \| |  | | | | __|
| |_| | |_) |  __/ | | | |__| |_| | |_
 \___/| .__/ \___|_| |_|\____\__,_|\__|
      |_|
"""


def print_banner():
    console.print(Panel(
        BANNER + "  Open Source Video Editing Automation\n  github.com/opencut",
        style="bold cyan",
        box=box.ROUNDED,
        padding=(0, 2),
    ))


@click.group()
@click.version_option(version=__version__, prog_name="opencut")
def cli():
    """OpenCut - Open source video editing automation for Premiere Pro."""
    pass


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), default=None, help="Output XML file path")
@click.option("-t", "--threshold", type=float, default=-30.0, help="Silence threshold in dB (default: -30)")
@click.option("-d", "--min-duration", type=float, default=0.5, help="Minimum silence duration in seconds (default: 0.5)")
@click.option("--padding-before", type=float, default=0.1, help="Padding before speech in seconds (default: 0.1)")
@click.option("--padding-after", type=float, default=0.1, help="Padding after speech in seconds (default: 0.1)")
@click.option("--min-speech", type=float, default=0.25, help="Minimum speech segment duration (default: 0.25)")
@click.option("--preset", type=click.Choice(["default", "aggressive", "conservative", "podcast", "youtube"]), default=None, help="Use a preset configuration")
@click.option("--format", "export_format", type=click.Choice(["premiere", "resolve"]), default="premiere", help="Export format (default: premiere)")
@click.option("--name", "seq_name", type=str, default="OpenCut Edit", help="Sequence name in the exported timeline")
@click.option("--dry-run", is_flag=True, help="Analyze only, don't export")
def silence(input_file, output, threshold, min_duration, padding_before, padding_after, min_speech, preset, export_format, seq_name, dry_run):
    """Remove silences from a video/audio file.

    Detects silent sections and exports a Premiere Pro XML timeline
    with only the speech segments, ready to import.
    """
    print_banner()

    from .core.silence import detect_speech, get_edit_summary
    from .export.premiere import export_premiere_xml
    from .utils.config import ExportConfig, SilenceConfig, get_preset

    # Apply preset or manual settings
    if preset:
        cfg = get_preset(preset)
        scfg = cfg.silence
    else:
        scfg = SilenceConfig(
            threshold_db=threshold,
            min_duration=min_duration,
            padding_before=padding_before,
            padding_after=padding_after,
            min_speech_duration=min_speech,
        )

    ecfg = ExportConfig(format=export_format, sequence_name=seq_name)

    # Default output path
    if output is None:
        base = os.path.splitext(input_file)[0]
        output = f"{base}_opencut.xml"

    # Detect speech
    console.print(f"\n[bold]Analyzing:[/bold] {input_file}")
    console.print(f"[dim]Threshold: {scfg.threshold_db} dB | Min silence: {scfg.min_duration}s | Padding: {scfg.padding_before}s/{scfg.padding_after}s[/dim]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Detecting silences...", total=None)
        start_time = time.time()

        segments = detect_speech(input_file, config=scfg)

        elapsed = time.time() - start_time
        progress.update(task, description=f"[green]Detection complete ({elapsed:.1f}s)")
        progress.stop()

    # Show summary
    summary = get_edit_summary(input_file, segments)
    _print_summary(summary, segments)

    if dry_run:
        console.print("\n[yellow]Dry run — no file exported.[/yellow]")
        return

    # Export
    console.print(f"\n[bold]Exporting:[/bold] {output}")
    export_premiere_xml(input_file, segments, output, config=ecfg)
    console.print("[green bold]Done![/green bold] Import this XML into Premiere Pro via File > Import.\n")


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), default=None, help="Output subtitle file path")
@click.option("--format", "sub_format", type=click.Choice(["srt", "vtt", "json"]), default="srt", help="Subtitle format (default: srt)")
@click.option("--model", type=str, default="base", help="Whisper model: tiny, base, small, medium, large-v3, turbo")
@click.option("--language", type=str, default=None, help="Language code (auto-detect if not set)")
@click.option("--translate", is_flag=True, help="Translate to English")
@click.option("--word-timestamps/--no-word-timestamps", default=True, help="Enable word-level timestamps")
def captions(input_file, output, sub_format, model, language, translate, word_timestamps):
    """Generate captions/subtitles using AI speech recognition.

    Uses OpenAI Whisper to transcribe speech and generate subtitle files.
    Supports 99+ languages with auto-detection.
    """
    print_banner()

    from .core.captions import check_whisper_available, transcribe
    from .export.srt import export_json, export_srt, export_vtt
    from .utils.config import CaptionConfig

    # Check Whisper availability
    available, backend = check_whisper_available()
    if not available:
        console.print("[red bold]Error:[/red bold] No Whisper backend installed.\n")
        console.print("Install one of:")
        console.print("  [cyan]pip install openai-whisper[/cyan]        # Reference implementation")
        console.print("  [cyan]pip install faster-whisper[/cyan]        # Fastest (recommended)")
        console.print("  [cyan]pip install whisperx[/cyan]              # Best word timestamps")
        sys.exit(1)

    console.print(f"[dim]Using Whisper backend: {backend}[/dim]")

    config = CaptionConfig(
        model=model,
        language=language,
        word_timestamps=word_timestamps,
        translate=translate,
        output_format=sub_format,
    )

    # Default output path
    if output is None:
        base = os.path.splitext(input_file)[0]
        output = f"{base}.{sub_format}"

    console.print(f"\n[bold]Transcribing:[/bold] {input_file}")
    console.print(f"[dim]Model: {model} | Language: {language or 'auto'} | Format: {sub_format}[/dim]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Transcribing audio...", total=None)
        start_time = time.time()

        result = transcribe(input_file, config=config)

        elapsed = time.time() - start_time
        progress.update(task, description=f"[green]Transcription complete ({elapsed:.1f}s)")
        progress.stop()

    # Export
    console.print(f"\n[bold]Language detected:[/bold] {result.language}")
    console.print(f"[bold]Segments:[/bold] {len(result.segments)}")
    console.print(f"[bold]Words:[/bold] {result.word_count}")

    if sub_format == "srt":
        export_srt(result, output)
    elif sub_format == "vtt":
        export_vtt(result, output)
    elif sub_format == "json":
        export_json(result, output)

    console.print(f"\n[green bold]Saved:[/green bold] {output}\n")


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), default=None, help="Output XML file path")
@click.option("--speakers", type=int, default=None, help="Expected number of speakers (auto-detect if not set)")
@click.option("--hf-token", type=str, default=None, help="HuggingFace auth token for pyannote")
@click.option("--min-segment", type=float, default=1.5, help="Minimum segment duration before camera switch (default: 1.5)")
@click.option("--name", "seq_name", type=str, default="OpenCut Podcast", help="Sequence name")
def podcast(input_file, output, speakers, hf_token, min_segment, seq_name):
    """Edit podcasts with automatic speaker-based camera switching.

    Uses AI speaker diarization to detect who is speaking and generates
    a multicam timeline that switches cameras at speaker changes.
    """
    print_banner()

    from .core.diarize import check_pyannote_available, diarize
    from .utils.config import DiarizeConfig

    if not check_pyannote_available():
        console.print("[red bold]Error:[/red bold] pyannote.audio not installed.\n")
        console.print("Install with: [cyan]pip install pyannote.audio[/cyan]")
        console.print("\nYou also need a HuggingFace token:")
        console.print("  1. Create account at [link]https://huggingface.co[/link]")
        console.print("  2. Accept model terms at [link]https://huggingface.co/pyannote/speaker-diarization-3.1[/link]")
        console.print("  3. Pass token via [cyan]--hf-token[/cyan] or [cyan]HUGGINGFACE_TOKEN[/cyan] env var")
        sys.exit(1)

    config = DiarizeConfig(
        num_speakers=speakers,
        hf_token=hf_token,
        min_segment_duration=min_segment,
    )

    if output is None:
        base = os.path.splitext(input_file)[0]
        output = f"{base}_podcast.xml"

    console.print(f"\n[bold]Diarizing:[/bold] {input_file}")
    console.print(f"[dim]Speakers: {speakers or 'auto'} | Min segment: {min_segment}s[/dim]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing speakers...", total=None)
        start_time = time.time()

        result = diarize(input_file, config=config)

        elapsed = time.time() - start_time
        progress.update(task, description=f"[green]Diarization complete ({elapsed:.1f}s)")
        progress.stop()

    # Show speaker breakdown
    console.print(f"\n[bold]Speakers detected:[/bold] {result.num_speakers}")
    durations = result.get_speaker_durations()

    table = Table(box=box.SIMPLE)
    table.add_column("Speaker", style="cyan")
    table.add_column("Duration", justify="right")
    table.add_column("Segments", justify="right")

    for speaker in result.speakers:
        dur = durations.get(speaker, 0)
        seg_count = len(result.get_speaker_segments(speaker))
        table.add_row(speaker, f"{dur:.1f}s", str(seg_count))

    console.print(table)

    # Convert to camera switches and export
    switches = result.to_camera_switches(min_segment_duration=min_segment)
    console.print(f"\n[bold]Camera switches:[/bold] {len(switches)}")

    # Export multicam XML
    from .core.multicam_xml import generate_multicam_xml

    # Build source files mapping from command args or auto-detect
    source_map = {}
    for i, speaker in enumerate(result.speakers):
        source_map[speaker] = input_file  # Same file, different tracks

    xml_result = generate_multicam_xml(
        cuts=switches,
        source_files=source_map,
        sequence_name=f"Multicam - {os.path.basename(input_file)}",
        output_path=output,
    )
    console.print(f"\n[green]Multicam XML exported:[/green] {xml_result['output']}")
    console.print(f"  Cuts: {xml_result['cuts_count']}, Duration: {xml_result['duration']:.1f}s\n")


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), default=None, help="Output directory for all generated files")
@click.option("--preset", type=click.Choice(["default", "aggressive", "conservative", "podcast", "youtube"]), default="youtube", help="Preset configuration")
@click.option("--skip-captions", is_flag=True, help="Skip caption generation")
@click.option("--skip-zoom", is_flag=True, help="Skip auto-zoom")
@click.option("--name", "seq_name", type=str, default="OpenCut Full Edit", help="Sequence name")
def full(input_file, output, preset, skip_captions, skip_zoom, seq_name):
    """Run the full editing pipeline: silence removal + captions + zoom.

    Combines all OpenCut features into a single workflow.
    Generates Premiere Pro XML and subtitle files.
    """
    print_banner()

    from .core.silence import detect_speech, get_edit_summary
    from .core.zoom import generate_zoom_events
    from .export.premiere import export_premiere_xml
    from .utils.config import ExportConfig, get_preset

    cfg = get_preset(preset)
    ecfg = ExportConfig(sequence_name=seq_name)

    # Setup output paths
    base = os.path.splitext(input_file)[0]
    if output:
        os.makedirs(output, exist_ok=True)
        base = os.path.join(output, os.path.basename(base))

    xml_path = f"{base}_opencut.xml"
    srt_path = f"{base}_opencut.srt"

    console.print(f"\n[bold]Full pipeline:[/bold] {input_file}")
    console.print(f"[dim]Preset: {preset}[/dim]\n")

    # Step 1: Silence detection
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), TimeElapsedColumn(), console=console) as progress:
        task = progress.add_task("[bold]Step 1/3:[/bold] Detecting silences...", total=None)
        segments = detect_speech(input_file, config=cfg.silence)
        progress.update(task, description=f"[green]Step 1/3: Found {len(segments)} speech segments")
        progress.stop()

    summary = get_edit_summary(input_file, segments)
    _print_summary(summary, segments)

    # Step 2: Zoom detection
    zoom_events = None
    if not skip_zoom:
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), TimeElapsedColumn(), console=console) as progress:
            task = progress.add_task("[bold]Step 2/3:[/bold] Analyzing zoom points...", total=None)
            zoom_events = generate_zoom_events(input_file, config=cfg.zoom, speech_segments=segments)
            progress.update(task, description=f"[green]Step 2/3: Found {len(zoom_events)} zoom points")
            progress.stop()
    else:
        console.print("[dim]Step 2/3: Zoom skipped[/dim]")

    # Step 3: Captions
    if not skip_captions:
        from .core.captions import check_whisper_available
        available, backend = check_whisper_available()
        if available:
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), TimeElapsedColumn(), console=console) as progress:
                task = progress.add_task("[bold]Step 3/3:[/bold] Generating captions...", total=None)
                from .core.captions import transcribe
                from .export.srt import export_srt
                result = transcribe(input_file, config=cfg.captions)
                export_srt(result, srt_path)
                progress.update(task, description=f"[green]Step 3/3: Generated {len(result.segments)} caption segments")
                progress.stop()
            console.print(f"[bold]Captions saved:[/bold] {srt_path}")
        else:
            console.print("[yellow]Step 3/3: Whisper not installed, skipping captions[/yellow]")
    else:
        console.print("[dim]Step 3/3: Captions skipped[/dim]")

    # Export timeline
    console.print(f"\n[bold]Exporting timeline:[/bold] {xml_path}")
    export_premiere_xml(input_file, segments, xml_path, config=ecfg, zoom_events=zoom_events)
    console.print("\n[green bold]Done![/green bold] Import the XML into Premiere Pro via File > Import.\n")


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
def info(input_file):
    """Display media file information."""
    print_banner()

    from .utils.media import probe

    info = probe(input_file)

    console.print(f"\n[bold]File:[/bold] {info.filename}")
    console.print(f"[bold]Path:[/bold] {os.path.abspath(info.path)}")
    console.print(f"[bold]Duration:[/bold] {info.duration:.3f}s")
    console.print(f"[bold]Format:[/bold] {info.format_name}")

    if info.has_video:
        v = info.video
        console.print("\n[bold cyan]Video:[/bold cyan]")
        console.print(f"  Resolution: {v.width}x{v.height}")
        console.print(f"  Frame Rate: {v.fps:.3f} fps (timebase={v.effective_timebase}, ntsc={v.ntsc})")
        console.print(f"  Codec: {v.codec}")
        console.print(f"  PAR: {v.pixel_aspect_ratio}")

    if info.has_audio:
        a = info.audio
        console.print("\n[bold cyan]Audio:[/bold cyan]")
        console.print(f"  Sample Rate: {a.sample_rate} Hz")
        console.print(f"  Channels: {a.channels}")
        console.print(f"  Bit Depth: {a.bit_depth}")
        console.print(f"  Codec: {a.codec}")

    console.print()


def _print_summary(summary: dict, segments):
    """Print edit summary table."""
    table = Table(title="Edit Summary", box=box.ROUNDED, show_header=False, padding=(0, 2))
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right", style="cyan")

    table.add_row("Original duration", summary["original_formatted"])
    table.add_row("Kept duration", summary["kept_formatted"])
    table.add_row("Removed duration", f"[red]{summary['removed_formatted']}[/red]")
    table.add_row("Reduction", f"[green]{summary['reduction_percent']:.1f}%[/green]")
    table.add_row("Speech segments", str(summary["segments_count"]))

    console.print(table)


@cli.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
@click.option("--port", type=int, default=5679, help="Port to listen on (default: 5679)")
@click.option("--debug", is_flag=True, help="Enable debug mode")
def server(host, port, debug):
    """Start the OpenCut backend server for the Premiere Pro panel.

    The server runs on localhost:5679 and handles processing requests
    from the CEP panel inside Premiere Pro.
    """
    from .server import run_server
    run_server(host=host, port=port, debug=debug)


def _resolve_output_dir(input_file, output_dir):
    """Resolve output directory, creating it if needed. Returns base path."""
    base = os.path.splitext(input_file)[0]
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.join(output_dir, os.path.basename(base))
    return base


@cli.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--provider", type=click.Choice(["ollama", "openai", "anthropic", "gemini"]), default="ollama", help="LLM provider")
@click.option("--model", default="llama3", help="LLM model name")
@click.option("--api-key", default="", help="API key (for OpenAI/Anthropic/Gemini)")
@click.option("--max-chapters", type=click.IntRange(1, 100), default=15, help="Maximum chapters to generate (1-100)")
@click.option("--whisper-model", default="turbo", help="Whisper model for transcription (turbo, base, small, medium, large-v3, distil-large-v3)")
@click.option("--output", "-o", default=None, help="Output text file path")
def chapters(file, provider, model, api_key, max_chapters, whisper_model, output):
    """Generate YouTube chapter timestamps from video transcript."""
    print_banner()

    try:
        from .core.captions import transcribe_audio
        from .core.chapter_gen import generate_chapters
        from .core.llm import LLMConfig
    except ImportError as e:
        console.print(f"[red bold]Error:[/red bold] Missing dependency: {e}")
        console.print("Ensure opencut[chapters] extras are installed.")
        sys.exit(1)

    # Validate whisper model — same allowlist used by the route layer.
    from .security import VALID_WHISPER_MODELS
    if whisper_model not in VALID_WHISPER_MODELS:
        console.print(f"[red bold]Error:[/red bold] Invalid --whisper-model '{whisper_model}'.")
        console.print(f"Valid models: {', '.join(sorted(VALID_WHISPER_MODELS))}")
        sys.exit(2)

    llm_cfg = LLMConfig(provider=provider, model=model, api_key=api_key)

    console.print(f"\n[bold]Generating chapters:[/bold] {file}")
    console.print(f"[dim]Provider: {provider} | Model: {model} | Whisper: {whisper_model} | Max chapters: {max_chapters}[/dim]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Transcribing audio...", total=None)
        start_time = time.time()
        segments = transcribe_audio(file, model=whisper_model)
        elapsed = time.time() - start_time
        progress.update(task, description=f"[green]Transcription complete ({elapsed:.1f}s)")
        progress.stop()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generating chapter timestamps...", total=None)
        start_time = time.time()
        result = generate_chapters(segments, llm_cfg, max_chapters=max_chapters)
        elapsed = time.time() - start_time
        progress.update(task, description=f"[green]Chapters generated ({elapsed:.1f}s)")
        progress.stop()

    description_block = result.get("description_block", "") if isinstance(result, dict) else getattr(result, "description_block", "")

    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(description_block)
        console.print(f"\n[green bold]Saved:[/green bold] {output}\n")
    else:
        console.print("\n[bold]Chapter Timestamps:[/bold]\n")
        console.print(description_block)


@cli.command("repeat-detect")
@click.argument("file", type=click.Path(exists=True))
@click.option("--threshold", default=0.6, help="Similarity threshold (0-1)")
@click.option("--model", default="base", help="Whisper model")
@click.option("--output", "-o", default=None, help="Output JSON path")
def repeat_detect(file, threshold, model, output):
    """Detect and list repeated/fumbled takes in a recording."""
    print_banner()

    try:
        from .core.captions import transcribe_audio
        from .core.repeat_detect import detect_repeated_takes
    except ImportError as e:
        console.print(f"[red bold]Error:[/red bold] Missing dependency: {e}")
        console.print("Ensure opencut[repeat-detect] extras are installed.")
        sys.exit(1)

    console.print(f"\n[bold]Detecting repeated takes:[/bold] {file}")
    console.print(f"[dim]Whisper model: {model} | Similarity threshold: {threshold}[/dim]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Transcribing audio...", total=None)
        start_time = time.time()
        segments = transcribe_audio(file, model=model)
        elapsed = time.time() - start_time
        progress.update(task, description=f"[green]Transcription complete ({elapsed:.1f}s)")
        progress.stop()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing repeated takes...", total=None)
        start_time = time.time()
        result = detect_repeated_takes(segments, threshold=threshold)
        elapsed = time.time() - start_time
        progress.update(task, description=f"[green]Analysis complete ({elapsed:.1f}s)")
        progress.stop()

    repeats = result.get("repeats", []) if isinstance(result, dict) else result
    console.print(f"\n[bold]Repeated takes found:[/bold] {len(repeats)}\n")

    if repeats:
        if output:
            import json
            with open(output, "w", encoding="utf-8") as f:
                json.dump([r if isinstance(r, dict) else vars(r) for r in repeats], f, indent=2)
            console.print(f"[green bold]Saved:[/green bold] {output}\n")
        else:
            table = Table(title="Repeated Takes", box=box.ROUNDED)
            table.add_column("Take #", style="cyan", justify="right")
            table.add_column("Start", justify="right")
            table.add_column("End", justify="right")
            table.add_column("Similarity", justify="right", style="yellow")
            table.add_column("Text", style="dim")

            for i, r in enumerate(repeats, 1):
                data = r if isinstance(r, dict) else vars(r)
                table.add_row(
                    str(i),
                    f"{data.get('start', 0):.2f}s",
                    f"{data.get('end', 0):.2f}s",
                    f"{data.get('similarity', 0):.2f}",
                    str(data.get('text', ''))[:60],
                )
            console.print(table)
    else:
        console.print("[green]No repeated takes detected.[/green]\n")


@cli.group()
def search():
    """Footage search commands."""
    pass


@search.command("index")
@click.argument("files", nargs=-1, type=click.Path(exists=True))
@click.option("--model", default="base", help="Whisper model")
def search_index(files, model):
    """Index media files for content search."""
    print_banner()

    try:
        from .core.captions import transcribe_audio
        from .core.footage_search import index_file
    except ImportError as e:
        console.print(f"[red bold]Error:[/red bold] Missing dependency: {e}")
        console.print("Ensure opencut[search] extras are installed.")
        sys.exit(1)

    if not files:
        console.print("[yellow]No files provided to index.[/yellow]")
        return

    console.print(f"\n[bold]Indexing {len(files)} file(s):[/bold]")
    console.print(f"[dim]Whisper model: {model}[/dim]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Indexing {len(files)} file(s)...", total=None)
        start_time = time.time()
        for filepath in files:
            segments = transcribe_audio(filepath, model=model)
            index_file(filepath, segments)
        elapsed = time.time() - start_time
        progress.update(task, description=f"[green]Indexing complete ({elapsed:.1f}s)")
        progress.stop()

    console.print(f"[green bold]Indexed {len(files)} file(s) successfully.[/green bold]\n")


@search.command("query")
@click.argument("query")
@click.option("--top-k", default=10, help="Number of results")
def search_query(query, top_k):
    """Search indexed footage by spoken content."""
    print_banner()

    try:
        from .core.footage_search import search_footage
    except ImportError as e:
        console.print(f"[red bold]Error:[/red bold] Missing dependency: {e}")
        console.print("Ensure opencut[search] extras are installed.")
        sys.exit(1)

    console.print(f"\n[bold]Searching:[/bold] {query!r}")
    console.print(f"[dim]Top {top_k} results[/dim]\n")

    results = search_footage(query, top_k=top_k)

    if not results:
        console.print("[yellow]No results found. Have you indexed any files with [cyan]opencut search index[/cyan]?[/yellow]\n")
        return

    table = Table(title=f'Search results for "{query}"', box=box.ROUNDED)
    table.add_column("File", style="cyan")
    table.add_column("Start", justify="right")
    table.add_column("End", justify="right")
    table.add_column("Score", justify="right", style="yellow")
    table.add_column("Text", style="dim")

    for r in results:
        data = r if isinstance(r, dict) else vars(r)
        table.add_row(
            os.path.basename(str(data.get('path', data.get('file', '')))),
            f"{data.get('start', 0):.2f}s",
            f"{data.get('end', 0):.2f}s",
            f"{data.get('score', 0):.3f}",
            str(data.get('text', ''))[:60],
        )
    console.print(table)
    console.print()


@cli.command("color-match")
@click.argument("source", type=click.Path(exists=True))
@click.argument("reference", type=click.Path(exists=True))
@click.option("--output-dir", "-o", default=None)
@click.option("--strength", default=1.0, help="Match strength 0-1")
def color_match(source, reference, output_dir, strength):
    """Match the color profile of source clip to reference clip."""
    print_banner()

    try:
        from .core.color_match import color_match_video
    except ImportError as e:
        console.print(f"[red bold]Error:[/red bold] Missing dependency: {e}")
        console.print("Ensure opencv-python and numpy are installed.")
        sys.exit(1)

    base = _resolve_output_dir(source, output_dir)
    output_path = f"{base}_color_matched{os.path.splitext(source)[1]}"

    console.print(f"\n[bold]Color matching:[/bold] {source}")
    console.print(f"[bold]Reference:[/bold] {reference}")
    console.print(f"[dim]Strength: {strength}[/dim]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Applying color match...", total=None)
        start_time = time.time()
        color_match_video(source, reference, output_path, strength=strength)
        elapsed = time.time() - start_time
        progress.update(task, description=f"[green]Color match complete ({elapsed:.1f}s)")
        progress.stop()

    console.print(f"\n[green bold]Saved:[/green bold] {output_path}\n")


@cli.command("loudness-match")
@click.argument("files", nargs=-1, type=click.Path(exists=True), required=True)
@click.option("--target-lufs", default=-14.0, help="Target LUFS level")
@click.option("--output-dir", "-o", default=None)
def loudness_match(files, target_lufs, output_dir):
    """Normalize loudness across multiple clips to a target LUFS level."""
    print_banner()

    try:
        from .core.loudness_match import batch_loudness_match
    except ImportError as e:
        console.print(f"[red bold]Error:[/red bold] Missing dependency: {e}")
        console.print("FFmpeg must be installed for loudness normalization.")
        sys.exit(1)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    console.print(f"\n[bold]Loudness normalizing {len(files)} file(s):[/bold]")
    console.print(f"[dim]Target: {target_lufs} LUFS[/dim]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Normalizing {len(files)} file(s)...", total=None)
        start_time = time.time()
        results = batch_loudness_match(list(files), output_dir=output_dir or os.path.dirname(files[0]), target_lufs=target_lufs)
        elapsed = time.time() - start_time
        progress.update(task, description=f"[green]Normalization complete ({elapsed:.1f}s)")
        progress.stop()

    console.print(f"\n[green bold]Normalized {len(files)} file(s):[/green bold]")
    for r in results:
        path = r.get("output", r) if isinstance(r, dict) else r
        console.print(f"  {path}")
    console.print()


@cli.command("auto-zoom")
@click.argument("file", type=click.Path(exists=True))
@click.option("--zoom-amount", default=1.15, help="Zoom factor (e.g. 1.15 = 15%% zoom)")
@click.option("--easing", default="ease_in_out", type=click.Choice(["linear", "ease_in", "ease_out", "ease_in_out"]))
@click.option("--output-dir", "-o", default=None)
@click.option("--apply/--keyframes-only", default=True, help="Bake zoom into output or return keyframe JSON")
def auto_zoom(file, zoom_amount, easing, output_dir, apply):
    """Apply face-tracked auto-zoom effect to a talking-head clip."""
    print_banner()

    try:
        from .core.auto_zoom import generate_zoom_keyframes
    except ImportError as e:
        console.print(f"[red bold]Error:[/red bold] Missing dependency: {e}")
        console.print("Ensure opencv-python is installed: pip install opencv-python-headless")
        sys.exit(1)

    base = _resolve_output_dir(file, output_dir)
    ext = os.path.splitext(file)[1]
    keyframes_path = f"{base}_autozoom_keyframes.json"

    console.print(f"\n[bold]Auto-zoom:[/bold] {file}")
    console.print(f"[dim]Zoom: {zoom_amount}x | Easing: {easing} | Apply: {apply}[/dim]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Detecting faces and generating keyframes...", total=None)
        start_time = time.time()
        result = generate_zoom_keyframes(
            file, zoom_amount=zoom_amount, easing=easing,
        )
        elapsed = time.time() - start_time
        progress.update(task, description=f"[green]Keyframe generation complete ({elapsed:.1f}s)")
        progress.stop()

    keyframes = result.get("keyframes", []) if isinstance(result, dict) else result
    console.print(f"\n[bold]Generated {len(keyframes)} zoom keyframes[/bold]")

    if apply and keyframes:
        # Apply zoom via FFmpeg zoompan filter
        from .helpers import get_ffmpeg_path, get_video_info, run_ffmpeg
        info = get_video_info(file)
        fps = info.get("fps", 30)
        output_path = f"{base}_autozoom{ext}"
        console.print("[bold]Applying zoom to video...[/bold]")
        # Build zoompan filter — simplified: use first keyframe zoom for now
        zoom_val = keyframes[0].get("zoom", zoom_amount) if keyframes else zoom_amount
        run_ffmpeg([
            get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
            "-i", file,
            "-vf", f"zoompan=z={zoom_val}:d=1:s={info['width']}x{info['height']}:fps={fps}",
            "-c:a", "copy", output_path,
        ])
        console.print(f"\n[green bold]Saved:[/green bold] {output_path}\n")
    else:
        import json as _json
        with open(keyframes_path, "w", encoding="utf-8") as f:
            _json.dump(keyframes, f, indent=2)
        console.print(f"\n[green bold]Keyframes saved:[/green bold] {keyframes_path}\n")


@cli.command()
@click.option("--sequence-json", required=True, type=click.Path(exists=True), help="Sequence data JSON file")
@click.option("--output-dir", "-o", default=None)
@click.option("--type", "doc_type", default="all", type=click.Choice(["all", "vfx", "adr", "music", "asset"]))
def deliverables(sequence_json, output_dir, doc_type):
    """Generate post-production deliverable documents from sequence data."""
    print_banner()

    import json as _json

    try:
        from .core.deliverables import (
            generate_adr_list,
            generate_asset_list,
            generate_music_cue_sheet,
            generate_vfx_sheet,
        )
    except ImportError as e:
        console.print(f"[red bold]Error:[/red bold] Missing dependency: {e}")
        sys.exit(1)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Load sequence data JSON
    try:
        with open(sequence_json, "r", encoding="utf-8") as f:
            seq_data = _json.load(f)
    except Exception as e:
        console.print(f"[red bold]Error:[/red bold] Could not read sequence JSON: {e}")
        sys.exit(1)

    console.print(f"\n[bold]Generating deliverables:[/bold] {sequence_json}")
    console.print(f"[dim]Type: {doc_type} | Output dir: {output_dir or '(same as input)'}[/dim]\n")

    generators = {
        "vfx": ("VFX Sheet", generate_vfx_sheet),
        "adr": ("ADR List", generate_adr_list),
        "music": ("Music Cue Sheet", generate_music_cue_sheet),
        "asset": ("Asset List", generate_asset_list),
    }

    if doc_type == "all":
        gen_list = list(generators.values())
    else:
        gen_list = [generators[doc_type]] if doc_type in generators else []

    docs = []
    out_dir = output_dir or os.path.dirname(sequence_json)
    for label, gen_fn in gen_list:
        console.print(f"  Generating {label}...")
        base_name = os.path.splitext(os.path.basename(sequence_json))[0]
        out_path = os.path.join(out_dir, f"{base_name}_{label.lower().replace(' ', '_')}.csv")
        result = gen_fn(seq_data, out_path)
        if isinstance(result, dict):
            docs.append(result.get("output", out_path))
        else:
            docs.append(out_path)

    console.print(f"\n[green bold]Generated {len(docs)} document(s):[/green bold]")
    for doc in docs:
        console.print(f"  {doc}")
    console.print()


@cli.command()
@click.argument("command_text")
@click.option("--file", type=click.Path(), default=None, help="Media file to operate on")
@click.option("--provider", default="ollama")
@click.option("--model", default="llama3")
@click.option("--api-key", default="")
def nlp(command_text, file, provider, model, api_key):
    """Parse a natural language editing command and show the matched route.

    Uses the LLM (when configured) or the keyword matcher to map the
    English instruction to an OpenCut backend route plus parameters.
    Currently displays the result; pass ``--file`` together with a
    running backend to execute it by POSTing to the matched route.
    """
    print_banner()

    try:
        from .core.llm import LLMConfig
        from .core.nlp_command import parse_command
    except ImportError as e:
        console.print(f"[red bold]Error:[/red bold] Missing dependency: {e}")
        console.print("Ensure opencut[nlp] extras are installed.")
        sys.exit(1)

    llm_cfg = LLMConfig(provider=provider, model=model, api_key=api_key)

    console.print(f"\n[bold]NLP command:[/bold] {command_text!r}")
    if file:
        console.print(f"[bold]File:[/bold] {file}")
    console.print(f"[dim]Provider: {provider} | Model: {model}[/dim]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Parsing command...", total=None)
        start_time = time.perf_counter()
        parsed = parse_command(command_text, llm_cfg)
        elapsed = time.perf_counter() - start_time
        progress.update(task, description=f"[green]Parsed ({elapsed:.2f}s)")
        progress.stop()

    if parsed is None:
        console.print("[red]Could not parse command.[/red]")
        sys.exit(1)

    route = parsed.get("route", "")
    params = parsed.get("params", {})
    confidence = parsed.get("confidence", 0.0)
    source = parsed.get("param_source", "?")
    console.print(f"[bold]Matched route:[/bold] [cyan]{route}[/cyan]  "
                  f"[dim](confidence={confidence:.2f}, via={source})[/dim]")
    console.print(f"[bold]Parameters:[/bold] {params}\n")

    if not file:
        console.print("[dim]Pass --file to execute the command against a "
                      "running backend.[/dim]\n")
        return

    # Execute against the local backend by POSTing to the matched route.
    try:
        import json as _json
        import urllib.error
        import urllib.request

        body = dict(params)
        body["filepath"] = file
        url = f"http://127.0.0.1:5679{route}"
        # Refresh CSRF first
        try:
            with urllib.request.urlopen("http://127.0.0.1:5679/health", timeout=5) as resp:
                csrf = _json.loads(resp.read()).get("csrf_token", "")
        except Exception:
            csrf = ""

        req = urllib.request.Request(
            url,
            data=_json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "X-OpenCut-Token": csrf},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = _json.loads(resp.read())
        console.print(f"\n[green bold]Done:[/green bold] {result}\n")
    except urllib.error.URLError as exc:
        console.print(f"\n[red bold]Backend unreachable[/red bold] ({exc}).")
        console.print("Start the OpenCut server first: [cyan]python -m opencut.server[/cyan]\n")
        sys.exit(1)
    except Exception as exc:
        console.print(f"\n[red bold]Execution failed:[/red bold] {exc}\n")
        sys.exit(1)


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), default=None, help="Output file path")
@click.option("--method", type=click.Choice(["afftdn", "highpass", "gate"]), default="afftdn", help="Denoise method")
@click.option("--strength", type=float, default=0.5, help="Denoise strength 0-1")
def denoise(input_file, output, method, strength):
    """Remove background noise from audio/video."""
    print_banner()

    from .helpers import get_ffmpeg_path, run_ffmpeg
    from .helpers import output_path as _out_path

    if output is None:
        output = _out_path(input_file, "denoised")

    console.print(f"\n[bold]Denoising:[/bold] {input_file}")
    console.print(f"[dim]Method: {method} | Strength: {strength}[/dim]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Applying noise reduction...", total=None)
        start_time = time.time()

        if method == "afftdn":
            nf = int(strength * 97)  # 0-97 range
            af = f"afftdn=nf=-{max(1, nf)}"
        elif method == "highpass":
            freq = int(100 + strength * 200)  # 100-300 Hz
            af = f"highpass=f={freq}"
        else:
            af = f"agate=threshold={0.01 + strength * 0.04}"

        run_ffmpeg([
            get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
            "-i", input_file, "-af", af,
            "-c:v", "copy", output,
        ])
        elapsed = time.time() - start_time
        progress.update(task, description=f"[green]Denoise complete ({elapsed:.1f}s)")
        progress.stop()

    console.print(f"\n[green bold]Saved:[/green bold] {output}\n")


@cli.command("scene-detect")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--method", type=click.Choice(["ffmpeg", "ml", "pyscenedetect"]), default="ffmpeg", help="Detection method")
@click.option("--threshold", type=float, default=0.3, help="Detection sensitivity 0-1")
@click.option("--output", "-o", type=click.Path(), default=None, help="Output JSON file")
def scene_detect(input_file, method, threshold, output):
    """Detect scene boundaries/cuts in a video."""
    print_banner()

    try:
        from .core.scene_detect import detect_scenes, detect_scenes_ml
    except ImportError as e:
        console.print(f"[red bold]Error:[/red bold] Missing dependency: {e}")
        sys.exit(1)

    console.print(f"\n[bold]Scene detection:[/bold] {input_file}")
    console.print(f"[dim]Method: {method} | Threshold: {threshold}[/dim]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Detecting scenes...", total=None)
        start_time = time.time()
        if method == "ml":
            scenes = detect_scenes_ml(input_file, threshold=threshold)
        else:
            scenes = detect_scenes(input_file, threshold=threshold, method=method)
        elapsed = time.time() - start_time
        progress.update(task, description=f"[green]Detection complete ({elapsed:.1f}s)")
        progress.stop()

    scene_list = scenes if isinstance(scenes, list) else scenes.get("scenes", []) if isinstance(scenes, dict) else []
    console.print(f"\n[bold]Scenes found:[/bold] {len(scene_list)}")

    if scene_list:
        table = Table(title="Scene Boundaries", box=box.ROUNDED)
        table.add_column("#", style="cyan", justify="right")
        table.add_column("Start", justify="right")
        table.add_column("End", justify="right")
        table.add_column("Duration", justify="right")

        for i, s in enumerate(scene_list[:30], 1):
            data = s if isinstance(s, dict) else {"start": s}
            start = data.get("start", 0)
            end = data.get("end", start)
            dur = end - start
            table.add_row(str(i), f"{start:.2f}s", f"{end:.2f}s", f"{dur:.2f}s")

        if len(scene_list) > 30:
            console.print(f"[dim](showing first 30 of {len(scene_list)})[/dim]")
        console.print(table)

    if output:
        import json as _json
        with open(output, "w", encoding="utf-8") as f:
            _json.dump(scene_list, f, indent=2)
        console.print(f"\n[green bold]Saved:[/green bold] {output}")

    console.print()


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), default=None, help="Output directory")
@click.option("--port", type=int, default=None, help="Backend port (default: auto-scan 5679-5689)")
@click.option("--no-repeats", is_flag=True, help="Skip repeated-take detection")
@click.option("--no-fillers", is_flag=True, help="Skip filler-word removal")
@click.option("--no-chapters", is_flag=True, help="Skip chapter generation")
@click.option("--timeout", type=int, default=7200, help="Job timeout in seconds (default 2h)")
def polish(input_file, output, port, no_repeats, no_fillers, no_chapters, timeout):
    """Run the full interview polish pipeline on an audio/video file.

    Requires a running OpenCut backend (starts one from source with
    `opencut-server`). Submits the job to /interview-polish, polls until
    complete, then prints the result paths.

    Designed for shell scripts, task schedulers, and CI — no Premiere
    required. Works on raw podcast / interview recordings.
    """
    import json as _json
    import time as _time
    import urllib.error
    import urllib.request

    print_banner()

    input_file = os.path.abspath(input_file)
    output_dir = os.path.abspath(output) if output else ""

    # Locate the running server. Explicit --port wins; otherwise scan.
    def _probe(p):
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{p}/health", timeout=0.5
            ) as resp:
                return _json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None

    backend_port = None
    health = None
    if port:
        health = _probe(port)
        if health:
            backend_port = port
    else:
        for candidate in range(5679, 5690):
            health = _probe(candidate)
            if health:
                backend_port = candidate
                break

    if not backend_port or not health:
        console.print(
            "\n[red bold]No OpenCut backend found.[/red bold] "
            "Start one with [cyan]opencut-server[/cyan] and retry."
        )
        raise SystemExit(2)

    csrf_token = health.get("csrf_token", "")
    base = f"http://127.0.0.1:{backend_port}"
    console.print(f"\n[bold]Polishing:[/bold] {input_file}")
    console.print(f"[dim]Backend: {base} (v{health.get('version', '?')})[/dim]\n")

    # Submit the job
    payload = {
        "filepath": input_file,
        "output_dir": output_dir,
        "detect_repeats": not no_repeats,
        "remove_fillers": not no_fillers,
        "generate_chapters": not no_chapters,
        "diarize": False,
    }
    req = urllib.request.Request(
        f"{base}/interview-polish",
        data=_json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-OpenCut-Token": csrf_token},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            submit = _json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            err = _json.loads(e.read().decode("utf-8"))
            console.print(f"[red]Rejected: {err.get('error', e)}[/red]")
        except Exception:
            console.print(f"[red]Rejected: {e}[/red]")
        raise SystemExit(1)

    job_id = submit.get("job_id")
    if not job_id:
        console.print(f"[red]Unexpected submit response: {submit}[/red]")
        raise SystemExit(1)

    # Poll for completion
    last_msg = ""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Queued…", total=None)
        deadline = _time.time() + timeout
        while _time.time() < deadline:
            try:
                with urllib.request.urlopen(
                    f"{base}/status/{job_id}", timeout=5
                ) as r:
                    job = _json.loads(r.read().decode("utf-8"))
            except Exception:
                _time.sleep(1.0)
                continue
            msg = job.get("message") or job.get("status", "")
            if msg and msg != last_msg:
                progress.update(task, description=msg)
                last_msg = msg
            status = job.get("status")
            if status in ("complete", "error", "cancelled"):
                break
            _time.sleep(1.0)
        else:
            console.print("[red]Timed out waiting for job[/red]")
            raise SystemExit(1)

    if status != "complete":
        console.print(
            f"\n[red bold]Job {status}:[/red bold] {job.get('error', job.get('message', ''))}"
        )
        raise SystemExit(1)

    result = job.get("result") or {}
    console.print("\n[green bold]Polish complete.[/green bold]\n")

    # Step summary
    steps = result.get("steps") or []
    if steps:
        table = Table(title="Steps", show_header=True, header_style="bold")
        table.add_column("Step", style="cyan")
        table.add_column("Result")
        table.add_column("Notes", style="dim")
        for s in steps:
            icon = "[green]OK[/green]" if s.get("ok") else (
                "[yellow]SKIP[/yellow]" if s.get("reason") else "[red]FAIL[/red]"
            )
            notes = s.get("reason", "")
            if s.get("ok"):
                for k in ("removed_fillers", "removed_ranges", "kept_segments",
                          "word_count", "count"):
                    if k in s:
                        notes = f"{k.replace('_', ' ')}: {s[k]}"
                        break
            table.add_row(s.get("label", s.get("key", "?")), icon, notes)
        console.print(table)
        console.print()

    # Output files
    for label, key in (
        ("Premiere XML", "xml_path"),
        ("Captions SRT", "srt_path"),
        ("Chapters MD", "chapters_path"),
    ):
        path = result.get(key, "")
        if path:
            console.print(f"[bold]{label}:[/bold] {path}")

    if "compression_ratio" in result:
        pct = int(result["compression_ratio"] * 100)
        console.print(f"\n[dim]Compressed to {pct}% of original "
                      f"({result.get('speech_duration', 0):.1f}s / "
                      f"{result.get('original_duration', 0):.1f}s)[/dim]\n")


def main():
    """Entry point."""
    cli()


if __name__ == "__main__":
    main()
