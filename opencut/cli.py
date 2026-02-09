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
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich import box

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
@click.version_option(version="0.1.0", prog_name="opencut")
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

    from .utils.config import SilenceConfig, ExportConfig, get_preset
    from .core.silence import detect_speech, get_edit_summary
    from .export.premiere import export_premiere_xml

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
    console.print(f"[green bold]Done![/green bold] Import this XML into Premiere Pro via File > Import.\n")


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

    from .core.captions import transcribe, check_whisper_available
    from .utils.config import CaptionConfig
    from .export.srt import export_srt, export_vtt, export_json

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

    from .core.diarize import diarize, check_pyannote_available
    from .utils.config import DiarizeConfig, ExportConfig

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

    # TODO: Export multicam XML
    console.print(f"\n[yellow]Multicam XML export coming in v0.2. Diarization data saved.[/yellow]\n")


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

    from .utils.config import get_preset, ExportConfig
    from .core.silence import detect_speech, get_edit_summary
    from .core.zoom import generate_zoom_events
    from .export.premiere import export_premiere_xml

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
    console.print(f"\n[green bold]Done![/green bold] Import the XML into Premiere Pro via File > Import.\n")


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
        console.print(f"\n[bold cyan]Video:[/bold cyan]")
        console.print(f"  Resolution: {v.width}x{v.height}")
        console.print(f"  Frame Rate: {v.fps:.3f} fps (timebase={v.effective_timebase}, ntsc={v.ntsc})")
        console.print(f"  Codec: {v.codec}")
        console.print(f"  PAR: {v.pixel_aspect_ratio}")

    if info.has_audio:
        a = info.audio
        console.print(f"\n[bold cyan]Audio:[/bold cyan]")
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


def main():
    """Entry point."""
    cli()


if __name__ == "__main__":
    main()
