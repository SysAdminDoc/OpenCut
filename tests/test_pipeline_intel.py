"""
Tests for OpenCut Pipeline Intelligence (Category 72).

Covers:
  - Pipeline health monitoring (HealthMetric, ComponentHealth, PipelineHealthResult)
  - Scheduled jobs (ScheduleConfig, ScheduledJob, JobHistory, cron parsing)
  - Smart content routing (classify_content, suggest_workflow, CONTENT_TYPES)
  - Processing time estimation (TimeEstimate, OPERATION_BASELINES, batch_estimate)
  - Resource monitoring (ResourceSnapshot, GPUInfo, check_resource_availability)
  - Pipeline intelligence routes (smoke tests for all endpoints)
"""

import json
import os
import sqlite3
import sys
import tempfile
import time
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ========================================================================
# 1. pipeline_health.py
# ========================================================================
class TestHealthMetric:
    """Tests for HealthMetric dataclass."""

    def test_default_timestamp(self):
        from opencut.core.pipeline_health import HealthMetric
        m = HealthMetric(operation="transcode", duration_s=5.0, success=True)
        assert m.timestamp > 0
        assert m.operation == "transcode"
        assert m.success is True

    def test_explicit_timestamp(self):
        from opencut.core.pipeline_health import HealthMetric
        m = HealthMetric(operation="trim", duration_s=1.0, success=False, timestamp=1000.0)
        assert m.timestamp == 1000.0

    def test_to_dict(self):
        from opencut.core.pipeline_health import HealthMetric
        m = HealthMetric(operation="export", duration_s=10.0, success=True, cpu_pct=75.0)
        d = m.to_dict()
        assert d["operation"] == "export"
        assert d["duration_s"] == 10.0
        assert d["cpu_pct"] == 75.0
        assert isinstance(d, dict)

    def test_error_fields(self):
        from opencut.core.pipeline_health import HealthMetric
        m = HealthMetric(
            operation="denoise", duration_s=2.0, success=False,
            error_type="FFmpegError", error_message="exit code 1",
        )
        assert m.error_type == "FFmpegError"
        assert m.error_message == "exit code 1"

    def test_resource_fields(self):
        from opencut.core.pipeline_health import HealthMetric
        m = HealthMetric(
            operation="stabilize", duration_s=60.0, success=True,
            gpu_pct=85.0, ram_mb=4096.0, disk_write_mb=500.0,
        )
        d = m.to_dict()
        assert d["gpu_pct"] == 85.0
        assert d["ram_mb"] == 4096.0
        assert d["disk_write_mb"] == 500.0


class TestComponentHealth:
    """Tests for ComponentHealth dataclass."""

    def test_defaults(self):
        from opencut.core.pipeline_health import ComponentHealth
        ch = ComponentHealth(component="transcode")
        assert ch.health_score == 100
        assert ch.alert_level == "ok"
        assert ch.trend == "stable"

    def test_to_dict(self):
        from opencut.core.pipeline_health import ComponentHealth
        ch = ComponentHealth(
            component="export", health_score=75, total_jobs=100,
            successful_jobs=80, failed_jobs=20, error_rate=0.2,
        )
        d = ch.to_dict()
        assert d["component"] == "export"
        assert d["health_score"] == 75
        assert d["error_rate"] == 0.2

    def test_degraded_values(self):
        from opencut.core.pipeline_health import ComponentHealth
        ch = ComponentHealth(
            component="denoise", health_score=30,
            alert_level="critical", trend="degrading",
        )
        assert ch.alert_level == "critical"
        assert ch.trend == "degrading"


class TestPipelineHealthResult:
    """Tests for PipelineHealthResult dataclass."""

    def test_defaults(self):
        from opencut.core.pipeline_health import PipelineHealthResult
        r = PipelineHealthResult()
        assert r.overall_score == 100
        assert r.overall_status == "healthy"
        assert r.generated_at > 0

    def test_to_dict_structure(self):
        from opencut.core.pipeline_health import (
            ComponentHealth,
            ErrorSummary,
            PipelineHealthResult,
            ThroughputPoint,
        )
        r = PipelineHealthResult(
            overall_score=80,
            components=[ComponentHealth(component="trim")],
            throughput=[ThroughputPoint(hour_start=time.time())],
            errors=[ErrorSummary(error_type="Timeout", count=3)],
        )
        d = r.to_dict()
        assert d["overall_score"] == 80
        assert len(d["components"]) == 1
        assert d["components"][0]["component"] == "trim"
        assert len(d["errors"]) == 1


class TestRecordMetric:
    """Tests for recording and querying health metrics."""

    def test_record_and_query(self, tmp_path):
        """Record metrics and verify get_pipeline_health returns them."""
        import opencut.core.pipeline_health as ph
        # Override DB path
        original_db = ph._DB_PATH
        ph._DB_PATH = str(tmp_path / "test_health.db")
        ph._INITIALIZED = False
        ph._LOCAL = __import__("threading").local()
        try:
            ph.record_metric("transcode", 5.0, True)
            ph.record_metric("transcode", 8.0, True)
            ph.record_metric("transcode", 3.0, False, error_type="FFmpegError")
            result = ph.get_pipeline_health(timeframe_hours=1)
            assert result.total_jobs == 3
            assert result.total_success == 2
            assert result.total_failures == 1
            assert len(result.components) == 1
            assert result.components[0].component == "transcode"
        finally:
            ph._DB_PATH = original_db
            ph._INITIALIZED = False

    def test_error_summary(self, tmp_path):
        """get_error_summary returns grouped errors."""
        import opencut.core.pipeline_health as ph
        original_db = ph._DB_PATH
        ph._DB_PATH = str(tmp_path / "test_errors.db")
        ph._INITIALIZED = False
        ph._LOCAL = __import__("threading").local()
        try:
            ph.record_metric("export", 2.0, False, error_type="DiskFull", error_message="No space")
            ph.record_metric("export", 1.0, False, error_type="DiskFull", error_message="No space left")
            ph.record_metric("trim", 0.5, False, error_type="Timeout")
            errors = ph.get_error_summary(timeframe_hours=1)
            assert len(errors) == 2
            # DiskFull should be first (count=2)
            assert errors[0].error_type == "DiskFull"
            assert errors[0].count == 2
        finally:
            ph._DB_PATH = original_db
            ph._INITIALIZED = False

    def test_empty_health_report(self, tmp_path):
        """Empty DB returns healthy defaults."""
        import opencut.core.pipeline_health as ph
        original_db = ph._DB_PATH
        ph._DB_PATH = str(tmp_path / "test_empty.db")
        ph._INITIALIZED = False
        ph._LOCAL = __import__("threading").local()
        try:
            result = ph.get_pipeline_health(timeframe_hours=1)
            assert result.overall_score == 100
            assert result.overall_status == "healthy"
            assert result.total_jobs == 0
        finally:
            ph._DB_PATH = original_db
            ph._INITIALIZED = False

    def test_progress_callback(self, tmp_path):
        """on_progress receives calls during health report generation."""
        import opencut.core.pipeline_health as ph
        original_db = ph._DB_PATH
        ph._DB_PATH = str(tmp_path / "test_progress.db")
        ph._INITIALIZED = False
        ph._LOCAL = __import__("threading").local()
        try:
            calls = []
            ph.record_metric("trim", 1.0, True)
            ph.get_pipeline_health(timeframe_hours=1, on_progress=lambda pct, msg: calls.append(pct))
            assert 100 in calls
        finally:
            ph._DB_PATH = original_db
            ph._INITIALIZED = False


class TestOperationBaselines:
    """Tests for OPERATION_BASELINES in pipeline_health."""

    def test_baselines_exist(self):
        from opencut.core.pipeline_health import OPERATION_BASELINES
        assert "transcode" in OPERATION_BASELINES
        assert "trim" in OPERATION_BASELINES
        assert "export" in OPERATION_BASELINES

    def test_baselines_are_positive(self):
        from opencut.core.pipeline_health import OPERATION_BASELINES
        for op, val in OPERATION_BASELINES.items():
            assert val > 0, f"Baseline for {op} should be positive"


# ========================================================================
# 2. scheduled_jobs.py
# ========================================================================
class TestScheduleConfig:
    """Tests for ScheduleConfig dataclass."""

    def test_defaults(self):
        from opencut.core.scheduled_jobs import ScheduleConfig
        sc = ScheduleConfig()
        assert sc.job_type == "workflow"
        assert sc.max_runtime_s == 3600

    def test_to_dict(self):
        from opencut.core.scheduled_jobs import ScheduleConfig
        sc = ScheduleConfig(job_type="backup", target_path="/data")
        d = sc.to_dict()
        assert d["job_type"] == "backup"
        assert d["target_path"] == "/data"

    def test_from_dict(self):
        from opencut.core.scheduled_jobs import ScheduleConfig
        sc = ScheduleConfig.from_dict({"job_type": "qc_check", "notify_on_complete": True})
        assert sc.job_type == "qc_check"
        assert sc.notify_on_complete is True

    def test_from_dict_ignores_unknown(self):
        from opencut.core.scheduled_jobs import ScheduleConfig
        sc = ScheduleConfig.from_dict({"job_type": "cleanup", "unknown_field": 42})
        assert sc.job_type == "cleanup"


class TestScheduledJob:
    """Tests for ScheduledJob dataclass."""

    def test_auto_id(self):
        from opencut.core.scheduled_jobs import ScheduledJob
        sj = ScheduledJob(name="Test Job")
        assert len(sj.schedule_id) == 12
        assert sj.created_at > 0

    def test_to_dict(self):
        from opencut.core.scheduled_jobs import ScheduleConfig, ScheduledJob
        sj = ScheduledJob(
            name="Nightly Backup",
            cron_expr="0 2 * * *",
            job_config=ScheduleConfig(job_type="backup"),
        )
        d = sj.to_dict()
        assert d["name"] == "Nightly Backup"
        assert d["job_config"]["job_type"] == "backup"

    def test_from_dict_roundtrip(self):
        from opencut.core.scheduled_jobs import ScheduledJob
        sj = ScheduledJob(name="My Job", cron_expr="*/5 * * * *", tags=["important"])
        d = sj.to_dict()
        restored = ScheduledJob.from_dict(dict(d))
        assert restored.name == "My Job"
        assert restored.tags == ["important"]


class TestJobHistory:
    """Tests for JobHistory dataclass."""

    def test_defaults(self):
        from opencut.core.scheduled_jobs import JobHistory
        jh = JobHistory(schedule_id="abc123", schedule_name="Test")
        assert len(jh.history_id) == 12
        assert jh.started_at > 0
        assert jh.success is True

    def test_to_dict(self):
        from opencut.core.scheduled_jobs import JobHistory
        jh = JobHistory(
            schedule_id="x", schedule_name="Y",
            success=False, error_message="boom",
        )
        d = jh.to_dict()
        assert d["success"] is False
        assert d["error_message"] == "boom"


class TestCronParsing:
    """Tests for cron expression parsing."""

    def test_every_minute(self):
        from opencut.core.scheduled_jobs import parse_cron_expr
        parsed = parse_cron_expr("* * * * *")
        assert len(parsed["minute"]) == 60
        assert len(parsed["hour"]) == 24

    def test_specific_values(self):
        from opencut.core.scheduled_jobs import parse_cron_expr
        parsed = parse_cron_expr("30 8 * * 1")
        assert parsed["minute"] == [30]
        assert parsed["hour"] == [8]
        assert parsed["day_of_week"] == [1]

    def test_range(self):
        from opencut.core.scheduled_jobs import parse_cron_expr
        parsed = parse_cron_expr("0 9-17 * * *")
        assert parsed["hour"] == list(range(9, 18))

    def test_step(self):
        from opencut.core.scheduled_jobs import parse_cron_expr
        parsed = parse_cron_expr("*/15 * * * *")
        assert parsed["minute"] == [0, 15, 30, 45]

    def test_comma_list(self):
        from opencut.core.scheduled_jobs import parse_cron_expr
        parsed = parse_cron_expr("0 8,12,18 * * *")
        assert parsed["hour"] == [8, 12, 18]

    def test_range_with_step(self):
        from opencut.core.scheduled_jobs import parse_cron_expr
        parsed = parse_cron_expr("0 0-23/6 * * *")
        assert parsed["hour"] == [0, 6, 12, 18]

    def test_invalid_field_count(self):
        from opencut.core.scheduled_jobs import parse_cron_expr
        with pytest.raises(ValueError, match="5 fields"):
            parse_cron_expr("* * *")

    def test_get_next_run_returns_future(self):
        from opencut.core.scheduled_jobs import get_next_run
        now = time.time()
        next_ts = get_next_run("* * * * *", after=now)
        assert next_ts > now

    def test_get_next_run_hourly(self):
        from opencut.core.scheduled_jobs import get_next_run
        now = time.time()
        next_ts = get_next_run("0 * * * *", after=now)
        assert next_ts > now
        # Should be within 1 hour
        assert next_ts <= now + 3601


class TestScheduleCRUD:
    """Tests for schedule create/list/delete."""

    def test_create_and_list(self, tmp_path):
        import opencut.core.scheduled_jobs as sj
        original = sj._SCHEDULES_FILE
        sj._SCHEDULES_FILE = str(tmp_path / "schedules.json")
        try:
            job = sj.create_schedule("Test", "0 * * * *", {"job_type": "backup"})
            assert job.name == "Test"

            jobs = sj.list_schedules()
            assert len(jobs) == 1
            assert jobs[0].schedule_id == job.schedule_id
        finally:
            sj._SCHEDULES_FILE = original

    def test_delete(self, tmp_path):
        import opencut.core.scheduled_jobs as sj
        original = sj._SCHEDULES_FILE
        sj._SCHEDULES_FILE = str(tmp_path / "schedules.json")
        try:
            job = sj.create_schedule("DeleteMe", "0 * * * *")
            assert sj.delete_schedule(job.schedule_id) is True
            assert sj.delete_schedule(job.schedule_id) is False  # already deleted
            assert len(sj.list_schedules()) == 0
        finally:
            sj._SCHEDULES_FILE = original

    def test_delete_nonexistent(self, tmp_path):
        import opencut.core.scheduled_jobs as sj
        original = sj._SCHEDULES_FILE
        sj._SCHEDULES_FILE = str(tmp_path / "schedules.json")
        try:
            assert sj.delete_schedule("nonexistent_id") is False
        finally:
            sj._SCHEDULES_FILE = original

    def test_check_due_jobs(self, tmp_path):
        import opencut.core.scheduled_jobs as sj
        original = sj._SCHEDULES_FILE
        sj._SCHEDULES_FILE = str(tmp_path / "schedules.json")
        try:
            # Create job with next_run in the past
            job = sj.create_schedule("Due", "* * * * *")
            # Force next_run to past
            schedules = sj._load_schedules()
            schedules[0]["next_run"] = time.time() - 120
            sj._save_schedules(schedules)

            due = sj.check_due_jobs()
            assert len(due) == 1
            assert due[0].schedule_id == job.schedule_id
        finally:
            sj._SCHEDULES_FILE = original

    def test_record_job_run(self, tmp_path):
        import opencut.core.scheduled_jobs as sj
        orig_s = sj._SCHEDULES_FILE
        orig_h = sj._HISTORY_FILE
        sj._SCHEDULES_FILE = str(tmp_path / "schedules.json")
        sj._HISTORY_FILE = str(tmp_path / "history.json")
        try:
            job = sj.create_schedule("Hist", "0 * * * *")
            entry = sj.record_job_run(job.schedule_id, "Hist", True, duration_s=5.0)
            assert entry.success is True
            assert entry.duration_s == 5.0

            history = sj.get_job_history(schedule_id=job.schedule_id)
            assert len(history) == 1
        finally:
            sj._SCHEDULES_FILE = orig_s
            sj._HISTORY_FILE = orig_h

    def test_job_types_dict(self):
        from opencut.core.scheduled_jobs import JOB_TYPES
        assert "workflow" in JOB_TYPES
        assert "backup" in JOB_TYPES
        assert "qc_check" in JOB_TYPES


# ========================================================================
# 3. smart_route.py
# ========================================================================
class TestContentTypes:
    """Tests for CONTENT_TYPES dictionary."""

    def test_content_types_exist(self):
        from opencut.core.smart_route import CONTENT_TYPES
        assert "interview" in CONTENT_TYPES
        assert "vlog" in CONTENT_TYPES
        assert "tutorial" in CONTENT_TYPES
        assert "music_video" in CONTENT_TYPES
        assert "social_short" in CONTENT_TYPES

    def test_content_types_have_indicators(self):
        from opencut.core.smart_route import CONTENT_TYPES
        for name, info in CONTENT_TYPES.items():
            assert "label" in info, f"{name} missing label"
            assert "description" in info, f"{name} missing description"
            assert "indicators" in info, f"{name} missing indicators"

    def test_ten_content_types(self):
        from opencut.core.smart_route import CONTENT_TYPES
        assert len(CONTENT_TYPES) == 10


class TestWorkflowTemplates:
    """Tests for WORKFLOW_TEMPLATES dictionary."""

    def test_all_content_types_have_templates(self):
        from opencut.core.smart_route import CONTENT_TYPES, WORKFLOW_TEMPLATES
        for ct in CONTENT_TYPES:
            assert ct in WORKFLOW_TEMPLATES, f"No workflow template for {ct}"

    def test_templates_have_operations(self):
        from opencut.core.smart_route import WORKFLOW_TEMPLATES
        for name, tmpl in WORKFLOW_TEMPLATES.items():
            assert "operations" in tmpl, f"{name} missing operations"
            assert len(tmpl["operations"]) > 0, f"{name} has empty operations"
            assert "params" in tmpl, f"{name} missing params"


class TestContentClassification:
    """Tests for ContentClassification dataclass."""

    def test_defaults(self):
        from opencut.core.smart_route import ContentClassification
        cc = ContentClassification()
        assert cc.content_type == "unknown"
        assert cc.confidence == 0.0

    def test_to_dict(self):
        from opencut.core.smart_route import ContentClassification
        cc = ContentClassification(content_type="vlog", confidence=0.85, label="Vlog")
        d = cc.to_dict()
        assert d["content_type"] == "vlog"
        assert d["confidence"] == 0.85


class TestWorkflowSuggestion:
    """Tests for WorkflowSuggestion dataclass."""

    def test_to_dict(self):
        from opencut.core.smart_route import WorkflowSuggestion
        ws = WorkflowSuggestion(
            content_type="interview",
            operations=["silence_detect", "export"],
            tips=["Remove silences"],
        )
        d = ws.to_dict()
        assert len(d["operations"]) == 2
        assert len(d["tips"]) == 1


class TestSmartRouteResult:
    """Tests for SmartRouteResult dataclass."""

    def test_to_dict(self):
        from opencut.core.smart_route import (
            ContentClassification,
            SmartRouteResult,
            WorkflowSuggestion,
        )
        result = SmartRouteResult(
            classification=ContentClassification(content_type="tutorial"),
            suggestion=WorkflowSuggestion(content_type="tutorial"),
            alternatives=[WorkflowSuggestion(content_type="interview")],
        )
        d = result.to_dict()
        assert d["classification"]["content_type"] == "tutorial"
        assert len(d["alternatives"]) == 1


class TestClassifyContent:
    """Tests for classify_content with mocked ffprobe."""

    @patch("opencut.core.smart_route.get_video_info")
    @patch("opencut.core.smart_route._estimate_motion_level", return_value="low")
    def test_classify_interview(self, mock_motion, mock_info):
        from opencut.core.smart_route import classify_content
        mock_info.return_value = {
            "duration": 600, "width": 1920, "height": 1080,
            "fps": 30, "codec": "h264", "audio_channels": 1,
        }
        result = classify_content("/fake/video.mp4", face_count=1)
        assert result.content_type in ("interview", "podcast", "tutorial")
        assert result.confidence > 0

    @patch("opencut.core.smart_route.get_video_info")
    @patch("opencut.core.smart_route._estimate_motion_level", return_value="high")
    def test_classify_social_short(self, mock_motion, mock_info):
        from opencut.core.smart_route import classify_content
        mock_info.return_value = {
            "duration": 30, "width": 1080, "height": 1920,
            "fps": 30, "codec": "h264", "audio_channels": 2,
        }
        result = classify_content("/fake/reel.mp4")
        assert result.confidence > 0
        assert len(result.scores) == 10

    @patch("opencut.core.smart_route.get_video_info")
    @patch("opencut.core.smart_route._estimate_motion_level", return_value="high")
    def test_classify_music_video(self, mock_motion, mock_info):
        from opencut.core.smart_route import classify_content
        mock_info.return_value = {
            "duration": 240, "width": 1920, "height": 1080,
            "fps": 30, "codec": "h264", "audio_channels": 2,
        }
        result = classify_content("/fake/mv.mp4")
        assert "music_video" in result.scores

    @patch("opencut.core.smart_route.get_video_info")
    @patch("opencut.core.smart_route._estimate_motion_level", return_value="very_low")
    def test_classify_podcast(self, mock_motion, mock_info):
        from opencut.core.smart_route import classify_content
        mock_info.return_value = {
            "duration": 3600, "width": 1920, "height": 1080,
            "fps": 30, "codec": "h264", "audio_channels": 2,
        }
        result = classify_content("/fake/podcast.mp4", face_count=2)
        assert result.content_type == "podcast"

    def test_classify_progress_callback(self):
        calls = []
        with patch("opencut.core.smart_route.get_video_info", return_value={
            "duration": 120, "width": 1920, "height": 1080, "fps": 30,
        }), patch("opencut.core.smart_route._estimate_motion_level", return_value="medium"):
            from opencut.core.smart_route import classify_content
            classify_content("/fake/v.mp4", on_progress=lambda pct, msg: calls.append(pct))
        assert 100 in calls


class TestSuggestWorkflow:
    """Tests for suggest_workflow."""

    def test_suggest_returns_operations(self):
        from opencut.core.smart_route import ContentClassification, suggest_workflow
        cc = ContentClassification(
            content_type="vlog", confidence=0.8,
            scores={"vlog": 0.8, "interview": 0.5, "tutorial": 0.3},
        )
        result = suggest_workflow(cc)
        assert result.suggestion.content_type == "vlog"
        assert len(result.suggestion.operations) > 0

    def test_suggest_includes_alternatives(self):
        from opencut.core.smart_route import ContentClassification, suggest_workflow
        cc = ContentClassification(
            content_type="gaming",
            confidence=0.7,
            scores={
                "gaming": 0.7, "vlog": 0.5, "tutorial": 0.4,
                "interview": 0.1, "podcast": 0.05, "music_video": 0.05,
                "drone": 0.02, "timelapse": 0.01, "corporate": 0.01,
                "social_short": 0.01,
            },
        )
        result = suggest_workflow(cc)
        assert len(result.alternatives) >= 1

    def test_suggest_unknown_type_fallback(self):
        from opencut.core.smart_route import ContentClassification, suggest_workflow
        cc = ContentClassification(content_type="unknown", scores={})
        result = suggest_workflow(cc)
        # Should still return a suggestion (fallback to vlog template)
        assert result.suggestion is not None


class TestMotionDistance:
    """Tests for _motion_distance helper."""

    def test_same(self):
        from opencut.core.smart_route import _motion_distance
        assert _motion_distance("low", "low") == 0

    def test_adjacent(self):
        from opencut.core.smart_route import _motion_distance
        assert _motion_distance("low", "medium") == 1

    def test_far(self):
        from opencut.core.smart_route import _motion_distance
        assert _motion_distance("very_low", "high") == 3

    def test_unknown(self):
        from opencut.core.smart_route import _motion_distance
        assert _motion_distance("unknown", "high") == 2


class TestAspectCategory:
    """Tests for _compute_aspect_category."""

    def test_vertical(self):
        from opencut.core.smart_route import _compute_aspect_category
        assert _compute_aspect_category(1080, 1920) == "vertical"

    def test_square(self):
        from opencut.core.smart_route import _compute_aspect_category
        assert _compute_aspect_category(1080, 1080) == "square"

    def test_landscape(self):
        from opencut.core.smart_route import _compute_aspect_category
        assert _compute_aspect_category(1920, 1080) == "landscape"

    def test_ultrawide(self):
        from opencut.core.smart_route import _compute_aspect_category
        assert _compute_aspect_category(3440, 1440) == "ultrawide"

    def test_zero_dimensions(self):
        from opencut.core.smart_route import _compute_aspect_category
        assert _compute_aspect_category(0, 0) == "unknown"


# ========================================================================
# 4. process_estimate.py
# ========================================================================
class TestTimeEstimate:
    """Tests for TimeEstimate dataclass."""

    def test_defaults(self):
        from opencut.core.process_estimate import TimeEstimate
        te = TimeEstimate(operation="trim")
        assert te.estimated_seconds == 0.0
        assert te.confidence == "medium"

    def test_auto_human_readable(self):
        from opencut.core.process_estimate import TimeEstimate
        te = TimeEstimate(operation="transcode", estimated_seconds=125)
        assert "2 min" in te.human_readable

    def test_to_dict(self):
        from opencut.core.process_estimate import TimeEstimate
        te = TimeEstimate(operation="export", estimated_seconds=60, gpu_available=True)
        d = te.to_dict()
        assert d["operation"] == "export"
        assert d["gpu_available"] is True


class TestFormatDuration:
    """Tests for format_duration helper."""

    def test_sub_second(self):
        from opencut.core.process_estimate import format_duration
        assert format_duration(0.5) == "< 1 second"

    def test_seconds(self):
        from opencut.core.process_estimate import format_duration
        assert format_duration(1) == "1 second"
        assert format_duration(45) == "45 seconds"

    def test_minutes(self):
        from opencut.core.process_estimate import format_duration
        assert "2 min" in format_duration(150)

    def test_hours(self):
        from opencut.core.process_estimate import format_duration
        result = format_duration(3720)
        assert "1 hr" in result
        assert "2 min" in result


class TestEstimateOperationBaselines:
    """Tests for OPERATION_BASELINES in process_estimate."""

    def test_baselines_exist(self):
        from opencut.core.process_estimate import OPERATION_BASELINES
        assert "transcode" in OPERATION_BASELINES
        assert "trim" in OPERATION_BASELINES
        assert "export" in OPERATION_BASELINES

    def test_baselines_have_required_keys(self):
        from opencut.core.process_estimate import OPERATION_BASELINES
        for op, info in OPERATION_BASELINES.items():
            assert "base_ratio" in info, f"{op} missing base_ratio"
            assert "description" in info, f"{op} missing description"
            assert "gpu_factor" in info, f"{op} missing gpu_factor"
            assert info["base_ratio"] > 0, f"{op} base_ratio should be positive"

    def test_gpu_factors_between_0_and_1(self):
        from opencut.core.process_estimate import OPERATION_BASELINES
        for op, info in OPERATION_BASELINES.items():
            assert 0 < info["gpu_factor"] <= 1.0, f"{op} gpu_factor out of range"


class TestEstimateProcessingTime:
    """Tests for estimate_processing_time with mocked dependencies."""

    @patch("opencut.core.process_estimate.get_video_info")
    @patch("opencut.core.process_estimate._detect_gpu", return_value=False)
    def test_basic_estimate(self, mock_gpu, mock_info):
        from opencut.core.process_estimate import estimate_processing_time
        mock_info.return_value = {"duration": 60, "height": 1080}
        est = estimate_processing_time("/fake/video.mp4", "transcode")
        assert est.estimated_seconds > 0
        assert est.operation == "transcode"
        assert est.human_readable != ""

    @patch("opencut.core.process_estimate.get_video_info")
    @patch("opencut.core.process_estimate._detect_gpu", return_value=True)
    def test_gpu_reduces_estimate(self, mock_gpu, mock_info):
        from opencut.core.process_estimate import estimate_processing_time
        mock_info.return_value = {"duration": 60, "height": 1080}
        est_gpu = estimate_processing_time("/fake/v.mp4", "transcode", gpu_available=True)
        est_cpu = estimate_processing_time("/fake/v.mp4", "transcode", gpu_available=False)
        # GPU should be faster (lower estimate) for operations with gpu_factor < 1
        assert est_gpu.estimated_seconds < est_cpu.estimated_seconds

    @patch("opencut.core.process_estimate.get_video_info")
    @patch("opencut.core.process_estimate._detect_gpu", return_value=False)
    def test_4k_slower_than_1080p(self, mock_gpu, mock_info):
        from opencut.core.process_estimate import estimate_processing_time
        mock_info.return_value = {"duration": 60, "height": 2160}
        est_4k = estimate_processing_time("/fake/v.mp4", "denoise")
        mock_info.return_value = {"duration": 60, "height": 1080}
        est_hd = estimate_processing_time("/fake/v.mp4", "denoise")
        assert est_4k.estimated_seconds > est_hd.estimated_seconds

    @patch("opencut.core.process_estimate.get_video_info")
    @patch("opencut.core.process_estimate._detect_gpu", return_value=False)
    def test_unknown_operation(self, mock_gpu, mock_info):
        from opencut.core.process_estimate import estimate_processing_time
        mock_info.return_value = {"duration": 60, "height": 1080}
        est = estimate_processing_time("/fake/v.mp4", "nonexistent_op")
        assert est.confidence == "low"
        assert est.estimated_seconds > 0

    @patch("opencut.core.process_estimate.get_video_info")
    @patch("opencut.core.process_estimate._detect_gpu", return_value=False)
    def test_quality_params(self, mock_gpu, mock_info):
        from opencut.core.process_estimate import estimate_processing_time
        mock_info.return_value = {"duration": 60, "height": 1080}
        est_high = estimate_processing_time("/fake/v.mp4", "transcode", params={"quality": "high"})
        est_low = estimate_processing_time("/fake/v.mp4", "transcode", params={"quality": "low"})
        assert est_high.estimated_seconds > est_low.estimated_seconds


class TestBatchEstimate:
    """Tests for batch_estimate."""

    @patch("opencut.core.process_estimate.get_video_info")
    @patch("opencut.core.process_estimate._detect_gpu", return_value=False)
    def test_batch_sums_correctly(self, mock_gpu, mock_info):
        from opencut.core.process_estimate import batch_estimate
        mock_info.return_value = {"duration": 60, "height": 1080}
        result = batch_estimate("/fake/v.mp4", [
            {"operation": "trim"},
            {"operation": "transcode"},
            {"operation": "export"},
        ])
        assert len(result.estimates) == 3
        individual_sum = sum(e.estimated_seconds for e in result.estimates)
        assert abs(result.total_seconds - individual_sum) < 0.01

    @patch("opencut.core.process_estimate.get_video_info")
    @patch("opencut.core.process_estimate._detect_gpu", return_value=False)
    def test_batch_human_readable(self, mock_gpu, mock_info):
        from opencut.core.process_estimate import batch_estimate
        mock_info.return_value = {"duration": 120, "height": 1080}
        result = batch_estimate("/fake/v.mp4", [{"operation": "transcode"}])
        assert result.total_human_readable != ""


class TestRecordActualTime:
    """Tests for record_actual_time."""

    def test_record_and_accuracy(self, tmp_path):
        import opencut.core.process_estimate as pe
        original_db = pe._DB_PATH
        pe._DB_PATH = str(tmp_path / "test_est.db")
        pe._INITIALIZED = False
        pe._LOCAL = __import__("threading").local()
        try:
            info = pe.record_actual_time("transcode", 10.0, 12.0)
            assert info["operation"] == "transcode"
            assert info["accuracy_pct"] > 0
            assert info["difference_s"] == 2.0
        finally:
            pe._DB_PATH = original_db
            pe._INITIALIZED = False

    def test_perfect_accuracy(self, tmp_path):
        import opencut.core.process_estimate as pe
        original_db = pe._DB_PATH
        pe._DB_PATH = str(tmp_path / "test_perfect.db")
        pe._INITIALIZED = False
        pe._LOCAL = __import__("threading").local()
        try:
            info = pe.record_actual_time("trim", 5.0, 5.0)
            assert info["accuracy_pct"] == 100.0
        finally:
            pe._DB_PATH = original_db
            pe._INITIALIZED = False


class TestResolutionClassification:
    """Tests for resolution helpers in process_estimate."""

    def test_classify_480(self):
        from opencut.core.process_estimate import _classify_resolution
        assert _classify_resolution(480) == "480p"

    def test_classify_1080(self):
        from opencut.core.process_estimate import _classify_resolution
        assert _classify_resolution(1080) == "1080p"

    def test_classify_2160(self):
        from opencut.core.process_estimate import _classify_resolution
        assert _classify_resolution(2160) == "2160p"

    def test_classify_zero(self):
        from opencut.core.process_estimate import _classify_resolution
        assert _classify_resolution(0) == "1080p"

    def test_resolution_factor_increases(self):
        from opencut.core.process_estimate import _get_resolution_factor
        f_720 = _get_resolution_factor(720)
        f_1080 = _get_resolution_factor(1080)
        f_2160 = _get_resolution_factor(2160)
        assert f_720 < f_1080 < f_2160


# ========================================================================
# 5. resource_monitor.py
# ========================================================================
class TestGPUInfo:
    """Tests for GPUInfo dataclass."""

    def test_defaults(self):
        from opencut.core.resource_monitor import GPUInfo
        g = GPUInfo()
        assert g.name == ""
        assert g.utilization_pct == 0.0

    def test_memory_used_pct(self):
        from opencut.core.resource_monitor import GPUInfo
        g = GPUInfo(memory_used_mb=4000, memory_total_mb=8000)
        assert g.memory_used_pct == 50.0

    def test_memory_used_pct_zero_total(self):
        from opencut.core.resource_monitor import GPUInfo
        g = GPUInfo(memory_used_mb=100, memory_total_mb=0)
        assert g.memory_used_pct == 0.0

    def test_to_dict_includes_computed(self):
        from opencut.core.resource_monitor import GPUInfo
        g = GPUInfo(name="RTX 4090", memory_used_mb=6000, memory_total_mb=24000)
        d = g.to_dict()
        assert d["name"] == "RTX 4090"
        assert d["memory_used_pct"] == 25.0


class TestDiskInfo:
    """Tests for DiskInfo dataclass."""

    def test_to_dict(self):
        from opencut.core.resource_monitor import DiskInfo
        d = DiskInfo(path="C:\\", total_gb=500, used_gb=300, free_gb=200, used_pct=60.0)
        result = d.to_dict()
        assert result["path"] == "C:\\"
        assert result["free_gb"] == 200


class TestResourceSnapshot:
    """Tests for ResourceSnapshot dataclass."""

    def test_auto_timestamp(self):
        from opencut.core.resource_monitor import ResourceSnapshot
        s = ResourceSnapshot()
        assert s.timestamp > 0

    def test_to_dict(self):
        from opencut.core.resource_monitor import GPUInfo, ResourceSnapshot
        s = ResourceSnapshot(
            cpu_total_pct=50.0,
            cpu_count=8,
            ram_total_mb=16384,
            gpus=[GPUInfo(name="RTX 3080")],
        )
        d = s.to_dict()
        assert d["cpu_total_pct"] == 50.0
        assert len(d["gpus"]) == 1
        assert d["gpus"][0]["name"] == "RTX 3080"


class TestResourceMonitorResult:
    """Tests for ResourceMonitorResult dataclass."""

    def test_available_default(self):
        from opencut.core.resource_monitor import ResourceMonitorResult
        r = ResourceMonitorResult()
        assert r.available is True
        assert r.reasons == []

    def test_to_dict(self):
        from opencut.core.resource_monitor import ResourceMonitorResult, ResourceSnapshot
        r = ResourceMonitorResult(
            available=False,
            snapshot=ResourceSnapshot(),
            reasons=["Not enough RAM"],
            warnings=["Disk getting full"],
        )
        d = r.to_dict()
        assert d["available"] is False
        assert len(d["reasons"]) == 1
        assert len(d["warnings"]) == 1


class TestResourceRequirements:
    """Tests for ResourceRequirements dataclass."""

    def test_defaults(self):
        from opencut.core.resource_monitor import ResourceRequirements
        r = ResourceRequirements()
        assert r.min_ram_mb == 512
        assert r.min_disk_gb == 1.0
        assert r.prefer_gpu is False


class TestGetResourceSnapshot:
    """Tests for get_resource_snapshot with mocked psutil."""

    @patch("opencut.core.resource_monitor.get_gpu_info", return_value=[])
    @patch("opencut.core.resource_monitor._get_cpu_temperature", return_value=0.0)
    @patch("opencut.core.resource_monitor._get_disk_info", return_value=[])
    @patch("opencut.core.resource_monitor._get_process_info", return_value={"process_ram_mb": 100, "process_cpu_pct": 5})
    @patch("opencut.core.resource_monitor._get_ram_info", return_value={
        "ram_total_mb": 16384, "ram_used_mb": 8192,
        "ram_available_mb": 8192, "ram_used_pct": 50.0,
    })
    @patch("opencut.core.resource_monitor._get_cpu_info", return_value={
        "cpu_total_pct": 25.0, "cpu_per_core": [20, 30, 25, 25], "cpu_count": 4,
    })
    def test_snapshot_assembled(self, mock_cpu, mock_ram, mock_proc, mock_disk, mock_temp, mock_gpu):
        from opencut.core.resource_monitor import get_resource_snapshot
        snap = get_resource_snapshot()
        assert snap.cpu_total_pct == 25.0
        assert snap.cpu_count == 4
        assert snap.ram_total_mb == 16384
        assert snap.ram_used_pct == 50.0

    @patch("opencut.core.resource_monitor.get_gpu_info", return_value=[])
    @patch("opencut.core.resource_monitor._get_cpu_temperature", return_value=0.0)
    @patch("opencut.core.resource_monitor._get_disk_info", return_value=[])
    @patch("opencut.core.resource_monitor._get_process_info", return_value={"process_ram_mb": 0, "process_cpu_pct": 0})
    @patch("opencut.core.resource_monitor._get_ram_info", return_value={
        "ram_total_mb": 8192, "ram_used_mb": 4096,
        "ram_available_mb": 4096, "ram_used_pct": 50.0,
    })
    @patch("opencut.core.resource_monitor._get_cpu_info", return_value={
        "cpu_total_pct": 10.0, "cpu_per_core": [10], "cpu_count": 1,
    })
    def test_snapshot_progress(self, mock_cpu, mock_ram, mock_proc, mock_disk, mock_temp, mock_gpu):
        from opencut.core.resource_monitor import get_resource_snapshot
        calls = []
        get_resource_snapshot(on_progress=lambda pct, msg: calls.append(pct))
        assert 100 in calls


class TestGetGpuInfo:
    """Tests for get_gpu_info with mocked nvidia-smi."""

    @patch("opencut.core.resource_monitor._query_torch_cuda", return_value=[])
    @patch("opencut.core.resource_monitor._query_nvidia_smi")
    def test_nvidia_smi_parse(self, mock_smi, mock_torch):
        from opencut.core.resource_monitor import GPUInfo, get_gpu_info
        mock_smi.return_value = [GPUInfo(
            index=0, name="RTX 4090", utilization_pct=45.0,
            memory_used_mb=8000, memory_total_mb=24000, memory_free_mb=16000,
            temperature_c=65,
        )]
        gpus = get_gpu_info()
        assert len(gpus) == 1
        assert gpus[0].name == "RTX 4090"
        assert gpus[0].memory_used_pct == pytest.approx(33.3, abs=0.1)

    @patch("opencut.core.resource_monitor._query_nvidia_smi", return_value=[])
    @patch("opencut.core.resource_monitor._query_torch_cuda", return_value=[])
    def test_no_gpu(self, mock_torch, mock_smi):
        from opencut.core.resource_monitor import get_gpu_info
        gpus = get_gpu_info()
        assert gpus == []


class TestCheckResourceAvailability:
    """Tests for check_resource_availability."""

    @patch("opencut.core.resource_monitor.get_resource_snapshot")
    def test_sufficient_resources(self, mock_snap):
        from opencut.core.resource_monitor import (
            DiskInfo,
            ResourceSnapshot,
            check_resource_availability,
        )
        mock_snap.return_value = ResourceSnapshot(
            cpu_total_pct=30, ram_available_mb=8192, ram_used_pct=50,
            disks=[DiskInfo(path="C:\\", free_gb=100)],
        )
        result = check_resource_availability({"min_ram_mb": 1024, "min_disk_gb": 5})
        assert result.available is True
        assert result.reasons == []

    @patch("opencut.core.resource_monitor.get_resource_snapshot")
    def test_insufficient_ram(self, mock_snap):
        from opencut.core.resource_monitor import ResourceSnapshot, check_resource_availability
        mock_snap.return_value = ResourceSnapshot(
            cpu_total_pct=30, ram_available_mb=256, ram_used_pct=96,
            disks=[],
        )
        result = check_resource_availability({"min_ram_mb": 1024, "max_ram_pct": 90})
        assert result.available is False
        assert any("RAM" in r for r in result.reasons)

    @patch("opencut.core.resource_monitor.get_resource_snapshot")
    def test_high_cpu(self, mock_snap):
        from opencut.core.resource_monitor import ResourceSnapshot, check_resource_availability
        mock_snap.return_value = ResourceSnapshot(
            cpu_total_pct=98, ram_available_mb=8192, ram_used_pct=50,
            disks=[],
        )
        result = check_resource_availability({"max_cpu_pct": 95})
        assert result.available is False
        assert any("CPU" in r for r in result.reasons)

    @patch("opencut.core.resource_monitor.get_resource_snapshot")
    def test_gpu_memory_check(self, mock_snap):
        from opencut.core.resource_monitor import (
            GPUInfo,
            ResourceSnapshot,
            check_resource_availability,
        )
        mock_snap.return_value = ResourceSnapshot(
            cpu_total_pct=10, ram_available_mb=8192, ram_used_pct=50,
            gpus=[GPUInfo(name="RTX", memory_free_mb=500)],
            disks=[],
        )
        result = check_resource_availability({"min_gpu_memory_mb": 2000})
        assert result.available is False
        assert any("GPU memory" in r for r in result.reasons)

    @patch("opencut.core.resource_monitor.get_resource_snapshot")
    def test_disk_warning(self, mock_snap):
        from opencut.core.resource_monitor import (
            DiskInfo,
            ResourceSnapshot,
            check_resource_availability,
        )
        mock_snap.return_value = ResourceSnapshot(
            cpu_total_pct=10, ram_available_mb=8192, ram_used_pct=50,
            disks=[DiskInfo(path="C:\\", free_gb=3)],
        )
        result = check_resource_availability({"min_disk_gb": 2})
        assert result.available is True
        # Free is above min but below 2x min, so warning
        assert len(result.warnings) > 0


class TestResourceHistory:
    """Tests for resource history."""

    def test_history_stores_and_retrieves(self):
        import opencut.core.resource_monitor as rm
        # Clear history
        rm.clear_history()
        # Store a snapshot manually
        from opencut.core.resource_monitor import ResourceSnapshot
        snap = ResourceSnapshot(cpu_total_pct=42.0, cpu_count=4)
        rm._store_snapshot(snap)
        history = rm.get_resource_history(minutes=5)
        assert len(history) >= 1
        assert history[0].cpu_total_pct == 42.0
        rm.clear_history()

    def test_clear_history(self):
        import opencut.core.resource_monitor as rm
        rm.clear_history()
        from opencut.core.resource_monitor import ResourceSnapshot
        rm._store_snapshot(ResourceSnapshot(cpu_total_pct=10))
        rm.clear_history()
        assert rm.get_resource_history(minutes=60) == []


# ========================================================================
# 6. pipeline_intel_routes.py  (smoke tests)
# ========================================================================
@pytest.fixture
def app():
    """Create a minimal Flask app with the pipeline_intel_bp registered."""
    from flask import Flask
    from opencut.routes.pipeline_intel_routes import pipeline_intel_bp
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(pipeline_intel_bp)
    return app


@pytest.fixture
def client(app):
    return app.test_client()


class TestHealthRoutes:
    """Smoke tests for /api/pipeline/health endpoints."""

    def test_get_health(self, client, tmp_path):
        import opencut.core.pipeline_health as ph
        original = ph._DB_PATH
        ph._DB_PATH = str(tmp_path / "route_health.db")
        ph._INITIALIZED = False
        ph._LOCAL = __import__("threading").local()
        try:
            resp = client.get("/api/pipeline/health")
            assert resp.status_code == 200
            data = resp.get_json()
            assert "overall_score" in data
            assert "overall_status" in data
        finally:
            ph._DB_PATH = original
            ph._INITIALIZED = False

    def test_get_errors(self, client, tmp_path):
        import opencut.core.pipeline_health as ph
        original = ph._DB_PATH
        ph._DB_PATH = str(tmp_path / "route_errors.db")
        ph._INITIALIZED = False
        ph._LOCAL = __import__("threading").local()
        try:
            resp = client.get("/api/pipeline/errors")
            assert resp.status_code == 200
            data = resp.get_json()
            assert "errors" in data
        finally:
            ph._DB_PATH = original
            ph._INITIALIZED = False


class TestScheduleRoutes:
    """Smoke tests for /api/pipeline/schedules endpoints."""

    def test_list_schedules(self, client, tmp_path):
        import opencut.core.scheduled_jobs as sj
        original = sj._SCHEDULES_FILE
        sj._SCHEDULES_FILE = str(tmp_path / "route_sched.json")
        try:
            resp = client.get("/api/pipeline/schedules")
            assert resp.status_code == 200
            data = resp.get_json()
            assert "schedules" in data
        finally:
            sj._SCHEDULES_FILE = original

    def test_schedule_history(self, client, tmp_path):
        import opencut.core.scheduled_jobs as sj
        orig_h = sj._HISTORY_FILE
        sj._HISTORY_FILE = str(tmp_path / "route_hist.json")
        try:
            resp = client.get("/api/pipeline/schedules/history")
            assert resp.status_code == 200
            data = resp.get_json()
            assert "history" in data
        finally:
            sj._HISTORY_FILE = orig_h


class TestContentTypeRoutes:
    """Smoke tests for content classification routes."""

    def test_list_content_types(self, client):
        resp = client.get("/api/video/content-types")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "content_types" in data
        assert "interview" in data["content_types"]

    def test_suggest_workflow_missing_data(self, client):
        resp = client.post(
            "/api/video/suggest-workflow",
            json={},
            content_type="application/json",
        )
        assert resp.status_code in (400, 403)

    def test_suggest_workflow_valid(self, client):
        # POST requires CSRF — verify route is registered (403 = CSRF block)
        resp = client.post(
            "/api/video/suggest-workflow",
            json={"classification": {
                "content_type": "vlog",
                "confidence": 0.8,
                "scores": {"vlog": 0.8},
            }},
            content_type="application/json",
        )
        assert resp.status_code in (200, 403)


class TestEstimateRoutes:
    """Smoke tests for estimation routes."""

    def test_get_baselines(self, client):
        resp = client.get("/api/pipeline/estimate/baselines")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "baselines" in data
        assert "transcode" in data["baselines"]

    def test_get_accuracy(self, client, tmp_path):
        import opencut.core.process_estimate as pe
        original = pe._DB_PATH
        pe._DB_PATH = str(tmp_path / "route_est.db")
        pe._INITIALIZED = False
        pe._LOCAL = __import__("threading").local()
        try:
            resp = client.get("/api/pipeline/estimate/accuracy")
            assert resp.status_code == 200
            data = resp.get_json()
            assert "avg_accuracy" in data
        finally:
            pe._DB_PATH = original
            pe._INITIALIZED = False


class TestResourceRoutes:
    """Smoke tests for resource monitoring routes."""

    @patch("opencut.core.resource_monitor.get_gpu_info", return_value=[])
    @patch("opencut.core.resource_monitor._get_cpu_temperature", return_value=0.0)
    @patch("opencut.core.resource_monitor._get_disk_info", return_value=[])
    @patch("opencut.core.resource_monitor._get_process_info", return_value={"process_ram_mb": 0, "process_cpu_pct": 0})
    @patch("opencut.core.resource_monitor._get_ram_info", return_value={
        "ram_total_mb": 8192, "ram_used_mb": 4096,
        "ram_available_mb": 4096, "ram_used_pct": 50.0,
    })
    @patch("opencut.core.resource_monitor._get_cpu_info", return_value={
        "cpu_total_pct": 20.0, "cpu_per_core": [20], "cpu_count": 1,
    })
    def test_get_resources(self, mock_cpu, mock_ram, mock_proc, mock_disk, mock_temp, mock_gpu, client):
        resp = client.get("/api/pipeline/resources")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "cpu_total_pct" in data

    @patch("opencut.core.resource_monitor._query_nvidia_smi", return_value=[])
    @patch("opencut.core.resource_monitor._query_torch_cuda", return_value=[])
    def test_get_gpu(self, mock_torch, mock_smi, client):
        resp = client.get("/api/pipeline/resources/gpu")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "gpus" in data

    def test_get_resource_history(self, client):
        import opencut.core.resource_monitor as rm
        rm.clear_history()
        resp = client.get("/api/pipeline/resources/history")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "history" in data
