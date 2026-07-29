"""Contract generation must not mutate the machine it is describing.

Building a route/readiness/OpenAPI manifest is an inspection of the
registration graph, but it used to boot a production app: that swept the
user's temp directory, started the disk-monitor thread, ran credential-vault
migrations, initialised Sentry, and imported and registered whatever plugins
happened to be installed under ``~/.opencut/plugins``. A manifest ``--check``
in CI or a release gate is not supposed to touch any of that.
"""

from __future__ import annotations

import threading

import pytest

from opencut.config import OpenCutConfig


@pytest.fixture()
def sentinels(monkeypatch):
    """Trip a flag if a boundary an introspection build must not cross is."""
    tripped: dict[str, int] = {}

    def trip(name):
        def _fn(*_args, **_kwargs):
            tripped[name] = tripped.get(name, 0) + 1
            return {"loaded": [], "failed": []}

        return _fn

    from opencut import credential_store, server
    from opencut.core import disk_monitor, plugins, temp_cleanup

    monkeypatch.setattr(temp_cleanup, "run_startup_sweep", trip("temp_sweep"))
    monkeypatch.setattr(temp_cleanup, "start_background_sweep", trip("temp_background"))
    monkeypatch.setattr(disk_monitor, "start_background", trip("disk_monitor"))
    monkeypatch.setattr(credential_store, "run_startup_migrations", trip("migrations"))
    monkeypatch.setattr(plugins, "load_all_plugins", trip("plugins"))
    monkeypatch.setattr(server, "_init_sentry_if_configured", trip("sentry"))
    return tripped


def test_introspection_app_crosses_no_runtime_boundary(sentinels):
    from opencut.server import create_app

    before = threading.active_count()
    app = create_app(config=OpenCutConfig(), introspection=True)
    after = threading.active_count()

    assert sentinels == {}, f"introspection build had side effects: {sentinels}"
    # No background worker should be left running behind a read-only build.
    assert after <= before, f"introspection started {after - before} thread(s)"
    # It is still a usable registration graph — that is the whole point.
    assert len(list(app.url_map.iter_rules())) > 100


def test_production_app_still_performs_startup_work(sentinels):
    """The guard must be introspection-only, not a silent global disable."""
    from opencut.server import create_app

    create_app(config=OpenCutConfig(), testing=False)

    for boundary in ("temp_sweep", "disk_monitor", "migrations", "plugins", "sentry"):
        assert boundary in sentinels, f"production boot skipped {boundary}"


def test_route_manifest_generation_is_side_effect_free(sentinels):
    from opencut.tools.dump_route_manifest import build_manifest

    manifest = build_manifest()

    assert sentinels == {}, f"manifest generation had side effects: {sentinels}"
    assert manifest["routes"]


def test_repeated_generation_is_deterministic():
    """Two builds in one process must agree, byte for byte on the payload."""
    from opencut.tools.dump_route_manifest import build_manifest

    first = build_manifest()
    second = build_manifest()

    # generated_at is a timestamp by design; everything else must be stable.
    first.pop("generated_at", None)
    second.pop("generated_at", None)
    assert first == second


def test_feature_readiness_generation_is_side_effect_free(sentinels):
    from opencut.tools.dump_feature_readiness import build_manifest

    manifest = build_manifest()

    assert sentinels == {}, f"readiness generation had side effects: {sentinels}"
    assert manifest["records"]


def test_introspection_app_registers_no_third_party_plugin_routes(monkeypatch):
    """A contract describes what OpenCut ships.

    Importing locally installed plugin modules to build it would both run
    their code and make the manifest depend on whatever the developer happens
    to have in ``~/.opencut/plugins``.
    """
    from flask import Blueprint

    from opencut.core import plugins
    from opencut.server import create_app

    def fake_loader(app):
        marker = Blueprint("fixture_plugin", __name__)
        marker.add_url_rule("/fixture-plugin-marker", "marker", lambda: "ok")
        app.register_blueprint(marker, url_prefix="/plugins/fixture")
        return {"loaded": ["fixture"], "failed": []}

    monkeypatch.setattr(plugins, "load_all_plugins", fake_loader)

    def marker_routes(app):
        return [
            str(rule)
            for rule in app.url_map.iter_rules()
            if "fixture-plugin-marker" in str(rule)
        ]

    introspection_app = create_app(config=OpenCutConfig(), introspection=True)
    assert marker_routes(introspection_app) == []

    # The same loader must still run for a real server, or the guard would be
    # silently disabling the plugin system rather than scoping it.
    production_app = create_app(config=OpenCutConfig(), testing=False)
    assert marker_routes(production_app) == ["/plugins/fixture/fixture-plugin-marker"]
