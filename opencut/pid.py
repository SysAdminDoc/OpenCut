"""
OpenCut PID / Port Management

Server lifecycle helpers extracted from opencut.server to keep that module
focused on app-factory and startup concerns.

Covers:
- PID file read / write / removal
- Port availability checks (live-listener aware; see ``_check_port``)
- Kill strategies (HTTP shutdown, PID kill, netstat)
- Master "nuke" sequence that tries all strategies in order
"""

import json
import logging
import os
import socket
import subprocess as _sp
import sys
import time
from contextlib import suppress

logger = logging.getLogger("opencut")

PID_FILE = os.path.join(os.path.expanduser("~"), ".opencut", "server.pid")


# ---------------------------------------------------------------------------
# PID File Management
# ---------------------------------------------------------------------------

def _write_pid(port: int):
    """Write current PID and port to file so future instances can find us."""
    import tempfile
    try:
        pid_dir = os.path.dirname(PID_FILE)
        os.makedirs(pid_dir, exist_ok=True)
        # Atomic write: temp file + rename to prevent partial reads
        fd, tmp_path = tempfile.mkstemp(dir=pid_dir, suffix=".tmp", prefix="server.pid.")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(f"{os.getpid()}\n{port}\n")
            os.replace(tmp_path, PID_FILE)
        except BaseException:
            with suppress(OSError):
                os.unlink(tmp_path)
            raise
        logger.debug("Wrote PID file: pid=%d port=%d", os.getpid(), port)
    except Exception as e:
        logger.warning("Could not write PID file: %s", e)


def _read_pid():
    """Read PID and port from file. Returns (pid, port) or (None, None)."""
    try:
        if os.path.exists(PID_FILE):
            with open(PID_FILE, "r", encoding="utf-8") as f:
                lines = f.read().strip().split("\n")
            pid = int(lines[0]) if lines else None
            port = int(lines[1]) if len(lines) > 1 else None
            return pid, port
    except Exception:
        pass
    return None, None


def _remove_pid():
    """Remove PID file."""
    try:
        if os.path.exists(PID_FILE):
            os.unlink(PID_FILE)
    except Exception:
        pass


def _is_pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is still running."""
    if pid is None:
        return False
    try:
        if sys.platform == "win32":
            result = _sp.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"],
                capture_output=True, text=True, timeout=5, check=False
            )
            return f'"{pid}"' in result.stdout
        else:
            os.kill(pid, 0)  # Signal 0 = check if alive
            return True
    except (OSError, _sp.TimeoutExpired):
        return False


# ---------------------------------------------------------------------------
# Port Checking (a live listener wins; TIME_WAIT remnants are reusable)
# ---------------------------------------------------------------------------

def _port_has_listener(host: str, port: int, timeout: float = 0.5) -> bool:
    """Return True when a process is actively accepting on ``host:port``.

    This is the authoritative "is the port taken" signal. A bind probe is not:
    on Windows ``SO_REUSEADDR`` lets a second socket bind over one that another
    process is *actively* holding (POSIX only permits that for TIME_WAIT), so a
    bind-based check reported a busy port as free and OpenCut started a second
    server on it. See ``_check_port``.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            return s.connect_ex((host, port)) == 0
    except OSError:
        return False


def _bind_probe(host: str, port: int, *, tolerate_time_wait: bool) -> bool:
    """Return True when this process could bind ``host:port`` right now."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if tolerate_time_wait:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            elif hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                # Windows-only. Refuses the bind if anyone else holds the port,
                # which is the semantic POSIX gives a plain bind for free.
                s.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            s.settimeout(1)
            s.bind((host, port))
            return True
    except OSError:
        return False


def _check_port(host: str, port: int) -> bool:
    """Return True when ``port`` is free for this process to own.

    A live listener always wins: no amount of socket-option tolerance makes it
    safe to start a second server against the same user data directory. Only
    once nothing is listening do we accept a TIME_WAIT socket left behind by a
    recently-killed server, which is what the original ``SO_REUSEADDR`` probe
    was reaching for.
    """
    if _port_has_listener(host, port):
        return False
    if _bind_probe(host, port, tolerate_time_wait=False):
        return True
    # Bind refused with nothing listening: a TIME_WAIT remnant is safe to reuse.
    return _bind_probe(host, port, tolerate_time_wait=True)


def _is_opencut_on_port(host: str, port: int) -> bool:
    """Return True when an OpenCut server is responding on ``port``."""
    import urllib.request
    try:
        req = urllib.request.Request(f"http://{host}:{port}/health", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read())
            return data.get("status") == "ok"
    except Exception:
        return False


def _wait_for_port(host: str, port: int, timeout: float = 8.0) -> bool:
    """Poll until ``port`` becomes available or ``timeout`` elapses."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _check_port(host, port):
            return True
        time.sleep(0.4)
    return False


# ---------------------------------------------------------------------------
# Kill Strategies (tried in order by _nuke_old_servers, each more aggressive)
# ---------------------------------------------------------------------------

def _kill_via_shutdown_endpoint(host: str, port: int) -> bool:
    """Strategy 1: Ask the server to shut itself down via HTTP."""
    import urllib.request
    try:
        csrf_token = ""
        try:
            health_req = urllib.request.Request(f"http://{host}:{port}/health", method="GET")
            with urllib.request.urlopen(health_req, timeout=2) as health_resp:
                health_data = json.loads(health_resp.read())
                csrf_token = health_data.get("csrf_token", "")
        except Exception:
            pass

        req = urllib.request.Request(
            f"http://{host}:{port}/shutdown",
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-OpenCut-Token": csrf_token,
            },
            data=b"{}",
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
        old_pid = data.get("pid")
        logger.info("Shutdown request accepted by server on :%d (pid=%s)", port, old_pid)
        return True
    except Exception:
        return False


def _kill_via_pid(pid: int) -> bool:
    """Strategy 2: Kill a specific PID directly."""
    if pid is None or not _is_pid_alive(pid):
        return False
    try:
        print(f"  Killing old server process (PID {pid})...")
        logger.info("Killing PID %d", pid)
        if sys.platform == "win32":
            _sp.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                    capture_output=True, timeout=10, check=False)
        else:
            os.kill(pid, 9)  # SIGKILL
            with suppress(ChildProcessError):
                os.waitpid(pid, os.WNOHANG)

        for _ in range(6):
            time.sleep(0.3)
            if not _is_pid_alive(pid):
                return True
        logger.warning("PID %d still alive after kill attempt", pid)
        return False
    except Exception as e:
        logger.warning("PID kill failed for %d: %s", pid, e)
        return False


def _kill_via_netstat(host: str, port: int) -> bool:
    """Strategy 3: force-kill a verified OpenCut process holding the port."""
    if not _is_opencut_on_port(host, port):
        logger.warning(
            "Refusing netstat kill for port %d: listener is not verified as OpenCut",
            port,
        )
        return False

    killed_any = False
    try:
        if sys.platform == "win32":
            result = _sp.run(
                ["netstat", "-ano", "-p", "TCP"],
                capture_output=True, text=True, timeout=10, check=False
            )
            for line in result.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 5 and parts[3] == "LISTENING":
                    local_addr = parts[1]
                    if local_addr.endswith(f":{port}"):
                        pid = parts[4]
                        if pid.isdigit() and int(pid) != os.getpid():
                            print(f"  Found process {pid} on port {port}, killing...")
                            logger.info("Killing PID %s found on port %d", pid, port)
                            _sp.run(["taskkill", "/F", "/T", "/PID", pid],
                                    capture_output=True, timeout=10, check=False)
                            killed_any = True
        else:
            result = _sp.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True, text=True, timeout=5, check=False
            )
            for pid in result.stdout.strip().split():
                if pid.isdigit() and int(pid) != os.getpid():
                    print(f"  Found process {pid} on port {port}, killing...")
                    logger.info("Killing PID %s found on port %d", pid, port)
                    os.kill(int(pid), 9)
                    killed_any = True
    except Exception as e:
        logger.warning("Netstat kill failed for port %d: %s", port, e)
    return killed_any


# ---------------------------------------------------------------------------
# Master Kill Sequence
# ---------------------------------------------------------------------------

def _nuke_old_servers(host: str, port: int) -> bool:
    """Aggressively kill any existing server(s) to reclaim ``port``.

    Tries multiple strategies in order.  Returns True when the port is free.
    """
    print(f"  Port {port} is in use. Cleaning up...")
    logger.info("Port %d busy — starting kill sequence", port)

    # Step 1: graceful HTTP shutdown
    _kill_via_shutdown_endpoint(host, port)
    if _wait_for_port(host, port, timeout=3.0):
        print("  Graceful shutdown succeeded.")
        return True

    # Step 2: kill via PID file
    old_pid, _old_port = _read_pid()
    if old_pid:
        _kill_via_pid(old_pid)
        if _wait_for_port(host, port, timeout=3.0):
            # Only drop the PID file once the process it names is actually
            # gone. Removing it first meant a failed kill left a live server
            # with no PID file, so the caller could no longer name the process
            # holding the port and reported "an unknown PID" instead.
            _remove_pid()
            print(f"  Killed old server via PID file (PID {old_pid}).")
            return True

    # Step 3: kill via netstat
    _kill_via_netstat(host, port)
    if _wait_for_port(host, port, timeout=4.0):
        print(f"  Killed process holding port {port}.")
        return True

    # Step 4: last check with SO_REUSEADDR (TIME_WAIT is OK)
    if _check_port(host, port):
        print(f"  Port {port} available (socket in TIME_WAIT, safe to reuse).")
        return True

    print(f"  Could not free port {port}.")
    logger.warning("All kill strategies failed for port %d", port)
    return False
