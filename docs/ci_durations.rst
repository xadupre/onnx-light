CI Job Durations (Past Two Months)
===================================

This page shows the duration of CI workflow runs over the past two months for the
`onnx-light <https://github.com/xadupre/onnx-light>`_ repository.
Durations are fetched from the GitHub REST API and plotted as time-series charts —
one chart per CI workflow.

.. note::

    This page queries the public GitHub REST API. No authentication token is required
    for a public repository, but anonymous requests are subject to rate-limiting
    (60 requests/hour per IP). When the API cannot be reached (offline build,
    rate-limit exceeded, …) the chart will be empty and a warning is printed to the
    console. Retrieved runs are cached per workflow as CSV files for two weeks.
    By default the cache is stored in the per-user cache directory
    (``$XDG_CACHE_HOME/onnx-light/ci_durations_workflows`` or
    ``~/.cache/onnx-light/ci_durations_workflows``). Setting the environment
    variable ``ONNX_LIGHT_DOCS_CACHE_IN_REPO=1`` switches the cache to
    ``<repo>/cache_data/ci_durations_workflows`` so it can be committed and
    reused across documentation builds. When the API is unreachable, previously
    cached data is displayed (along with the timestamp of the last successful
    fetch) instead of an empty chart.

.. note::

    **Persisting the in-repo cache on CI.** The
    ``ONNX_LIGHT_DOCS_CACHE_IN_REPO`` opt-in is only useful when the
    documentation job is also allowed to commit and push the resulting
    ``cache_data/`` directory back to the repository. To enable this on the
    official ``xadupre/onnx-light`` docs workflow:

    1. In ``.github/workflows/docs.yml``, grant the job write access to the
       repository contents and export the opt-in variable, e.g.::

           permissions:
             contents: write
           env:
             ONNX_LIGHT_DOCS_CACHE_IN_REPO: "1"

    2. After the ``sphinx-build`` step, add a step that commits ``cache_data/``
       and pushes it back to ``main`` using the workflow's ``GITHUB_TOKEN``
       (no extra secret is required — the default token is sufficient as long
       as ``permissions.contents`` is set to ``write`` and the repository
       setting *Settings → Actions → General → Workflow permissions* allows
       "Read and write permissions"). Branch protection rules on ``main`` may
       additionally need to allow pushes from GitHub Actions, or the push must
       target a dedicated branch.

    Forks, pull requests from forks and local builds should leave the
    variable unset: they keep the per-user cache and never attempt to write
    inside the repository.

.. runpython::
    :rst:

    import datetime
    import os

    print(
        f"*Page generated on {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.*"
    )

.. plot::
    :include-source: false

    """Query the GitHub API and plot CI workflow run durations."""

    import csv
    import datetime
    import json
    import os
    import urllib.request

    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.ticker
    import numpy as np

    _OWNER = "xadupre"
    _REPO = "onnx-light"
    _API_BASE = f"https://api.github.com/repos/{_OWNER}/{_REPO}"
    _HEADERS = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    _CACHE_MAX_AGE_DAYS = 14
    # Cache location.
    # By default the cache is stored in the per-user cache directory.
    # Set the environment variable ``ONNX_LIGHT_DOCS_CACHE_IN_REPO=1`` (e.g.
    # on the official xadupre/onnx-light documentation job that publishes to
    # xadupre.github.io) to persist the cache inside the repository under
    # ``cache_data/`` so it can be committed and reused across builds. The
    # repo-root is taken from ``GITHUB_WORKSPACE`` when available, otherwise
    # from the current working directory.
    _CACHE_IN_REPO = os.environ.get("ONNX_LIGHT_DOCS_CACHE_IN_REPO", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if _CACHE_IN_REPO:
        _REPO_ROOT = os.environ.get("GITHUB_WORKSPACE") or os.getcwd()
        _CACHE_DIR = os.path.join(_REPO_ROOT, "cache_data", "ci_durations_workflows")
    else:
        _USER_CACHE_DIR = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
        _CACHE_DIR = os.path.join(_USER_CACHE_DIR, "onnx-light", "ci_durations_workflows")
    _WORKFLOWS_CACHE_PATH = os.path.join(_CACHE_DIR, "_workflows.json")

    # Workflows that are NOT CI (skip Copilot agents and setup workflows).
    # All other workflows — including style/lint/spelling/typing/docs jobs that
    # run on pull requests — are considered CI and tracked on this page.
    _SKIP_PATTERNS = ("copilot",)


    def _gh_get(path, params=""):
        """Performs a GET request to the GitHub REST API and returns parsed JSON.

        Returns:
            dict | None: Parsed JSON response, or None on network or decode error.
        """
        import urllib.error

        url = f"{_API_BASE}/{path}"
        if params:
            url = f"{url}?{params}"
        req = urllib.request.Request(url, headers=_HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode())
        except (urllib.error.URLError, json.JSONDecodeError, OSError):
            return None


    def _fetch_workflow_runs(workflow_id, since_iso):
        """Fetches all workflow runs for *workflow_id* created on or after *since_iso*.

        Returns:
            list[dict]: List of workflow run objects returned by the GitHub API.
        """
        runs = []
        page = 1
        while True:
            data = _gh_get(
                f"actions/workflows/{workflow_id}/runs",
                f"per_page=100&page={page}&created=>={since_iso}&branch=main",
            )
            if data is None or not data.get("workflow_runs"):
                break
            runs.extend(data["workflow_runs"])
            if len(data["workflow_runs"]) < 100:
                break
            page += 1
        return runs


    def _cache_path(workflow_id):
        """Returns the cache file path for one workflow.

        Returns:
            str: Absolute path to the CSV cache file for this workflow.
        """
        return os.path.join(_CACHE_DIR, f"{workflow_id}.csv")


    def _load_cached_runs(workflow_id):
        """Loads cached workflow runs for one workflow.

        Returns:
            list[dict[str, str]]: Cached run rows.
        """
        cache_path = _cache_path(workflow_id)
        if not os.path.exists(cache_path):
            return []
        with open(cache_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = []
            for row in reader:
                rows.append(
                    {
                        "created_at": row.get("created_at", ""),
                        "updated_at": row.get("updated_at", ""),
                        "status": row.get("status", ""),
                        "conclusion": row.get("conclusion", ""),
                    }
                )
            return rows


    def _save_cached_runs(workflow_id, runs):
        """Saves cached workflow runs for one workflow.

        Returns:
            None
        """
        os.makedirs(_CACHE_DIR, exist_ok=True)
        cache_path = _cache_path(workflow_id)
        with open(cache_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["created_at", "updated_at", "status", "conclusion"]
            )
            writer.writeheader()
            for run in runs:
                writer.writerow(
                    {
                        "created_at": run.get("created_at", ""),
                        "updated_at": run.get("updated_at", ""),
                        "status": run.get("status", ""),
                        "conclusion": run.get("conclusion", ""),
                    }
                )


    def _cache_is_recent(workflow_id, now):
        """Checks whether a workflow cache file is newer than the cache max age.

        Returns:
            bool: True when the cache entry is recent enough.
        """
        cache_path = _cache_path(workflow_id)
        if not os.path.exists(cache_path):
            return False
        try:
            fetched_ts = os.path.getmtime(cache_path)
            fetched_dt = datetime.datetime.fromtimestamp(fetched_ts, tz=datetime.timezone.utc)
        except OSError:
            return False
        age = now - fetched_dt
        return age <= datetime.timedelta(days=_CACHE_MAX_AGE_DAYS)


    def _run_duration_minutes(run):
        """Computes the wall-clock duration of a successfully completed run.

        Returns:
            float | None: Duration in minutes, or None when unavailable.
        """
        if run.get("status") != "completed":
            return None
        if run.get("conclusion") != "success":
            return None
        created = run.get("created_at")
        updated = run.get("updated_at")
        if not created or not updated:
            return None
        try:
            fmt = "%Y-%m-%dT%H:%M:%SZ"
            t0 = datetime.datetime.strptime(created, fmt)
            t1 = datetime.datetime.strptime(updated, fmt)
            minutes = (t1 - t0).total_seconds() / 60.0
            return round(minutes, 2) if minutes >= 0 else None
        except ValueError:
            return None


    def _collect_data():
        """Collects workflow duration data from cache and/or GitHub API.

        Returns:
            tuple: ``(data, last_fetch)`` where ``data`` is a mapping
            ``workflow_name -> list[(datetime, duration_min)]`` and ``last_fetch``
            is a ``datetime`` of the most recent successful API fetch (workflow list
            or runs), or ``None`` when no cache exists.
        """
        now = datetime.datetime.now(datetime.timezone.utc)
        cutoff = (now - datetime.timedelta(days=62)).strftime("%Y-%m-%d")

        workflows_data = _gh_get("actions/workflows")
        if workflows_data:
            # Persist the workflow list so we can render the page even when
            # the API later becomes unreachable.
            try:
                os.makedirs(_CACHE_DIR, exist_ok=True)
                with open(_WORKFLOWS_CACHE_PATH, "w", encoding="utf-8") as f:
                    json.dump(workflows_data, f)
            except OSError:
                pass
        elif os.path.exists(_WORKFLOWS_CACHE_PATH):
            try:
                with open(_WORKFLOWS_CACHE_PATH, "r", encoding="utf-8") as f:
                    workflows_data = json.load(f)
            except (OSError, json.JSONDecodeError):
                workflows_data = None

        if not workflows_data:
            return {}, None

        last_fetch = None
        result = {}
        for wf in workflows_data.get("workflows", []):
            name = wf.get("name", "")
            wf_id = wf.get("id")
            path = wf.get("path", "")
            # Skip non-CI workflows (Copilot agents live under ``dynamic/`` and
            # the Copilot Setup Steps workflow under ``.github/workflows/``).
            filename = path.split("/")[-1].lower() if "/" in path else path.lower()
            if path.startswith("dynamic/") or any(
                filename.startswith(p) for p in _SKIP_PATTERNS
            ):
                continue

            cache_key = str(wf_id)
            cached_runs = _load_cached_runs(cache_key)
            if _cache_is_recent(cache_key, now):
                runs = cached_runs
            else:
                fetched = _fetch_workflow_runs(wf_id, cutoff)
                if fetched:
                    runs = fetched
                    _save_cached_runs(cache_key, runs)
                else:
                    runs = cached_runs
            # Track most-recent successful fetch from cache mtime.
            cache_path = _cache_path(cache_key)
            if os.path.exists(cache_path):
                try:
                    mtime = datetime.datetime.fromtimestamp(
                        os.path.getmtime(cache_path), tz=datetime.timezone.utc
                    )
                    if last_fetch is None or mtime > last_fetch:
                        last_fetch = mtime
                except OSError:
                    pass
            points = []
            for run in runs:
                dur = _run_duration_minutes(run)
                if dur is None:
                    continue
                try:
                    fmt = "%Y-%m-%dT%H:%M:%SZ"
                    dt = datetime.datetime.strptime(run["created_at"], fmt)
                    points.append((dt, dur))
                except (KeyError, ValueError):
                    continue
            if points:
                points.sort(key=lambda x: x[0])
                result[name] = points

        return result, last_fetch


    _AVG_WINDOW = 10

    data, _last_fetch = _collect_data()

    if not data:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(
            0.5,
            0.5,
            "No data available (GitHub API not reachable and no cache found).",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
        )
        ax.axis("off")
        plt.tight_layout()
    else:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        if _last_fetch is not None:
            _suptitle = (
                f"Last successful data fetch: "
                f"{_last_fetch.strftime('%Y-%m-%d %H:%M UTC')}"
            )
        else:
            _suptitle = "Last successful data fetch: unknown"

        for idx, (wf_name, points) in enumerate(sorted(data.items())):
            fig, ax = plt.subplots(figsize=(10, 4))
            dates = [p[0] for p in points]
            durations = [p[1] for p in points]

            color = colors[idx % len(colors)]
            ax.plot(dates, durations, "-", color=color, linewidth=1.2, label=wf_name)

            # Rolling average
            if len(durations) >= _AVG_WINDOW:
                kernel = np.ones(_AVG_WINDOW) / _AVG_WINDOW
                avg = np.convolve(durations, kernel, mode="valid")
                offset = (_AVG_WINDOW - 1) // 2
                avg_dates = dates[offset : offset + len(avg)]
                ax.plot(
                    avg_dates,
                    avg,
                    "--",
                    color="orange",
                    linewidth=1.5,
                    alpha=0.9,
                    label=f"{_AVG_WINDOW}-run avg",
                )

            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
            ax.tick_params(axis="x", rotation=30, labelsize=8)
            ax.set_title(wf_name, fontsize=12)
            ax.set_ylabel("Duration (min)", fontsize=10)
            fig.text(0.99, 0.01, _suptitle, ha="right", va="bottom", fontsize=8,
                     color="gray", style="italic")
            _y_fmt = matplotlib.ticker.ScalarFormatter()
            _y_fmt.set_useOffset(False)
            _y_fmt.set_scientific(False)
            ax.yaxis.set_major_formatter(_y_fmt)
            ax.grid(True, linestyle="--", alpha=0.4)
            if len(durations) >= _AVG_WINDOW:
                ax.legend(fontsize=9)
            fig.autofmt_xdate()
            plt.tight_layout()
