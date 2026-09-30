---
name: reduce-ci-waste
description: Reduces lost CI compute when investigating failed or cancelled GitHub Actions runs or preparing a PR update that will rerun CI.
---

# Reduce lost CI compute

The lost-compute dashboard counts the duration of failed and cancelled workflow
runs. Use this checklist to avoid repeating expensive runs without hiding failures.

1. Diagnose before changing code.
   - List recent runs and inspect failed job logs, not just the workflow conclusion.
     Identify the failing job, step, commit, and whether the failure is reproducible.
   - If a run has no failed jobs, check whether it was skipped, blocked before jobs
     started, or superseded; do not treat its conclusion as a test failure.
   - For cancelled runs, check whether a newer run replaced the same branch under
     `cancel-in-progress: true`. Investigate the surviving run instead of rerunning
     the cancelled one.
   - Distinguish scheduled and post-`ci-1-core` `workflow_run` failures from PR
     failures. Fix the cause in the relevant workflow or code, not by weakening checks.

2. Validate locally before pushing.
   - Select the existing test, formatter, or build command for the affected files;
     run the smallest relevant checks first. Reproduce CI failures locally when
     feasible and verify the fix before triggering another CI cycle.
   - Review the diff and changed-file list. Avoid unrelated formatting, generated
     artifacts, dependency changes, and multiple speculative fixes in one push.
   - Do not disable tests, suppress errors, or mark failures successful merely to
     improve the dashboard's success ratio.

3. Minimize redundant runs.
   - Batch related, locally verified changes into one PR update when possible.
     Each additional push can cancel in-flight runs and start their replacements.
   - If a follow-up is needed, use the latest completed run's logs to decide what
     to change; do not blindly rerun an unchanged failing workflow.
   - After submitting a change while CI is running, report that CI is pending and
     return control. Do not wait or poll just to make another speculative push.
