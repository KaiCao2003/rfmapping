"""Logging failures must not leave FFmpeg waiting on an open video pipe."""

from concurrent.futures import Future
import io
import json
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from EBB import ebc_video as video


class _Overlay:
    def __init__(self, data, width, height, fps):
        self.size = width, height
        self.valid = np.ones(len(data["frame_ids"]), dtype=bool)

    def draw(self, frame, frame_id, row):
        return frame


class _EncoderInput(io.BytesIO):
    def __init__(self, child, fail):
        super().__init__()
        self.child, self.fail = child, fail

    def write(self, value):
        if self.fail:
            self.child.returncode = 1
            self.child.log.write(b"encoder rejected input: regression diagnostic\n")
            self.child.log.flush()
            raise BrokenPipeError("encoder pipe closed")
        return super().write(value)


class _Child:
    def __init__(self, role, payload, log, *, encoder_fails=False, ignores_terminate=False):
        self.role, self.log = role, log
        self.stdout = io.BytesIO(payload) if role == "decoder" else None
        self.stdin = _EncoderInput(self, encoder_fails) if role == "encoder" else None
        self.payload_size = len(payload)
        self.ignores_terminate = ignores_terminate
        self.returncode = None
        self.terminated = self.killed = False
        self.events = []

    def poll(self):
        return self.returncode

    def terminate(self):
        self.events.append(("terminate",))
        self.terminated = True

    def kill(self):
        self.events.append(("kill",))
        self.killed = True

    def wait(self, timeout=None):
        self.events.append(("wait", timeout))
        if self.returncode is not None:
            return self.returncode
        # Turn the real circular wait into an immediate test failure.
        if self.role == "encoder" and not self.stdin.closed:
            raise AssertionError("Waiting for encoder while its input pipe is open")
        if self.role == "decoder" and not self.stdout.closed:
            if self.stdout.tell() < self.payload_size:
                raise AssertionError("Waiting for decoder with an undrained output pipe")
        if self.killed:
            self.returncode = -9
        elif self.terminated:
            assert timeout is not None and 0 < timeout <= 10, "Cleanup wait must be bounded"
            if self.ignores_terminate:
                raise subprocess.TimeoutExpired(self.role, timeout)
            self.returncode = -15
        else:
            self.returncode = 0
        return self.returncode


class _SharedRenderer:
    """Exercise the parallel loop without creating a real worker pool."""

    def __init__(self, data, width, height, fps, workers, overlay_type):
        self.slots = workers * 2
        self.input_bytes = width * height * 3
        self.payloads = {}
        self.pool = SimpleNamespace(submit=self.submit)
        self.closed = False

    @staticmethod
    def submit(*args):
        result = Future()
        result.set_result(None)
        return result

    def read(self, stream, slot):
        self.payloads[slot] = stream.read(self.input_bytes)
        return len(self.payloads[slot])

    def write(self, stream, slot, preview=None):
        stream.write(self.payloads.pop(slot))

    def close(self):
        self.closed = True


@pytest.fixture
def rendering(tmp_path, monkeypatch):
    count, width, height = 1005, 4, 4
    data = dict(frame_ids=np.arange(count), times=np.arange(count) / 50,
                session=tmp_path, phase="baseline", arena_type="rectangle")
    state = SimpleNamespace(children=[], renderers=[], encoder_fails=False,
                            ignores_terminate=False)
    stream = dict(width=width, height=height, nb_frames=str(count), avg_frame_rate="50/1")

    def run(command, **kwargs):
        assert command[0] == "ffprobe"
        return SimpleNamespace(stdout=json.dumps({"streams": [stream]}))

    def popen(command, **kwargs):
        role = "decoder" if "pipe:1" in command else "encoder"
        child = _Child(role, bytes(width * height * 3 * count), kwargs["stderr"],
                       encoder_fails=state.encoder_fails,
                       ignores_terminate=state.ignores_terminate)
        state.children.append(child)
        return child

    def renderer(*args):
        result = _SharedRenderer(*args)
        state.renderers.append(result)
        return result

    monkeypatch.setattr(video, "subprocess", SimpleNamespace(
        run=run, Popen=popen, PIPE=subprocess.PIPE, TimeoutExpired=subprocess.TimeoutExpired))
    monkeypatch.setattr(video, "_select_encoder", lambda *args: "libx264")
    monkeypatch.setattr(video, "_SharedRenderer", renderer)

    def export(workers):
        return video.export_ebc_overlay(data, tmp_path / "source.avi", tmp_path / "overlay.mp4",
                                        workers=workers, overlay_type=_Overlay)

    state.export = export
    return state


def _closed_stdout_at(monkeypatch, stage):
    def output(message, **kwargs):
        if ((stage == "startup" and "drawing workers" in message)
                or (stage == "progress" and "1,000/" in message)):
            raise BrokenPipeError("closed stdout")

    monkeypatch.setattr(video, "print", output, raising=False)


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("stage", ["startup", "progress"])
def test_closed_stdout_exits_without_waiting_on_ffmpeg_pipes(rendering, monkeypatch, stage, workers):
    _closed_stdout_at(monkeypatch, stage)
    with pytest.raises(BrokenPipeError, match="closed stdout"):
        rendering.export(workers)
    assert len(rendering.children) == (1 if stage == "startup" else 2)
    for child in rendering.children:
        assert child.terminated
        assert child.returncode is not None
        assert (child.stdin if child.role == "encoder" else child.stdout).closed
    assert all(renderer.closed for renderer in rendering.renderers)


@pytest.mark.parametrize("workers", [1, 2])
def test_encoder_pipe_failure_retains_ffmpeg_diagnostic(rendering, workers):
    rendering.encoder_fails = True
    with pytest.raises(RuntimeError, match="encoder rejected input: regression diagnostic"):
        rendering.export(workers)
    encoder = next(child for child in rendering.children if child.role == "encoder")
    assert encoder.returncode == 1
    assert encoder.stdin.closed
    assert all(child.returncode is not None for child in rendering.children)
    assert all(renderer.closed for renderer in rendering.renderers)


def test_cleanup_closes_pipes_and_kills_only_its_unresponsive_children(rendering, monkeypatch):
    rendering.ignores_terminate = True
    _closed_stdout_at(monkeypatch, "progress")
    with pytest.raises(BrokenPipeError, match="closed stdout"):
        rendering.export(2)
    assert len(rendering.children) == 2
    for child in rendering.children:
        assert child.terminated and child.killed
        assert child.returncode == -9
        assert child.events[0] == ("terminate",)
        assert child.events[1][0] == "wait" and child.events[1][1] is not None
        assert child.events[2] == ("kill",)
    assert rendering.renderers[0].closed
