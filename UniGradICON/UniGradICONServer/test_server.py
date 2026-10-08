"""Tests of the registration server with a fake registration, so they need no PyTorch.

    pip install fastapi python-multipart httpx pytest
    pytest test_server.py
"""

import json
import os
import threading
import time

from fastapi.testclient import TestClient

import server
from UniGradICONHelpers import registration

PARAMS = {
  "fixed_modality": "MRI", "moving_modality": "MRI", "model": "unigradicon",
  "loss": "LNCC", "io_steps": 3, "device": "CPU",
}


class FakeRegistration:
  """Writes small result files; can be held at a step until released."""

  def __init__(self, fail=False):
    self.fail = fail
    self.release = threading.Event()
    self.release.set()
    self.calls = []
    self.masks = []

  def __call__(self, fixed_path, moving_path, fixed_modality, moving_modality, model_name, loss, io_steps,
               device, weights_dir, out_dir, call_back=None, warp=True, fixed_mask_path=None):
    self.calls.append((open(fixed_path, "rb").read(), open(moving_path, "rb").read(), io_steps, device))
    self.masks.append(open(fixed_mask_path, "rb").read() if fixed_mask_path else None)
    for step in range(io_steps):
      call_back(io_steps, step, float(step), float(io_steps - step))
      self.release.wait(5)
    if self.fail:
      raise RuntimeError("images do not overlap")
    transform_path = os.path.join(out_dir, "transform.tfm")
    open(transform_path, "w").write("transform")
    if not warp:
      return transform_path, None
    warped_path = os.path.join(out_dir, "warped.nrrd")
    open(warped_path, "wb").write(b"warped")
    return transform_path, warped_path


def client_for(fake, token=None):
  return TestClient(server.create_app(token=token, register=fake, devices=["CPU"]))


def submit(client, params=PARAMS, headers=None, fixed_name="fixed.nrrd"):
  return client.post(
    "/jobs",
    files={"fixed": (fixed_name, b"FIXED"), "moving": ("moving.nii.gz", b"MOVING")},
    data={"params": json.dumps(params)},
    headers=headers or {},
  )


def wait_for(client, job_id, states, headers=None, timeout=5):
  deadline = time.time() + timeout
  while time.time() < deadline:
    status = client.get(f"/jobs/{job_id}", headers=headers or {}).json()
    if status["state"] in states:
      return status
    time.sleep(0.02)
  raise AssertionError(f"job {job_id} never reached {states}, last {status}")


def test_info_lists_devices_and_models():
  info = client_for(FakeRegistration()).get("/info").json()
  assert info["devices"] == ["CPU"]
  assert set(info["models"]) == set(registration.MODEL_WEIGHTS)


def test_job_runs_and_returns_results():
  fake = FakeRegistration()
  client = client_for(fake)
  job = submit(client).json()
  status = wait_for(client, job["id"], ("done",))
  assert status["progress"]["total"] == 3
  assert fake.calls == [(b"FIXED", b"MOVING", 3, "CPU")]
  assert client.get(f"/jobs/{job['id']}/transform").content == b"transform"
  assert client.get(f"/jobs/{job['id']}/warped").content == b"warped"


def test_fixed_mask_is_passed_on():
  fake = FakeRegistration()
  client = client_for(fake)
  job = client.post(
    "/jobs",
    files={"fixed": ("fixed.nrrd", b"FIXED"), "moving": ("moving.nrrd", b"MOVING"),
           "fixed_mask": ("mask.nrrd", b"MASK")},
    data={"params": json.dumps(PARAMS)},
  ).json()
  wait_for(client, job["id"], ("done",))
  job_without = submit(client).json()
  wait_for(client, job_without["id"], ("done",))
  assert fake.masks == [b"MASK", None]
  assert client.post(
    "/jobs",
    files={"fixed": ("fixed.nrrd", b"F"), "moving": ("moving.nrrd", b"M"), "fixed_mask": ("mask.png", b"X")},
    data={"params": json.dumps(PARAMS)},
  ).status_code == 400


def test_transform_only_job_has_no_warped_image():
  client = client_for(FakeRegistration())
  job = submit(client, dict(PARAMS, warp=False)).json()
  wait_for(client, job["id"], ("done",))
  assert client.get(f"/jobs/{job['id']}/transform").content == b"transform"
  assert client.get(f"/jobs/{job['id']}/warped").status_code == 404
  assert submit(client, dict(PARAMS, warp="no")).status_code == 400


def test_failure_is_reported():
  client = client_for(FakeRegistration(fail=True))
  job = submit(client).json()
  status = wait_for(client, job["id"], ("failed",))
  assert "images do not overlap" in status["error"]
  assert client.get(f"/jobs/{job['id']}/warped").status_code == 409


def test_running_job_can_be_cancelled():
  fake = FakeRegistration()
  fake.release.clear()  # hold the job at its first step
  client = client_for(fake)
  job = submit(client).json()
  wait_for(client, job["id"], ("running",))
  assert client.delete(f"/jobs/{job['id']}").json()["state"] == "cancelling"
  fake.release.set()
  deadline = time.time() + 5
  while client.get(f"/jobs/{job['id']}").status_code != 404:
    assert time.time() < deadline, "cancelled job was never removed"
    time.sleep(0.02)


def test_queued_jobs_report_their_position():
  fake = FakeRegistration()
  fake.release.clear()
  client = client_for(fake)
  first = submit(client).json()
  wait_for(client, first["id"], ("running",))
  second = submit(client).json()
  third = submit(client).json()
  assert client.get(f"/jobs/{third['id']}").json()["queuePosition"] == 1
  assert client.delete(f"/jobs/{second['id']}").json()["state"] == "removed"
  assert client.get(f"/jobs/{third['id']}").json()["queuePosition"] == 0
  fake.release.set()
  wait_for(client, third["id"], ("done",))
  assert len(fake.calls) == 2  # the cancelled one never ran


def test_jobs_run_concurrently_on_each_gpu():
  fake = FakeRegistration()
  fake.release.clear()
  client = TestClient(server.create_app(register=fake, devices=["CPU", "GPU"], gpus=2, jobs_per_gpu=2))
  info = client.get("/info").json()
  assert (info["gpus"], info["jobsPerGpu"]) == (2, 2)
  gpu_jobs = [submit(client, dict(PARAMS, device="GPU")).json() for _ in range(5)]
  cpu_job = submit(client).json()
  for job in gpu_jobs[:4] + [cpu_job]:
    wait_for(client, job["id"], ("running",))
  assert client.get(f"/jobs/{gpu_jobs[4]['id']}").json()["queuePosition"] == 0  # CPU jobs do not count
  fake.release.set()
  for job in gpu_jobs + [cpu_job]:
    wait_for(client, job["id"], ("done",))
  assert len(fake.calls) == 6


def test_invalid_parameters_are_rejected():
  client = client_for(FakeRegistration())
  assert submit(client, dict(PARAMS, device="GPU")).status_code == 400
  assert submit(client, dict(PARAMS, io_steps=-1)).status_code == 400
  assert submit(client, dict(PARAMS, fixed_modality="PET")).status_code == 400
  assert submit(client, fixed_name="fixed.png").status_code == 400


def test_token_is_required_when_set():
  client = client_for(FakeRegistration(), token="secret")
  assert client.get("/info").status_code == 401
  assert client.get("/info", headers={"Authorization": "Bearer wrong"}).status_code == 401
  headers = {"Authorization": "Bearer secret"}
  assert client.get("/info", headers=headers).status_code == 200
  job = submit(client, headers=headers).json()
  wait_for(client, job["id"], ("done",), headers=headers)
