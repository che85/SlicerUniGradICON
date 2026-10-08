"""UniGradICON registration server.

Runs uniGradICON for clients that cannot run it themselves, or not fast enough: a Slicer whose
Python has no working PyTorch, or a laptop without a GPU. A client uploads a fixed and a moving
image, polls the job, and downloads the transform and the warped moving image.

Start it from the UniGradICON module ("Start server"), or without Slicer:

    pip install -r requirements.txt
    python server.py --host 0.0.0.0 --port 8899 --token <secret>

Jobs run in the order they arrive: one at a time on the CPU, and --jobs-per-gpu at a time on each
GPU. Each of those runs in its own process, which keeps the model loaded between jobs.
"""

import argparse
import hmac
import json
import logging
import multiprocessing
import os
import queue
import shutil
import sys
import tempfile
import threading
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # for UniGradICONHelpers

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse

from UniGradICONHelpers import registration

API_VERSION = 1
DEFAULT_PORT = 8899
DEFAULT_WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "model_checkpoints")
TOKEN_ENVIRONMENT_VARIABLE = "UNIGRADICON_SERVER_TOKEN"
JOB_LIFETIME_SECONDS = 3600  # finished jobs and their files are removed after this
MODALITIES = ("MRI", "CT/CBCT")
LOSSES = ("LNCC", "Squared LNCC", "MIND-SSC")
MAX_IO_STEPS = 1000
IMAGE_SUFFIXES = (".nrrd", ".nhdr", ".nii.gz", ".nii", ".mha", ".mhd")

QUEUED, RUNNING, DONE, FAILED, CANCELLED = "queued", "running", "done", "failed", "cancelled"
FINISHED_STATES = (DONE, FAILED, CANCELLED)


class Job:
  def __init__(self, params, directory, order):
    self.id = uuid.uuid4().hex
    self.order = order  # arrival order, for the queue position
    self.params = params
    self.directory = directory
    self.state = QUEUED
    self.progress = None
    self.error = None
    self.finished = None
    self.cancel = threading.Event()
    self.transform_path = None
    self.warped_path = None
    self.fixed_mask_path = None

  def status(self, position=None):
    status = {"id": self.id, "state": self.state, "progress": self.progress, "error": self.error}
    if position is not None:
      status["queuePosition"] = position
    return status


class ThreadRegistrationRunner:
  """Runs a registration function in the server's worker thread; for tests with a fake one."""

  def __init__(self, register):
    self.register = register

  def run(self, job, arguments):
    def call_back(total, step, elapsed=None, remaining=None):
      if job.cancel.is_set():
        raise registration.RegistrationCancelled()
      job.progress = {"total": total, "step": step, "elapsed": elapsed, "remaining": remaining}

    return self.register(**arguments, call_back=call_back)


class ProcessRegistrationRunner:
  """Runs registrations in a child process, which keeps the model loaded between jobs.

  A registration holds the GIL for long stretches (ITK resampling, conversions); in a thread of
  the server it would stall the server until it could not even answer a status request.
  """

  def __init__(self, gpu=None):
    self.context = multiprocessing.get_context("spawn")
    self.gpu = gpu
    self.process = None

  def stop(self):
    if self.process is not None and self.process.is_alive():
      self.process.terminate()

  def _ensure_process(self):
    if self.process is not None and self.process.is_alive():
      return
    self.requests = self.context.Queue()
    self.messages = self.context.Queue()
    self.cancel_flag = self.context.Value("b", 0)
    self.process = self.context.Process(
      target=registration.serve_registrations, args=(self.requests, self.messages, self.cancel_flag, self.gpu),
      name="registration", daemon=True)
    self.process.start()

  def run(self, job, arguments):
    self._ensure_process()
    self.cancel_flag.value = 0
    self.requests.put(arguments)
    while True:
      if job.cancel.is_set():
        self.cancel_flag.value = 1
      try:
        kind, payload = self.messages.get(timeout=0.5)
      except queue.Empty:
        if not self.process.is_alive():
          raise RuntimeError("The registration process stopped unexpectedly")
        continue
      if kind == "progress":
        job.progress = payload
      elif kind == "done":
        return payload
      elif kind == "cancelled":
        raise registration.RegistrationCancelled()
      else:
        raise RuntimeError(payload)


def image_suffix(file_name):
  lower = (file_name or "").lower()
  for suffix in IMAGE_SUFFIXES:
    if lower.endswith(suffix):
      return suffix
  raise HTTPException(400, f"Unsupported image file '{file_name}'. Use one of: {', '.join(IMAGE_SUFFIXES)}")


def validate_params(params, devices):
  def choice(name, allowed):
    if params.get(name) not in allowed:
      raise HTTPException(400, f"'{name}' must be one of {list(allowed)}, not {params.get(name)!r}")
    return params[name]

  try:
    io_steps = int(params.get("io_steps", 0))
  except (TypeError, ValueError):
    raise HTTPException(400, "'io_steps' must be an integer")
  if not 0 <= io_steps <= MAX_IO_STEPS:
    raise HTTPException(400, f"'io_steps' must be between 0 and {MAX_IO_STEPS}")
  warp = params.get("warp", True)  # older clients always want the warped image
  if not isinstance(warp, bool):
    raise HTTPException(400, "'warp' must be true or false")
  return {
    "fixed_modality": choice("fixed_modality", MODALITIES),
    "moving_modality": choice("moving_modality", MODALITIES),
    "model": choice("model", registration.MODEL_WEIGHTS),
    "loss": choice("loss", LOSSES),
    "io_steps": io_steps,
    "device": choice("device", devices),
    "warp": warp,
  }


def create_app(weights_dir=DEFAULT_WEIGHTS_DIR, token=None, register=None, devices=None, jobs_per_gpu=1, gpus=None):
  """The server application. ``register``, ``devices`` and ``gpus`` (their number) are replaceable for testing."""
  app = FastAPI(title="UniGradICON registration server")
  jobs = {}
  pending = {registration.DEVICE_CPU: queue.Queue(), registration.DEVICE_GPU: queue.Queue()}
  lock = threading.Lock()
  work_root = tempfile.mkdtemp(prefix="unigradicon-server-")
  device_list = devices

  def server_devices():
    nonlocal device_list
    if device_list is None:
      device_list = registration.available_devices()
    return device_list

  def check_token(authorization: str = Header(default="")):
    if token and not hmac.compare_digest(authorization, f"Bearer {token}"):
      raise HTTPException(401, "Missing or wrong access token")

  def remove_job(job):
    shutil.rmtree(job.directory, ignore_errors=True)
    jobs.pop(job.id, None)

  def remove_expired_jobs():
    now = time.time()
    with lock:
      for job in list(jobs.values()):
        if job.state in FINISHED_STATES and job.finished and now - job.finished > JOB_LIFETIME_SECONDS:
          remove_job(job)

  def queue_position(job):
    if job.state != QUEUED:
      return None
    return sum(1 for other in jobs.values() if other.state == QUEUED and other.order < job.order
               and other.params["device"] == job.params["device"])

  def run_job(job, runner, place):
    params = job.params
    logging.info("Job %s started on %s", job.id[:8], place)
    started = time.time()
    arguments = {
      "fixed_path": job.fixed_path, "moving_path": job.moving_path,
      "fixed_modality": params["fixed_modality"], "moving_modality": params["moving_modality"],
      "model_name": params["model"], "loss": params["loss"], "io_steps": params["io_steps"],
      "device": params["device"], "weights_dir": weights_dir, "out_dir": job.directory,
      "warp": params["warp"],
      "fixed_mask_path": job.fixed_mask_path,
    }
    try:
      job.transform_path, job.warped_path = runner.run(job, arguments)
      job.state = CANCELLED if job.cancel.is_set() else DONE
    except registration.RegistrationCancelled:
      job.state = CANCELLED
    except Exception as e:
      logging.exception("Job %s failed", job.id[:8])
      job.state = FAILED
      job.error = f"{type(e).__name__}: {e}"
    job.finished = time.time()
    if job.state != FAILED:
      logging.info("Job %s %s after %.0f s", job.id[:8], job.state, job.finished - started)
    if job.state == CANCELLED:
      with lock:
        remove_job(job)

  def worker(device, gpu=None):
    runner = ThreadRegistrationRunner(register) if register else ProcessRegistrationRunner(gpu)
    runners.append(runner)
    place = device if gpu is None else f"{device} {gpu}"

    def loop():
      while True:
        job = pending[device].get()
        if job.cancel.is_set():
          continue  # cancelled while queued; already removed
        job.state = RUNNING
        run_job(job, runner, place)

    threading.Thread(target=loop, name=f"registration-worker-{place}", daemon=True).start()

  runners = []
  worker(registration.DEVICE_CPU)
  gpu_total = 0
  if registration.DEVICE_GPU in server_devices():
    gpu_total = registration.gpu_count() if gpus is None else gpus
    for gpu in range(gpu_total):
      for _ in range(jobs_per_gpu):
        worker(registration.DEVICE_GPU, gpu)
  app.state.runners = runners
  if gpu_total:
    logging.info("Registrations run one at a time on the CPU and %d at a time on each of %d GPU(s).",
                 jobs_per_gpu, gpu_total)
  order = iter(range(1 << 62))

  def get_job(job_id):
    job = jobs.get(job_id)
    if job is None:
      raise HTTPException(404, f"No job {job_id}")
    return job

  @app.get("/info", dependencies=[Depends(check_token)])
  def info():
    return {
      "server": "UniGradICON",
      "apiVersion": API_VERSION,
      "devices": server_devices(),
      "models": list(registration.MODEL_WEIGHTS),
      "modalities": list(MODALITIES),
      "losses": list(LOSSES),
      "gpus": gpu_total,
      "jobsPerGpu": jobs_per_gpu,
    }

  @app.post("/jobs", dependencies=[Depends(check_token)])
  def submit(fixed: UploadFile = File(...), moving: UploadFile = File(...), params: str = Form(...),
             fixed_mask: UploadFile = File(None)):
    remove_expired_jobs()
    try:
      raw_params = json.loads(params)
    except json.JSONDecodeError:
      raise HTTPException(400, "'params' must be JSON")
    job_params = validate_params(raw_params, server_devices())
    job = Job(job_params, tempfile.mkdtemp(dir=work_root), next(order))
    job.fixed_path = os.path.join(job.directory, "fixed" + image_suffix(fixed.filename))
    job.moving_path = os.path.join(job.directory, "moving" + image_suffix(moving.filename))
    uploads = [(fixed, job.fixed_path), (moving, job.moving_path)]
    if fixed_mask is not None:
      job.fixed_mask_path = os.path.join(job.directory, "fixed_mask" + image_suffix(fixed_mask.filename))
      uploads.append((fixed_mask, job.fixed_mask_path))
    for upload, path in uploads:
      with open(path, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    with lock:
      jobs[job.id] = job
    pending[job_params["device"]].put(job)
    logging.info("Job %s received: %s %s -> %s, %s, %d IO steps on %s%s", job.id[:8], job_params["model"],
                 job_params["moving_modality"], job_params["fixed_modality"], job_params["loss"],
                 job_params["io_steps"], job_params["device"], ", fixed mask" if job.fixed_mask_path else "")
    return job.status(queue_position(job))

  @app.get("/jobs/{job_id}", dependencies=[Depends(check_token)])
  def status(job_id: str):
    job = get_job(job_id)
    return job.status(queue_position(job))

  def result_file(job_id, attribute, download_name):
    job = get_job(job_id)
    if job.state != DONE:
      raise HTTPException(409, f"Job {job_id} is {job.state}, not done")
    path = getattr(job, attribute)
    if path is None:
      raise HTTPException(404, f"Job {job_id} has no {download_name}: it was not requested")
    return FileResponse(path, filename=download_name)

  @app.get("/jobs/{job_id}/transform", dependencies=[Depends(check_token)])
  def transform(job_id: str):
    return result_file(job_id, "transform_path", "transform.tfm")

  @app.get("/jobs/{job_id}/warped", dependencies=[Depends(check_token)])
  def warped(job_id: str):
    return result_file(job_id, "warped_path", "warped.nrrd")

  @app.delete("/jobs/{job_id}", dependencies=[Depends(check_token)])
  def delete(job_id: str):
    job = get_job(job_id)
    if job.state not in FINISHED_STATES:
      logging.info("Job %s cancel requested", job.id[:8])
    job.cancel.set()
    with lock:
      if job.state == RUNNING:
        return {"id": job.id, "state": "cancelling"}  # the worker removes it once it stops
      job.state = CANCELLED if job.state == QUEUED else job.state
      remove_job(job)
    return {"id": job.id, "state": "removed"}

  return app


def exit_with(pids, runners):
  """Stop the server once any of the processes is gone, as Slicer and its launcher are."""
  def watch():
    while all(registration.process_exists(pid) for pid in pids):
      time.sleep(2)
    logging.info("The process that started the server is gone; stopping.")
    for runner in runners:
      if hasattr(runner, "stop"):
        runner.stop()
    os._exit(0)
  threading.Thread(target=watch, name="parent-watch", daemon=True).start()


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  parser.add_argument("--host", default="127.0.0.1",
                      help="address to listen on; 0.0.0.0 makes it reachable from other machines")
  parser.add_argument("--port", type=int, default=DEFAULT_PORT)
  parser.add_argument("--weights-dir", default=DEFAULT_WEIGHTS_DIR,
                      help="folder with the model weights; missing weights are downloaded there")
  parser.add_argument("--token", default=os.environ.get(TOKEN_ENVIRONMENT_VARIABLE),
                      help=f"access token clients must send (default: ${TOKEN_ENVIRONMENT_VARIABLE})")
  parser.add_argument("--jobs-per-gpu", type=int, default=1,
                      help="registrations run at the same time on each GPU; each needs its own GPU memory")
  parser.add_argument("--exit-with-parent", action="store_true",
                      help="stop when the process that started the server exits (used by Slicer)")
  parser.add_argument("--parent-pid", type=int, action="append", default=[],
                      help="also stop when this process exits; can be repeated")
  args = parser.parse_args(argv)
  if args.jobs_per_gpu < 1:
    parser.error("--jobs-per-gpu must be at least 1")

  # the same look as uvicorn's own messages
  logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(message)s")
  if args.host not in ("127.0.0.1", "localhost") and not args.token:
    logging.warning("Listening on %s without an access token: anyone who can reach it can use it.", args.host)

  import uvicorn
  app = create_app(args.weights_dir, args.token, jobs_per_gpu=args.jobs_per_gpu)
  watched = args.parent_pid + ([os.getppid()] if args.exit_with_parent else [])
  if watched:
    exit_with(watched, app.state.runners)
  print(f"UniGradICON server on http://{args.host}:{args.port}", flush=True)
  # no line per request: clients poll their jobs every second; the job events are logged instead
  uvicorn.run(app, host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
  main()
