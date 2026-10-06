# UniGradICON registration server

Runs uniGradICON registrations for Slicers that cannot run them locally (for example, an Intel
macOS Slicer, whose PyTorch cannot use Slicer's NumPy 2) or not fast enough (no GPU). In the
UniGradICON module, check **Remote processing**, enter the server address (`hostname:port`) and,
if the server has one, the token, and press **Connect**.

## Start it from Slicer

On a computer where UniGradICON runs locally, open **Local registration server**, choose the port,
enter a token so that only clients that know it can use the server, and press **Start server**.
The server address (`hostname:port`) is shown below the button; **Copy** puts it on the
clipboard, ready for the Server field of a client. **Log to GUI** shows the messages of the server
in the module's log, **Log to Console** in the Python console.

Other computers can only connect if the port is open in the firewall of the server computer. A
"Connection refused" from other computers while `curl http://localhost:<port>/info` works on the
server computer itself means the firewall blocks it, e.g. on RHEL/Rocky:
`sudo firewall-cmd --add-port=<port>/tcp --permanent && sudo firewall-cmd --reload`. Without
admin rights, an SSH tunnel works too: `ssh -N -L <port>:localhost:<port> user@server`, then
connect to `localhost:<port>`.

## Start it without Slicer

```bash
python -m venv unigradicon-server && source unigradicon-server/bin/activate
pip install torch                      # the build for your GPU, see https://pytorch.org
pip install -r requirements.txt
UNIGRADICON_SERVER_TOKEN=<secret> python server.py --host 0.0.0.0 --port 8899
```

Run it from a checkout of this repository: it imports `../UniGradICONHelpers`. The model weights
are read from `--weights-dir` (default: the repository's `model_checkpoints`) and downloaded there
on first use if missing.

## Notes

- The images are sent to the server. Use a token, and outside a trusted network put the server
  behind HTTPS (for example a reverse proxy); the server itself speaks plain HTTP.
- Jobs run one at a time in the order they arrive. Finished jobs and their files are removed
  after the client downloads the results, or after an hour.
- `pytest test_server.py` tests the server with a fake registration (needs `httpx` and `pytest`).
