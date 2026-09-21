IMAGE QUALITY CHECKER (shadow + blur detection)
================================================

Everything the app needs is in this folder:

  app.py               the web app
  templates\index.html the web page
  models\              the trained models: shadow_model.onnx, blur_model.pt,
                       docres.pkl, RealESRGAN_x4plus.pth, big-lama.pt
                       (UVDoc's best_model.pkl comes from C:\UVDoc)
  requirements.txt     Python packages (exact versions)
  setup.bat            one-time install
  start_server.bat     starts the app
  allow_firewall.bat   optional: lets other PCs on the network open the app

REQUIREMENTS
  * Windows 10/11, 64-bit
  * Python 3.12 (64-bit) from https://www.python.org/downloads/
    (during install tick "Add python.exe to PATH")
  * Internet connection for the first setup (about 1 GB of packages is downloaded)

FIRST TIME
  1. Double-click setup.bat and wait until it says "Setup finished".
     It creates a .venv folder here; nothing is installed system-wide.

EVERY TIME
  1. Double-click start_server.bat
  2. Open http://127.0.0.1:5000 in a browser.
  3. Keep the black window open while using the app; close it (or Ctrl+C) to stop.

OPENING FROM OTHER DEVICES (optional)
  * Right-click allow_firewall.bat -> Run as administrator (once).
  * Start the app; the window prints a network address like http://192.168.x.x:5000
    which other devices on the same network can open.

LIVE PROGRESS
  After "Process images" the files are uploaded and a run page opens. Its table
  shows, for every image, each step as it happens:
    Dewarp detection -> Dewarping (UVDoc, from C:\UVDoc; warped = bending
    score >= 0.02) -> Shadow detection -> Shadow removal -> Blur detection ->
    Blur removal -> Resolution check -> Upscaling (Real-ESRGAN 4x) ->
    Enhancement (DocRes "appearance": lighting, background and colour) ->
    Damage detection (tears / missing paper) -> Inpainting (big-LaMa)
  The enhancement has no check: every image gets it. Damage detection and
  inpainting are the last step; tears are found against a dark background
  behind the page, so if an earlier step removed that background the check
  reports "no dark backdrop" and nothing is inpainted.
  Each check runs on the image as corrected by the steps before it. At the end,
  images whose width or height is below 1200 px are upscaled 4x with
  models\RealESRGAN_x4plus.pth (once only). A correction step only runs when
  that problem is found, otherwise it shows "Not needed".
  Runs are processed one at a time; a second upload waits for the first. The
  run page can be refreshed or bookmarked while the server keeps running.

PROCESSING A WHOLE FOLDER
  In the browser: click "Choose a folder" instead of choosing images, then
  "Process images". When it is done the page offers "Download restored folder
  (ZIP)" and "Download report (CSV)". Every run is also saved on the PC that
  runs the app, in results\<date-time-id>\ (output\, report.csv and the ZIP);
  runs older than 7 days are deleted automatically (CONFIG in app.py).

  Without a browser: in this folder, with the .venv active, run:
      python process_folder.py C:\path\to\input C:\path\to\output
  * Every image is saved to the output folder with the same name: corrected
    images are fixed (blur first, then shadow), good images are copied as-is.
  * output\report.csv lists the probabilities and corrections for each image.
  * Stopped half-way? Run the same command again: finished images are skipped.
    Add --overwrite to redo them, or --checks with any of upscale, blur and
    shadow to run only those steps (e.g. --checks blur shadow skips upscaling).

GPU (NVIDIA)
  The blur, DocRes and Real-ESRGAN models run on the NVIDIA GPU when there is
  one (CONFIG "device": "auto"), otherwise on the CPU. The start-up window
  prints "Device: GPU ..." or "Device: CPU". This needs the CUDA build of
  PyTorch from requirements.txt:
      pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
  (CUDA 12.6 builds still support older cards such as the Quadro P4000.)
  If the GPU runs out of memory on a very large image, that step runs on the
  CPU instead. The small shadow model always runs on the CPU.

SETTINGS
  Thresholds, port, etc. are in the CONFIG block near the top of app.py.

Mac / Linux: instead of the .bat files run
    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
    .venv/bin/python app.py
  (on Mac, remove "+cpu" from the torch/torchvision lines in requirements.txt first)
