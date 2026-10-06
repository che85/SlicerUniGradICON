import vtk, qt, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

import json
import glob
import os
import numpy as np
import SimpleITK as sitk
import logging
import sys

# Slicer settings (not scene) keys
REMOTE_PROCESSING_SETTING = "UniGradICON/RemoteProcessing"
SERVER_URL_SETTING = "UniGradICON/ServerUrl"
SERVER_TOKENS_SETTING_GROUP = "UniGradICON/ServerTokens"


class UniGradICON(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "UniGradICON"
    self.parent.categories = ["Registration"]
    self.parent.dependencies = []
    self.parent.associatedNodeTypes = ["vtkMRMLScriptedModuleNode"]
    self.parent.contributors = ["Basar Demir (University of North Carolina at Chapel Hill), Lin Tian (University of North Carolina at Chapel Hill), Hastings Greer (University of North Carolina at Chapel Hill), Marc Niethammer (University of North Carolina at Chapel Hill)"]
    self.parent.helpText = """
    This module performs medical image registration using the family of foundational GradICON deep registration models. For more information, visit the <a href="https://github.com/uncbiag/uniGradICON">uniGradICON</a> repository.
    """
    self.parent.acknowledgementText = ""


def _checkPyTorchNumPyCompatibility():
  """Check if PyTorch can be imported with the current NumPy version.

  Some PyTorch versions compiled against NumPy 1.x fail to import when
  NumPy 2.x is installed due to C API changes. This function detects
  that specific error and downgrades NumPy if needed.

  This is future-proof: when PyTorch supports NumPy 2.x, the import
  will succeed and no downgrade will occur.
  """
  try:
    import torch
  except ImportError as e:
    error_msg = str(e)
    if "_ARRAY_API not found" in error_msg or "numpy.core.multiarray" in error_msg:
      import numpy as np
      if np.__version__.startswith("2."):
        if not slicer.util.confirmOkCancelDisplay(
          "NumPy 2.x is incompatible with the installed PyTorch.\n\n"
          "Do you want to downgrade NumPy to a compatible version?\n"
          "You will need to restart 3D Slicer after the change.",
          windowTitle="UniGradICON - NumPy compatibility"
        ):
          raise RuntimeError(
            "UniGradICON requires NumPy < 2 for PyTorch compatibility. "
            "You can run 'slicer.util.pip_install(\"numpy<2\")' in the Python Console and restart Slicer."
          )
        slicer.util.pip_install("numpy<2")
        slicer.util.infoDisplay(
          "NumPy has been downgraded for PyTorch compatibility.\n"
          "Please restart 3D Slicer for the changes to take effect.",
          windowTitle="UniGradICON"
        )
        raise RuntimeError(
          "NumPy was downgraded for PyTorch compatibility. "
          "Please restart 3D Slicer."
        )
    raise


#
# unigradiconWidget
#

class UniGradICONWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  REMOTE_PROCESSING_SETTING = REMOTE_PROCESSING_SETTING
  SERVER_URL_SETTING = SERVER_URL_SETTING
  SERVER_URL_HISTORY_SETTING = "UniGradICON/ServerUrlHistory"
  SERVER_PORT_SETTING = "UniGradICON/ServerPort"
  LOG_CONSOLE_SETTING = "UniGradICON/ServerLogToConsole"
  LOG_GUI_SETTING = "UniGradICON/ServerLogToGui"
  DEFAULT_SERVER_PORT = 8899
  WEIGHTS_FILE_MB = 283  # each model's weights file

  def __init__(self, parent=None):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.__init__(self, parent)
    VTKObservationMixin.__init__(self)  # needed for parameter node observation
    self.logic = None
    self._parameterNode = None
    self._updatingGUIFromParameterNode = False
    self.checkpointFolder = self.resourcePath("UI") + "/../../../model_checkpoints/"
    self._localBackendProblem = None
    self._serverInfo = None
    self._remoteRegistration = None
    self._remoteState = None
    self._registrationStart = None
    self._webServer = None
    self._serverUrl = None
    self._lastReconnectAttempt = None

    global sitkUtils
    global SampleData
    import sitkUtils
    import SampleData

  def ensureLocalBackend(self):
    """See UniGradICONLogic.ensureLocalBackend."""
    self.logic.ensureLocalBackend()

  def setup(self):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.setup(self)

    # Load widget from .ui file (created by Qt Designer).
    # Additional widgets can be instantiated manually and added to self.layout.
    uiWidget = slicer.util.loadUI(self.resourcePath('UI/UniGradICON.ui'))
    self.layout.addWidget(uiWidget)
    self.ui = slicer.util.childWidgetVariables(uiWidget)
    
    # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
    # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
    # "setMRMLScene(vtkMRMLScene*)" slot.
    uiWidget.setMRMLScene(slicer.mrmlScene)

    # Create logic class. Logic implements all computations that should be possible to run
    # in batch mode, without a graphical user interface.
    self.logic = UniGradICONLogic(self.checkpointFolder)

    self.ui.modelComboBox.addItems(PresetManager().getPresetNames())
    self.ui.stagesPresetsComboBox = self.ui.modelComboBox  # its name from SlicerANTs, for scripts
    # Connections

    # These connections ensure that we update parameter node when scene is closed
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

    # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
    # (in the selected parameter node).
    self.ui.lossComboBox.connect("currentIndexChanged(int)", self.updateParameterNodeFromGUI)
    self.ui.outputTransformComboBox.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.outputVolumeComboBox.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.ioSpinBox.connect("valueChanged(int)", self.updateParameterNodeFromGUI)
    self.ui.deviceComboBox.connect("currentIndexChanged(int)", self.updateParameterNodeFromGUI)

    self.ui.fixedImageNodeComboBox.connect("currentNodeChanged(vtkMRMLNode*)", self.updateImageSettingsFromGUI)
    self.ui.movingImageNodeComboBox.connect("currentNodeChanged(vtkMRMLNode*)", self.updateImageSettingsFromGUI)

    self.ui.fixedModalityComboBox.connect("currentIndexChanged(int)", self.updateImageSettingsFromGUI)
    self.ui.movingModalityComboBox.connect("currentIndexChanged(int)", self.updateImageSettingsFromGUI)

    # Model: each preset file in Resources/Presets is one model choice
    self.ui.modelComboBox.currentTextChanged.connect(self.onPresetSelected)

    # Buttons
    self.ui.runRegistrationButton.connect('clicked(bool)', self.onRunRegistrationButton)
    
    self.ui.progressBar.hide()
    self.ui.time.hide()

    # Remote processing and the local registration server, laid out as in MONAIAuto3DSeg
    settings = qt.QSettings()
    self._localBackendProblem = self.detectLocalBackendProblem()
    self.updateServerUrlGUIFromSettings()
    self.ui.serverComboBox.lineEdit().setPlaceholderText("server:8899")
    self.ui.remoteProcessingCheckBox.checked = (
      settings.value(self.REMOTE_PROCESSING_SETTING) == "true" or self._localBackendProblem is not None)
    self.ui.remoteProcessingCheckBox.toggled.connect(self.onRemoteProcessingCheckBoxToggled)
    self.ui.remoteServerButton.toggled.connect(self.onRemoteServerButtonToggled)
    self.ui.remoteTokenLineEdit.text = self.logic.serverToken(self.ui.serverComboBox.currentText)
    self.ui.serverComboBox.currentTextChanged.connect(self.onServerUrlChanged)
    self.ui.remoteTokenLineEdit.textChanged.connect(self.onServerSettingsChanged)

    self.ui.portSpinBox.value = int(settings.value(self.SERVER_PORT_SETTING) or self.DEFAULT_SERVER_PORT)
    self.ui.portSpinBox.valueChanged.connect(lambda port: qt.QSettings().setValue(self.SERVER_PORT_SETTING, str(port)))
    for checkBox, key in ((self.ui.logConsoleCheckBox, self.LOG_CONSOLE_SETTING),
                          (self.ui.logGuiCheckBox, self.LOG_GUI_SETTING)):
      if settings.value(key) is not None:
        checkBox.checked = settings.value(key) == "true"
      checkBox.toggled.connect(lambda checked, key=key: qt.QSettings().setValue(key, "true" if checked else "false"))
    self.ui.serverButton.toggled.connect(self.onServerButtonToggled)
    self.ui.copyServerAddressButton.clicked.connect(self.copyServerAddress)
    slicer.app.connect("aboutToQuit()", self.stopServer)  # cleanup() does not run on exit

    self.addLog(self._localBackendProblem or self.localReadinessMessage())
    self.updateDeviceChoices()
    self.updateProcessingGUI()

    # Make sure parameter node is initialized (needed for module reload)
    self.initializeParameterNode()
  def nodeEditable(self, node):
    """This module edits its own parameter nodes, also ones created or chosen by other modules."""
    return 0.7 if node and node.GetAttribute("ModuleName") == self.moduleName else 0.0

  def setEditedNode(self, node, role="", context=""):
    if not self.nodeEditable(node):
      return False
    self.setParameterNode(node)
    return True

  def cleanup(self):
    """
    Called when the application closes and the module widget is destroyed.
    """
    if self._remoteRegistration:
      self._remoteRegistration.cancel()
    self.stopServer()
    self.removeObservers()

  #
  # Remote processing and the local registration server
  #

  def detectLocalBackendProblem(self):
    """Why this Slicer cannot run the registration itself, or None.

    A missing PyTorch is no problem: it is installed on the first local registration. One that
    is installed but cannot use this Slicer's NumPy is, and no install fixes it here.
    """
    import importlib.util
    if importlib.util.find_spec("torch") is None:
      return None
    try:
      import torch
      torch.from_numpy(np.zeros(1))
    except Exception as e:
      if "Numpy is not available" in str(e) or "_ARRAY_API" in str(e):
        version = getattr(sys.modules.get("torch"), "__version__", "")
        return (f"PyTorch {version} was built for NumPy 1.x and cannot use NumPy {np.__version__}, "
                "which this Slicer runs on. Use remote processing instead.")
      return f"PyTorch cannot run in this Slicer ({e}). Use remote processing instead."
    return None

  def localReadinessMessage(self):
    """What a registration in this Slicer still needs, found without installing anything."""
    import importlib.util
    from UniGradICONHelpers import registration
    if importlib.util.find_spec("PyTorchUtils") is None:
      return ("To register in this Slicer, install the PyTorch extension from the Extensions Manager "
              "and restart Slicer; or use remote processing.")
    packages = [name for name, module in (("PyTorch", "torch"), ("icon_registration", "icon_registration"))
                if importlib.util.find_spec(module) is None]
    missingWeights = [name for name, (fileName, _) in registration.MODEL_WEIGHTS.items()
                      if not os.path.exists(os.path.join(self.checkpointFolder, fileName))]
    steps = []
    if packages:
      steps.append(f"install {' and '.join(packages)}")
    if missingWeights:
      steps.append(f"download the model weights (about {len(missingWeights) * self.WEIGHTS_FILE_MB} MB)")
    if not steps:
      return "Ready to register in this Slicer."
    return f"The first registration in this Slicer will {' and '.join(steps)}."

  def addLog(self, text):
    """Append text to the log window."""
    statusLog = self.ui.statusLog
    if len(statusLog.plainText) > 1024 * 256:
      statusLog.clear()
      statusLog.insertPlainText("Log cleared\n")
    statusLog.moveCursor(qt.QTextCursor.End)
    statusLog.insertPlainText(text + "\n")
    statusLog.ensureCursorVisible()

  def addServerLog(self, *lines):
    for line in lines:
      if self.ui.logConsoleCheckBox.checked:
        print(line)
      if self.ui.logGuiCheckBox.checked:
        self.addLog(line)

  def localDevices(self):
    if self._localBackendProblem is not None:
      return ["CPU"]
    try:
      import torch
      return ["CPU", "GPU"] if torch.cuda.is_available() else ["CPU"]
    except ImportError:
      return ["CPU"]

  def updateDeviceChoices(self):
    """The devices of where the registration runs: this Slicer, or the connected server.

    While the server's devices are not known (not connected yet), the device stored in the parameter node is kept
    as a choice, so showing a node does not change it.
    """
    storedDevice = (self.logic.deviceName(self._parameterNode.GetParameter(self.logic.DEVICE_PARAM))
                    if self._parameterNode is not None and self._parameterNode.GetParameter(self.logic.DEVICE_PARAM)
                    else None)
    if self.ui.remoteProcessingCheckBox.checked:
      if self._serverInfo:
        devices = list(self._serverInfo["devices"])
      else:
        devices = ["CPU"] + ([storedDevice] if storedDevice and storedDevice != "CPU" else [])
    else:
      devices = self.localDevices()
    comboBox = self.ui.deviceComboBox
    current = comboBox.currentText
    wasBlocked = comboBox.blockSignals(True)
    comboBox.clear()
    comboBox.addItems(devices)
    if storedDevice in devices:
      comboBox.currentText = storedDevice
    else:
      comboBox.currentText = current if current in devices else devices[0]
    comboBox.blockSignals(wasBlocked)
    self.updateParameterNodeFromGUI()

  def updateProcessingGUI(self):
    remote = self.ui.remoteProcessingCheckBox.checked
    localProblem = self._localBackendProblem
    # without a working local backend, remote processing is the only way
    self.ui.remoteProcessingCheckBox.enabled = not (localProblem and remote)
    self.ui.remoteProcessingCheckBox.toolTip = localProblem or "Register on a UniGradICON registration server instead of in this Slicer."
    self.ui.serverConnectionFrame.enabled = remote
    self.ui.remoteTokenLabel.visible = self.ui.remoteTokenLineEdit.visible = remote

    connected = self._serverInfo is not None
    wasBlocked = self.ui.remoteServerButton.blockSignals(True)
    self.ui.remoteServerButton.checked = connected
    self.ui.remoteServerButton.blockSignals(wasBlocked)
    self.ui.remoteServerButton.text = "Connected" if connected else "Connect"

    self.ui.serverCollapsibleButton.enabled = localProblem is None
    self.ui.serverCollapsibleButton.toolTip = localProblem or ""
    serverRunning = self._webServer is not None and self._webServer.isRunning()
    wasBlocked = self.ui.serverButton.blockSignals(True)
    self.ui.serverButton.checked = serverRunning
    self.ui.serverButton.blockSignals(wasBlocked)
    self.ui.serverButton.text = "Running ..." if serverRunning else "Start server"
    self.ui.portSpinBox.enabled = self.ui.serverTokenLineEdit.enabled = not serverRunning
    for widget in (self.ui.serverAddressTitleLabel, self.ui.serverAddressLabel, self.ui.copyServerAddressButton):
      widget.visible = serverRunning
    self.ui.serverAddressLabel.text = self._webServer.address if serverRunning else ""

  def onRemoteProcessingCheckBoxToggled(self, checked):
    qt.QSettings().setValue(self.REMOTE_PROCESSING_SETTING, "true" if checked else "false")
    self.updateParameterNodeFromGUI()
    self.updateDeviceChoices()
    self.updateProcessingGUI()
    if checked:
      self.scheduleReconnect()

  def onServerUrlChanged(self, url):
    # the token remembered for this server, if any
    wasBlocked = self.ui.remoteTokenLineEdit.blockSignals(True)
    self.ui.remoteTokenLineEdit.text = self.logic.serverToken(url.strip())
    self.ui.remoteTokenLineEdit.blockSignals(wasBlocked)
    self.updateParameterNodeFromGUI()
    self.onServerSettingsChanged()

  def onServerSettingsChanged(self, text=None):
    # another server or token: connect again
    if self._serverInfo is not None:
      self._serverInfo = None
      self.updateDeviceChoices()
    self.updateProcessingGUI()

  def serverUrlHistory(self):
    value = qt.QSettings().value(self.SERVER_URL_HISTORY_SETTING)
    return [url for url in (value or "").split(";") if url]

  def saveServerUrl(self, url):
    settings = qt.QSettings()
    settings.setValue(self.SERVER_URL_SETTING, url)
    history = [url] + [other for other in self.serverUrlHistory() if other != url]
    settings.setValue(self.SERVER_URL_HISTORY_SETTING, ";".join(history[:10]))
    self.updateServerUrlGUIFromSettings()

  def updateServerUrlGUIFromSettings(self):
    comboBox = self.ui.serverComboBox
    wasBlocked = comboBox.blockSignals(True)
    comboBox.clear()
    comboBox.addItems(self.serverUrlHistory())
    comboBox.setCurrentText(qt.QSettings().value(self.SERVER_URL_SETTING) or "")
    comboBox.blockSignals(wasBlocked)

  def connectToServer(self, timeout=None):
    from UniGradICONHelpers import remote
    url = self.ui.serverComboBox.currentText.strip()
    if not url:
      raise remote.ServerError("Enter the address of a registration server.")
    token = self.ui.remoteTokenLineEdit.text
    client = remote.RegistrationServerClient(url, token)
    self._serverInfo = client.info(timeout=timeout) if timeout else client.info()
    self._serverUrl = client.url
    self.saveServerUrl(url)
    # remembered on this computer, for registrations run without this panel (never in the scene)
    self.logic.setServerToken(url, token)
    self.addLog(f"Connected to {client.url} (devices: {', '.join(self._serverInfo['devices'])}).")
    self.updateDeviceChoices()
    self.updateProcessingGUI()
    return client

  RECONNECT_TIMEOUT = 5  # seconds; a server that does not answer that fast just shows as not connected

  def scheduleReconnect(self):
    """Check the server again once the GUI is updated (see reconnectToServerQuietly)."""
    qt.QTimer.singleShot(0, self.reconnectToServerQuietly)

  def reconnectToServerQuietly(self):
    """Connect to the server of the shown parameter node again, without dialogs.

    Registrations do not need it, they reach the server by its address, but it shows whether the server answers
    and which devices it offers. Tried once per server and token, until they change or the module is entered again.
    """
    from UniGradICONHelpers import remote
    if (self._parameterNode is None or not self.ui.remoteProcessingCheckBox.checked
        or self._serverInfo is not None or self._remoteRegistration is not None):
      return
    url = self.ui.serverComboBox.currentText.strip()
    attempt = (url, self.ui.remoteTokenLineEdit.text)
    if not url or attempt == self._lastReconnectAttempt:
      return
    self._lastReconnectAttempt = attempt
    try:
      with slicer.util.WaitCursor():
        self.connectToServer(timeout=self.RECONNECT_TIMEOUT)
    except remote.ServerError as e:
      self._serverInfo = None
      self.addLog(f"Not connected to {url}: {e}")
      self.updateProcessingGUI()

  def onRemoteServerButtonToggled(self, checked):
    from UniGradICONHelpers import remote
    if checked:
      try:
        with slicer.util.WaitCursor():
          self.connectToServer()
      except remote.ServerError as e:
        self._serverInfo = None
        self.addLog(f"Connection failed: {e}")
        slicer.util.warningDisplay(f"{e}\n\nPlease check the address, port, token and connection.")
    else:
      self._serverInfo = None
    self.updateDeviceChoices()
    self.updateProcessingGUI()

  def ensureServerPackages(self):
    import importlib.util
    if all(importlib.util.find_spec(name) for name in ("fastapi", "uvicorn")) and (
        importlib.util.find_spec("python_multipart") or importlib.util.find_spec("multipart")):
      return
    if not slicer.util.confirmOkCancelDisplay(
        "Running a registration server needs the Python packages fastapi, uvicorn and python-multipart. "
        "Install them now?"):
      raise RuntimeError("The server packages are not installed.")
    self.addLog("Installing fastapi, uvicorn and python-multipart...")
    slicer.util.pip_install("fastapi uvicorn python-multipart")

  def onServerButtonToggled(self, checked):
    if checked:
      self.startServer()
    else:
      self.stopServer()

  def startServer(self):
    try:
      self.ensureLocalBackend()
      self.ensureServerPackages()
    except Exception as e:
      self.addLog(f"Failed to start the server: {e}")
      slicer.util.errorDisplay(f"Failed to start the registration server: {e}")
      self.updateProcessingGUI()
      return
    token = self.ui.serverTokenLineEdit.text
    self._webServer = RegistrationServerProcess(
      self.ui.portSpinBox.value, token, self.checkpointFolder,
      logCallback=self.addServerLog, stoppedCallback=self.onServerStopped)
    self._webServer.start()
    if token:
      self.addLog(f"Server started at {self._webServer.address}; clients need the token.")
    else:
      self.addLog(f"Server started at {self._webServer.address}. It has no token: anyone who can reach "
                  "this computer can use it.")
    self.updateProcessingGUI()

  def stopServer(self):
    if self._webServer is not None:
      self._webServer.stop()  # reports through onServerStopped

  def onServerStopped(self, returnCode, stoppedByUser):
    self._webServer = None
    if not hasattr(self, "ui"):
      return
    if stoppedByUser:
      self.addLog("Server was stopped.")
    else:
      self.addLog(f"Server exited with code {returnCode}. Enable 'Log to GUI' and start it again for details.")
    self.updateProcessingGUI()

  def copyServerAddress(self):
    """Put the address of the running server on the clipboard, as a client enters it to connect."""
    address = self.ui.serverAddressLabel.text
    qt.QApplication.clipboard().setText(address)
    slicer.util.showStatusMessage(f"Copied {address} to the clipboard.", 3000)

  def enter(self):
    """
    Called each time the user opens this module.
    """
    # Make sure parameter node exists and observed
    self.initializeParameterNode()
    self._lastReconnectAttempt = None  # entering the module checks the server again
    self.scheduleReconnect()

  def exit(self):
    """
    Called each time the user opens a different module.
    """
    # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
    self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

  def onSceneStartClose(self, caller, event):
    """
    Called just before the scene is closed.
    """
    # Parameter node will be reset, do not use it anymore
    self.setParameterNode(None)

  def onSceneEndClose(self, caller, event):
    """
    Called just after the scene is closed.
    """
    # If this module is shown while the scene is closed then recreate a new parameter node immediately
    if self.parent.isEntered:
      self.initializeParameterNode()

  def initializeParameterNode(self):
    """
    Ensure parameter node exists and observed.
    """
    # Parameter node stores all user choices in parameter values, node selections, etc.
    # so that when the scene is saved and reloaded, these settings are restored.
    self.setParameterNode(self.logic.getParameterNode() if not self._parameterNode else self._parameterNode)

  def setParameterNode(self, inputParameterNode):
    """
    Set and observe parameter node.
    Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
    """

    if inputParameterNode:
      self.logic.setDefaultParameters(inputParameterNode)

    # Unobserve previously selected parameter node and add an observer to the newly selected.
    # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
    # those are reflected immediately in the GUI.
    if self._parameterNode is not None:
      self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
    self._parameterNode = inputParameterNode
    if self._parameterNode is not None:
      self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    # Initial GUI update
    self.updateGUIFromParameterNode()

  def updateGUIFromParameterNode(self, caller=None, event=None):
    """
    This method is called whenever parameter node is changed.
    The module GUI is updated to show the current state of the parameter node.
    """

    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return

    # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
    self._updatingGUIFromParameterNode = True

    self.updateImageSettingsGUIFromParameter()

    self.ui.outputTransformComboBox.setCurrentNode(self._parameterNode.GetNodeReference(self.logic.OUTPUT_TRANSFORM_REF))
    self.ui.outputVolumeComboBox.setCurrentNode(self._parameterNode.GetNodeReference(self.logic.OUTPUT_VOLUME_REF))
    self.ui.lossComboBox.currentText = self._parameterNode.GetParameter(self.logic.LOSS_PARAM)
    self.ui.deviceComboBox.currentText = self._parameterNode.GetParameter(self.logic.DEVICE_PARAM)
    
    self.ui.ioSpinBox.value = int(self._parameterNode.GetParameter(self.logic.IO_PARAM))

    # where to register; a Slicer that cannot run it locally always uses a server
    remoteProcessing = self.logic.isRemoteProcessing(self._parameterNode) or self._localBackendProblem is not None
    processingChanged = remoteProcessing != self.ui.remoteProcessingCheckBox.checked
    wasBlocked = self.ui.remoteProcessingCheckBox.blockSignals(True)
    self.ui.remoteProcessingCheckBox.checked = remoteProcessing
    self.ui.remoteProcessingCheckBox.blockSignals(wasBlocked)
    serverUrl = self._parameterNode.GetParameter(self.logic.SERVER_URL_PARAM)
    if serverUrl != self.ui.serverComboBox.currentText:
      for widget in (self.ui.serverComboBox, self.ui.remoteTokenLineEdit):
        widget.blockSignals(True)
      self.ui.serverComboBox.setCurrentText(serverUrl)
      self.ui.remoteTokenLineEdit.text = self.logic.serverToken(serverUrl)
      for widget in (self.ui.serverComboBox, self.ui.remoteTokenLineEdit):
        widget.blockSignals(False)
      self._serverInfo = None
      processingChanged = True

    # the transformed volume is optional: the transform alone can be applied to anything
    self.ui.runRegistrationButton.enabled = bool(self.ui.fixedImageNodeComboBox.currentNodeID and self.ui.movingImageNodeComboBox.currentNodeID and
                                                 self.ui.outputTransformComboBox.currentNodeID)

    # All the GUI updates are done
    self._updatingGUIFromParameterNode = False
    if processingChanged:
      self.updateDeviceChoices()
      self.updateProcessingGUI()
      self.scheduleReconnect()

  def updateImageSettingsGUIFromParameter(self):
    """Modalities from the stored settings (StagesJson, named after its SlicerANTs origin)."""
    presetParameters = json.loads(self._parameterNode.GetParameter(self.logic.SETTINGS_JSON_PARAM))
    # the model of the shown node, without applying its preset again
    wasBlocked = self.ui.modelComboBox.blockSignals(True)
    self.ui.modelComboBox.currentText = presetParameters['modelSettings']['model']
    self.ui.modelComboBox.blockSignals(wasBlocked)
    fixedModality, movingModality = self.logic.modalityNames(
      presetParameters['image']['modality-fixed'], presetParameters['image']['modality-moving'])
    self.ui.fixedModalityComboBox.currentText = fixedModality
    self.ui.movingModalityComboBox.currentText = movingModality
    self.logic.changeModelSettings(presetParameters)

  def updateParameterNodeFromGUI(self, caller=None, event=None):
    """
    This method is called when the user makes any change in the GUI.
    The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
    """

    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return

    wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch
    
    self._parameterNode.SetNodeReferenceID(self.logic.OUTPUT_TRANSFORM_REF, self.ui.outputTransformComboBox.currentNodeID)
    self._parameterNode.SetNodeReferenceID(self.logic.OUTPUT_VOLUME_REF, self.ui.outputVolumeComboBox.currentNodeID)
    self._parameterNode.SetParameter(self.logic.LOSS_PARAM, self.ui.lossComboBox.currentText)
    self._parameterNode.SetParameter(self.logic.IO_PARAM, str(self.ui.ioSpinBox.value))
    self._parameterNode.SetParameter(self.logic.DEVICE_PARAM, self.ui.deviceComboBox.currentText)
    self._parameterNode.SetParameter(self.logic.REMOTE_PROCESSING_PARAM,
                                     "true" if self.ui.remoteProcessingCheckBox.checked else "false")
    self._parameterNode.SetParameter(self.logic.SERVER_URL_PARAM, self.ui.serverComboBox.currentText.strip())

    self._parameterNode.EndModify(wasModified)

  def updateImageSettingsFromGUI(self):
    """Store the selected images and their modalities in the settings (StagesJson)."""
    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return
    presetParameters = json.loads(self._parameterNode.GetParameter(self.logic.SETTINGS_JSON_PARAM))
    presetParameters['image']['fixed'] = self.ui.fixedImageNodeComboBox.currentNodeID
    presetParameters['image']['moving'] = self.ui.movingImageNodeComboBox.currentNodeID
    
    presetParameters['image']['modality-fixed'] = self.ui.fixedModalityComboBox.currentText
    presetParameters['image']['modality-moving'] = self.ui.movingModalityComboBox.currentText
  
    self._parameterNode.SetParameter(self.logic.SETTINGS_JSON_PARAM, json.dumps(presetParameters))

  def updateProgress(self, allProgress, currentProgress, elapsedTime=None, remainingTime=None):
    self.ui.progressBar.maximum = allProgress
    self.ui.progressBar.value = currentProgress-1
    self.ui.progressBar.show()
    
    if elapsedTime is not None and remainingTime is not None:
      self.ui.time.text = f"Elapsed Time: {elapsedTime:.2f}s Remaining Time: {remainingTime:.2f}s"
      self.ui.time.show()
    else:
      self.ui.time.hide()
    if allProgress == currentProgress:
      self.ui.progressBar.hide()
      
  def onPresetSelected(self, presetName):
    if presetName == 'Select...' or self._parameterNode is None or self._updatingGUIFromParameterNode:
      return
    wasModified = self._parameterNode.StartModify()  # Modify in a single batch
    presetParameters = PresetManager().getPresetParametersByName(presetName)
    presetParameters['image']['fixed'] = self.ui.fixedImageNodeComboBox.currentNodeID
    presetParameters['image']['moving'] = self.ui.movingImageNodeComboBox.currentNodeID
    presetParameters['image']['modality-fixed'] = self.ui.fixedModalityComboBox.currentText
    presetParameters['image']['modality-moving'] = self.ui.movingModalityComboBox.currentText
      
    self._parameterNode.SetParameter(self.logic.SETTINGS_JSON_PARAM, json.dumps(presetParameters))
    self._parameterNode.EndModify(wasModified)
    
    self.logic.changeModelSettings(presetParameters)

  def onRunRegistrationButton(self):
    if self._remoteRegistration:
      self.addLog("Cancelling the registration on the server...")
      self._remoteRegistration.cancel()
      return

    import time
    parameters = self.logic.createProcessParameters(self._parameterNode)
    if self.ui.remoteProcessingCheckBox.checked:
      self.startRemoteRegistration(parameters)
      return

    try:
      self.ensureLocalBackend()
    except Exception as e:
      self.addLog(f"UniGradICON cannot run in this Slicer: {e}")
      slicer.util.errorDisplay(f"UniGradICON cannot run in this Slicer: {e}\n\n"
                               "You can use remote processing instead.")
      return
    self.addLog("Registering in this Slicer...")
    start = time.time()
    self.logic.process(**parameters, call_back=self.updateProgress)
    self.addLog(f"Registration completed in {time.time() - start:.0f} s.")

    #show message box
    slicer.util.infoDisplay("Registration is completed!")

  def startRemoteRegistration(self, parameters):
    import time
    from UniGradICONHelpers import remote
    try:
      with slicer.util.WaitCursor():
        client = self.connectToServer() if self._serverInfo is None else remote.RegistrationServerClient(
          self.ui.serverComboBox.currentText, self.ui.remoteTokenLineEdit.text)
        self.addLog(f"Uploading the images to {client.url}...")
        slicer.app.processEvents()
        registration = RemoteRegistration(self.logic, client, parameters,
                                          self.onRemoteStatus, self.onRemoteFinished)
        registration.start()
    except remote.ServerError as e:
      self.addLog(f"Registration on the server failed: {e}")
      slicer.util.errorDisplay(str(e))
      self.updateProcessingGUI()
      return
    self._remoteRegistration = registration
    self._remoteState = None
    self._registrationStart = time.time()
    self.addLog("Images uploaded.")
    self.ui.runRegistrationButton.text = "Cancel"

  def onRemoteStatus(self, status):
    state = (status["state"], status.get("queuePosition"))
    if state != self._remoteState:  # log changes, not every poll
      self._remoteState = state
      if status["state"] == "queued":
        self.addLog(f"Waiting for the server ({status.get('queuePosition', 0)} registrations ahead)...")
      elif status["state"] == "running":
        self.addLog("Registering on the server...")
    progress = status.get("progress")
    if progress and progress.get("total"):
      self.updateProgress(progress["total"], progress["step"], progress.get("elapsed"), progress.get("remaining"))

  def onRemoteFinished(self, error, cancelled):
    import time
    self._remoteRegistration = None
    self.ui.runRegistrationButton.text = "Run Registration"
    self.ui.progressBar.hide()
    self.ui.time.hide()
    if error:
      self.addLog(f"Registration on the server failed: {error}")
      slicer.util.errorDisplay(f"Registration on the server failed: {error}")
    elif cancelled:
      self.addLog("Registration cancelled.")
    else:
      self.addLog(f"Registration completed in {time.time() - self._registrationStart:.0f} s.")
      slicer.util.infoDisplay("Registration is completed!")
    
#
# unigradiconLogic
#
class UniGradICONLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  OUTPUT_TRANSFORM_REF = "OutputTransform"
  OUTPUT_VOLUME_REF = "OutputVolume"
  # LOSS_PARAM, IO_PARAM and DEVICE_PARAM are the parameter names (not values) stored in scenes;
  # kept as they are for compatibility with existing scenes and older versions
  LOSS_PARAM = "LNCC"
  SETTINGS_JSON_PARAM = "StagesJson"  # stored name from SlicerANTs; kept for scene compatibility
  STAGES_JSON_PARAM = SETTINGS_JSON_PARAM  # old constant name, for scripts
  IO_PARAM = "0"
  DEVICE_PARAM = "CPU"
  REMOTE_PROCESSING_PARAM = "RemoteProcessing"
  SERVER_URL_PARAM = "ServerUrl"

  def __init__(self, weights_location=None):
    """
    Called when the logic class is instantiated. Can be used for initializing member variables.
    """
    ScriptedLoadableModuleLogic.__init__(self)
    # the repository's model_checkpoints folder, as the module widget uses, when created without
    # one (e.g. by another module)
    self.weights_location = weights_location or os.path.join(
      os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_checkpoints") + os.sep
    # the network is loaded on the first local registration, so the module works without
    # PyTorch when it registers on a server
    self.modelName = "unigradicon"
    self.loss = "LNCC"
    self._localBackendReady = False
    self.cancelRequested = False

  def ensureLocalBackend(self):
    """Download the weights and install PyTorch and icon_registration, for running in this Slicer.

    Not needed when registering on a server, so it runs on the first local registration
    rather than when the module is opened.
    """
    if self._localBackendReady:
      return

    import SampleData
    import shutil

    if not os.path.exists(self.weights_location):
      os.makedirs(self.weights_location)
        
    if not os.path.exists(self.weights_location + "unigradicon_weights.pth"):
      slicer.progressWindow = slicer.util.createProgressDialog()
      self.sampleDataLogic = SampleData.SampleDataLogic()
      self.sampleDataLogic.logMessage = self.reportProgress
              
      weights_location = self.sampleDataLogic.downloadFileIntoCache("https://github.com/uncbiag/uniGradICON/releases/download/unigradicon_weights/Step_2_final.trch", "unigradicon_weights.trch")
      if self.sampleDataLogic.downloadPercent and self.sampleDataLogic.downloadPercent == 100:
        shutil.copyfile(weights_location, self.weights_location + "unigradicon_weights.pth")
        slicer.progressWindow.close()
    
    if not os.path.exists(self.weights_location + "multigradicon_weights.pth"):
      slicer.progressWindow = slicer.util.createProgressDialog()
      self.sampleDataLogic = SampleData.SampleDataLogic()
      self.sampleDataLogic.logMessage = self.reportProgress
      weights_location = self.sampleDataLogic.downloadFileIntoCache("https://github.com/uncbiag/uniGradICON/releases/download/multigradicon_weights/Step_2_final.trch", "Step_2_final.trch")
      if self.sampleDataLogic.downloadPercent and self.sampleDataLogic.downloadPercent == 100:
        shutil.copyfile(weights_location, self.weights_location + "multigradicon_weights.pth")
        slicer.progressWindow.close()
    
    # Install PyTorch
    try:
      import PyTorchUtils
    except ModuleNotFoundError as e:
      raise RuntimeError("This module requires PyTorch extension. Install it from the Extensions Manager.")

    minimumTorchVersion = "1.12"
    torchLogic = PyTorchUtils.PyTorchUtilsLogic()
    if not torchLogic.torchInstalled():
        print('PyTorch Python package is required. Installing... (it may take several minutes)')
        torch = torchLogic.installTorch(askConfirmation=True, torchVersionRequirement = f">={minimumTorchVersion}")
        if torch is None:
            raise RuntimeError("This module requires PyTorch extension. Install it from the Extensions Manager.")
    else:
        # torch is installed, check version
        from packaging import version
        if version.parse(torchLogic.torch.__version__) < version.parse(minimumTorchVersion):
            raise RuntimeError(f'PyTorch version {torchLogic.torch.__version__} is not compatible with this module.'
                              + f' Minimum required version is {minimumTorchVersion}. You can use "PyTorch Util" module to install PyTorch'
                              + f' with version requirement set to: >={minimumTorchVersion}')
    _checkPyTorchNumPyCompatibility()
    import torch
    try:
      import icon_registration
    except ModuleNotFoundError:
      if slicer.util.confirmOkCancelDisplay("'icon_registration' is missing. Click OK to install it."): 
        slicer.util.pip_install("icon_registration")
    try:
      # the registration code imports these where it uses them; here they only check the installation
      import icon_registration  # noqa: F401
      import itk  # noqa: F401
      import UniGradICONHelpers.icon_helper  # noqa: F401
    except ModuleNotFoundError:
      raise RuntimeError("There is a problem about the installation of 'icon' package. Please try again to install!")
    self._localBackendReady = True
    
  def reportProgress(self, msg, level=None):
    if slicer.progressWindow.wasCanceled:
        raise Exception("Download aborted")
    slicer.progressWindow.show()
    slicer.progressWindow.activateWindow()
    slicer.progressWindow.setValue(int(self.sampleDataLogic.downloadPercent))
    slicer.progressWindow.setLabelText("Downloading checkpoints...")
    slicer.app.processEvents()

  def setDefaultParameters(self, parameterNode):
    """
    Initialize parameter node with default settings.
    """
    presetParameters = PresetManager().getPresetParametersByName()
    if not parameterNode.GetParameter(self.SETTINGS_JSON_PARAM):
      parameterNode.SetParameter(self.SETTINGS_JSON_PARAM, json.dumps(presetParameters))
    if not parameterNode.GetNodeReference(self.OUTPUT_TRANSFORM_REF):
      parameterNode.SetNodeReferenceID(self.OUTPUT_TRANSFORM_REF, "")
    if not parameterNode.GetNodeReference(self.OUTPUT_VOLUME_REF):
      parameterNode.SetNodeReferenceID(self.OUTPUT_VOLUME_REF, "")
    if not parameterNode.GetParameter(self.LOSS_PARAM):
      parameterNode.SetParameter(self.LOSS_PARAM, str(presetParameters["modelSettings"]["loss"]))
    if not parameterNode.GetParameter(self.IO_PARAM):
      parameterNode.SetParameter(self.IO_PARAM, str(presetParameters["modelSettings"]["io_steps"]))
    if not parameterNode.GetParameter(self.DEVICE_PARAM):
      parameterNode.SetParameter(self.DEVICE_PARAM, self.deviceName(presetParameters["modelSettings"]["device"]))
    # where to register: the last choice made in the module is the default for a new node
    settings = qt.QSettings()
    if not parameterNode.GetParameter(self.REMOTE_PROCESSING_PARAM):
      parameterNode.SetParameter(self.REMOTE_PROCESSING_PARAM,
                                 "true" if settings.value(REMOTE_PROCESSING_SETTING) == "true" else "false")
    if not parameterNode.GetParameter(self.SERVER_URL_PARAM):
      parameterNode.SetParameter(self.SERVER_URL_PARAM, settings.value(SERVER_URL_SETTING) or "")

  def isRemoteProcessing(self, parameterNode):
    return parameterNode.GetParameter(self.REMOTE_PROCESSING_PARAM) == "true"

  @staticmethod
  def _serverTokenKey(url):
    from urllib.parse import quote
    from UniGradICONHelpers import remote
    # one address however it was typed, and no slashes, which QSettings reads as groups
    return f"{SERVER_TOKENS_SETTING_GROUP}/{quote(remote.RegistrationServerClient(url).url, safe='')}"

  @classmethod
  def serverToken(cls, url):
    """The token remembered for a server: in this computer's Slicer settings, never in a scene."""
    if not url:
      return ""
    return qt.QSettings().value(cls._serverTokenKey(url)) or ""

  @classmethod
  def setServerToken(cls, url, token):
    if not url:
      return
    if token:
      qt.QSettings().setValue(cls._serverTokenKey(url), token)
    else:
      qt.QSettings().remove(cls._serverTokenKey(url))

  def registerVolumesUsingParameterNode(self, parameterNode, fixedVolumeNode, movingVolumeNode, outputTransformNode,
                                        fixedMaskNode=None, initialTransformNode=None, logCallback=None):
    """Register two volumes with the settings of a parameter node, in this Slicer or on its server.

    For callers without the module's GUI, such as other modules. Writes the transform
    that resamples the moving volume onto the fixed one into outputTransformNode, and blocks until
    done. A fixedMaskNode (labelmap) limits the fixed image to its nonzero region.
    """
    import time
    from UniGradICONHelpers import remote
    log = logCallback or logging.info
    self.cancelRequested = False
    parameters = self.createProcessParameters(parameterNode)
    parameters['outputSettings']['transform'] = outputTransformNode
    parameters['outputSettings']['volume'] = None
    temporaryNodes = []
    try:
      movingInput = movingVolumeNode
      if initialTransformNode is not None:
        movingInput = slicer.modules.volumes.logic().CloneVolume(
          slicer.mrmlScene, movingVolumeNode, movingVolumeNode.GetName() + "_initialized")
        temporaryNodes.append(movingInput)
        movingInput.SetAndObserveTransformNodeID(initialTransformNode.GetID())
        slicer.vtkSlicerTransformLogic().hardenTransform(movingInput)
      parameters['image']['fixed'] = fixedVolumeNode
      parameters['image']['moving'] = movingInput

      start = time.time()
      if self.isRemoteProcessing(parameterNode):
        url = parameterNode.GetParameter(self.SERVER_URL_PARAM)
        if not url:
          raise ValueError("Remote processing is selected, but no server address is set.")
        client = remote.RegistrationServerClient(url, self.serverToken(url))
        log(f"Registering {movingVolumeNode.GetName()} to {fixedVolumeNode.GetName()} on {client.url}...")
        lastState = []

        def logStatus(status):
          if status["state"] not in lastState:
            lastState.append(status["state"])
            log(f"  {status['state']} on the server")

        registration = RemoteRegistration(self, client, parameters, statusCallback=logStatus, fixedMask=fixedMaskNode)
        registration.start()
        error, cancelled = registration.waitForCompletion(lambda: self.cancelRequested)
        if cancelled:
          raise ValueError("User requested cancel.")
        if error:
          raise RuntimeError(f"Registration on {client.url} failed: {error}")
      else:
        log(f"Registering {movingVolumeNode.GetName()} to {fixedVolumeNode.GetName()} in this Slicer...")
        self.ensureLocalBackend()

        def logProgress(total, step, elapsed=None, remaining=None):
          if step:
            log(f"  instance optimization step {step}/{total}")

        try:
          self.process(**parameters, call_back=logProgress, fixedMask=fixedMaskNode)
        except Exception as e:
          from UniGradICONHelpers import registration
          if isinstance(e, registration.RegistrationCancelled):
            raise ValueError("User requested cancel.")
          raise

      if initialTransformNode is not None:
        # the result maps the initialized moving volume: put the initial transform under it
        initialCopy = slicer.mrmlScene.AddNewNodeByClass(initialTransformNode.GetClassName())
        temporaryNodes.append(initialCopy)
        initialCopy.CopyContent(initialTransformNode)
        initialCopy.SetAndObserveTransformNodeID(outputTransformNode.GetID())
        slicer.vtkSlicerTransformLogic().hardenTransform(initialCopy)
        outputTransformNode.SetAndObserveTransformToParent(initialCopy.GetTransformToParent())
      log(f"  done in {time.time() - start:.0f} s")
    finally:
      for node in temporaryNodes:
        slicer.mrmlScene.RemoveNode(node)
      
  def createProcessParameters(self, paramNode):
    parameters = json.loads(paramNode.GetParameter(self.SETTINGS_JSON_PARAM))

    parameters['outputSettings'] = {}
    parameters['outputSettings']['transform'] = paramNode.GetNodeReference(self.OUTPUT_TRANSFORM_REF)
    parameters['outputSettings']['volume'] = paramNode.GetNodeReference(self.OUTPUT_VOLUME_REF)
    parameters['outputSettings']['loss'] = paramNode.GetParameter(self.LOSS_PARAM)

    parameters['generalSettings'] = {}
    parameters['generalSettings']['io_steps'] = int(paramNode.GetParameter(self.IO_PARAM))
    parameters['generalSettings']['device'] = str(paramNode.GetParameter(self.DEVICE_PARAM))
    
    return parameters

  def itk2sitk(self, itk_image):
    import itk
    sitkImage = sitk.GetImageFromArray(itk.GetArrayFromImage(itk_image))

    sitkImage.SetOrigin(tuple(itk_image.GetOrigin()))
    sitkImage.SetSpacing(tuple(itk_image.GetSpacing()))
    sitkImage.SetDirection(itk.GetArrayFromMatrix(itk_image.GetDirection()).flatten()) 
    
    return sitkImage
  
  def sitk2itk(self, sitk_image):
    import itk
    itk_image = itk.GetImageFromArray(sitk.GetArrayFromImage(sitk_image))
    image_dimension = 3
    
    itk_image.SetOrigin(sitk_image.GetOrigin())
    itk_image.SetSpacing(sitk_image.GetSpacing())
    itk_image.SetDirection(itk.GetMatrixFromArray(np.reshape(np.array(sitk_image.GetDirection()), [image_dimension]*2)))
    
    return itk_image
  
  def changeModelSettings(self, presetParameters):
    self.modelName = presetParameters['modelSettings']['model']
    self.loss = presetParameters['modelSettings']['loss']

  # the modalities the preset files carry; they never matched the module's lists, which stayed at MRI
  PRESET_MODALITIES = ("mri", "ct")

  @staticmethod
  def modalityName(modality):
    """MRI or CT/CBCT for a modality in any spelling the module or uniGradICON uses, None if unknown."""
    name = str(modality).strip().upper()
    if name in ("MRI", "MR"):
      return "MRI"
    if name in ("CT", "CBCT", "CT/CBCT"):
      return "CT/CBCT"
    return None

  @classmethod
  def modalityNames(cls, fixedModality, movingModality):
    """The fixed and moving modality, as the module and uniGradICON's preprocessing name them.

    Settings still holding the presets' untouched "mri"/"ct" run as MRI/MRI, as they always did: the
    module's modality lists did not have those names and stayed at MRI. Unknown names count as MRI.
    """
    if (fixedModality, movingModality) == cls.PRESET_MODALITIES:
      return "MRI", "MRI"
    names = []
    for modality in (fixedModality, movingModality):
      name = cls.modalityName(modality)
      if name is None:
        logging.warning(f"Unknown modality {modality!r}; registering it as MRI.")
        name = "MRI"
      names.append(name)
    return tuple(names)

  @staticmethod
  def deviceName(device):
    """CPU or GPU, as the module and the server name them; the presets say "cpu"."""
    return "GPU" if str(device).strip().upper() in ("GPU", "CUDA") else "CPU"

  def registrationSettings(self, image, outputSettings, modelSettings=None, generalSettings=None):
    """What a registration runs with, the same for this Slicer and a server."""
    fixedModality, movingModality = self.modalityNames(image['modality-fixed'], image['modality-moving'])
    return {
      "fixed_modality": fixedModality,
      "moving_modality": movingModality,
      "model": (modelSettings or {}).get('model') or self.modelName,
      "loss": outputSettings.get('loss') or self.loss,
      "io_steps": int(generalSettings["io_steps"]),
      "device": self.deviceName(generalSettings['device']),
      # without an output volume there is no need to resample the moving image
      "warp": outputSettings.get('volume') is not None,
    }

  def loadResults(self, outputSettings, transformPath, warpedImage=None):
    """Put a registration result into the output transform node, and volume node if there is one."""
    import sitkUtils
    transformNode = outputSettings['transform']
    storageNode = transformNode.CreateDefaultStorageNode()
    storageNode.SetFileName(transformPath)
    storageNode.ReadData(transformNode)

    outputVolumeNode = outputSettings.get('volume')
    if outputVolumeNode is None or warpedImage is None:
      return
    outputVolumeNode.CreateDefaultDisplayNodes()
    sitkUtils.PushVolumeToSlicer(warpedImage, outputVolumeNode)

  @staticmethod
  def resampleMovingVolume(movingVolume, fixedVolume, transformNode, outputVolume):
    """Resample the moving volume onto the fixed volume's grid through the registration transform.

    Done in Slicer (BRAINSResample, linear) for registrations on a server, so only the transform has
    to be downloaded; it matches the server's or a local ITK resampling to rounding.
    """
    if isinstance(movingVolume, str):
      movingVolume = slicer.util.getNode(movingVolume)
    if isinstance(fixedVolume, str):
      fixedVolume = slicer.util.getNode(fixedVolume)
    outputVolume.CreateDefaultDisplayNodes()
    parameters = {
      "inputVolume": movingVolume.GetID(),
      "referenceVolume": fixedVolume.GetID(),
      "outputVolume": outputVolume.GetID(),
      "warpTransform": transformNode.GetID(),
      "interpolationMode": "Linear",
      "pixelType": "float",
    }
    cliNode = slicer.cli.runSync(slicer.modules.brainsresample, None, parameters)
    try:
      if cliNode.GetStatus() & cliNode.ErrorsMask:
        raise RuntimeError(f"Resampling the moving volume failed: {cliNode.GetErrorText()}")
    finally:
      slicer.mrmlScene.RemoveNode(cliNode)

  def process(self, image, outputSettings, modelSettings=None, generalSettings=None, wait_for_completion=False, call_back=None,
              fixedMask=None):
    """Register in this Slicer, with the dictionaries createProcessParameters makes.

    :param image: fixed and moving volume nodes (or IDs) and their modalities ('modality-fixed', 'modality-moving')
    :param outputSettings: output transform node, optional output volume node, and the similarity loss
    :param modelSettings: the model ('unigradicon' or 'multigradicon'), as in the preset files
    :param generalSettings: instance optimization steps ('io_steps') and device ('CPU' or 'GPU')
    :param wait_for_completion: unused; the registration always runs to completion
    :param call_back: progress callback(total, step[, elapsed, remaining])
    :param fixedMask: optional labelmap that limits the fixed image to its nonzero region
    """
    
    import itk
    import sitkUtils
    from UniGradICONHelpers import registration

    settings = self.registrationSettings(image, outputSettings, modelSettings, generalSettings)
    model = registration.load_model(self.weights_location, settings["model"], settings["loss"], settings["device"])

    #convert to itk by preserving metadata
    fixed = self.sitk2itk(sitkUtils.PullVolumeFromSlicer(image['fixed']))
    moving = self.sitk2itk(sitkUtils.PullVolumeFromSlicer(image['moving']))
    fixed_mask = self.sitk2itk(sitkUtils.PullVolumeFromSlicer(fixedMask)) if fixedMask is not None else None

    def progress(*args):
      # a cancel request stops the registration at its next instance optimization step
      if self.cancelRequested:
        raise registration.RegistrationCancelled()
      if call_back is not None:
        call_back(*args)

    io_steps = settings["io_steps"]
    if io_steps != 0 and call_back is not None:
      call_back(io_steps, 0)
    phi_AB, warped_moving_image = registration.register_images(
      model, fixed, moving, settings["fixed_modality"], settings["moving_modality"], io_steps, progress,
      warp=settings["warp"], fixed_mask=fixed_mask)
    if io_steps != 0 and call_back is not None:
      call_back(io_steps, io_steps)

    transformPath = f'{slicer.app.temporaryPath}/transform.tfm'
    itk.transformwrite([phi_AB], transformPath)
    self.loadResults(outputSettings, transformPath,
                     self.itk2sitk(warped_moving_image) if warped_moving_image is not None else None)
  
  
class RegistrationServerProcess:
  """A registration server (UniGradICONServer/server.py) run with this Slicer's Python.

  Its output is read on a thread, so a full pipe never stalls the server, and handed to
  ``logCallback`` on the main thread; ``stoppedCallback(returnCode, stoppedByUser)`` reports its end.
  """

  OUTPUT_CHECK_INTERVAL_MS = 200

  def __init__(self, port, token=None, weightsDir=None, host="0.0.0.0", logCallback=None, stoppedCallback=None):
    import socket
    self.port = port
    self.token = token
    self.weightsDir = weightsDir
    self.host = host
    self.logCallback = logCallback
    self.stoppedCallback = stoppedCallback
    self.address = f"{socket.gethostname() if host == '0.0.0.0' else host}:{port}"
    self.process = None
    self._timer = None
    self._stopping = False

  def start(self):
    import queue
    import threading
    script = os.path.join(os.path.dirname(__file__), "UniGradICONServer", "server.py")
    arguments = ["--host", self.host, "--port", str(self.port)]
    if self.weightsDir:
      arguments += ["--weights-dir", self.weightsDir]
    # sys.executable is the PythonSlicer launcher, which runs the actual Python as its child:
    # stopping the launcher would leave that running, so the server stops itself once the
    # launcher or this Slicer is gone
    arguments += ["--exit-with-parent", "--parent-pid", str(os.getpid())]
    environment = {"UNIGRADICON_SERVER_TOKEN": self.token} if self.token else None
    self.process = slicer.util.launchConsoleProcess([sys.executable, script] + arguments,
                                                    useStartupEnvironment=False, updateEnvironment=environment)
    self._output = queue.Queue()
    threading.Thread(target=self._readOutput, daemon=True).start()
    self._timer = qt.QTimer()
    self._timer.setInterval(self.OUTPUT_CHECK_INTERVAL_MS)
    self._timer.timeout.connect(self._checkProcess)
    self._timer.start()

  def isRunning(self):
    return self.process is not None and self.process.poll() is None

  def _readOutput(self):
    for line in self.process.stdout:
      self._output.put(line.rstrip())

  def _drainOutput(self):
    import queue
    lines = []
    while True:
      try:
        lines.append(self._output.get_nowait())
      except queue.Empty:
        break
    if lines and self.logCallback:
      self.logCallback(*lines)

  def _checkProcess(self):
    self._drainOutput()
    if not self.isRunning():
      self._finish()

  def _finish(self):
    if self._timer is None:
      return
    self._timer.stop()
    self._timer = None
    self._drainOutput()
    if self.stoppedCallback:
      self.stoppedCallback(self.process.returncode, self._stopping)

  def stop(self):
    self._stopping = True
    if self.isRunning():
      self.process.terminate()
      try:
        self.process.wait(5)
      except Exception:
        self.process.kill()
        self.process.wait()
    self._finish()


class RemoteRegistration:
  """A registration on a server: uploads the inputs, polls the job and loads the results."""

  POLL_INTERVAL_MS = 1000
  MAX_POLL_FAILURES = 10  # a busy server may miss a status request now and then

  def __init__(self, logic, client, parameters, statusCallback=None, finishedCallback=None, fixedMask=None):
    self.logic = logic
    self.fixedMask = fixedMask
    self.error = None
    self.cancelled = False
    self.client = client
    self.parameters = parameters
    self.statusCallback = statusCallback
    self.finishedCallback = finishedCallback  # (error or None, cancelled)
    self.jobId = None
    self.finished = False
    self.pollFailures = 0
    self.workDir = None
    self.timer = qt.QTimer()
    self.timer.setInterval(self.POLL_INTERVAL_MS)
    self.timer.timeout.connect(self.poll)

  def start(self):
    import tempfile
    import sitkUtils
    self.workDir = tempfile.mkdtemp(prefix="UniGradICON-", dir=slicer.app.temporaryPath)
    paths = {}
    for role in ("fixed", "moving"):
      paths[role] = os.path.join(self.workDir, f"{role}.nrrd")
      sitk.WriteImage(sitkUtils.PullVolumeFromSlicer(self.parameters["image"][role]), paths[role], True)
    maskPath = None
    if self.fixedMask is not None:
      maskPath = os.path.join(self.workDir, "fixed_mask.nrrd")
      sitk.WriteImage(sitkUtils.PullVolumeFromSlicer(self.fixedMask), maskPath, True)
    settings = self.logic.registrationSettings(
      self.parameters["image"], self.parameters["outputSettings"],
      self.parameters.get("modelSettings"), self.parameters["generalSettings"])
    settings["warp"] = False  # the transformed volume is made here, from the transform: less to download
    self.jobId = self.client.submit(paths["fixed"], paths["moving"], settings, maskPath)["id"]
    self.timer.start()

  def poll(self):
    from UniGradICONHelpers import remote
    try:
      try:
        status = self.client.status(self.jobId)
        self.pollFailures = 0
      except remote.ServerError:
        self.pollFailures += 1
        if self.pollFailures < self.MAX_POLL_FAILURES:
          return
        raise
      if status["state"] == "done":
        self.timer.stop()
        transformPath = self.client.download(self.jobId, "transform", os.path.join(self.workDir, "transform.tfm"))
        self.client.cancel(self.jobId)  # done with it: the server can remove its files
        outputSettings = self.parameters["outputSettings"]
        self.logic.loadResults(outputSettings, transformPath)
        if outputSettings.get("volume") is not None:
          self.logic.resampleMovingVolume(self.parameters["image"]["moving"], self.parameters["image"]["fixed"],
                                          outputSettings["transform"], outputSettings["volume"])
        self._finish(None)
      elif status["state"] == "failed":
        self._finish(status.get("error") or "unknown error")
      elif status["state"] == "cancelled":
        self._finish(None, cancelled=True)
      elif self.statusCallback:
        self.statusCallback(status)
    except remote.ServerError as e:
      self._finish(str(e))
    except Exception as e:
      logging.exception("Loading the registration result failed")
      self._finish(f"{type(e).__name__}: {e}")

  def cancel(self):
    from UniGradICONHelpers import remote
    if self.finished:
      return
    if self.jobId:
      try:
        self.client.cancel(self.jobId)
      except remote.ServerError:
        pass  # gone already, or the server is unreachable: nothing more to do
    self._finish(None, cancelled=True)

  def waitForCompletion(self, cancelRequested=None):
    """Block until the job is finished, keeping Slicer responsive; returns (error, cancelled)."""
    import time
    while not self.finished:
      if cancelRequested is not None and cancelRequested():
        self.cancel()
        break
      slicer.app.processEvents()  # runs the polling timer
      time.sleep(0.05)
    return self.error, self.cancelled

  def _finish(self, error, cancelled=False):
    import shutil
    self.timer.stop()
    self.finished = True
    self.error = error
    self.cancelled = cancelled
    if self.workDir:
      shutil.rmtree(self.workDir, ignore_errors=True)
    if self.finishedCallback:
      self.finishedCallback(error, cancelled)


class PresetManager:
  """The model choices: one JSON file per model in Resources/Presets, with its default settings."""

  def __init__(self):
      self.presetPath = os.path.join(os.path.dirname(__file__), 'Resources', 'Presets')

  def getPresetParametersByName(self, name='unigradicon'):
    presetFilePath = os.path.join(self.presetPath, name + '.json')
    with open(presetFilePath) as presetFile:
      return json.load(presetFile)

  def getPresetNames(self):
    G = glob.glob(os.path.join(self.presetPath, '*.json'))
    return [os.path.splitext(os.path.basename(g))[0] for g in G]

#
# unigradiconTest
#
class UniGradICONTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  # the registration tests take minutes and need PyTorch (and, for the remote test, the server
  # packages), so they run only when asked for
  FULL_TESTS_ENVIRONMENT_VARIABLE = "UNIGRADICON_FULL_TESTS"

  @classmethod
  def fullTestsEnabled(cls):
    return os.environ.get(cls.FULL_TESTS_ENVIRONMENT_VARIABLE, "").lower() in ("1", "true", "yes", "on")

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear()

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_UniGradICON()
    self.setUp()
    self.test_remote()
    self.setUp()
    self.test_parameterNodeRegistration()

  def test_parameterNodeRegistration(self):
    """Register with a parameter node and a fixed mask, in this Slicer and on a server, as plugins do."""
    if self.fullTestsEnabled():
      self.runParameterNodeRegistrationTest()
    else:
      logging.warning('Parameter node registration test was skipped. Enable it by starting Slicer with '
                      f'{self.FULL_TESTS_ENVIRONMENT_VARIABLE}=1.')

  def runParameterNodeRegistrationTest(self, port=8897):
    import time
    import vtk
    import SampleData
    from UniGradICONHelpers import remote

    self.delayDisplay("Registering with a parameter node and a fixed mask")
    sampleDataLogic = SampleData.SampleDataLogic()
    fixed = sampleDataLogic.downloadMRBrainTumor1()
    moving = sampleDataLogic.downloadMRBrainTumor2()
    mask = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode', 'HalfMask')
    maskArray = np.zeros(slicer.util.arrayFromVolume(fixed).shape, dtype=np.uint8)
    maskArray[:, :, : maskArray.shape[2] // 2] = 1
    slicer.util.updateVolumeFromArray(mask, maskArray)
    ijkToRas = vtk.vtkMatrix4x4()
    fixed.GetIJKToRASMatrix(ijkToRas)
    mask.SetIJKToRASMatrix(ijkToRas)

    logic = UniGradICONLogic()
    logic.isSingletonParameterNode = False
    token = "test-token"
    url = f"127.0.0.1:{port}"
    weightsFolder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_checkpoints")
    server = RegistrationServerProcess(port, token=token, weightsDir=weightsFolder, host="127.0.0.1")
    server.start()
    try:
      results = {}
      for location in ("local", "remote"):
        parameterNode = logic.createParameterNode()
        slicer.mrmlScene.AddNode(parameterNode)
        logic.setDefaultParameters(parameterNode)
        stages = json.loads(parameterNode.GetParameter(logic.SETTINGS_JSON_PARAM))
        stages["image"]["modality-fixed"] = stages["image"]["modality-moving"] = "MRI"
        parameterNode.SetParameter(logic.SETTINGS_JSON_PARAM, json.dumps(stages))
        parameterNode.SetParameter(logic.REMOTE_PROCESSING_PARAM, "true" if location == "remote" else "false")
        parameterNode.SetParameter(logic.SERVER_URL_PARAM, url)
        if location == "remote":
          UniGradICONLogic.setServerToken(url, token)
          deadline = time.time() + 120
          while True:
            try:
              remote.RegistrationServerClient(url, token).info()
              break
            except remote.ServerError:
              self.assertLess(time.time(), deadline, "the registration server did not start")
              slicer.app.processEvents()
              time.sleep(0.5)
        outputTransform = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLTransformNode', f"{location}Transform")
        logic.registerVolumesUsingParameterNode(parameterNode, fixed, moving, outputTransform, fixedMaskNode=mask)
        self.assertIsNotNone(outputTransform.GetTransformToParent())
        results[location] = outputTransform
        # the token stays in this computer's settings, out of the scene
        self.assertFalse(any(token in parameterNode.GetParameter(name) for name in parameterNode.GetParameterNames()))

      points = [[10.0, -20.0, 5.0], [-30.0, 15.0, 20.0]]
      for point in points:
        local = results["local"].GetTransformToParent().TransformPoint(point)
        remoteResult = results["remote"].GetTransformToParent().TransformPoint(point)
        self.assertLess(max(abs(a - b) for a, b in zip(local, remoteResult)), 1e-3)
      self.delayDisplay('Parameter node registration test passed!')
    finally:
      UniGradICONLogic.setServerToken(url, "")
      server.stop()

  def test_remote(self):
    """Register through a registration server started on this computer."""
    # Needs PyTorch and the server packages (fastapi, uvicorn, python-multipart), and takes
    # a few minutes on a CPU: disabled by default, like the logic tests below.
    testRemote = self.fullTestsEnabled()
    if testRemote:
      self.runRemoteRegistrationTest()
    else:
      logging.warning('Remote registration test was skipped. Enable it by starting Slicer with '
                      f'{self.FULL_TESTS_ENVIRONMENT_VARIABLE}=1.')

  def runRemoteRegistrationTest(self, port=8898):
    import time
    import SampleData
    from UniGradICONHelpers import remote

    self.delayDisplay("Registering on a registration server on this computer")
    sampleDataLogic = SampleData.SampleDataLogic()
    fixed = sampleDataLogic.downloadMRBrainTumor1()
    moving = sampleDataLogic.downloadMRBrainTumor2()
    outputVolume = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode')
    outputTransform = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLTransformNode')
    weightsFolder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_checkpoints")

    def logServer(*lines):
      for line in lines:
        logging.info("UniGradICON server: " + line)
    server = RegistrationServerProcess(port, weightsDir=weightsFolder, host="127.0.0.1", logCallback=logServer)
    server.start()
    process = server.process
    try:
      client = remote.RegistrationServerClient(f"http://127.0.0.1:{port}")
      deadline = time.time() + 120
      while True:
        try:
          client.info()
          break
        except remote.ServerError:
          if time.time() > deadline or process.poll() is not None:
            raise
          time.sleep(0.5)

      parameters = {
        "image": {"fixed": fixed, "moving": moving, "modality-fixed": "MRI", "modality-moving": "MRI"},
        "modelSettings": {"model": "unigradicon"},
        "outputSettings": {"transform": outputTransform, "volume": outputVolume, "loss": "LNCC"},
        "generalSettings": {"io_steps": 0, "device": "CPU"},
      }
      results = []
      registration = RemoteRegistration(UniGradICONLogic(weightsFolder), client, parameters,
                                        finishedCallback=lambda error, cancelled: results.append((error, cancelled)))
      registration.start()
      deadline = time.time() + 900
      while not results:
        self.assertLess(time.time(), deadline, "the remote registration did not finish")
        slicer.app.processEvents()
        time.sleep(0.1)

      self.assertEqual(results[0], (None, False))
      self.assertEqual(slicer.util.arrayFromVolume(outputVolume).shape, slicer.util.arrayFromVolume(fixed).shape)
      self.assertIsNotNone(outputTransform.GetTransformToParent())
      self.delayDisplay('Remote registration test passed!')
    finally:
      server.stop()
    
    
  def test_wo_IO(self):
    parameters = {}

    parameters['image'] = {}
    parameters['image']['fixed'] = self.fixed
    parameters['image']['moving'] = self.moving
    
    parameters['image']['modality-fixed'] = 'MRI'
    parameters['image']['modality-moving'] = 'MRI'
    
    parameters['outputSettings'] = {}
    parameters['outputSettings']['transform'] = self.outputTransform
    parameters['outputSettings']['volume'] = self.outputVolume
    parameters['outputSettings']['loss'] = 'LNCC'
    
    parameters['generalSettings'] = {}
    parameters['generalSettings']['io_steps'] = int(0)
    parameters['generalSettings']['device'] = str('CPU')
    
    return parameters
  
  def test_w_LNCC(self):
    parameters = {}

    parameters['image'] = {}
    parameters['image']['fixed'] = self.fixed
    parameters['image']['moving'] = self.moving
    
    parameters['image']['modality-fixed'] = 'MRI'
    parameters['image']['modality-moving'] = 'MRI'
    
    parameters['outputSettings'] = {}
    parameters['outputSettings']['transform'] = self.outputTransform
    parameters['outputSettings']['volume'] = self.outputVolume
    parameters['outputSettings']['loss'] = 'LNCC'
    
    parameters['generalSettings'] = {}
    parameters['generalSettings']['io_steps'] = int(1)
    parameters['generalSettings']['device'] = str('CPU')
    
    return parameters
  
  def test_w_SquaredLNCC(self):
    parameters = {}

    parameters['image'] = {}
    parameters['image']['fixed'] = self.fixed
    parameters['image']['moving'] = self.moving
    
    parameters['image']['modality-fixed'] = 'MRI'
    parameters['image']['modality-moving'] = 'MRI'
    
    parameters['outputSettings'] = {}
    parameters['outputSettings']['transform'] = self.outputTransform
    parameters['outputSettings']['volume'] = self.outputVolume
    parameters['outputSettings']['loss'] = 'Squared LNCC'
    
    parameters['generalSettings'] = {}
    parameters['generalSettings']['io_steps'] = int(1)
    parameters['generalSettings']['device'] = str('CPU')
    
    return parameters
  
  def test_w_MINDSSC(self):
    parameters = {}

    parameters['image'] = {}
    parameters['image']['fixed'] = self.fixed
    parameters['image']['moving'] = self.moving
    
    parameters['image']['modality-fixed'] = 'MRI'
    parameters['image']['modality-moving'] = 'MRI'
    
    parameters['outputSettings'] = {}
    parameters['outputSettings']['transform'] = self.outputTransform
    parameters['outputSettings']['volume'] = self.outputVolume
    parameters['outputSettings']['loss'] = 'MIND-SSC'
    
    parameters['generalSettings'] = {}
    parameters['generalSettings']['io_steps'] = int(1)
    parameters['generalSettings']['device'] = str('CPU')
    
    return parameters
  
  def test_GPU(self):
    parameters = {}

    parameters['image'] = {}
    parameters['image']['fixed'] = self.fixed
    parameters['image']['moving'] = self.moving
    
    parameters['image']['modality-fixed'] = 'MRI'
    parameters['image']['modality-moving'] = 'MRI'
    
    parameters['outputSettings'] = {}
    parameters['outputSettings']['transform'] = self.outputTransform
    parameters['outputSettings']['volume'] = self.outputVolume
    parameters['outputSettings']['loss'] = 'LNCC'
    
    parameters['generalSettings'] = {}
    parameters['generalSettings']['io_steps'] = 0
    parameters['generalSettings']['device'] = str('GPU')
    
    return parameters
  
  def test_UniGradICON(self):
    """ Ideally you should have several levels of tests.  At the lowest level
    tests should exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

    self.delayDisplay("Starting the test")

    import SampleData
    sampleDataLogic = SampleData.SampleDataLogic()
    self.fixed = sampleDataLogic.downloadMRBrainTumor1()
    self.moving = sampleDataLogic.downloadMRBrainTumor2()

    self.outputVolume = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode')
    self.outputTransform = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLTransformNode')
    
    self.sampleDataLogic = SampleData.SampleDataLogic()
    weights_location = self.sampleDataLogic.downloadFileIntoCache("https://github.com/uncbiag/uniGradICON/releases/download/unigradicon_weights/Step_2_final.trch", "unigradicon_weights.pth")
    
    logic = UniGradICONLogic(weights_location.split("unigradicon_weights.pth")[0])
    
    # Logic testing is disabled by default to not overload automatic build machines.
    testLogic = self.fullTestsEnabled()
    
    if testLogic:
      parameters = self.test_wo_IO()
      logic.process(**parameters)
      print("w/o IO test passed!")
      
      parameters = self.test_w_LNCC()
      logic.process(**parameters)
      print("w/ LNCC test passed!")
      
      parameters = self.test_w_SquaredLNCC()
      logic.process(**parameters)
      print("w/ SquaredLNCC test passed!")
      
      parameters = self.test_w_MINDSSC()
      logic.process(**parameters)
      print("w/ MINDSSC test passed!")
      
      import torch
      if torch.cuda.is_available():
        parameters = self.test_GPU()
        logic.process(**parameters)
        print("w/ GPU test passed!")
      
      self.delayDisplay('Test passed!')
    else:
      logging.warning('Tests was skipped. Enable it by starting Slicer with '
                      f'{self.FULL_TESTS_ENVIRONMENT_VARIABLE}=1.')