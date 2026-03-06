#define MyAppName "MPS"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Neumaier Lab"
#define MyAppExeName "launcher.cmd"

[Setup]
AppId={{A3D9D4F3-6A9C-4B11-9F2C-EXAMPLE000001}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}

; Installs under Program Files (OK, because we will write env/logs to LocalAppData instead)
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}

OutputBaseFilename={#MyAppName}_Setup_{#MyAppVersion}
Compression=lzma
SolidCompression=yes
WizardStyle=modern

; Installer icon + Add/Remove Programs icon
SetupIconFile=icons\neumaierlabdesign.ico
UninstallDisplayIcon={app}\icons\neumaierlabdesign.ico

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &Desktop icon"; GroupDescription: "Additional icons:"; Flags: checkedonce

[Files]
; Copy everything EXCEPT Python bytecode
Source: "*"; DestDir: "{app}"; \
    Flags: ignoreversion recursesubdirs createallsubdirs; \
    Excludes: "*\__pycache__\*;__pycache__\*;*.pyc;*.pyo"

[Icons]
; Use cmd.exe explicitly for maximum reliability with .cmd
Name: "{group}\{#MyAppName}"; \
  Filename: "{sys}\cmd.exe"; \
  Parameters: "/c ""{app}\{#MyAppExeName}"""; \
  WorkingDir: "{app}"; \
  IconFilename: "{app}\icons\neumaierlabdesign.ico"

Name: "{autodesktop}\{#MyAppName}"; \
  Filename: "{sys}\cmd.exe"; \
  Parameters: "/c ""{app}\{#MyAppExeName}"""; \
  Tasks: desktopicon; \
  WorkingDir: "{app}"; \
  IconFilename: "{app}\icons\neumaierlabdesign.ico"

[Run]
; Offer to launch after install
Filename: "{sys}\cmd.exe"; \
  Parameters: "/c ""{app}\{#MyAppExeName}"""; \
  Description: "Launch {#MyAppName}"; \
  WorkingDir: "{app}"; \
  Flags: nowait postinstall skipifsilent