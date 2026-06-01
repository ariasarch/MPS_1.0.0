#define MyAppName "MPS"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Neumaier Lab"
#define MyAppExeName "launcher.cmd"

[Setup]
AppId={{014D43F6-96A5-4665-93FD-DE1EDA61E20E}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}

; --- Per-user install, no elevation ------------------------------------------
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}

; --- Force 64-bit install location -------------------------------------------
; Without this, a 32-bit installer process resolves {autopf} to
; "Program Files (x86)" on 64-bit Windows. Lock it to 64-bit.
; (If using Inno Setup < 6.3, replace "x64compatible" with "x64".)
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

OutputBaseFilename={#MyAppName}_Setup_{#MyAppVersion}
Compression=lzma
SolidCompression=yes
WizardStyle=modern

SetupIconFile=icons\neumaierlabdesign.ico
UninstallDisplayIcon={app}\icons\neumaierlabdesign.ico

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &Desktop icon"; GroupDescription: "Additional icons:"; Flags: checkedonce

[Files]
; Copy everything EXCEPT Python bytecode and any runtime state that the
; launcher creates on first run (env/, logs/, conda_path.txt). Also exclude
; the .iss itself in case it's sitting in the source tree.
Source: "*"; DestDir: "{app}"; \
    Flags: ignoreversion recursesubdirs createallsubdirs; \
    Excludes: "*\__pycache__\*;__pycache__\*;*.pyc;*.pyo;env\*;logs\*;conda_path.txt;*.iss"

[Icons]
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
Filename: "{sys}\cmd.exe"; \
  Parameters: "/c ""{app}\{#MyAppExeName}"""; \
  Description: "Launch {#MyAppName}"; \
  WorkingDir: "{app}"; \
  Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\env"
Type: filesandordirs; Name: "{app}\logs"
Type: files;          Name: "{app}\conda_path.txt"
