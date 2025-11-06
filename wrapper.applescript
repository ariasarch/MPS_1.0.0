on run
  -- Absolute path to the .app bundle on disk
  set appPath to POSIX path of (path to me)
  -- Payload lives inside the bundle here:
  set payloadPath to appPath & "Contents/Resources/payload"

  -- Sanity check (no { } grouping, no tricky quotes)
  set checkCmd to "if [ ! -d " & quoted form of payloadPath & " ]; then echo 'Payload missing:' " & quoted form of payloadPath & " ; exit 66; fi"
  do shell script checkCmd

  -- User-writable target (keeps signature intact)
  set targetPath to (POSIX path of (path to home folder)) & "Library/Application Support/MPS"
  do shell script "/bin/mkdir -p " & quoted form of targetPath

  -- Mirror code, but keep user env/logs between versions
  set rsyncCmd to "/usr/bin/rsync -aE --delete --exclude env/ --exclude logs/ " & quoted form of (payloadPath & "/") & " " & quoted form of (targetPath & "/")
  do shell script rsyncCmd

  -- Ensure launcher is executable
  do shell script "/bin/chmod +x " & quoted form of (targetPath & "/launcher.command")

  -- Launch WITHOUT Apple Events: open Terminal to run the .command file
  do shell script "/usr/bin/open -a Terminal " & quoted form of (targetPath & "/launcher.command")
end run
