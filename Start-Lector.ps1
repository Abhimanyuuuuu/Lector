[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Test-TcpPort {
  param(
    [Parameter(Mandatory = $true)][string]$HostName,
    [Parameter(Mandatory = $true)][int]$Port
  )

  $client = [System.Net.Sockets.TcpClient]::new()
  try {
    $connect = $client.BeginConnect($HostName, $Port, $null, $null)
    if (-not $connect.AsyncWaitHandle.WaitOne(500)) { return $false }
    $client.EndConnect($connect)
    return $true
  }
  catch {
    return $false
  }
  finally {
    $client.Dispose()
  }
}

function Test-OllamaReady {
  try {
    $response = Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/version' -TimeoutSec 3
    return -not [string]::IsNullOrWhiteSpace([string]$response.version)
  }
  catch {
    return $false
  }
}

function Test-LectorBackendReady {
  try {
    $response = Invoke-RestMethod -Uri 'http://127.0.0.1:7130/api/v1/health' -TimeoutSec 3
    return $response.status -eq 'ready' -and $response.service -eq 'lector-rag-api'
  }
  catch {
    return $false
  }
}

function Test-LectorFrontendReady {
  try {
    $response = Invoke-WebRequest -Uri 'http://localhost:7127' -TimeoutSec 4 -UseBasicParsing
    if ($response.StatusCode -lt 200 -or $response.StatusCode -ge 300) { return $false }
    # Identify the expected Next.js reader without tying readiness to title text.
    return $response.Content -match '__NEXT_DATA__'
  }
  catch {
    return $false
  }
}

function Wait-ForService {
  param(
    [Parameter(Mandatory = $true)][scriptblock]$Probe,
    [Parameter(Mandatory = $true)][int]$TimeoutSeconds,
    [Parameter(Mandatory = $true)][string]$ServiceName
  )

  $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
  while ([DateTime]::UtcNow -lt $deadline) {
    if (& $Probe) { return $true }
    Start-Sleep -Seconds 2
  }
  throw "$ServiceName did not become ready within $TimeoutSeconds seconds. Its terminal remains open for logs."
}

function ConvertTo-PowerShellLiteral {
  param([Parameter(Mandatory = $true)][string]$Value)
  return "'" + $Value.Replace("'", "''") + "'"
}

function Start-LectorTerminal {
  param(
    [Parameter(Mandatory = $true)][string]$Title,
    [Parameter(Mandatory = $true)][string]$Command,
    [switch]$PassGroqKey
  )

  $terminalScript = "`$Host.UI.RawUI.WindowTitle = '$Title'`r`n$Command"
  $encodedCommand = [Convert]::ToBase64String(
    [Text.Encoding]::Unicode.GetBytes($terminalScript)
  )
  $previousGroqKey = [Environment]::GetEnvironmentVariable('GROQ_API_KEY', 'Process')
  if (-not $PassGroqKey) {
    [Environment]::SetEnvironmentVariable('GROQ_API_KEY', $null, 'Process')
  }
  try {
    Start-Process -FilePath $script:WindowsPowerShell -ArgumentList @(
      '-NoProfile', '-NoExit', '-EncodedCommand', $encodedCommand
    ) | Out-Null
  }
  finally {
    if (-not $PassGroqKey) {
      [Environment]::SetEnvironmentVariable('GROQ_API_KEY', $previousGroqKey, 'Process')
    }
  }
}

function Set-BackendEnvironment {
  param([Parameter(Mandatory = $true)][string]$ApiKey)

  $values = @{
    LECTOR_OLLAMA_BASE_URL = 'http://127.0.0.1:11434'
    LECTOR_EMBEDDING_MODEL = 'nomic-embed-text:latest'
    LECTOR_EMBEDDING_DIMENSIONS = '768'
    LECTOR_STORY_EXTRACTION_MODEL = 'qwen2.5:3b-instruct-q4_K_M'
    GROQ_MODEL = 'openai/gpt-oss-120b'
    LECTOR_CORS_ORIGINS = 'http://localhost:7127,http://127.0.0.1:7127'
    GROQ_API_KEY = $ApiKey
  }

  $previous = @{}
  foreach ($name in $values.Keys) {
    $previous[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
    [Environment]::SetEnvironmentVariable($name, $values[$name], 'Process')
  }
  return $previous
}

function Restore-BackendEnvironment {
  param([Parameter(Mandatory = $true)][hashtable]$Previous)
  foreach ($name in $Previous.Keys) {
    [Environment]::SetEnvironmentVariable($name, $Previous[$name], 'Process')
  }
}

function ConvertTo-WindowsCommandLineArgument {
  param([Parameter(Mandatory = $true)][AllowEmptyString()][string]$Value)

  if ($Value.Length -gt 0 -and $Value -notmatch '[\s"]') { return $Value }

  $builder = [System.Text.StringBuilder]::new()
  [void]$builder.Append('"')
  $slashes = 0
  foreach ($character in $Value.ToCharArray()) {
    if ($character -eq '\') {
      $slashes++
      continue
    }
    if ($character -eq '"') {
      [void]$builder.Append(('\' * (2 * $slashes + 1)))
      [void]$builder.Append('"')
      $slashes = 0
      continue
    }
    if ($slashes -gt 0) {
      [void]$builder.Append(('\' * $slashes))
      $slashes = 0
    }
    [void]$builder.Append($character)
  }
  if ($slashes -gt 0) { [void]$builder.Append(('\' * (2 * $slashes))) }
  [void]$builder.Append('"')
  return $builder.ToString()
}

function New-EncodedPowerShellCommand {
  param([Parameter(Mandatory = $true)][string]$ScriptText)
  return [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($ScriptText))
}

function Start-LectorWindowsTerminalTabs {
  param(
    [Parameter(Mandatory = $true)][string]$WindowsTerminalPath,
    [Parameter(Mandatory = $true)][object[]]$Tabs
  )

  $arguments = [System.Collections.Generic.List[string]]::new()
  $arguments.Add('-w')
  $arguments.Add('new')
  $first = $true
  foreach ($tab in $Tabs) {
    if (-not $first) { $arguments.Add(';') }
    $arguments.Add('new-tab')
    $arguments.Add('--title')
    $arguments.Add([string]$tab.Title)
    $arguments.Add('--suppressApplicationTitle')
    $arguments.Add('-d')
    $arguments.Add([string]$tab.Directory)
    $arguments.Add([string]$tab.PowerShell)
    $arguments.Add('-NoProfile')
    $arguments.Add('-NoExit')
    $arguments.Add('-EncodedCommand')
    $arguments.Add([string]$tab.EncodedCommand)
    $first = $false
  }

  $argumentLine = (($arguments | ForEach-Object {
    ConvertTo-WindowsCommandLineArgument -Value $_
  }) -join ' ')
  Start-Process -FilePath $WindowsTerminalPath -ArgumentList $argumentLine -ErrorAction Stop | Out-Null
}

try {
  $projectRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
  $backendDir = Join-Path $projectRoot 'apps\rag-api'
  $readerDir = Join-Path $projectRoot 'apps\reader'
  $pythonExe = Join-Path $backendDir '.venv\Scripts\python.exe'

  if (-not (Test-Path -LiteralPath $backendDir -PathType Container)) {
    throw "Backend directory was not found: $backendDir"
  }
  if (-not (Test-Path -LiteralPath $readerDir -PathType Container)) {
    throw "Reader directory was not found: $readerDir"
  }
  if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
    throw "Backend Python environment was not found: $pythonExe"
  }

  $ollamaCommand = Get-Command 'ollama' -ErrorAction SilentlyContinue
  if (-not $ollamaCommand) { throw 'Ollama was not found on PATH. Install it before starting Lector.' }
  $pnpmCommand = Get-Command 'pnpm.cmd' -ErrorAction SilentlyContinue
  if (-not $pnpmCommand) { throw 'pnpm.cmd was not found on PATH. Install/configure pnpm before starting Lector.' }
  $script:WindowsPowerShell = (Get-Command 'powershell.exe' -ErrorAction Stop).Source

  $secureKey = Read-Host -Prompt 'Groq API key (input is hidden)' -AsSecureString
  $keyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
  try {
    $apiKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPointer)
  }
  finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPointer)
  }
  if ([string]::IsNullOrWhiteSpace($apiKey)) {
    $apiKey = $null
    throw 'A Groq API key is required. Run Start-Lector.ps1 again and enter the key when prompted.'
  }

  $ollamaLiteral = ConvertTo-PowerShellLiteral -Value $ollamaCommand.Source
  $backendLiteral = ConvertTo-PowerShellLiteral -Value $backendDir
  $readerLiteral = ConvertTo-PowerShellLiteral -Value $readerDir
  $pythonLiteral = ConvertTo-PowerShellLiteral -Value $pythonExe
  $pnpmLiteral = ConvertTo-PowerShellLiteral -Value $pnpmCommand.Source
  $wtCommand = Get-Command 'wt.exe' -ErrorAction SilentlyContinue

  # Inspect every port before launching anything, so a partial startup never
  # masks a conflict on a later service.
  $ollamaHealthy = Test-OllamaReady
  $backendHealthy = Test-LectorBackendReady
  $frontendHealthy = Test-LectorFrontendReady
  if (-not $ollamaHealthy -and (Test-TcpPort -HostName '127.0.0.1' -Port 11434)) {
    throw 'Port 11434 is occupied by a service that did not pass Ollama health checks. No process was stopped.'
  }
  if (-not $backendHealthy -and (Test-TcpPort -HostName '127.0.0.1' -Port 7130)) {
    throw 'Port 7130 is occupied by a service that did not pass Lector health checks. No process was stopped.'
  }
  if (-not $frontendHealthy -and (Test-TcpPort -HostName '127.0.0.1' -Port 7127)) {
    throw 'Port 7127 is occupied by a service that did not pass Lector reader checks. No process was stopped.'
  }

  $ollamaScript = if ($ollamaHealthy) {
    "Write-Host 'Ollama is already healthy at 127.0.0.1:11434; reusing it.'"
  } else {
    "Set-Location -LiteralPath $backendLiteral; & $ollamaLiteral serve"
  }
  $backendPipeName = $null
  if ($backendHealthy) {
    $backendScript = "Write-Host 'A healthy Lector API is already running at 127.0.0.1:7130; reusing it.'; Write-Host 'The key entered in this launcher cannot change an already-running process.'"
    Write-Warning 'The API is already healthy; the entered Groq key will not be applied to its existing process.'
  } elseif ($wtCommand) {
    # The key travels through a short-lived local named pipe, never through the
    # WT command line or its shared environment inherited by the other tabs.
    $backendPipeName = 'LectorGroq_' + [Guid]::NewGuid().ToString('N')
    $backendScript = @"
`$pipe = [System.IO.Pipes.NamedPipeClientStream]::new('.', '$backendPipeName', [System.IO.Pipes.PipeDirection]::In)
try {
  `$pipe.Connect(30000)
  `$reader = [System.IO.BinaryReader]::new(`$pipe, [Text.Encoding]::UTF8, `$true)
  `$env:GROQ_API_KEY = `$reader.ReadString()
  `$reader.Dispose()
} finally {
  `$pipe.Dispose()
}
`$env:LECTOR_OLLAMA_BASE_URL = 'http://127.0.0.1:11434'
`$env:LECTOR_EMBEDDING_MODEL = 'nomic-embed-text:latest'
`$env:LECTOR_EMBEDDING_DIMENSIONS = '768'
`$env:LECTOR_STORY_EXTRACTION_MODEL = 'qwen2.5:3b-instruct-q4_K_M'
`$env:GROQ_MODEL = 'openai/gpt-oss-120b'
`$env:LECTOR_CORS_ORIGINS = 'http://localhost:7127,http://127.0.0.1:7127'
Set-Location -LiteralPath $backendLiteral
& $pythonLiteral -m uvicorn lector_rag_api.main:create_app --factory --port 7130
"@
  } else {
    $backendScript = "Write-Host 'Waiting for the launcher to pass the Groq key to this backend terminal.'; `$env:LECTOR_OLLAMA_BASE_URL = 'http://127.0.0.1:11434'; `$env:LECTOR_EMBEDDING_MODEL = 'nomic-embed-text:latest'; `$env:LECTOR_EMBEDDING_DIMENSIONS = '768'; `$env:LECTOR_STORY_EXTRACTION_MODEL = 'qwen2.5:3b-instruct-q4_K_M'; `$env:GROQ_MODEL = 'openai/gpt-oss-120b'; `$env:LECTOR_CORS_ORIGINS = 'http://localhost:7127,http://127.0.0.1:7127'; Set-Location -LiteralPath $backendLiteral; & $pythonLiteral -m uvicorn lector_rag_api.main:create_app --factory --port 7130"
  }
  $frontendScript = if ($frontendHealthy) {
    "Write-Host 'A healthy Lector reader is already running at http://localhost:7127; reusing it.'"
  } else {
    "Set-Location -LiteralPath $readerLiteral; & $pnpmLiteral dev"
  }

  if ($wtCommand) {
    Write-Host 'Opening one Windows Terminal window with Ollama, Lector Backend, and Lector Frontend tabs.'
    $pipeServer = $null
    try {
      $tabs = @(
        [pscustomobject]@{ Title = 'Ollama'; Directory = $backendDir; PowerShell = $script:WindowsPowerShell; EncodedCommand = (New-EncodedPowerShellCommand $ollamaScript) },
        [pscustomobject]@{ Title = 'Lector Backend'; Directory = $backendDir; PowerShell = $script:WindowsPowerShell; EncodedCommand = (New-EncodedPowerShellCommand $backendScript) },
        [pscustomobject]@{ Title = 'Lector Frontend'; Directory = $readerDir; PowerShell = $script:WindowsPowerShell; EncodedCommand = (New-EncodedPowerShellCommand $frontendScript) }
      )
      if ($backendPipeName) {
        $pipeServer = [System.IO.Pipes.NamedPipeServerStream]::new(
          $backendPipeName,
          [System.IO.Pipes.PipeDirection]::Out,
          1,
          [System.IO.Pipes.PipeTransmissionMode]::Byte,
          [System.IO.Pipes.PipeOptions]::Asynchronous
        )
        $connectionTask = $pipeServer.WaitForConnectionAsync()
      }
      Start-LectorWindowsTerminalTabs -WindowsTerminalPath $wtCommand.Source -Tabs $tabs
      if ($backendPipeName) {
        if (-not $connectionTask.Wait(30000)) {
          throw 'The backend tab did not connect to the secure Groq-key handoff pipe.'
        }
        $keyWriter = [System.IO.BinaryWriter]::new($pipeServer, [Text.Encoding]::UTF8, $true)
        try {
          $keyWriter.Write($apiKey)
          $keyWriter.Flush()
        } finally {
          $keyWriter.Dispose()
        }
      }
    } finally {
      if ($pipeServer) { $pipeServer.Dispose() }
      $apiKey = $null
      $secureKey = $null
    }
  } else {
    Write-Warning 'Windows Terminal (wt.exe) is unavailable. Opening three separate visible PowerShell windows instead.'
    Start-LectorTerminal -Title 'Ollama' -Command $ollamaScript
    if ($backendHealthy) {
      Start-LectorTerminal -Title 'Lector Backend' -Command $backendScript
      $apiKey = $null
    } else {
      $previousEnvironment = Set-BackendEnvironment -ApiKey $apiKey
      try {
        Start-LectorTerminal -Title 'Lector Backend' -Command "Set-Location -LiteralPath $backendLiteral; & $pythonLiteral -m uvicorn lector_rag_api.main:create_app --factory --port 7130" -PassGroqKey
      } finally {
        Restore-BackendEnvironment -Previous $previousEnvironment
        $apiKey = $null
      }
    }
    Start-LectorTerminal -Title 'Lector Frontend' -Command $frontendScript
    $secureKey = $null
  }

  if ($ollamaHealthy) { Write-Host 'Ollama is already healthy; reusing it.' }
  if ($backendHealthy) { Write-Host 'Lector Backend is already healthy; reusing it.' }
  if ($frontendHealthy) { Write-Host 'Lector Frontend is already healthy; reusing it.' }

  # All new service tabs have been started before any readiness wait begins.
  Wait-ForService -Probe { Test-OllamaReady } -TimeoutSeconds 60 -ServiceName 'Ollama' | Out-Null
  Wait-ForService -Probe { Test-LectorBackendReady } -TimeoutSeconds 120 -ServiceName 'Lector Backend' | Out-Null
  Wait-ForService -Probe { Test-LectorFrontendReady } -TimeoutSeconds 240 -ServiceName 'Lector Frontend' | Out-Null

  try {
    Start-Process -FilePath 'http://localhost:7127' -ErrorAction Stop | Out-Null
    Write-Host 'All services are ready; opened Lector at http://localhost:7127.'
  } catch {
    Write-Warning "All services are ready, but automatic browser launch failed: $($_.Exception.Message) Open http://localhost:7127 manually."
  }
}
catch {
  [Console]::Error.WriteLine("Lector startup failed: {0}", $_.Exception.Message)
  Read-Host 'Press Enter to close the launcher'
  exit 1
}
