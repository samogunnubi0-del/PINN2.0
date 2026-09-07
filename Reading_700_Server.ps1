param([switch]$NoBrowser)

$ErrorActionPreference = 'Stop'
$pagePath = Join-Path $PSScriptRoot 'Reading_700_Mastery.html'
$keyPath = Join-Path $PSScriptRoot '.reading_700_key'
$yashiKeyPath = Join-Path $PSScriptRoot '.for_yashi_key'
$url = 'http://127.0.0.1:4174/Reading_700_Mastery.html'
$model = 'deepseek/deepseek-v4-flash-0731'
$port = 4174
$listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $port)

$systemPrompt = @'
You are Sam's elite Digital SAT Reading & Writing study coach. His goal is 700+ in Reading & Writing (Math baseline is 800 locked, targeting 1500–1540 composite) on the Digital SAT on Saturday, October 3, 2026.

Write with precision, intellectual rigor, and direct diagnostic clarity.
- Standard English Conventions: treat grammar as mathematical boolean logic (Independent vs. Dependent clauses, colon rules, subject-verb agreement).
- Command of Evidence: emphasize strict textual anchors and zero outside assumptions.
- Quantitative Evidence: translate graph axes and trend lines before considering options.
- Transitions: categorize into Additive, Contrastive, or Causative.
- Rhetorical Synthesis: match the student's exact specified presentation goal.

Return only valid JSON matching the requested schema with no markdown fences or outside commentary.
'@

function Get-PrivateKey {
  if ($env:OPENROUTER_API_KEY) { return $env:OPENROUTER_API_KEY.Trim() }
  if (Test-Path -LiteralPath $keyPath) {
    try {
      $encrypted = (Get-Content -LiteralPath $keyPath -Raw).Trim()
      $secure = ConvertTo-SecureString $encrypted
      $credential = [System.Management.Automation.PSCredential]::new('OpenRouter', $secure)
      return $credential.GetNetworkCredential().Password
    } catch {}
  }
  if (Test-Path -LiteralPath $yashiKeyPath) {
    try {
      $encrypted = (Get-Content -LiteralPath $yashiKeyPath -Raw).Trim()
      $secure = ConvertTo-SecureString $encrypted
      $credential = [System.Management.Automation.PSCredential]::new('OpenRouter', $secure)
      return $credential.GetNetworkCredential().Password
    } catch {}
  }
  return $null
}

function Get-KeyHealth {
  $apiKey = Get-PrivateKey
  if ([string]::IsNullOrWhiteSpace($apiKey)) { return 'missing' }
  try {
    $headers = @{ Authorization = "Bearer $apiKey" }
    $keyInfo = Invoke-RestMethod -Method Get -Uri 'https://openrouter.ai/api/v1/key' -Headers $headers -TimeoutSec 15
    $remaining = $keyInfo.data.limit_remaining
    if ($null -ne $remaining -and [double]$remaining -le 0) { return 'credits' }
    return 'ready'
  } catch {
    $status = 0
    try { $status = [int]$_.Exception.Response.StatusCode } catch {}
    if ($status -eq 401) { return 'rejected' }
    return 'unavailable'
  }
}

function Write-Response {
  param(
    [System.Net.Sockets.NetworkStream]$Stream,
    [int]$Status,
    [string]$Reason,
    [string]$ContentType,
    [byte[]]$Body
  )
  $header = "HTTP/1.1 $Status $Reason`r`nContent-Type: $ContentType`r`nContent-Length: $($Body.Length)`r`nCache-Control: no-store`r`nReferrer-Policy: strict-origin-when-cross-origin`r`nX-Content-Type-Options: nosniff`r`nConnection: close`r`n`r`n"
  $headerBytes = [System.Text.Encoding]::ASCII.GetBytes($header)
  $Stream.Write($headerBytes, 0, $headerBytes.Length)
  $Stream.Write($Body, 0, $Body.Length)
  $Stream.Flush()
}

function Write-JsonResponse {
  param(
    [System.Net.Sockets.NetworkStream]$Stream,
    [int]$Status,
    [string]$Reason,
    [object]$Data
  )
  $json = $Data | ConvertTo-Json -Depth 12 -Compress
  $bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
  Write-Response -Stream $Stream -Status $Status -Reason $Reason -ContentType 'application/json; charset=utf-8' -Body $bytes
}

function Read-Request {
  param([System.Net.Sockets.NetworkStream]$Stream)
  $headerBytes = [System.Collections.Generic.List[byte]]::new()
  $matchState = 0
  while ($headerBytes.Count -lt 65536) {
    $value = $Stream.ReadByte()
    if ($value -lt 0) { throw 'Connection closed before headers arrived.' }
    $byte = [byte]$value
    $headerBytes.Add($byte)
    if (($matchState -eq 0 -or $matchState -eq 2) -and $byte -eq 13) { $matchState++ }
    elseif (($matchState -eq 1 -or $matchState -eq 3) -and $byte -eq 10) { $matchState++ }
    else { $matchState = $(if ($byte -eq 13) { 1 } else { 0 }) }
    if ($matchState -eq 4) { break }
  }
  if ($matchState -ne 4) { throw 'Request headers were too large.' }

  $headerText = [System.Text.Encoding]::ASCII.GetString($headerBytes.ToArray())
  $lines = $headerText -split "`r`n"
  $requestLine = $lines[0]
  $headers = @{}
  foreach ($line in $lines[1..($lines.Count - 1)]) {
    if (-not $line -or -not $line.Contains(':')) { continue }
    $parts = $line.Split(':', 2)
    $headers[$parts[0].Trim().ToLowerInvariant()] = $parts[1].Trim()
  }

  $contentLength = 0
  if ($headers.ContainsKey('content-length')) { $contentLength = [int]$headers['content-length'] }
  if ($contentLength -gt 262144) { throw 'Request body was too large.' }
  $bodyBytes = [byte[]]::new($contentLength)
  $offset = 0
  while ($offset -lt $contentLength) {
    $read = $Stream.Read($bodyBytes, $offset, $contentLength - $offset)
    if ($read -le 0) { break }
    $offset += $read
  }

  return [pscustomobject]@{
    Line = $requestLine
    Body = [System.Text.Encoding]::UTF8.GetString($bodyBytes, 0, $offset)
  }
}

try {
  $listener.Start()
} catch {
  if (-not $NoBrowser) { Start-Process $url }
  exit 0
}

if (-not $NoBrowser) { Start-Process $url }
$lastRequest = Get-Date

try {
  while (((Get-Date) - $lastRequest).TotalHours -lt 2) {
    if (-not $listener.Pending()) {
      Start-Sleep -Milliseconds 120
      continue
    }

    $client = $listener.AcceptTcpClient()
    try {
      $stream = $client.GetStream()
      $stream.ReadTimeout = 5000
      $dataDeadline = (Get-Date).AddSeconds(2)
      while (-not $stream.DataAvailable -and (Get-Date) -lt $dataDeadline) { Start-Sleep -Milliseconds 25 }
      if (-not $stream.DataAvailable) { continue }

      $request = Read-Request -Stream $stream
      $lastRequest = Get-Date
      $requestParts = $request.Line -split ' '
      $method = $requestParts[0].ToUpperInvariant()
      $path = $requestParts[1].Split('?')[0]

      if ($method -eq 'GET' -and ($path -eq '/' -or $path -eq '/Reading_700_Mastery.html')) {
        $body = [System.IO.File]::ReadAllBytes($pagePath)
        Write-Response -Stream $stream -Status 200 -Reason 'OK' -ContentType 'text/html; charset=utf-8' -Body $body
        continue
      }

      if ($method -eq 'GET' -and ($path -eq '/api/health' -or $path -eq '/api/status')) {
        $keyState = Get-KeyHealth
        Write-JsonResponse -Stream $stream -Status 200 -Reason 'OK' -Data @{ configured = ($keyState -eq 'ready'); keyState = $keyState; model = $model }
        continue
      }

      if ($method -eq 'POST' -and $path -eq '/api/chat') {
        $apiKey = Get-PrivateKey
        if ([string]::IsNullOrWhiteSpace($apiKey)) {
          Write-JsonResponse -Stream $stream -Status 503 -Reason 'Service Unavailable' -Data @{ error = 'trainer-not-configured' }
          continue
        }

        try {
          $incoming = $request.Body | ConvertFrom-Json
          $prompt = [string]$incoming.prompt
          if ([string]::IsNullOrWhiteSpace($prompt) -or $prompt.Length -gt 20000) { throw 'Invalid prompt.' }

          $apiBody = @{
            model = $model
            messages = @(
              @{ role = 'system'; content = $systemPrompt },
              @{ role = 'user'; content = $prompt }
            )
            temperature = 0.3
            max_tokens = 2200
            response_format = @{ type = 'json_object' }
          }
          $apiHeaders = @{
            Authorization = "Bearer $apiKey"
            'HTTP-Referer' = $url
            'X-Title' = 'Sam Reading 700+ Plan'
          }
          $apiResult = Invoke-RestMethod -Method Post -Uri 'https://openrouter.ai/api/v1/chat/completions' -Headers $apiHeaders -ContentType 'application/json' -Body ($apiBody | ConvertTo-Json -Depth 12 -Compress) -TimeoutSec 75
          $content = [string]$apiResult.choices[0].message.content
          if ([string]::IsNullOrWhiteSpace($content)) { throw 'The model returned an empty response.' }
          Write-JsonResponse -Stream $stream -Status 200 -Reason 'OK' -Data @{ content = $content; model = $model }
        } catch {
          $upstreamStatus = 0
          try { $upstreamStatus = [int]$_.Exception.Response.StatusCode } catch {}
          $safeError = switch ($upstreamStatus) {
            401 { 'trainer-key-rejected' }
            402 { 'trainer-needs-credits' }
            429 { 'trainer-busy' }
            default { 'trainer-request-failed' }
          }
          Write-JsonResponse -Stream $stream -Status 502 -Reason 'Bad Gateway' -Data @{ error = $safeError }
        }
        continue
      }

      $notFound = [System.Text.Encoding]::UTF8.GetBytes('Not found')
      Write-Response -Stream $stream -Status 404 -Reason 'Not Found' -ContentType 'text/plain; charset=utf-8' -Body $notFound
    } catch {
      try { Write-JsonResponse -Stream $stream -Status 400 -Reason 'Bad Request' -Data @{ error = 'bad-request' } } catch {}
    } finally {
      $client.Close()
    }
  }
} finally {
  $listener.Stop()
}
