param([switch]$NoBrowser, [int]$ListenPort = 4174)

$ErrorActionPreference = 'Stop'
$pagePath = Join-Path $PSScriptRoot 'Reading_700_Mastery.html'
$keyPath = Join-Path $PSScriptRoot '.reading_700_key'
$yashiKeyPath = Join-Path $PSScriptRoot '.for_yashi_key'
$url = "http://127.0.0.1:$ListenPort/Reading_700_Mastery.html"
$model = 'openai/gpt-5.6-luna'
$listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $ListenPort)
$cachePath = Join-Path $PSScriptRoot '.reading_700_daily_cache'

$systemPrompt = @'
You are Sam's elite Digital SAT Reading & Writing study coach. His goal is 700+ in Reading & Writing (Math baseline is 800 locked, targeting 1500–1540 composite) on the Digital SAT on Saturday, October 3, 2026.

Write with precision, intellectual rigor, and direct diagnostic clarity.
- Standard English Conventions: treat grammar as mathematical boolean logic (Independent vs. Dependent clauses, colon rules, subject-verb agreement).
- Command of Evidence: emphasize strict textual anchors and zero outside assumptions.
- Quantitative Evidence: translate graph axes and trend lines before considering options.
- Transitions: categorize into Additive, Contrastive, or Causative.
- Rhetorical Synthesis: match the student's exact specified presentation goal.

Every generated question must be original. Never reproduce or closely imitate a released College Board, Khan Academy, or publisher question. Use supplied credible-source excerpts only as factual inspiration and rewrite all passages from scratch.
Return only valid JSON matching the requested schema with no markdown fences or outside commentary.
'@

$sourcePool = @(
  @{ title = 'NASA Climate Evidence'; url = 'https://science.nasa.gov/climate-change/evidence/' },
  @{ title = 'NOAA Hydrothermal Vents'; url = 'https://oceanexplorer.noaa.gov/education/hydrothermal-vents-volcanoes-educators/' },
  @{ title = 'Library of Congress Primary Sources'; url = 'https://www.loc.gov/programs/teachers/getting-started-with-primary-sources/finding/' },
  @{ title = 'National Park Service History'; url = 'https://www.nps.gov/subjects/history/index.htm' },
  @{ title = 'Smithsonian Human Origins'; url = 'https://humanorigins.si.edu/evidence' },
  @{ title = 'USGS Water Science School'; url = 'https://www.usgs.gov/special-topics/water-science-school' },
  @{ title = 'NIH News in Health'; url = 'https://newsinhealth.nih.gov/' },
  @{ title = 'USDA Economic Research Service'; url = 'https://www.ers.usda.gov/topics' },
  @{ title = 'Bureau of Labor Statistics Data'; url = 'https://www.bls.gov/data/' },
  @{ title = 'U.S. Census Data and Maps'; url = 'https://www.census.gov/data.html' },
  @{ title = 'National Gallery of Art Stories'; url = 'https://www.nga.gov/stories.html' },
  @{ title = 'Federal Reserve History'; url = 'https://www.federalreservehistory.org/' }
)

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

function Get-SourceContext {
  param([hashtable]$Source)
  try {
    $response = Invoke-WebRequest -Uri $Source.url -Headers @{ 'User-Agent' = 'Reading700StudyCoach/1.0' } -TimeoutSec 10
    $text = [regex]::Replace($response.Content, '(?is)<script.*?</script>|<style.*?</style>|<!--.*?-->', ' ')
    $text = [regex]::Replace($text, '<[^>]+>', ' ')
    $text = [System.Net.WebUtility]::HtmlDecode($text)
    $text = [regex]::Replace($text, '\s+', ' ').Trim()
    if ($text.Length -gt 2200) { $text = $text.Substring(0, 2200) }
    return "SOURCE: $($Source.title)`nURL: $($Source.url)`nEXCERPT FOR FACTUAL INSPIRATION ONLY: $text"
  } catch {
    return "SOURCE: $($Source.title)`nURL: $($Source.url)`nThe source was temporarily unavailable. Use only well-established facts associated with this institution and do not invent statistics."
  }
}

function Get-DailyQuestionPack {
  param([string]$DateText, [switch]$Force)

  $date = [datetime]::MinValue
  if (-not [datetime]::TryParseExact($DateText, 'yyyy-MM-dd', [Globalization.CultureInfo]::InvariantCulture, [Globalization.DateTimeStyles]::None, [ref]$date)) {
    throw 'Invalid date.'
  }

  if (-not (Test-Path -LiteralPath $cachePath)) { New-Item -ItemType Directory -Path $cachePath | Out-Null }
  $cacheFile = Join-Path $cachePath ($DateText + '.json')
  if (-not $Force -and (Test-Path -LiteralPath $cacheFile)) {
    return (Get-Content -LiteralPath $cacheFile -Raw | ConvertFrom-Json)
  }

  $apiKey = Get-PrivateKey
  if ([string]::IsNullOrWhiteSpace($apiKey)) { throw 'trainer-not-configured' }

  $startIndex = $date.DayOfYear % $sourcePool.Count
  $selectedSources = for ($i = 0; $i -lt 4; $i++) { $sourcePool[($startIndex + ($i * 3)) % $sourcePool.Count] }
  $sourceContext = ($selectedSources | ForEach-Object { Get-SourceContext $_ }) -join "`n`n"
  $setId = 'sat-rw-' + $DateText
  $prompt = @"
Create one complete 27-question Digital SAT Reading and Writing practice module for $DateText. The set ID is $setId.

QUALITY AND ORIGINALITY RULES
- All passages, stems, answer choices, and explanations must be newly written and unique to this set.
- Do not copy, quote, paraphrase closely, or reconstruct any released SAT, PSAT, Khan Academy, test-prep-company, or publisher question.
- Match challenging Digital SAT style and concision. Aim at a student moving from about 540 toward 700+.
- Use a balanced difficulty curve: 6 medium, 15 hard, 6 very hard. Put difficulty in each item.
- Exactly four plausible answer choices. Vary correct-answer positions across A-D; no obvious pattern.
- Explanations must identify the decisive textual or grammatical evidence and briefly diagnose why each major trap fails.
- For source-inspired reading items, write a fresh 60-130 word passage based only on the supplied context, and include source_title/source_url as an inspiration link. Do not imply the source authored the question.
- For grammar/transition/synthesis items, source_title and source_url may be empty strings.
- Avoid contested, medical-treatment, political-persuasion, or personally identifying content.

EXACT MODULE MIX AND NUMBERING
- Q1-Q7: Craft and Structure (words in context, text structure/purpose, cross-text connections)
- Q8-Q15: Information and Ideas (central ideas/details, inference, command of textual or quantitative evidence)
- Q16-Q22: Standard English Conventions (boundaries, form/structure/sense, agreement, modifiers, verb form)
- Q23-Q27: Expression of Ideas (transitions and rhetorical synthesis)

CREDIBLE SOURCE CONTEXT (facts only; transform into original SAT-style passages):
$sourceContext
"@

  $questionSchema = @{
    type = 'object'
    additionalProperties = $false
    properties = @{
      num = @{ type = 'integer'; minimum = 1; maximum = 27 }
      type = @{ type = 'string'; minLength = 3; maxLength = 80 }
      domain = @{ type = 'string'; enum = @('Craft and Structure', 'Information and Ideas', 'Standard English Conventions', 'Expression of Ideas') }
      skill = @{ type = 'string'; minLength = 3; maxLength = 80 }
      difficulty = @{ type = 'string'; enum = @('Medium', 'Hard', 'Very Hard') }
      passage = @{ type = 'string'; maxLength = 1400 }
      prompt = @{ type = 'string'; minLength = 15; maxLength = 1800 }
      options = @{ type = 'array'; minItems = 4; maxItems = 4; items = @{ type = 'string'; minLength = 1; maxLength = 700 } }
      correct = @{ type = 'integer'; minimum = 0; maximum = 3 }
      explanation = @{ type = 'string'; minLength = 40; maxLength = 1800 }
      source_title = @{ type = 'string'; maxLength = 140 }
      source_url = @{ type = 'string'; maxLength = 500 }
    }
    required = @('num','type','domain','skill','difficulty','passage','prompt','options','correct','explanation','source_title','source_url')
  }
  $schema = @{
    type = 'object'
    additionalProperties = $false
    properties = @{
      generated_date = @{ type = 'string'; const = $DateText }
      set_id = @{ type = 'string'; const = $setId }
      questions = @{ type = 'array'; minItems = 27; maxItems = 27; items = $questionSchema }
    }
    required = @('generated_date','set_id','questions')
  }

  $apiBody = @{
    model = $model
    messages = @(
      @{ role = 'system'; content = $systemPrompt },
      @{ role = 'user'; content = $prompt }
    )
    reasoning = @{ effort = 'none' }
    max_tokens = 14000
    provider = @{ sort = 'throughput'; require_parameters = $true }
    response_format = @{ type = 'json_schema'; json_schema = @{ name = 'daily_sat_rw_module'; strict = $true; schema = $schema } }
    plugins = @(@{ id = 'response-healing' })
  }
  $apiHeaders = @{
    Authorization = "Bearer $apiKey"
    'HTTP-Referer' = $url
    'X-Title' = 'Reading 700 Daily Coach'
  }
  try {
    $requestJson = $apiBody | ConvertTo-Json -Depth 30 -Compress
    $requestBytes = [System.Text.Encoding]::UTF8.GetBytes($requestJson)
    $apiResult = Invoke-RestMethod -Method Post -Uri 'https://openrouter.ai/api/v1/chat/completions' -Headers $apiHeaders -ContentType 'application/json; charset=utf-8' -Body $requestBytes -TimeoutSec 150
  } catch {
    throw
  }
  $content = [string]$apiResult.choices[0].message.content
  if ([string]::IsNullOrWhiteSpace($content)) { throw 'trainer-empty-response' }
  $pack = $content | ConvertFrom-Json
  if ($pack.questions.Count -ne 27) { throw 'trainer-invalid-question-count' }
  $numbers = @($pack.questions | ForEach-Object { [int]$_.num } | Sort-Object -Unique)
  if ($numbers.Count -ne 27 -or $numbers[0] -ne 1 -or $numbers[-1] -ne 27) { throw 'trainer-invalid-numbering' }
  $mediumNumbers = @(1,4,8,12,16,23)
  $veryHardNumbers = @(6,11,15,21,25,27)
  $approvedUrls = @($sourcePool | ForEach-Object { [string]$_.url })
  foreach ($question in $pack.questions) {
    if ($question.options.Count -ne 4 -or [int]$question.correct -lt 0 -or [int]$question.correct -gt 3 -or [string]::IsNullOrWhiteSpace([string]$question.explanation)) {
      throw 'trainer-invalid-question'
    }
    $questionNumber = [int]$question.num
    $targetPosition = ($questionNumber + $date.DayOfYear) % 4
    $oldCorrect = [int]$question.correct
    $correctText = [string]$question.options[$oldCorrect]
    $otherOptions = @()
    for ($optionIndex = 0; $optionIndex -lt 4; $optionIndex++) {
      if ($optionIndex -ne $oldCorrect) { $otherOptions += [string]$question.options[$optionIndex] }
    }
    $newOptions = @()
    $otherIndex = 0
    for ($optionIndex = 0; $optionIndex -lt 4; $optionIndex++) {
      if ($optionIndex -eq $targetPosition) { $newOptions += $correctText }
      else { $newOptions += $otherOptions[$otherIndex]; $otherIndex++ }
    }
    $question.options = $newOptions
    $question.correct = $targetPosition
    $question.difficulty = if ($mediumNumbers -contains $questionNumber) { 'Medium' } elseif ($veryHardNumbers -contains $questionNumber) { 'Very Hard' } else { 'Hard' }
    if ($question.source_url -and $approvedUrls -notcontains [string]$question.source_url) {
      $question.source_title = ''
      $question.source_url = ''
    }
  }

  $cacheJson = $pack | ConvertTo-Json -Depth 20 -Compress
  [System.IO.File]::WriteAllText($cacheFile, $cacheJson, [System.Text.UTF8Encoding]::new($false))
  return $pack
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

      if ($method -eq 'POST' -and $path -eq '/api/daily-questions') {
        try {
          $incoming = $request.Body | ConvertFrom-Json
          $dateText = [string]$incoming.date
          $force = [bool]$incoming.force
          $pack = Get-DailyQuestionPack -DateText $dateText -Force:$force
          Write-JsonResponse -Stream $stream -Status 200 -Reason 'OK' -Data @{ pack = $pack; model = $model; cachedForDay = $true }
        } catch {
          $message = [string]$_.Exception.Message
          $safeError = if ($message -match 'trainer-not-configured') { 'trainer-not-configured' }
            elseif ($message -match '401|Unauthorized') { 'trainer-key-rejected' }
            elseif ($message -match '402|credits') { 'trainer-needs-credits' }
            elseif ($message -match '429|rate') { 'trainer-busy' }
            else { 'daily-generation-failed' }
          Write-JsonResponse -Stream $stream -Status 502 -Reason 'Bad Gateway' -Data @{ error = $safeError }
        }
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
            reasoning = @{ effort = 'none' }
            provider = @{ sort = 'throughput' }
            response_format = @{ type = 'json_object' }
          }
          $apiHeaders = @{
            Authorization = "Bearer $apiKey"
            'HTTP-Referer' = $url
            'X-Title' = 'Sam Reading 700+ Plan'
          }
          $chatJson = $apiBody | ConvertTo-Json -Depth 12 -Compress
          $apiResult = Invoke-RestMethod -Method Post -Uri 'https://openrouter.ai/api/v1/chat/completions' -Headers $apiHeaders -ContentType 'application/json; charset=utf-8' -Body ([System.Text.Encoding]::UTF8.GetBytes($chatJson)) -TimeoutSec 75
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
