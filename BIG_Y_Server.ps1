$ErrorActionPreference = 'Stop'
$pagePath = Join-Path $PSScriptRoot 'BIG_Y_Plan.html'
$url = 'http://127.0.0.1:4173/BIG_Y_Plan.html'
$listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 4173)

try {
  $listener.Start()
} catch {
  Start-Process $url
  exit 0
}

Start-Process $url
$lastRequest = Get-Date

try {
  while (((Get-Date) - $lastRequest).TotalHours -lt 2) {
    if (-not $listener.Pending()) {
      Start-Sleep -Milliseconds 150
      continue
    }

    $client = $listener.AcceptTcpClient()
    try {
      $stream = $client.GetStream()
      $reader = [System.IO.StreamReader]::new($stream, [System.Text.Encoding]::ASCII, $false, 1024, $true)
      $requestLine = $reader.ReadLine()
      while (($line = $reader.ReadLine()) -ne $null -and $line -ne '') {}
      $lastRequest = Get-Date

      if ($requestLine -match '^GET\s+/(?:BIG_Y_Plan\.html)?(?:\?.*)?\s+HTTP/') {
        $body = [System.IO.File]::ReadAllBytes($pagePath)
        $header = "HTTP/1.1 200 OK`r`nContent-Type: text/html; charset=utf-8`r`nContent-Length: $($body.Length)`r`nCache-Control: no-store`r`nReferrer-Policy: strict-origin-when-cross-origin`r`nConnection: close`r`n`r`n"
      } else {
        $body = [System.Text.Encoding]::UTF8.GetBytes('Not found')
        $header = "HTTP/1.1 404 Not Found`r`nContent-Type: text/plain; charset=utf-8`r`nContent-Length: $($body.Length)`r`nConnection: close`r`n`r`n"
      }

      $headerBytes = [System.Text.Encoding]::ASCII.GetBytes($header)
      $stream.Write($headerBytes, 0, $headerBytes.Length)
      $stream.Write($body, 0, $body.Length)
      $stream.Flush()
    } finally {
      $client.Close()
    }
  }
} finally {
  $listener.Stop()
}
