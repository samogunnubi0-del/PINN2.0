$ErrorActionPreference = 'Stop'
$keyPath = Join-Path $PSScriptRoot '.for_yashi_key'

Write-Host ''
Write-Host 'The 1400 Plan trainer setup' -ForegroundColor DarkRed
Write-Host 'Paste a NEW OpenRouter key. The key already shared in chat should be revoked first.'
Write-Host 'Your new key will be encrypted for this Windows account and will not be placed in the HTML.'
Write-Host ''

$secureKey = Read-Host 'New OpenRouter key' -AsSecureString
$credential = [System.Management.Automation.PSCredential]::new('OpenRouter', $secureKey)
$plainKey = $credential.GetNetworkCredential().Password
if ([string]::IsNullOrWhiteSpace($plainKey) -or $plainKey -notmatch '^sk-or-v1-[A-Za-z0-9_-]+$') {
  Write-Host ''
  Write-Host 'That does not look like a valid OpenRouter key. Nothing was saved.' -ForegroundColor Red
  Write-Host 'Press Enter to close.'
  [void](Read-Host)
  exit 1
}
$plainKey = $null
$encryptedKey = ConvertFrom-SecureString $secureKey
Set-Content -LiteralPath $keyPath -Value $encryptedKey -Encoding UTF8 -NoNewline
try { (Get-Item -LiteralPath $keyPath).Attributes = 'Hidden' } catch {}

Write-Host ''
Write-Host 'The trainer is ready. The app will open next.' -ForegroundColor DarkGreen
Write-Host 'Press Enter to close.'
[void](Read-Host)
