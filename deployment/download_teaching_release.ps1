param(
    [string]$SshTarget = 'root@YOUR_SERVER',
    [int]$Port = 22,
    [string]$Destination = 'D:\actquant'
)

$ErrorActionPreference = 'Stop'
$Release = 'pi05-teaching-20261019-01'
$ArchiveName = "$Release.tar"
$RemoteRoot = '/workspace/actquant/exports'
New-Item -ItemType Directory -Force -Path $Destination | Out-Null
$Archive = Join-Path $Destination $ArchiveName
$Checksum = Join-Path $Destination "$ArchiveName.sha256"
$Extracted = Join-Path $Destination $Release
if (Test-Path -LiteralPath $Extracted) {
    throw "Release directory already exists; refusing to overwrite: $Extracted"
}

Write-Host "Downloading teaching release to $Destination (approximately 3.4 GB)..."
& scp.exe -P $Port "${SshTarget}:$RemoteRoot/$ArchiveName" "${SshTarget}:$RemoteRoot/$ArchiveName.sha256" $Destination
if ($LASTEXITCODE -ne 0) { throw 'SCP download failed. Check SSH connectivity and retry.' }

$Expected = ((Get-Content -Raw -LiteralPath $Checksum).Trim() -split '\s+')[0].ToLowerInvariant()
$Actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Archive).Hash.ToLowerInvariant()
if ($Actual -ne $Expected) { throw 'SHA256 mismatch; archive will not be extracted.' }
Write-Host "SHA256 verified: $Actual"

& tar.exe -xf $Archive -C $Destination
if ($LASTEXITCODE -ne 0) { throw 'Extraction failed; keep the archive and inspect the destination.' }
if (-not (Test-Path -LiteralPath (Join-Path $Extracted 'reproduction\results\pi05-quantized-demo-01\model\pi05.gguf'))) {
    throw 'Extracted model was not found.'
}
Write-Host "Transfer and extraction complete: $Extracted"
Write-Host "Start with: $(Join-Path $Extracted 'README.md')"
