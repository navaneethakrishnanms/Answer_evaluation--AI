# Poppler Auto-Installer
# This script downloads and installs Poppler for Windows

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "Poppler Auto-Installer" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# Define paths
$popplerDir = "C:\poppler"
$downloadUrl = "https://github.com/oschwartz10612/poppler-windows/releases/download/v24.08.0-0/Release-24.08.0-0.zip"
$zipFile = "$env:TEMP\poppler.zip"

# Create directory
Write-Host "Creating Poppler directory..." -ForegroundColor Yellow
if (-not (Test-Path $popplerDir)) {
    New-Item -ItemType Directory -Path $popplerDir -Force | Out-Null
}

# Download Poppler
Write-Host "Downloading Poppler (this may take a few minutes)..." -ForegroundColor Yellow
try {
    Invoke-WebRequest -Uri $downloadUrl -OutFile $zipFile -UseBasicParsing
    Write-Host "[SUCCESS] Download complete!" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Download failed: $_" -ForegroundColor Red
    exit 1
}

# Extract
Write-Host "Extracting Poppler..." -ForegroundColor Yellow
try {
    Expand-Archive -Path $zipFile -DestinationPath $popplerDir -Force
    Write-Host "[SUCCESS] Extraction complete!" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Extraction failed: $_" -ForegroundColor Red
    exit 1
}

# Find the bin directory
$binPath = Get-ChildItem -Path $popplerDir -Recurse -Filter "bin" -Directory | Select-Object -First 1
if ($binPath) {
    $fullBinPath = $binPath.FullName
    Write-Host "[SUCCESS] Found Poppler bin at: $fullBinPath" -ForegroundColor Green
} else {
    # Try alternative path
    $fullBinPath = "$popplerDir\poppler-24.08.0\Library\bin"
    if (-not (Test-Path $fullBinPath)) {
        $fullBinPath = "$popplerDir\Library\bin"
    }
}

# Add to PATH
Write-Host "Adding Poppler to system PATH..." -ForegroundColor Yellow
$currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($currentPath -notlike "*$fullBinPath*") {
    [Environment]::SetEnvironmentVariable("Path", "$currentPath;$fullBinPath", "User")
    Write-Host "[SUCCESS] Added to PATH!" -ForegroundColor Green
} else {
    Write-Host "[SUCCESS] Already in PATH!" -ForegroundColor Green
}

# Cleanup
Remove-Item $zipFile -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "=====================================" -ForegroundColor Green
Write-Host "Installation Complete!" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green
Write-Host ""
Write-Host "[WARNING] IMPORTANT: You must restart your terminal/command prompt for changes to take effect!" -ForegroundColor Yellow
Write-Host ""
Write-Host "After restarting, verify installation by running:" -ForegroundColor Cyan
Write-Host "  pdfinfo -v" -ForegroundColor White
Write-Host ""

pause
