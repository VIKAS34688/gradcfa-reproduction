# Launch 6 parallel shards for Obesity Phase 1 evaluation (config C_nopert, seed 42)
$dir = (Get-Location).ProviderPath
$python = "$dir\gradcfa_env\Scripts\python.exe"

0..5 | ForEach-Object {
    $shard = "$_/6"
    $log = "$dir\logs\shard_$_.log"
    $arg = "-Command `"& '$python' -u obesity_eval.py --config C_nopert --seeds 42 --shard $shard *>&1 | Tee-Object -FilePath '$log'; Start-Sleep -Seconds 86400`""
    Write-Host "Launching Shard $shard -> logs\shard_$_.log"
    Start-Process -FilePath "$PSHOME\powershell.exe" -WorkingDirectory $dir -ArgumentList $arg
}
