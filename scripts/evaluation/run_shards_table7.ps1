# Launch 6 parallel shards for Obesity Phase 2 Table VII (Fixed Feature = Weight, all 368 queries, seed 42)
$dir = (Get-Location).ProviderPath
$python = "$dir\gradcfa_env\Scripts\python.exe"

0..5 | ForEach-Object {
    $shard = "$_/6"
    $log = "$dir\logs\shard_t7_${_}.log"
    $arg = "-Command `"& '$python' -u obesity_fixed_feature.py --fixed Weight --shard $shard *>&1 | Tee-Object -FilePath '$log'; Start-Sleep -Seconds 86400`""
    Write-Host "Launching Table 7 Shard $shard -> logs\shard_t7_${_}.log"
    Start-Process -FilePath "$PSHOME\powershell.exe" -WorkingDirectory $dir -ArgumentList $arg
}
