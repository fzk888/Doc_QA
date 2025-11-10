# 简单的模型缓存清理脚本
Write-Host "开始清理Hugging Face缓存模型..."

# 定义缓存目录
$cacheDir = "$env:USERPROFILE\.cache\huggingface\hub"
Write-Host "缓存目录: $cacheDir"

# 删除所有模型文件夹
if (Test-Path $cacheDir) {
    Write-Host "正在删除模型文件..."
    Get-ChildItem -Path "$cacheDir\models--*" -Directory | ForEach-Object {
        $modelName = $_.Name -replace "models--", "" -replace "--", "/"
        Write-Host "- 删除: $modelName"
        Remove-Item -Path $_.FullName -Recurse -Force
    }
}

# 验证删除结果
Write-Host "\n验证删除结果:" -ForegroundColor Green
$remainingModels = Get-ChildItem -Path "$cacheDir\models--*" -Directory
if ($remainingModels.Count -eq 0) {
    Write-Host "✓ 所有缓存模型已成功删除！" -ForegroundColor Green
} else {
    Write-Host "仍有以下模型文件:" -ForegroundColor Red
    $remainingModels | ForEach-Object {
        $modelName = $_.Name -replace "models--", "" -replace "--", "/"
        Write-Host "- $modelName" -ForegroundColor Red
    }
}

Write-Host "\n注意：项目目录中的D:\大模型应用开发\RAG\Doc_QA\model文件夹内容不会被删除。"