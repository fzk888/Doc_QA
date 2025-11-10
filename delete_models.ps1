# 删除Hugging Face缓存中的模型文件

# 定义Hugging Face缓存目录路径
$cacheDir = "$env:USERPROFILE\.cache\huggingface\hub"

# 检查缓存目录是否存在
if (Test-Path $cacheDir) {
    Write-Host "找到Hugging Face缓存目录: $cacheDir"
    
    # 查找并显示模型文件夹
    $modelDirs = Get-ChildItem -Path $cacheDir -Directory -Filter "models--*"
    
    if ($modelDirs.Count -eq 0) {
        Write-Host "缓存目录中没有找到模型文件夹"
    } else {
        Write-Host "\n找到以下模型文件夹:"
        $modelDirs | ForEach-Object {
            $modelName = $_.Name -replace "models--", "" -replace "--", "/"
            $size = (Get-ChildItem -Path $_.FullName -Recurse | Measure-Object -Property Length -Sum).Sum / 1GB
            Write-Host "- $modelName (约 $([math]::Round($size, 2)) GB)"
        }
        
        # 删除指定的模型文件夹
        Write-Host "\n正在删除BAAI相关模型..."
        $modelsToDelete = @(
            "models--BAAI--bge-large-zh-v1.5",
            "models--BAAI--bge-reranker-large"
        )
        
        foreach ($model in $modelsToDelete) {
            $modelPath = Join-Path -Path $cacheDir -ChildPath $model
            if (Test-Path $modelPath) {
                Write-Host "删除 $modelPath"
                Remove-Item -Path $modelPath -Recurse -Force
                Write-Host "已删除 $model"
            } else {
                Write-Host "模型不存在: $model"
            }
        }
        
        Write-Host "\n清理完成!"
    }
} else {
    Write-Host "未找到Hugging Face缓存目录: $cacheDir"
}

# 显示脚本执行完成
Write-Host "\n按任意键退出..."
$null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')