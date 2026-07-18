param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $SkinArgs
)

python -m nanobot.cli.terminal_skin @SkinArgs
if ($LASTEXITCODE -ne 0) {
    Write-Error "背景图切换失败，退出码: $LASTEXITCODE"
}
