# Windows Terminal 背景图切换

支持的图片格式：JPG、JPEG、PNG、WebP、BMP、GIF。图片放在本目录后会自动出现在列表中。

本目录随 nanobot 工程位于 `nanobot/fork/cmdSkins`。脚本和本文档可提交到 Git；壁纸图片由仓库 `.gitignore` 排除，需在每台机器上自行放入。

## 普通 CMD / PowerShell

```text
skin              # 交互选择
skin list         # 列出图片，* 表示当前背景
skin next         # 下一张
skin prev         # 上一张
skin random       # 随机一张（避免连续重复）
skin 3            # 按列表编号切换
skin 10.jpg       # 按完整文件名切换
```

脚本只修改 Windows Terminal 的 `profiles.defaults.backgroundImage`，因此 CMD、PowerShell 和 nanobot 共用同一背景。

首次切换时，会在 Windows Terminal 配置旁创建：

```text
settings.skin-backup.json
```

该备份只创建一次，不会被后续切换覆盖。

## nanobot

```text
/skin              # 弹出图片选择列表
/skin list         # 列出图片
/skin next
/skin prev
/skin random
/skin 3
/skin 10.jpg
```

`/skin` 是本地交互命令，不发送给模型、不保存为用户消息，也不会触发 agent turn。

## 配置

默认直接读取本目录。也可以在 `~/.nanobot/config.json` 中覆盖：

```json
{
  "agents": {
    "defaults": {
      "tuiSkinDir": "D:\\Wallpapers"
    }
  }
}
```

命令行临时指定目录时，`skin --dir D:\\Wallpapers list` 的优先级最高。
