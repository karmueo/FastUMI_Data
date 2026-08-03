<!-- 本文档记录 Ubuntu 24.04、XRDP、XFCE 环境下 SteamVR、VIVE Tracker 和基站的完整安装、配置及验收流程。 -->

# Ubuntu 24.04 + XRDP/XFCE + SteamVR + VIVE Tracker 配置教程

## 1. 文档目标

本文档用于在 Ubuntu 24.04 的 XRDP + XFCE 远程桌面环境中完成以下工作：

- 安装 Steam 和 SteamVR。
- 处理 Snap 下载代理和 XRDP D-Bus/cgroup 问题。
- 处理 Steam Snap 无法访问 VIVE USB/HID 设备的问题。
- 配置无头显运行，即只使用 VIVE Tracker 和 Lighthouse 基站。
- 配对 VIVE Tracker USB Dongle/Watchman Dongle。
- 启动并验证两个 SteamVR Base Station 2.0。
- 根据 SteamVR 日志判断 USB、Tracker 和基站状态。

本文档内容已在实际设备上验证。最终成功日志确认：

```text
Lighthouse IMU HID opened
LHR-B77A06A7: Connected to receiver 0E36592B06
SOB: add S-2
SOB: add S-1 also seeing S-2
```

## 2. 重要结论

当前环境的推荐运行链路如下：

```text
XRDP + XFCE
    └── 原生 Steam（/usr/games/steam）
          └── Steam Linux Runtime 3.0 (sniper)
                └── SteamVR
                      └── udev + plugdev
                            └── Watchman Dongle
                                  └── VIVE Tracker 3.0
                                        └── Base Station 2.0：S-1、S-2
```

关键结论：

1. Steam Snap 可以正常安装和启动 Steam 客户端。
2. Steam Snap 当前缺少可连接的 `raw-usb` 接口，SteamVR 无法读写 Watchman Dongle。
3. VIVE Tracker 配对和跟踪需要使用原生 Steam，即 Ubuntu 的 `steam-installer` 包。
4. XRDP 会话不属于本地物理 seat，默认 `uaccess` 权限通常会分配给 `lightdm`。需要通过 `plugdev` 和自定义 udev 规则授权 Dongle。
5. `requireHmd=false` 允许 SteamVR 在没有头显的情况下加载 Tracker 驱动。SteamVR 界面仍可能显示 `Please plug in your VR headset`，该提示不影响 Tracker 工作。
6. 每个 Watchman Dongle 通常只连接一个 Tracker。Dongle 已有 Tracker 时，配对窗口可能显示 `no pairable devices detected`。

### 2.1 推荐执行顺序

第一次配置时按以下顺序执行：

1. 安装 XFCE、XRDP、原生 Steam、`steam-devices` 和 32 位显卡库。
2. 配置 `~/.xsession`，退出并重新登录 XRDP。
3. 将当前用户加入 `plugdev`。
4. 安装 `61-vive-tracker.rules` udev 规则。
5. 再次退出并重新登录 XRDP，使组权限生效。
6. 插入 Watchman Dongle，并用 `lsusb` 验证 `28de:2101`。
7. 使用 `/usr/games/steam` 启动原生 Steam。
8. 安装 SteamVR，并选择 Steam Linux Runtime 3.0。
9. 设置 `requireHmd=false`，再修复 `vrcompositor-launcher` capability。
10. 启动 SteamVR并配对 Tracker。
11. 给两个基站接通电源，等待绿色或白色常亮。
12. 唤醒 Tracker，在 `vrserver.txt` 中验证 Dongle、Tracker、`S-1` 和 `S-2`。

## 3. 已验证环境

| 项目 | 验证值 |
| --- | --- |
| 操作系统 | Ubuntu 24.04.4 LTS |
| 远程桌面 | XRDP |
| 桌面环境 | XFCE |
| 当前用户 | 当前登录的普通用户 |
| GPU | NVIDIA RTX 5090 |
| NVIDIA 驱动 | 610.43.02 |
| Steam Snap | 1.0.0.85，revision 231，最终状态为 disabled |
| 原生 Steam 启动器 | `steam-installer` 1.0.0.79 |
| SteamVR App ID | 250820 |
| SteamVR 测试版本 | 2.16.7 |
| Dongle USB ID | `28de:2101` Valve Software Watchman Dongle |
| Tracker | VIVE Tracker 3.0 |
| 基站 | 两个 SteamVR Base Station 2.0 |

## 4. 系统准备

### 4.1 启用 Ubuntu multiverse 和 i386

Steam 和 SteamVR 需要 32 位运行库：

```bash
sudo dpkg --add-architecture i386
sudo add-apt-repository -y multiverse
sudo apt update
```

### 4.2 安装原生 Steam 和硬件规则

```bash
sudo apt install -y steam-installer steam-devices acl
```

NVIDIA 机器还需要与当前驱动版本一致的 32 位 OpenGL/Vulkan 库：

```bash
sudo apt install -y libnvidia-gl:i386
```

安装前可以检查 apt 的依赖计划，确认不会卸载当前 NVIDIA 驱动：

```bash
sudo apt install --simulate libnvidia-gl:i386
```

验证 64 位和 32 位 NVIDIA 库：

```bash
dpkg -l | grep -E 'libnvidia-gl:(amd64|i386)'
```

AMD 或 Intel GPU 不需要安装 `libnvidia-gl:i386`，应安装发行版提供的对应 Mesa 32 位库。

## 5. 配置 XRDP + XFCE 会话

### 5.1 安装桌面和远程服务

如果系统尚未安装 XRDP/XFCE：

```bash
sudo apt install -y xrdp xfce4 xfce4-goodies dbus-x11
sudo systemctl enable --now xrdp
```

### 5.2 配置 `~/.xsession`

使用文本编辑器打开：

```bash
nano ~/.xsession
```

文件内容：

```sh
# xrdp 用户会话入口：远程桌面使用 XFCE，并使用独立 DBus 会话避免与物理机桌面冲突。
export XDG_CURRENT_DESKTOP=XFCE
export XDG_SESSION_DESKTOP=xfce
export DESKTOP_SESSION=xfce

unset DBUS_SESSION_BUS_ADDRESS
unset SESSION_MANAGER

exec /usr/bin/dbus-run-session -- /usr/bin/startxfce4
```

设置执行权限：

```bash
chmod +x ~/.xsession
```

修改后退出 XRDP，并重新登录。

## 6. Steam Snap 安装记录

本节用于记录当前机器走过的 Snap 安装过程。VIVE Tracker 最终运行使用第 7 节的原生 Steam。

### 6.1 Snap 下载出现 `connection reset by peer`

典型错误：

```text
Download snap "core24" ... read: connection reset by peer
```

Shell 中的 `http_proxy` 不会自动传给 `snapd` 系统服务。需要单独配置 Snap 系统代理：

```bash
sudo snap set system \
  proxy.http=http://127.0.0.1:7890 \
  proxy.https=http://127.0.0.1:7890
```

验证：

```bash
sudo snap get system proxy.http proxy.https
```

安装 Steam Snap：

```bash
sudo snap install steam
```

如果使用 Mihomo/Clash，Steam 下载域名需要走可用的代理节点。常见域名包括：

```text
steampowered.com
steam-chat.com
steamgames.com
steamusercontent.com
steamcontent.com
steamstatic.com
steamcdn-a.akamaihd.net
steamstat.us
```

### 6.2 XRDP 中出现 `is not a snap cgroup`

典型错误：

```text
/user.slice/user-1004.slice/session-c4.scope is not a snap cgroup for tag snap.steam.steam
```

原因是 `~/.xsession` 创建了独立 D-Bus 会话，而 Snap 需要通过 systemd 用户总线建立应用 scope。

当前机器使用兼容包装器 `~/.local/bin/snap`。打开文件：

```bash
mkdir -p ~/.local/bin
nano ~/.local/bin/snap
```

写入：

```sh
#!/bin/sh
# 本文件为 XRDP 独立 D-Bus 会话提供 Snap 应用启动兼容层。
# 仅在执行 snap run 时切换到 systemd 用户总线，以便创建 Snap cgroup。

if [ "${1-}" = "run" ] && [ -n "${XDG_RUNTIME_DIR:-}" ] && [ -S "${XDG_RUNTIME_DIR}/bus" ]; then
    case " $* " in
        *" steam "*)
            export http_proxy="http://127.0.0.1:7890"
            export https_proxy="http://127.0.0.1:7890"
            export HTTP_PROXY="http://127.0.0.1:7890"
            export HTTPS_PROXY="http://127.0.0.1:7890"
            ;;
    esac
    DBUS_SESSION_BUS_ADDRESS="unix:path=${XDG_RUNTIME_DIR}/bus" exec /usr/bin/snap "$@"
fi

exec /usr/bin/snap "$@"
```

设置权限并检查语法：

```bash
chmod +x ~/.local/bin/snap
sh -n ~/.local/bin/snap
```

确认 `~/.local/bin` 位于 `PATH` 前部：

```bash
command -v snap
```

预期输出：

```text
/home/<用户名>/.local/bin/snap
```

### 6.3 Steam Snap 无法打开 Watchman Dongle

即使连接 `hardware-observe`：

```bash
sudo snap connect steam:hardware-observe
```

Snap 内仍可能出现：

```text
/dev/hidraw3: Operation not permitted
/dev/bus/usb/003/004: Operation not permitted
```

SteamVR 日志对应错误：

```text
CHidDevice: Can't open USB device VID 000028de, PID 00002101
Driver lighthouse has no suitable devices.
```

这是 Snap 设备隔离造成的限制。修改文件权限无法越过该限制，Tracker 配对阶段应切换到原生 Steam。

## 7. 切换到原生 Steam

### 7.1 新机器的推荐方式

完成第 4 节后直接启动：

```bash
/usr/games/steam
```

也可以从 XFCE 应用菜单打开 Steam。原生桌面入口的 `Exec` 应为：

```text
Exec=/usr/games/steam %U
```

### 7.2 复用已经由 Snap 下载的 SteamVR

当前机器的 Steam 数据约 8.5 GB，位于：

```text
~/snap/steam/common/.local/share/Steam
```

当前机器通过以下软链接复用这些文件：

```text
~/.local/share/Steam -> $HOME/snap/steam/common/.local/share/Steam
~/.steam/root         -> $HOME/snap/steam/common/.local/share/Steam
~/.steam/steam        -> $HOME/snap/steam/common/.local/share/Steam
```

在另一台机器迁移前，先完全退出 Steam 和 SteamVR，并确认目标路径没有需要保留的数据：

```bash
pgrep -a -x steam
pgrep -a -x vrserver
```

确认进程已经退出后再建立链接：

```bash
mkdir -p ~/.steam ~/.local/share
ln -s ~/snap/steam/common/.local/share/Steam ~/.local/share/Steam
ln -s ~/snap/steam/common/.local/share/Steam ~/.steam/root
ln -s ~/snap/steam/common/.local/share/Steam ~/.steam/steam
```

如果上述目标已经存在，应先备份，不能直接覆盖原 Steam 数据目录。

完成后启动原生 Steam：

```bash
/usr/games/steam
```

确认原生版本运行后禁用 Snap 入口：

```bash
sudo snap disable steam
```

当前数据仍存放在 `~/snap/steam/common`，因此不要执行：

```bash
sudo snap remove --purge steam
```

删除 Steam Snap 前，应先把 Steam 数据迁移到普通用户目录并更新软链接。

### 7.3 判断当前运行的是哪一种 Steam

```bash
pgrep -a -x steam
```

原生 Steam 的 cgroup 应类似：

```text
/user.slice/user-1004.slice/session-5.scope
```

如果出现 `snap.steam.steam-*.scope`，当前仍处于 Snap 限制中。

## 8. 配置 Watchman Dongle USB 权限

### 8.1 确认 USB 设备

插入 Dongle 后执行：

```bash
lsusb | grep -i '28de:2101'
```

预期输出类似：

```text
Bus 003 Device 004: ID 28de:2101 Valve Software Watchman Dongle
```

Bus 和 Device 编号会在重新插拔或重启后变化。

### 8.2 添加用户到 `plugdev`

```bash
sudo usermod -aG plugdev "$USER"
```

执行后需要退出 XRDP 并重新登录。验证：

```bash
id
```

输出中应包含 `plugdev`。

### 8.3 创建持久 udev 规则

使用管理员编辑器：

```bash
sudoedit /etc/udev/rules.d/61-vive-tracker.rules
```

写入：

```udev
# 本文件为 XRDP 会话中的 Vive Tracker Watchman Dongle 配置持久访问权限。
# 设备权限交由 plugdev 组管理，避免 uaccess 只授权本地物理座席用户。
SUBSYSTEM=="hidraw", KERNEL=="hidraw*", ATTRS{idVendor}=="28de", ATTRS{idProduct}=="2101", GROUP="plugdev", MODE="0660"
SUBSYSTEM=="usb", ENV{DEVTYPE}=="usb_device", ATTR{idVendor}=="28de", ATTR{idProduct}=="2101", GROUP="plugdev", MODE="0660"
```

重新加载规则：

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger --action=change --subsystem-match=usb
sudo udevadm trigger --action=change --subsystem-match=hidraw
```

也可以拔下并重新插入 Dongle。

### 8.4 验证设备节点

先查找对应的 `hidraw`：

```bash
for device in /dev/hidraw*; do
    udevadm info -q property -n "$device" 2>/dev/null \
        | grep -q 'HID_ID=0003:000028DE:00002101' \
        && echo "$device"
done
```

本次测试对应 `/dev/hidraw3`。检查权限：

```bash
ls -l /dev/hidraw3
getfacl /dev/hidraw3
```

预期设备组为 `plugdev`，权限为 `rw-rw----`。

如果还没有重新登录 XRDP，可以临时授权当前设备节点：

```bash
sudo setfacl -m u:"$USER":rw /dev/hidraw3 /dev/bus/usb/003/004
```

其中 `/dev/hidraw3` 和 `/dev/bus/usb/003/004` 必须根据当前 `lsusb` 和 `udevadm` 结果调整。重新插拔后临时 ACL 可能失效，持久方案仍是 `plugdev` + udev 规则 + 重新登录。

## 9. 安装和配置 SteamVR

### 9.1 安装 SteamVR

在 Steam 商店搜索 `SteamVR`，安装 App ID 250820。

可以通过以下命令查找 Steam 根目录：

```bash
STEAM_ROOT="$(readlink -f ~/.steam/root)"
STEAMVR_ROOT="$STEAM_ROOT/steamapps/common/SteamVR"
printf '%s\n' "$STEAM_ROOT" "$STEAMVR_ROOT"
```

当前机器得到：

```text
<STEAM_ROOT>
<STEAM_ROOT>/steamapps/common/SteamVR
```

### 9.2 强制使用 Steam Linux Runtime 3.0

在 Steam GUI 中操作：

1. 打开 Library。
2. 右键 SteamVR，选择 Properties。
3. 打开 Compatibility。
4. 勾选强制使用指定兼容工具。
5. 选择 `Steam Linux Runtime 3.0 (sniper)`。

当前配置内部名称为：

```text
steamlinuxruntime_sniper
```

### 9.3 配置无头显运行

推荐在用户配置中设置 `requireHmd`，用户配置更新后更容易保留：

```text
$(readlink -f ~/.steam/root)/config/steamvr.vrsettings
```

编辑前先退出 SteamVR：

```bash
nano "$(readlink -f ~/.steam/root)/config/steamvr.vrsettings"
```

如果文件只有基础配置，可以合并为以下结构：

```json
{
  "steamvr": {
    "requireHmd": false,
    "showAdvancedSettings": true
  }
}
```

如果文件已有 `DesktopUI`、`DismissedWarnings`、`installID` 等内容，只需在现有 `steamvr` 对象中加入：

```json
"requireHmd": false
```

注意 JSON 项目之间需要逗号，并且不能重复创建两个 `steamvr` 对象。

原始教程使用的默认配置文件是：

```text
$(readlink -f ~/.steam/root)/steamapps/common/SteamVR/resources/settings/default.vrsettings
```

其中对应项为：

```json
"requireHmd": false
```

SteamVR 更新可能覆盖 `default.vrsettings`，因此用户配置更适合作为长期配置。

### 9.4 在 XFCE 文件管理器中打开配置

1. 打开 Thunar 文件管理器。
2. 按 `Ctrl+H` 显示隐藏文件。
3. 按 `Ctrl+L` 打开位置输入框。
4. 输入 SteamVR 配置路径：

   ```text
   <STEAM_ROOT>/config/steamvr.vrsettings
   ```

5. 或打开 SteamVR 默认配置：

   ```text
   <STEAM_ROOT>/steamapps/common/SteamVR/resources/settings/default.vrsettings
   ```

对于普通原生安装，先执行 `readlink -f ~/.steam/root`，再将输出路径替换到上述位置。

### 9.5 修复 `SteamVR setup is incomplete`

典型提示：

```text
SteamVR setup is incomplete, some features might be missing.
```

日志中可能出现：

```text
pkexec must be setuid root
setcap of vrcompositor-launcher failed
```

手动授予实时调度能力：

```bash
STEAMVR_ROOT="$(readlink -f ~/.steam/root)/steamapps/common/SteamVR"
sudo setcap CAP_SYS_NICE=eip "$STEAMVR_ROOT/bin/linux64/vrcompositor-launcher"
```

验证：

```bash
getcap "$STEAMVR_ROOT/bin/linux64/vrcompositor-launcher"
```

预期输出：

```text
vrcompositor-launcher cap_sys_nice=eip
```

SteamVR 更新后如果提示再次出现，应重新检查 capability。

## 10. 启动 Steam 和 SteamVR

### 10.1 启动 Steam

退出后重新启动 Steam：

```bash
/usr/games/steam
```

也可以使用 XFCE 应用菜单中的 Steam 图标。

不要再使用：

```bash
snap run steam
```

### 10.2 启动 SteamVR

在 Steam Library 中点击 SteamVR 的 Play，或者执行：

```bash
/usr/games/steam 'steam://rungameid/250820'
```

检查进程：

```bash
pgrep -a -x steam
pgrep -a -x vrserver
pgrep -a -x vrmonitor
```

## 11. 配对 VIVE Tracker

### 11.1 准备设备

1. 插入 Watchman Dongle。
2. 启动原生 Steam 和 SteamVR。
3. 确认 Tracker 有电。
4. 每次只对一个 Tracker 和一个 Dongle 执行配对。

### 11.2 从 SteamVR GUI 配对

不同 SteamVR 版本的菜单文字可能略有差异，常见路径为：

```text
SteamVR 菜单 → Devices → Pair Controller
```

选择 VIVE Tracker，然后按界面提示操作。Tracker 进入配对时通常显示蓝色闪烁，连接成功后显示绿色。

### 11.3 `no pairable devices detected`

依次检查：

1. `lsusb` 是否有 `28de:2101`。
2. Steam 是否由 `/usr/games/steam` 启动。
3. `vrserver` 是否位于普通 session cgroup，而非 `snap.steam.steam-*.scope`。
4. Dongle 的 `hidraw` 节点是否归 `plugdev` 组。
5. 当前 XRDP 登录用户是否已经获得 `plugdev` 组。
6. Tracker 是否处于配对模式。
7. Dongle 是否已经连接了一个 Tracker。

本次实际情况中，修复 USB 访问后日志显示 Tracker 已经连接到接收器。此时配对窗口没有新的可配对设备属于正常状态。

## 12. 启动和布置基站

### 12.1 Base Station 2.0 开机

Base Station 2.0 没有独立电源按钮，连接原装适配器后自动启动。

状态灯含义：

| 状态灯 | 含义和处理 |
| --- | --- |
| 绿色或白色常亮 | 正常工作 |
| 蓝色或蓝色闪烁 | 正在稳定；长时间不变时检查安装牢固程度和振动 |
| 红色闪烁 | 硬件或固件错误 |
| 不亮 | 检查插座、适配器和电源接口 |

如果 SteamVR 电源管理曾让基站进入待机，可以先断电 10 秒，再重新连接。不同批次 Base Station 2.0 的待机灯效可能不同，判断工作状态时以绿色或白色常亮为准。

### 12.2 两个基站的安装要求

1. 安装在跟踪区域对角位置。
2. 正面朝向跟踪区域中心。
3. 使用稳定支架，避免桌面振动。
4. 保证 Tracker 与基站之间无遮挡。
5. 两个 Base Station 2.0 使用不同频道。
6. 基站启动完成后不要移动或改变角度。

Base Station 2.0 只需要电源。Tracker 通过红外扫描信号感知基站，基站不需要通过 USB 连接电脑。

### 12.3 如果使用 Base Station 1.0

- 单基站：设置为 `A`。
- 两个基站且没有同步线：设置为 `b` 和 `c`。
- 两个基站使用同步线：按设备说明设置 `A` 和 `b`。
- 暗绿色表示待机，紫色表示正在同步，紫色闪烁表示同步受阻。
- Base Station 1.0 和 2.0 不能混用于同一个跟踪区域。

### 12.4 唤醒 Tracker

短按 Tracker 电源键：

- 蓝灯表示正在连接。
- 绿灯表示已经连接。
- 长按通常用于关机或配对，应根据设备当前状态操作。

本次验证中，Tracker 进入待机后的日志为：

```text
Device LHR-B77A06A7 powering off upon entering standby.
Disconnected from receiver 0E36592B06
```

短按电源键后重新连接，并立即看到两个基站。

## 13. 最终验收

### 13.1 查找日志

```bash
VR_LOG="$(readlink -f ~/.steam/root)/logs/vrserver.txt"
printf '%s\n' "$VR_LOG"
```

实时查看：

```bash
tail -f "$VR_LOG"
```

筛选关键事件：

```bash
grep -E 'Lighthouse IMU HID opened|Connected to receiver|SOB: add|No base stations seen|Disconnected from receiver' "$VR_LOG"
```

如果安装了 ripgrep，也可以使用：

```bash
rg 'Lighthouse IMU HID opened|Connected to receiver|SOB: add|No base stations seen|Disconnected from receiver' "$VR_LOG"
```

### 13.2 成功标志

Dongle 成功打开：

```text
Lighthouse IMU HID opened
```

Tracker 成功连接：

```text
LHR-B77A06A7: Connected to receiver 0E36592B06
```

检测到一个 Base Station 2.0：

```text
SOB: add S-2
```

同时检测到两个 Base Station 2.0：

```text
SOB: add S-1 also seeing S-2
```

设备序列号会因硬件不同而变化，判断时重点查看事件文字。

## 14. 常见错误对照表

| 现象或日志 | 原因 | 处理方式 |
| --- | --- | --- |
| Snap 下载 `connection reset by peer` | `snapd` 没有使用 Shell 代理 | 使用 `sudo snap set system proxy.http=... proxy.https=...` |
| `is not a snap cgroup` | XRDP 私有 D-Bus 与 systemd 用户总线分离 | 使用第 6.2 节包装器；VIVE 最终运行使用原生 Steam |
| `Operation not permitted` 打开 `hidraw` | Snap 设备隔离 | 使用 `/usr/games/steam` 启动原生 Steam |
| `Permission denied` 打开 `hidraw` | XRDP 用户没有设备 ACL 或组权限 | 配置 udev、加入 `plugdev`、退出并重新登录 |
| `CHidDevice: Can't open USB device` | Dongle 权限或 Snap 隔离 | 先测试主机权限，再确认 Steam 进程不在 Snap cgroup |
| `SteamVR setup is incomplete` | `vrcompositor-launcher` 缺少 capability | 执行 `setcap CAP_SYS_NICE=eip` |
| `Please plug in your VR headset` | 当前没有 HMD | 设置 `requireHmd=false`；只使用 Tracker 时可忽略状态提示 |
| `no pairable devices detected` | Dongle 不可访问、Tracker 未进入配对或 Dongle 已被占用 | 检查 USB、udev、运行方式和 Tracker 状态 |
| `No base stations seen` | Tracker 没收到基站光学信号 | 开启基站、检查状态灯、朝向和遮挡 |
| `No optical frames in past 5 seconds` | 基站未扫描、Tracker 被遮挡或距离/角度不合适 | 重新布置基站和 Tracker，等待基站稳定 |
| Tracker 突然断开 | Tracker 关机、没电或进入待机 | 短按 Tracker 电源键并检查电量 |
| Steam 退出后不知道如何启动 | Snap 和原生入口混淆 | 使用 `/usr/games/steam` 或 XFCE 应用菜单 |

## 15. 更新后的复查事项

Steam 或 SteamVR 更新后建议检查：

```bash
getcap "$(readlink -f ~/.steam/root)/steamapps/common/SteamVR/bin/linux64/vrcompositor-launcher"
grep -n 'requireHmd' "$(readlink -f ~/.steam/root)/config/steamvr.vrsettings"
lsusb | grep -i '28de:2101'
id | grep plugdev
```

重点注意：

- SteamVR 更新可能覆盖 `default.vrsettings`。
- SteamVR 更新可能替换 `vrcompositor-launcher` 并清除 capability。
- USB Bus/Device 编号和 `hidrawN` 编号可能改变。
- 当前机器仍通过 Snap 用户数据目录保存 Steam 文件，迁移完成前保持 Steam Snap 为 disabled，不要 purge。

## 16. 参考资料

- [Valve Steam for Linux](https://github.com/ValveSoftware/steam-for-linux)
- [Valve SteamVR for Linux](https://github.com/ValveSoftware/SteamVR-for-Linux)
- [Steam Support：Index Base Station 与 Lighthouse Tracking](https://help.steampowered.com/en/faqs/view/1AF1-670B-FF5C-3323)
- [HTC VIVE：Base Station 2.0 安装说明](https://www.vive.com/us/support/vive-pro/category_howto/installing-the-base-stations.html)
- [HTC VIVE：基站状态灯说明](https://business.vive.com/eu/support/vive-pro2/faq/what-base-station-status-lights-means.html)
