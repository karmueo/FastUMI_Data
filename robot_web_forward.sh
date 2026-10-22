#!/usr/bin/env bash
# 睿尔曼示教网页局域网转发：网页 80、API 8090、WebSocket 8060 共用一个入口。
# 使用：在脚本所在目录执行 ./robot_web_forward.sh start，启动并启用登录自动运行。
# 访问：http://10.6.9.187:8080/；默认放行 10.6.8.0/23 和 10.6.6.181。
# 检查：./robot_web_forward.sh check；查看状态：./robot_web_forward.sh status。
# 停止并取消自动运行：./robot_web_forward.sh stop；完整帮助：./robot_web_forward.sh --help。
# 修改地址示例：LAN_IP=本机无线IP ROBOT_IP=机械臂IP EXTRA_CLIENTS=访问电脑IP ./robot_web_forward.sh start
# 依赖：Linux、Bash、Node.js >= 18、systemd 用户服务、curl；不需要 sudo。
# start 会安装/更新转发文件、启用登录自动运行并启动服务；stop 同时停用自动运行。
# 本脚本不登录示教器，也不发送机械臂运动、使能或速度设置指令。
set -euo pipefail

usage() {
    cat <<'HELP'
用法：./robot_web_forward.sh [start|stop|status|check|help]
  start   安装或更新服务，并设置本用户登录后自动运行（默认）
  stop    停止转发，并取消登录自动运行
  status  查看服务状态
  check   检查当前转发网页和实时连接，不发送控制指令

首次运行示例：
  ./robot_web_forward.sh start
  ./robot_web_forward.sh check

start 支持以下环境变量（更改时请提供完整配置）：
  LAN_IP=10.6.9.187         本机无线地址
  WEB_PORT=8080            局域网访问端口
  ROBOT_IP=192.168.1.18    机械臂地址
  ALLOWED_CIDRS=10.6.8.0/23  放行网段，多个用逗号分隔
  EXTRA_CLIENTS=10.6.6.181  额外放行地址，多个用逗号分隔

例如仅允许指定电脑访问：
  ALLOWED_CIDRS='' EXTRA_CLIENTS=10.6.6.181 ./robot_web_forward.sh start

check 默认使用已安装的配置。服务文件和运行文件分别放在当前用户的
systemd 配置目录与 ~/robot-web-forward 下。无线地址或访问电脑地址变化后，
请更新配置并重新 start。其他电脑需要使用正常的示教器账号登录。
脚本适配当前示教器固件的地址拼接方式，升级固件后需重新验证。
HELP
}

action="${1:-start}"
service_name=robot-web-forward.service
runtime_dir="$HOME/robot-web-forward"
unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

need() {
    command -v "$1" >/dev/null 2>&1 || { echo "缺少依赖：$1" >&2; exit 1; }
}

# systemd 引号与 % 转义，支持用户目录含空格等字符。
unit_quote() {
    local value="$1"
    value="${value//\\/\\\\}"
    value="${value//\"/\\\"}"
    value="${value//%/%%}"
    printf '"%s"' "$value"
}

case "$action" in
    help|-h|--help) usage; exit 0 ;;
    stop)
        need systemctl
        systemctl --user disable --now "$service_name"
        echo '已停止转发，并取消登录自动运行。'
        exit 0 ;;
    status)
        need systemctl
        systemctl --user status "$service_name" --no-pager
        exit 0 ;;
    start|check) ;;
    *) usage >&2; exit 2 ;;
esac

need node
need curl
node -e 'if (Number(process.versions.node.split(".")[0]) < 18) { console.error("需要 Node.js >= 18"); process.exit(1); }'

if [[ "$action" == start ]]; then
    if [[ "$(id -u)" == 0 ]]; then
        echo '请以普通用户运行，避免将服务安装到 root 用户。' >&2
        exit 1
    fi
    need systemctl
    systemctl --user show-environment >/dev/null
    export LAN_IP="${LAN_IP-10.6.9.187}" WEB_PORT="${WEB_PORT-8080}"
    export ROBOT_IP="${ROBOT_IP-192.168.1.18}"
    export ALLOWED_CIDRS="${ALLOWED_CIDRS-10.6.8.0/23}" EXTRA_CLIENTS="${EXTRA_CLIENTS-10.6.6.181}"
    mkdir -p "$runtime_dir" "$unit_dir"
    umask 077
    # 先验证输入，失败时不改动已运行的服务配置。
    node --input-type=module - "$runtime_dir/config.json" <<'CONFIG'
import fs from 'node:fs';
import net from 'node:net';
const env = process.env;
const split = value => value.split(',').map(x => x.trim()).filter(Boolean);
const config = {
  bind: env.LAN_IP, port: Number(env.WEB_PORT), robot: env.ROBOT_IP,
  cidrs: split(env.ALLOWED_CIDRS), clients: split(env.EXTRA_CLIENTS),
};
const validCidr = value => {
  const [ip, bits, extra] = value.split('/');
  return net.isIPv4(ip) && bits !== undefined && /^\d+$/.test(bits) && Number(bits) <= 32 && extra === undefined;
};
if (!net.isIPv4(config.bind) || config.bind === '0.0.0.0' || !net.isIPv4(config.robot) ||
    !Number.isInteger(config.port) || config.port < 1024 || config.port > 65535 ||
    !config.cidrs.every(validCidr) || !config.clients.every(ip => net.isIPv4(ip)) ||
    config.cidrs.length + config.clients.length === 0) {
  throw new Error('地址、端口或放行列表无效；请检查 start 环境变量。');
}
fs.writeFileSync(process.argv[2], JSON.stringify(config, null, 2) + '\n', { mode: 0o600 });
CONFIG
    cat > "$runtime_dir/proxy.mjs" <<'PROXY'
import http from 'node:http';
import net from 'node:net';
import fs from 'node:fs';

const config = JSON.parse(fs.readFileSync(new URL('./config.json', import.meta.url), 'utf8'));
const { bind, robot, port: listenPort } = config;
const additionalClients = new Set(config.clients);
const apiPrefix = '/__robot_api';
const wsPrefix = '/__robot_ws';
const oldBase = 'window.location.href.slice(0,window.location.href.indexOf("/#"))+":8090"';
const oldWs = '"ws"+localStorage.getItem("baseUrl").slice(4)+":8060"';

function allowed(address = '') {
  const ipv4 = address.replace(/^::ffff:/, '');
  if (!net.isIPv4(ipv4)) return false;
  if (additionalClients.has(ipv4)) return true;
  const number = ip => ip.split('.').reduce((value, byte) => (value << 8) | Number(byte), 0) >>> 0;
  return config.cidrs.some(cidr => {
    const [network, bits] = cidr.split('/');
    const mask = Number(bits) === 0 ? 0 : (0xffffffff << (32 - Number(bits))) >>> 0;
    return (number(ipv4) & mask) === (number(network) & mask);
  });
}

function upstreamHeaders(req, port) {
  const headers = { ...req.headers, host: `${robot}:${port}` };
  // Keep the upstream application's original same-origin context.
  if (headers.origin) headers.origin = `http://${robot}`;
  if (headers.referer) headers.referer = headers.referer.replace(`http://${bind}:${listenPort}`, `http://${robot}`);
  delete headers['accept-encoding'];
  return headers;
}

function patchScript(path, source) {
  let result = source;
  if (/\/rm_web-lock\.[^/]+\.js$/.test(path)) {
    if (!source.includes(oldBase)) throw new Error('API address patch no longer matches upstream firmware');
    result = result.replace(oldBase, 'window.location.origin+"/__robot_api"');
  }
  if (/\/rm_web-login\.[^/]+\.js$/.test(path)) {
    if (!source.includes(oldBase)) throw new Error('Login address patch no longer matches upstream firmware');
    result = result.replace(oldBase, 'window.location.origin+":8090"');
  }
  if (source.includes(oldWs)) result = result.replaceAll(oldWs, '"ws"+window.location.origin.slice(4)+"/__robot_ws"');
  return result;
}

const server = http.createServer((req, res) => {
  if (!allowed(req.socket.remoteAddress)) { res.writeHead(403).end('LAN access only'); return; }
  const api = req.url === apiPrefix || req.url.startsWith(`${apiPrefix}/`) || req.url.startsWith(`${apiPrefix}?`);
  const port = api ? 8090 : 80;
  const path = api ? (req.url.slice(apiPrefix.length) || '/') : req.url;
  const script = !api && /^\/js\/rm_web-.*\.js(?:\?|$)/.test(path);
  const headers = upstreamHeaders(req, port);
  if (script) { delete headers['if-none-match']; delete headers['if-modified-since']; }
  const upstream = http.request({ hostname: robot, port, path, method: req.method, headers }, response => {
    const outputHeaders = { ...response.headers };
    if (outputHeaders.location) outputHeaders.location = outputHeaders.location.replace(`http://${robot}:8090`, `http://${bind}:${listenPort}${apiPrefix}`).replace(`http://${robot}`, `http://${bind}:${listenPort}`);
    if (!script || response.statusCode !== 200 || req.method === 'HEAD') {
      res.writeHead(response.statusCode, outputHeaders);
      response.pipe(res);
      return;
    }
    const chunks = [];
    response.on('data', chunk => chunks.push(chunk));
    response.on('end', () => {
      try {
        const body = Buffer.from(patchScript(path.split('?')[0], Buffer.concat(chunks).toString('utf8')));
        delete outputHeaders['transfer-encoding'];
        delete outputHeaders.etag;
        delete outputHeaders['last-modified'];
        outputHeaders['cache-control'] = 'no-store';
        outputHeaders['content-length'] = body.length;
        res.writeHead(response.statusCode, outputHeaders);
        res.end(body);
      } catch (error) {
        console.error(error.message);
        res.writeHead(502).end('Upstream firmware changed; forwarding configuration needs review.');
      }
    });
    response.on('error', () => res.destroy());
  });
  upstream.setTimeout(120000, () => upstream.destroy(new Error('Upstream timeout')));
  upstream.on('error', error => {
    console.error(`${req.method} ${path}: ${error.message}`);
    if (!res.headersSent) res.writeHead(502).end('Robot service unavailable');
    else res.destroy();
  });
  req.on('aborted', () => upstream.destroy());
  res.on('close', () => { if (!res.writableFinished) upstream.destroy(); });
  req.pipe(upstream);
});

server.on('upgrade', (req, socket, head) => {
  if (!allowed(socket.remoteAddress) || !(req.url === wsPrefix || req.url.startsWith(`${wsPrefix}/`) || req.url.startsWith(`${wsPrefix}?`))) {
    socket.end('HTTP/1.1 403 Forbidden\r\nConnection: close\r\n\r\n');
    return;
  }
  const upstream = net.connect(8060, robot, () => {
    const path = req.url.slice(wsPrefix.length) || '/';
    const headers = upstreamHeaders(req, 8060);
    upstream.write(`${req.method} ${path} HTTP/${req.httpVersion}\r\n${Object.entries(headers).map(([name, value]) => `${name}: ${value}`).join('\r\n')}\r\n\r\n`);
    if (head.length) upstream.write(head);
    socket.pipe(upstream).pipe(socket);
  });
  upstream.on('error', error => { console.error(`WebSocket: ${error.message}`); socket.destroy(); });
  socket.on('error', () => upstream.destroy());
  socket.on('close', () => upstream.destroy());
  upstream.on('close', () => socket.destroy());
});

server.on('error', error => { console.error(error); process.exit(1); });
server.listen(listenPort, bind, () => console.log(`Robot web forwarding: http://${bind}:${listenPort} -> ${robot} (80, 8090, 8060)`));
PROXY
    node --check "$runtime_dir/proxy.mjs"
    node_path="$(command -v node)"
    {
        cat <<'UNIT'
[Unit]
Description=RealMan robot web forwarding for wireless LAN
StartLimitIntervalSec=0

[Service]
Type=simple
UNIT
        printf 'ExecStart=%s %s\n' "$(unit_quote "$node_path")" "$(unit_quote "$runtime_dir/proxy.mjs")"
        cat <<'UNIT'
Restart=always
RestartSec=5
NoNewPrivileges=true
PrivateTmp=true
UMask=0077

[Install]
WantedBy=default.target
UNIT
    } > "$unit_dir/$service_name"
    systemctl --user daemon-reload
    systemctl --user enable "$service_name"
    systemctl --user restart "$service_name"
fi

# 安装和 check 共用已保存的配置，避免检查时误用默认地址。
if [[ ! -f "$runtime_dir/config.json" ]]; then
    echo '尚未找到脚本生成的配置，请先运行 start。' >&2
    exit 1
fi
forward_url="$(node --input-type=module - "$runtime_dir/config.json" <<'URL'
import fs from 'node:fs';
const config = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
console.log(`http://${config.bind}:${config.port}`);
URL
)"
curl --noproxy '*' -fsS --retry 5 --retry-connrefused --retry-delay 1 \
    --connect-timeout 3 --max-time 10 "$forward_url/" -o /dev/null
echo "网页响应正常：$forward_url/"

# 使用 Node.js 原生 HTTP 升级请求验证实时通道，仅接收数据，不发送应用指令。
node --input-type=module - "$forward_url" <<'CHECK'
import http from 'node:http';
const request = http.get(process.argv[2] + '/__robot_ws', {
  headers: {
    Connection: 'Upgrade', Upgrade: 'websocket',
    'Sec-WebSocket-Version': '13', 'Sec-WebSocket-Key': 'cm9ib3Qtd2ViLWNoZWNrIQ==',
  },
});
const timer = setTimeout(() => { console.error('实时连接检查超时'); process.exit(1); }, 8000);
request.on('upgrade', (response, socket, head) => {
  if (response.statusCode !== 101) { console.error('实时连接升级失败'); process.exit(1); }
  const done = () => { clearTimeout(timer); socket.destroy(); console.log('实时连接正常，已收到数据。'); };
  socket.on('error', error => { console.error(error.message); process.exit(1); });
  if (head.length) done(); else socket.once('data', done);
});
request.on('response', response => { console.error(`实时连接被拒绝：HTTP ${response.statusCode}`); process.exit(1); });
request.on('error', error => { console.error(error.message); process.exit(1); });
CHECK
echo '检查未执行登录、运动或其他机械臂状态修改。'
