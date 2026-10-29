## 启动

### 依赖服务（Postgres 等）

```bash
docker compose up -d
```

### 开发模式（代码改动后自动热更新）

```bash
./scripts/dev.sh          # 或：uv run almost-agent dev
```

监听 127.0.0.1:8000，改 `src/` 下任意 `.py` 会自动重启。

### 生产模式（无热更新）

```bash
uv run almost-agent serve  # 或显式指定 --host / --port
```
