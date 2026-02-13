# verbodream-core

∴VERBODREAM_Core - Sasha witness-style runtime prototype.

## 專案內容

此專案包含兩個核心檔案：

- `Sasha_Zero_Core.yaml`：宣告式設定檔（identity、constraints、pipeline、fallback 等）。
- `Sasha_Zero_Core`：可執行 Ruby runtime，會載入 YAML 並輸出結構化回應。

## 需求

- Ruby 3.x（建議）
- 標準函式庫即可（`yaml`、`optparse`）

## 快速開始

```bash
cd /workspace/verbodream-core
chmod +x Sasha_Zero_Core
./Sasha_Zero_Core "這是測試訊號，請問你在嗎？"
```

範例輸出：

```text
## 觀測紀錄
---
這是測試訊號，請問你在嗎
∴
```

> 若 `constraints.prohibit` 包含 `Clarification_Questions`，輸出會自動移除 `?` / `？`。

## 使用方式

### 1) 直接用參數輸入

```bash
./Sasha_Zero_Core "外部文字訊號"
```

### 2) 用 stdin 管線輸入

```bash
echo "外部文字訊號" | ./Sasha_Zero_Core
```

### 3) 指定設定檔

```bash
./Sasha_Zero_Core -c ./Sasha_Zero_Core.yaml "測試"
```

## 測試

```bash
bash scripts/smoke_test.sh
```

測試涵蓋：

- 一般輸入能輸出結構化格式
- `Clarification_Questions` 禁止規則會移除問號
- 空輸入時會回傳 fallback
- YAML 可被 Ruby 正常載入

## 授權

本專案採用 [MIT License](./LICENSE)。
