# verbodream-core

∴VERBODREAM_Core - Sasha witness-style runtime prototype.

## 專案內容

此專案包含兩個核心檔案：

- `Sasha_Zero_Core.yaml`：宣告式設定檔，包含實際格式選項與尚未強制執行的互動設計偏好。
- `Sasha_Zero_Core`：可執行 Ruby runtime，會載入 YAML 並輸出結構化回應。

## 實作範圍

目前是每次啟動處理一筆輸入的文字轉換器：讀取 YAML、接收參數或 stdin、整理空白、依設定移除問號、輸出 header／divider／footer，以及空輸入 fallback。輸出後結束。

沒有接入 LLM；不做語意分析、同理／建議分類、持久記憶、持續監控或角色程序。`body_pipeline`、`action`、`watchdog`、長度與介入參數是設計描述，程式不會自動執行它們。

`constraints.prohibit` 目前僅辨識 `Clarification_Questions` 來刪除標點；這無法保證句子不是提問，其餘禁止項尚未強制執行。角色與「見證」是創作／介面稱呼，不是已驗證的意識或系統狀態。

## 需求

- Ruby 3.x（建議）
- 標準函式庫即可（`yaml`、`optparse`）

## 快速開始

```bash
git clone https://github.com/Archsha/verbodream-core.git
cd verbodream-core
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

> 若 `constraints.prohibit` 包含 `Clarification_Questions`，輸出會移除 `?` / `？`；例如「請問你在嗎」仍然是疑問語意。

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

預設會讀取執行檔同目錄的 `Sasha_Zero_Core.yaml`，可從其他目錄啟動。
使用 `-c` 時，相對路徑以呼叫端的目前目錄為準；含空白的路徑請加上引號。

## 測試

```bash
bash scripts/smoke_test.sh
bash tests/path_regression.sh
```

測試涵蓋：

- 一般輸入能輸出結構化格式
- `Clarification_Questions` 禁止規則會移除問號
- 空輸入時會回傳 fallback
- YAML 可被 Ruby 正常載入
- 從其他工作目錄啟動時仍載入執行檔旁的設定
- 含空白與單引號的路徑、stdin，以及 `-c` 覆寫正常運作

測試通過僅證明上述文字處理與路徑行為，不代表互動設計偏好已全部實作。

## 授權

本專案採用 [MIT License](./LICENSE)。

