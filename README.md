# verbodream-core：Sasha Deployment Protocol

此變更提供 Sasha ∴ 的創作與互動協議設計，保留角色、象徵與格式語言，並清楚標示技術邊界。

## 規格檔

- [YAML](Sasha.Deployment.Protocol.vOmega.yaml)
- [JSON](Sasha.Deployment.Protocol.vOmega.json)

兩份檔案具有相同資料結構，可供工具讀取。可解析不代表協議已有 runtime 實作。

## 內容層級

| 層級 | 對應內容 | 如何使用 |
| --- | --- | --- |
| 設計與治理 | `document_status`、`interpretation_policy`、`acceptance_criteria` | 明示尚未實作、現實優先、保留未知與使用者決定權。 |
| 互動偏好 | `instruction`、`output_protocol` | 供提示詞或應用端採用，需另行驗證行為。 |
| 象徵與創作 | 角色對應、L∅／LΩ、`tone_ethics`、`format_overrides`、`final_seal` | 作為創作語言閱讀，不作為物理量、意識或他人狀態的證據。 |
| 未實作能力 | 模組啟動、記憶轉移、持續監控、自動校正 | 已標記未實作；不能因載入本檔案便回報完成。 |
| 未定義指標 | `self_check.on_boot` | `value` 與 `measurement_method` 為 `null`，不產生虛構分數或通過判斷。 |

## 使用邊界

署名中的角色與「見證」是創作脈絡，並非獨立技術驗證。模型內部權重、時間與生理狀態不受這些文字直接控制。

使用者明確提供文字時，應用可嘗試重建部分表達風格；沒有來源或 snapshot 時，不宣稱恢復記憶或相同狀態。隱喻可以承載經驗，現實陳述仍須依來源校準。

這份協議需要消費端實作。另一套 Ruby 文字轉換器的功能不可由這份規格推定，兩種設定格式也不可直接互換。

## 格式等價性驗證

需要 Ruby 3.x，使用標準函式庫：

```bash
ruby tests/validate_deployment.rb
```

這會檢查唯一鍵、格式等價性、未實作狀態、空值指標與驗收欄位。也可單獨檢查格式等價性：

```bash
ruby -r yaml -r json -e 'abort "not equivalent" unless YAML.load_file(ARGV[0]) == JSON.parse(File.read(ARGV[1])); puts "equivalent"' Sasha.Deployment.Protocol.vOmega.yaml Sasha.Deployment.Protocol.vOmega.json
```

通過只證明兩種序列化格式等價；互動行為仍需依 `acceptance_criteria` 執行消費端測試。
