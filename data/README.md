# 数据目录

此目录只在 Git 中保留说明文件。训练 sampler、责任特征和 Qlib provider 体积较大，应通过版本化数据归档下载，并由 `manifest.json` 绑定身份。

Python pickle 不是安全的交换格式，反序列化不可信 pickle 可能执行任意代码。只从已确认身份的发布者获取数据，并在训练前核对归档和文件 SHA-256；校验值只能确认文件身份，不能替代对发布来源的信任。

## 预期布局

```text
data/
├── manifest.json
├── samplers/
│   ├── short_csi300/
│   │   ├── train.pkl
│   │   ├── valid.pkl
│   │   └── test.pkl
│   └── short_csi800_direct/
│       ├── train.pkl
│       ├── valid.pkl
│       └── test.pkl
├── responsibility/
│   ├── short_csi300/
│   │   ├── manifest.json
│   │   ├── train/{stock_state.npy,rule_components.npy}
│   │   ├── valid/{stock_state.npy,rule_components.npy}
│   │   └── test/{stock_state.npy,rule_components.npy}
│   └── short_csi800_direct/
│       └── ...
├── direction_scales/
│   ├── short_csi300.json
│   └── short_csi800_direct.json
└── qlib/
    ├── PROVIDER_MANIFEST.json
    └── cn_data/
        ├── calendars/
        ├── instruments/
        └── features/
```

两个 sampler 都使用 8 日窗口，每个样本最后一维共 222 个字段：158 个股因子、63 个市场特征和 1 个标签。责任特征与 sampler 的行顺序必须由相同 index SHA-256 绑定，不能仅按行数猜测对齐关系。

## `manifest.json` 格式

可复制 `manifest.example.json` 作为发布清单起点。顶层至少包含数据集路径映射：

```json
{
  "schema_version": "responsibility-data-manifest-v1",
  "data_version": "responsibility-short-v1",
  "archive_url": "<VERSIONED_DATA_ARCHIVE_URL>",
  "archive_sha256": "<OPTIONAL_ARCHIVE_SHA256>",
  "datasets": {
    "short_csi300": {
      "samplers": {
        "train": "samplers/short_csi300/train.pkl",
        "valid": "samplers/short_csi300/valid.pkl",
        "test": "samplers/short_csi300/test.pkl"
      },
      "responsibility_root": "responsibility/short_csi300",
      "direction_scale": "direction_scales/short_csi300.json"
    }
  },
  "qlib_provider": {
    "path": "qlib/cn_data",
    "manifest": "qlib/PROVIDER_MANIFEST.json"
  },
  "files": [
    {
      "path": "samplers/short_csi300/train.pkl",
      "bytes": 802335705,
      "sha256": "<64 lowercase hex characters>",
      "required": true
    }
  ]
}
```

约束如下：

- `datasets.<id>.samplers`、`responsibility_root` 和 `direction_scale` 是训练入口使用的稳定路径映射；
- `path` 必须是相对于 `data/` 的普通文件路径，不能是绝对路径，不能包含 `..`；
- `bytes` 是非负整数；
- `sha256` 若存在，必须是文件内容的 64 位十六进制摘要；初期清单允许只填写 `bytes`/`shape`，正式公开资产建议补齐全部哈希；
- `required` 缺省为 `true`；
- `files` 应覆盖所有科学计算输入，而不只是几个目录锚点；
- manifest 中不得记录访问令牌、Cookie 或机器私有路径。

`python scripts/verify_data.py --data-root data` 会检查 schema、固定必需路径、大小和 SHA-256。`--quick` 仅跳过内容哈希，不跳过必需文件和大小检查。

## 数据包拆分

如托管平台限制单文件大小，可以拆成以下互不覆盖的归档：

1. CSI300 samplers；
2. CSI800-Direct samplers；
3. 两个数据集的责任特征与 direction scales；
4. Qlib 最小回测 provider；
5. 最终 `manifest.json`。

每个归档内部都使用相对于本目录的路径。下载器禁止路径穿越、链接文件和覆盖既有文件，因此多个归档必须拥有互不冲突的 payload。

## 许可边界

代码的 MIT License 不自动覆盖行情、指数成分、sampler 或 provider。公开数据归档前必须单独确认数据来源和再分发权，并在归档中附数据来源、生成过程、版本和许可记录。
