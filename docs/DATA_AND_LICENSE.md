# 数据、来源与公开发布检查

Responsibility 的代码和数据应分开发布。GitHub 代码仓保存实现、配置、manifest schema 和校验脚本；大体积数据通过版本化资产单独提供。

## 发布数据前必须确认

1. 原始行情和指数成分的来源；
2. 是否允许公开再分发原始数据及其派生文件；
3. 是否要求署名、限制商业用途或限制适用地域；
4. sampler、责任特征和 Qlib provider 是否包含可还原的第三方数据；
5. 归档是否包含私有路径、用户名、访问令牌或日志中的凭据；
6. 每个文件是否有大小和 SHA-256，并与 `data/manifest.json` 一致。

如果任一数据资产的再分发权不明确，不要把它上传到公开 Release。可以只发布构建脚本和校验清单，让有合法数据访问权的用户在本地生成。

## 推荐资产命名

```text
responsibility-data-v1-csi300.tar.gz
responsibility-data-v1-csi800-direct.tar.gz
responsibility-data-v1-features.tar.gz
responsibility-data-v1-qlib-provider.tar.gz
responsibility-data-v1-manifest.tar.gz
```

各归档必须使用互不冲突的相对路径，且最终解压后形成 `data/README.md` 所述布局。每次发布新数据都增加版本号，不原地替换同名资产。

## GitHub 边界

普通 Git 历史不适合存放数百 MB 至数 GB 的 pickle。可选择：

- GitHub Release 分卷资产；
- Git LFS；
- 具有版本和校验能力的对象存储；
- 仅公开数据构建脚本，由用户本地生成。

无论采用哪种方式，都应把 URL 作为命令行参数传给 `scripts/download_data.py`，不要将短期签名 URL、Token 或 Cookie 提交到仓库。
