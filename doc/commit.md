## 概述

约定式提交（Conventional Commits）是一种为 Git 提交信息定义的轻量级规范，通过结构化的提交消息，使提交历史既便于阅读，也利于解析，以支持自动生成 CHANGELOG、自动化发布和语义化版本控制（SemVer）。

- **MAJOR**（主版本）：重大不兼容变更。
- **MINOR**（次版本）：向现有功能添加新功能。
- **PATCH**（补丁版本）：修复缺陷。

## 提交消息格式

所有提交消息**必须**遵循以下格式：

```text
<type>[optional scope]: <description>

[optional body]

[optional footer]
```

1. `<type>`：提交类型，必须为以下之一的小写英文单词，后接冒号和空格：

   - `feat`：新功能，对应 MINOR 版本升级；
   - `fix`：修复缺陷，对应 PATCH 版本升级；
   - `docs`：仅文档变更；
   - `style`：代码格式变更，不影响逻辑；
   - `refactor`：重构代码，不新增功能也不修复缺陷；
   - `perf`：性能优化；
   - `test`：新增或修改测试；
   - `build`：影响构建或依赖；
   - `ci`：持续集成配置变更；
   - `chore`：其他杂务（如依赖更新、脚本改动）；
   - `revert`：回滚先前提交。

2. `[optional scope]`：可选，位于类型后圆括号内，用于说明影响的模块或子系统。

3. `<description>`：简要描述，动词开头，全小写，不超过 50 个字符。

4. `[optional body]`：可选，空一行后，在描述下面详细阐述变更动机、实现细节或对比上下文，行宽建议不超过 72 个字符。

5. `[optional footer]`：可选，空一行后，用于列出元信息，比如：
   - `BREAKING CHANGE: <description>` —— 描述重大不兼容变更；
   - `Closes #<issue>` —— 自动关闭 Issue；
   - `Reviewed-by: <name>` 等标准 Git trailer。

> 如引入不兼容变更，也可在 `<type>` 或 `scope` 后添加 `!`，例如 `feat!: ...` 或 `feat(parser)!: ...`。

## 类型与语义对应

| 类型     | 含义          | SemVer | 示例                                   |
| -------- | ------------- | ------ | -------------------------------------- |
| feat     | 新功能        | MINOR  | `feat: 添加用户登录功能`               |
| fix      | 修复缺陷      | PATCH  | `fix: 修复注册接口密码验证错误`        |
| docs     | 文档变更      | —      | `docs: 更新 README 安装说明`           |
| style    | 代码格式调整  | —      | `style: 统一缩进为四个空格`            |
| refactor | 重构代码      | —      | `refactor: 拆分订单处理模块`           |
| perf     | 性能优化      | PATCH  | `perf: 缓存 API 响应减少数据库访问`    |
| test     | 测试相关      | —      | `test: 为订单模块添加单元测试`         |
| build    | 构建/依赖变更 | —      | `build: 升级 Node.js 镜像至 18-alpine` |
| ci       | CI 配置变更   | —      | `ci: 添加 GitHub Actions 流水线`       |
| chore    | 杂务          | —      | `chore: 同步依赖、更新 lock 文件`      |
| revert   | 回滚提交      | —      | `revert: feat(auth): 添加社交登录`     |

## 示例演示

### init：初始化项目

```bash
git commit -m "init: 初始化项目仓库，添加 README 和目录结构"
```

### feat：新增功能

```bash
# 简单
git commit -m "feat: 支持用户使用 OAuth2 登录"

# 带 scope 和 body
git commit -m "feat(auth): 添加第三方 OAuth2 登录支持

- 支持 GitHub、Google 登录
- 更新用户模型，记录登录来源

Closes #42"
```

### fix：修复缺陷

```bash
# 简单
git commit -m "fix: 修复登录超时未清理会话问题"

# 带 Reviewed-by
git commit -m "fix(auth): 修复 token 过期处理逻辑

- 重构校验方法，避免重复请求

Reviewed-by: 张三"
```

### docs & style：文档与格式

```bash
git commit -m "docs: 补充 API 调用示例"

git commit -m "style: 移除多余分号，统一代码风格"
```

### refactor & perf：重构与优化

```bash
git commit -m "refactor: 拆分用户服务模块，提取公共方法"

git commit -m "perf(cache): 添加 Redis 缓存，提升查询效率"
```

### test：测试相关

```bash
git commit -m "test: 为用户模块添加集成测试"
```

### build & ci：构建与 CI

```bash
git commit -m "build: 多阶段构建优化 Dockerfile"

git commit -m "ci: 修复 Windows Runner 下路径兼容问题"
```

### chore & revert：杂务与回滚

```bash
git commit -m "chore: 更新依赖，重新生成 lock 文件"

# 回滚提交
git revert abc1234
```

---

遵循 [Conventional Commits v1.0.0 中文规范](https://www.conventionalcommits.org/zh-hans/v1.0.0/)。
