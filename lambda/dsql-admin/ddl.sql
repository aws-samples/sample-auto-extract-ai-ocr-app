-- ========================================
-- Aurora DSQL DDL
-- 各 CREATE TABLE / CREATE INDEX は個別トランザクションで実行すること
-- FK 制約は非サポート。参照関係はコメントで明記。
-- ========================================

-- users
CREATE TABLE users (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cognito_sub    VARCHAR(255) NOT NULL UNIQUE,
    email          VARCHAR(255) NOT NULL UNIQUE,
    display_name   VARCHAR(255),
    department     VARCHAR(255),
    role           VARCHAR(20) NOT NULL DEFAULT 'reader',
    is_active      BOOLEAN NOT NULL DEFAULT true,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- groups
CREATE TABLE groups (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name           VARCHAR(255) NOT NULL UNIQUE,
    description    TEXT,
    source         VARCHAR(20) NOT NULL DEFAULT 'manual',
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- usecases (RBAC 権限管理用。OCR設定は SchemasTable に残す)
-- owner は user_usecases (permission='owner') で管理。複数 owner 対応。
CREATE TABLE usecases (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    app_name       VARCHAR(255) NOT NULL UNIQUE,
    created_by     UUID NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- tools
CREATE TABLE tools (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name           VARCHAR(255) NOT NULL UNIQUE,
    description    TEXT,
    is_active      BOOLEAN NOT NULL DEFAULT true
);

-- user_groups
CREATE TABLE user_groups (
    user_id        UUID NOT NULL,
    group_id       UUID NOT NULL,
    source         VARCHAR(20) NOT NULL DEFAULT 'manual',
    synced_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, group_id)
);

-- group_usecases
CREATE TABLE group_usecases (
    group_id       UUID NOT NULL,
    usecase_id     UUID NOT NULL,
    permission     VARCHAR(20) NOT NULL DEFAULT 'viewer',
    PRIMARY KEY (group_id, usecase_id)
);

-- user_usecases
CREATE TABLE user_usecases (
    user_id        UUID NOT NULL,
    usecase_id     UUID NOT NULL,
    permission     VARCHAR(20) NOT NULL DEFAULT 'viewer',
    PRIMARY KEY (user_id, usecase_id)
);

-- group_tools
CREATE TABLE group_tools (
    group_id       UUID NOT NULL,
    tool_id        UUID NOT NULL,
    PRIMARY KEY (group_id, tool_id)
);

-- user_tools
CREATE TABLE user_tools (
    user_id        UUID NOT NULL,
    tool_id        UUID NOT NULL,
    PRIMARY KEY (user_id, tool_id)
);

-- usecase_tools
CREATE TABLE usecase_tools (
    usecase_id     UUID NOT NULL,
    tool_id        UUID NOT NULL,
    PRIMARY KEY (usecase_id, tool_id)
);

-- migrations
ALTER TABLE tools DROP COLUMN IF EXISTS tool_name;

-- indexes (CREATE INDEX ASYNC)
CREATE INDEX ASYNC idx_usecases_created_by ON usecases(created_by);
CREATE INDEX ASYNC idx_user_usecases_usecase ON user_usecases(usecase_id);
CREATE INDEX ASYNC idx_group_usecases_usecase ON group_usecases(usecase_id);
CREATE INDEX ASYNC idx_group_tools_group ON group_tools(group_id);
CREATE INDEX ASYNC idx_group_tools_tool ON group_tools(tool_id);
CREATE INDEX ASYNC idx_user_tools_user ON user_tools(user_id);
CREATE INDEX ASYNC idx_user_tools_tool ON user_tools(tool_id);
CREATE INDEX ASYNC idx_usecase_tools_usecase ON usecase_tools(usecase_id);
CREATE INDEX ASYNC idx_usecase_tools_tool ON usecase_tools(tool_id);
