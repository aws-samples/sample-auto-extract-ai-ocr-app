---
inclusion: always
---

# Product Overview

## プロダクト名
AutoExtract - AI-OCR 帳票読み取りソリューション

## 概要
OCR + Amazon Bedrock を活用し、帳票からの情報抽出を半自動化するWebアプリケーション。人間によるデータ入力チェックをサポートする。

## 主要機能
- 帳票画像のアップロード・管理
- OCR による文字認識（PaddleOCR / DeepSeek OCR / Yomitoku）
- Bedrock (Claude) による構造化データ抽出
- スキーマ定義による抽出項目のカスタマイズ
- 抽出結果の人間によるレビュー・修正
- AI Agent による抽出結果の自動検証・補正（Experimental）
- S3 同期によるバッチ処理
- ユースケース単位の共有・権限管理（owner/editor/viewer、複数 owner 対応）
- 管理画面（ユーザー・グループ・ユースケース・ツール・画像の一元管理）

## ユーザー
- 帳票データの入力・確認を行う業務担当者
- スキーマ定義を管理する管理者

## ビジネスゴール
- 手作業によるデータ入力の工数削減
- 入力ミスの低減
- 処理時間の短縮
