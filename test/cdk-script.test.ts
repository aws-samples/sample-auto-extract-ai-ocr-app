import { chmodSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'fs';
import { tmpdir } from 'os';
import * as path from 'path';
import { spawnSync } from 'child_process';

const repoRoot = path.resolve(__dirname, '..');
const cdkScript = path.join(repoRoot, 'scripts', 'cdk.sh');

function writeExecutable(filePath: string, content: string): void {
  writeFileSync(filePath, content);
  chmodSync(filePath, 0o755);
}

describe('cdk wrapper stack preview', () => {
  let mockDir: string;
  let npxLog: string;

  beforeEach(() => {
    mockDir = mkdtempSync(path.join(tmpdir(), 'cdk-script-test-'));
    npxLog = path.join(mockDir, 'npx.log');

    writeExecutable(
      path.join(mockDir, 'npx'),
      `#!/bin/bash
set -euo pipefail
if [[ "$1" == "ts-node" ]]; then
  printf '%s\n' "$MOCK_STACK_PLAN"
  exit 0
fi
if [[ "$1" == "cdk" ]]; then
  echo "$*" >> "$MOCK_NPX_LOG"
  exit 0
fi
exit 64
`,
    );

    writeExecutable(
      path.join(mockDir, 'aws'),
      `#!/bin/bash
set -euo pipefail
if [[ "$1" == "sts" ]]; then
  printf '%s\n' '{"Account":"111111111111","Arn":"arn:aws:sts::111111111111:assumed-role/TestRole/test"}'
  exit 0
fi
if [[ "$1" == "cloudformation" && "$2" == "describe-stacks" ]]; then
  [[ "$MOCK_DESCRIBE_MODE" != "access-denied" ]] || {
    echo 'An error occurred (AccessDenied) when calling DescribeStacks' >&2
    exit 254
  }
  if [[ "$MOCK_DESCRIBE_MODE" == "delete-complete" ]]; then
    echo 'DELETE_COMPLETE'
    exit 0
  fi
  stack_name=""
  while [[ $# -gt 0 ]]; do
    [[ "$1" != "--stack-name" ]] || { stack_name="$2"; break; }
    shift
  done
  if [[ "$stack_name" == *-Waf ]]; then
    echo "An error occurred (ValidationError): Stack with id $stack_name does not exist" >&2
    exit 254
  fi
  echo 'UPDATE_COMPLETE'
  exit 0
fi
exit 64
`,
    );
  });

  afterEach(() => rmSync(mockDir, { recursive: true, force: true }));

  function run(mode = 'mixed') {
    return spawnSync(
      'bash',
      [cdkScript, 'deploy', 'base', '--region', 'ap-northeast-1', '-y'],
      {
        cwd: repoRoot,
        encoding: 'utf8',
        env: {
          ...process.env,
          PATH: `${mockDir}:${process.env.PATH ?? ''}`,
          MOCK_DESCRIBE_MODE: mode,
          MOCK_NPX_LOG: npxLog,
          MOCK_STACK_PLAN:
            'waf\tOcrAppStack-Waf\tus-east-1\napplication\tOcrAppStack\tap-northeast-1',
        },
      },
    );
  }

  test('shows CREATE for a missing WAF stack and UPDATE for an existing app stack', () => {
    const result = run();

    expect(result.status).toBe(0);
    expect(result.stdout).toMatch(
      /CREATE\s+OcrAppStack-Waf\s+\(us-east-1\) \[not found\]/,
    );
    expect(result.stdout).toMatch(
      /UPDATE\s+OcrAppStack\s+\(ap-northeast-1\) \[UPDATE_COMPLETE\]/,
    );
    expect(readFileSync(npxLog, 'utf8')).toContain('cdk deploy --all');
  });

  test('treats a DELETE_COMPLETE stack as CREATE', () => {
    const result = run('delete-complete');

    expect(result.status).toBe(0);
    expect(result.stdout).toMatch(
      /CREATE\s+OcrAppStack\s+\(ap-northeast-1\) \[DELETE_COMPLETE\]/,
    );
  });

  test('stops instead of treating an access error as CREATE', () => {
    const result = run('access-denied');

    expect(result.status).not.toBe(0);
    expect(result.stderr).toContain('存在確認に失敗しました');
    expect(result.stderr).toContain('AccessDenied');
    expect(() => readFileSync(npxLog, 'utf8')).toThrow();
  });
});
